"""
src/transcribe/transcriber.py — Batch transcription orchestrator.

:class:`Transcriber` is the only object that knows the *workflow*: read the
pending items from the manifest, run them through a backend, write transcripts,
and record the outcome.  It depends on the small
:class:`~src.transcribe.backends.base.TranscriptionBackend` interface (not on any
concrete backend), so swapping OpenAI for local Whisper changes nothing here.

Two robustness guarantees were added when this orchestrator was extracted, both
aimed at never silently losing a transcript:

* **Collision-safe output** — two recordings that derive the same output name
  no longer clobber each other; the second is disambiguated by message id.
* **Empty-result guard** — a blank transcript is treated as a *failure* (so it
  is retried on the next run) rather than written out as a "successful" but
  empty file.
"""

from __future__ import annotations

import logging
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.audio import AUDIO_EXTENSIONS, sha256_file, unique_filepath
from src.config import Config, TranscribeConfig
from src.manifest import Manifest
from src.persistence import PeriodicSaver, atomic_write_text
from src.transcribe.backends.base import TranscriptionBackend
from src.transcribe.backends.local_whisper import LocalWhisperBackend
from src.transcribe.backends.openai_backend import OpenAIBackend
from src.transcribe.backends.registry import (
    create_backend_registry,
    provider_model_name,
)
from src.transcribe.ensemble import EnsembleEngine
from src.transcribe.evidence import EvidenceStore
from src.transcribe.normalize import normalize_text
from src.transcribe.results import BatchStats, TranscriptionResult

log = logging.getLogger(__name__)

# Persist every paid transcription outcome before starting the next item.  A
# final save still runs at the end, but interval=1 prevents a crash late in a
# long batch from losing several completed (or failed) API calls.
_SAVE_INTERVAL: int = 1


class Transcriber:
    """
    Orchestrates batch transcription of pending audio files.

    Reads pending items from :class:`~src.manifest.Manifest`, writes ``.txt``
    files to ``transcriptions_dir``, and records each outcome back in the
    manifest.
    """

    def __init__(
        self,
        config: Config,
        manifest: Manifest,
        backend: TranscriptionBackend | None = None,
        *,
        audio_revision: str | None = None,
        audio_repository_branch: str | None = None,
    ) -> None:
        self._config = config
        self._manifest = manifest
        self._audio_revision = audio_revision
        self._audio_repository_branch = audio_repository_branch
        self._ensemble: EnsembleEngine | None = None
        self._model_names: dict[str, str] = {}
        if backend is not None:
            # Explicit backend injection retains the historical single-call
            # surface used by embedders and the compatibility test suite.
            self._backend = backend
        elif getattr(config.transcribe, "providers", ()):
            registry = create_backend_registry(config.transcribe)
            primary = config.transcribe.primary_provider
            self._backend = registry[primary]
            model_names = {
                name: provider_model_name(config.transcribe, name)
                for name in registry
            }
            self._model_names = model_names
            artifact_dir = (
                config.transcribe.artifacts_dir
                or config.transcribe.transcriptions_dir.parent / "candidates"
            )
            report_dir = (
                config.transcribe.reports_dir
                or config.transcribe.transcriptions_dir.parent / "reports"
            )
            self._ensemble = EnsembleEngine(
                config.transcribe,
                registry,
                model_names=model_names,
                evidence_store=EvidenceStore(artifact_dir, report_dir),
            )
        else:
            self._backend = self.create_backend(config.transcribe)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def create_backend(config: TranscribeConfig) -> TranscriptionBackend:
        """Factory: create the appropriate backend from config."""
        if config.backend == "openai":
            return OpenAIBackend(config)
        if config.backend == "local":
            return LocalWhisperBackend(config)
        raise ValueError(
            f"Unknown transcription backend {config.backend!r}. "
            "Expected 'openai' or 'local'."
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> BatchStats:
        """Transcribe all pending files.  Returns batch statistics."""
        stats = BatchStats()
        out_dir = self._config.transcribe.transcriptions_dir
        pending = self._reconcile_pending_transcriptions(out_dir)

        if not pending:
            log.info("No files pending transcription.")
            return stats

        tcfg = self._config.transcribe
        audio_dir = self._config.download.audio_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        provider_display = (
            ",".join(self._config.transcribe.providers)
            if self._ensemble is not None
            else self._backend.name
        )
        log.info(
            "Starting transcription of %d file(s) with provider(s) '%s'",
            len(pending),
            provider_display,
        )

        # Local backend cannot be parallelised easily (single GPU).
        use_parallel = (
            self._ensemble is None
            and self._backend.name == "openai"
            and tcfg.workers > 1
        )

        if use_parallel:
            self._run_parallel(pending, audio_dir, out_dir, stats)
        else:
            self._run_sequential(pending, audio_dir, out_dir, stats)

        self._manifest.save()
        log.info("%s", stats.summary())
        return stats

    def _reconcile_pending_transcriptions(self, out_dir: Path) -> list[dict]:
        """Return unfinished entries plus terminal entries whose text is absent.

        A manifest status alone is not enough to skip paid work forever.  The
        recorded output must still be a safe regular file and, when a
        transcript hash is present, its bytes must still match that digest.
        """
        pending: list[dict] = []
        for entry in self._manifest.pending_transcription():
            if entry.get("duplicate") is True:
                log.info(
                    "Skipping duplicate transcription: msg_id=%s canonical_msg_id=%s",
                    entry.get("telegram_msg_id") or entry.get("msg_id", ""),
                    entry.get("duplicate_of"),
                )
                continue
            pending.append(entry)
        pending_ids = {
            str(entry.get("telegram_msg_id") or entry.get("msg_id", ""))
            for entry in pending
        }
        for entry in self._manifest.all_entries().values():
            if not (entry.get("downloaded") and entry.get("filename")):
                continue
            msg_id = str(entry.get("telegram_msg_id") or entry.get("msg_id", ""))
            if entry.get("duplicate") is True:
                log.info(
                    "Skipping duplicate transcription: msg_id=%s canonical_msg_id=%s",
                    msg_id,
                    entry.get("duplicate_of"),
                )
                continue
            if msg_id in pending_ids:
                continue
            intact, reason = self._transcript_output_is_intact(entry, out_dir)
            if intact:
                transcription = entry.get("transcription") or {}
                log.info(
                    "Skipping completed transcription: msg_id=%s output=%s status=%s",
                    msg_id,
                    transcription.get("output_file"),
                    transcription.get("status"),
                )
                continue
            log.warning(
                "Re-queueing transcription: msg_id=%s reason=%s",
                msg_id,
                reason,
            )
            pending.append(dict(entry))
            pending_ids.add(msg_id)
        return pending

    @staticmethod
    def _transcript_output_is_intact(
        entry: dict, out_dir: Path
    ) -> tuple[bool, str]:
        transcription = entry.get("transcription")
        if not isinstance(transcription, dict):
            return False, "manifest has no transcription result"
        output_file = transcription.get("output_file")
        if not isinstance(output_file, str) or not output_file:
            return False, "manifest has no output_file"
        relative = Path(output_file)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in output_file
        ):
            return False, "manifest output_file is not a safe relative path"
        candidate = out_dir / relative
        if candidate.is_symlink() or not candidate.is_file():
            return False, "recorded transcript file is missing"
        try:
            candidate.resolve(strict=True).relative_to(out_dir.resolve())
        except (OSError, ValueError):
            return False, "recorded transcript escapes the output directory"
        recorded_hash = transcription.get("transcript_sha256")
        if isinstance(recorded_hash, str) and len(recorded_hash) == 64:
            if hashlib.sha256(candidate.read_bytes()).hexdigest() != recorded_hash:
                return False, "recorded transcript SHA-256 does not match"
        return True, ""

    # ------------------------------------------------------------------
    # Execution modes
    # ------------------------------------------------------------------

    def _run_sequential(
        self,
        pending: list[dict],
        audio_dir: Path,
        out_dir: Path,
        stats: BatchStats,
    ) -> None:
        saver = PeriodicSaver(self._manifest, interval=_SAVE_INTERVAL)
        for entry in pending:
            result = self._process_entry(entry, audio_dir, out_dir)
            if result is None:
                stats.skipped += 1
                continue
            stats.add(result)
            saver.tick()

    def _run_parallel(
        self,
        pending: list[dict],
        audio_dir: Path,
        out_dir: Path,
        stats: BatchStats,
    ) -> None:
        tcfg = self._config.transcribe
        saver = PeriodicSaver(self._manifest, interval=_SAVE_INTERVAL)

        with ThreadPoolExecutor(max_workers=tcfg.workers) as executor:
            future_to_entry = {
                executor.submit(self._process_entry, entry, audio_dir, out_dir): entry
                for entry in pending
            }
            for future in as_completed(future_to_entry):
                result = future.result()
                if result is None:
                    stats.skipped += 1
                    continue
                stats.add(result)
                saver.tick()

    # ------------------------------------------------------------------
    # Per-entry processing
    # ------------------------------------------------------------------

    def _process_entry(
        self,
        entry: dict,
        audio_dir: Path,
        out_dir: Path,
    ) -> TranscriptionResult | None:
        """
        Transcribe a single manifest entry.

        Returns ``None`` when the audio file is missing (skip, not failure).
        Returns a :class:`TranscriptionResult` for both success and failure.
        """
        msg_id: str = str(entry.get("telegram_msg_id") or entry.get("msg_id", ""))
        filename: str = entry.get("filename", "")
        title: str | None = entry.get("title")
        result_model = self._model_names.get(
            getattr(self._config.transcribe, "primary_provider", ""),
            self._backend.name,
        )

        t0 = time.monotonic()
        try:
            # Manifest filenames are untrusted input.  A tampered manifest must
            # never make a cloud backend upload a credential or arbitrary host
            # file via an absolute path, traversal, or symlink.
            relative_audio = Path(filename)
            if (
                not filename
                or relative_audio.is_absolute()
                or len(relative_audio.parts) != 1
                or "/" in filename
                or "\\" in filename
            ):
                raise RuntimeError(
                    "invalid manifest audio filename; expected one relative basename"
                )
            if relative_audio.suffix.lower() not in AUDIO_EXTENSIONS:
                raise RuntimeError(
                    "invalid manifest audio extension; refusing non-audio upload"
                )

            candidate = audio_dir / relative_audio
            if candidate.is_symlink():
                raise RuntimeError(
                    "manifest audio file is a symlink; refusing external upload"
                )
            if not candidate.exists():
                log.warning(
                    "Audio file not found, skipping msg_id=%s: %s",
                    msg_id,
                    candidate,
                )
                return None
            if not candidate.is_file():
                raise RuntimeError("manifest audio path is not a regular file")

            audio_dir_root = audio_dir.resolve()
            audio_path = candidate.resolve(strict=True)
            try:
                audio_path.relative_to(audio_dir_root)
            except ValueError as exc:
                raise RuntimeError(
                    "manifest audio file escapes the configured audio directory"
                ) from exc

            audio_repo = getattr(self._config, "audio_repo", None)
            audio_repo_root = Path(
                getattr(audio_repo, "path", audio_dir.parent)
            ).resolve()
            try:
                logical_audio_path = audio_path.relative_to(audio_repo_root)
            except ValueError as exc:
                raise RuntimeError(
                    "configured audio directory escapes the audio evidence repository"
                ) from exc

            # Collision-safe output path: if two recordings derive the same
            # name, disambiguate the later one rather than overwrite it.
            desired_name = self._output_filename(audio_path, title)
            actual_audio_hash = sha256_file(audio_path)
            log.info(
                "Processing audio evidence: msg_id=%s file=%s sha256=%s bytes=%d",
                msg_id,
                audio_path,
                actual_audio_hash,
                audio_path.stat().st_size,
            )
            recorded_hash = entry.get("hash")
            if (
                isinstance(recorded_hash, str)
                and len(recorded_hash) == 64
                and recorded_hash.lower() != actual_audio_hash
            ):
                raise RuntimeError(
                    "audio SHA-256 mismatch; refusing to transcribe changed evidence"
                )
            if not (
                isinstance(recorded_hash, str)
                and len(recorded_hash) == 64
                and recorded_hash == actual_audio_hash
            ):
                self._manifest.record_audio_hash(msg_id, actual_audio_hash)

            ensemble_outcome = None
            if self._ensemble is not None:
                ensemble_kwargs: dict[str, object] = {
                    "expected_sha256": actual_audio_hash,
                    "logical_audio_path": logical_audio_path,
                }
                if self._audio_revision:
                    ensemble_kwargs["audio_revision"] = self._audio_revision
                if self._audio_repository_branch:
                    ensemble_kwargs[
                        "audio_repository_branch"
                    ] = self._audio_repository_branch
                ensemble_outcome = self._ensemble.transcribe(
                    audio_path,
                    **ensemble_kwargs,
                )
                raw_text = ensemble_outcome.text
            else:
                raw_text = self._backend.transcribe(audio_path)
            text = normalize_text(raw_text)
            elapsed = time.monotonic() - t0

            # Empty-result guard: an empty transcript is almost always a backend
            # hiccup, not a genuinely silent recording.  Fail it so the next run
            # retries rather than recording a misleading "completed" empty file.
            if not text.strip():
                raise RuntimeError(
                    "transcription produced empty output (no speech recognised)"
                )

            quality_state = (
                ensemble_outcome.quality_state.value
                if ensemble_outcome is not None
                else ""
            )
            target_dir = (
                out_dir / "needs-review"
                if quality_state == "needs_review"
                else out_dir
            )
            target_dir.mkdir(parents=True, exist_ok=True)
            output_path = unique_filepath(target_dir, desired_name, msg_id)
            output_filename = str(output_path.relative_to(out_dir))
            text_repo = getattr(self._config, "text_repo", None)
            text_repo_root = Path(
                getattr(text_repo, "path", out_dir.parent)
            ).resolve()
            try:
                logical_output_path = output_path.resolve().relative_to(text_repo_root)
            except ValueError as exc:
                raise RuntimeError(
                    "configured transcription directory escapes the text evidence repository"
                ) from exc
            atomic_write_text(output_path, text)

            manifest_result: dict[str, object] = {
                "output_file": output_filename,
                "output_path": logical_output_path.as_posix(),
                "audio_path": logical_audio_path.as_posix(),
                "model": result_model,
                "transcript_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
            }
            evidence_report = ""
            unresolved = 0
            if ensemble_outcome is not None:
                quality_state = ensemble_outcome.quality_state.value
                if ensemble_outcome.evidence_report_path:
                    report_path = Path(ensemble_outcome.evidence_report_path)
                    report_root = getattr(
                        getattr(self._config, "text_repo", None),
                        "path",
                        self._config.transcribe.transcriptions_dir.parent,
                    )
                    try:
                        evidence_report = str(report_path.relative_to(report_root))
                    except ValueError:
                        evidence_report = str(report_path)
                unresolved = ensemble_outcome.unresolved_discrepancies
                ensemble_manifest: dict[str, object] = {
                    "quality_state": quality_state,
                    "providers": list(self._config.transcribe.providers),
                    "models": dict(self._model_names),
                    "primary_provider": self._config.transcribe.primary_provider,
                    "audio_sha256": ensemble_outcome.audio_sha256,
                    "duration_ms": ensemble_outcome.duration_ms,
                    "evidence_report": evidence_report,
                    "unresolved_discrepancies": unresolved,
                }
                if self._audio_revision:
                    ensemble_manifest["audio_revision"] = self._audio_revision
                if self._audio_repository_branch:
                    ensemble_manifest[
                        "audio_repository_branch"
                    ] = self._audio_repository_branch
                provider_failures = [
                    {
                        "provider": failure.provider,
                        "attempt": failure.attempt,
                        "role": failure.role,
                        "error": failure.error,
                    }
                    for failure in getattr(
                        ensemble_outcome, "provider_failures", ()
                    )
                ]
                if provider_failures:
                    ensemble_manifest["provider_failures"] = provider_failures
                manifest_result.update(ensemble_manifest)
            self._manifest.mark_transcribed(msg_id, manifest_result)

            log.info(
                "Transcription result: input=%s output=%s quality=%s "
                "unresolved_discrepancies=%d report=%s elapsed=%.1fs",
                filename,
                output_filename,
                quality_state or "machine_transcribed",
                unresolved,
                evidence_report or "(none)",
                elapsed,
            )
            return TranscriptionResult(
                msg_id=msg_id,
                filename=filename,
                success=True,
                output_file=output_filename,
                text=text,
                elapsed_seconds=elapsed,
                model=result_model,
                quality_state=quality_state,
                evidence_report=evidence_report,
                unresolved_discrepancies=unresolved,
            )

        except Exception as exc:
            elapsed = time.monotonic() - t0
            error_str = str(exc)
            for secret in (
                getattr(self._config.transcribe, "api_key", None),
                getattr(self._config.transcribe, "xai_api_key", None),
                getattr(self._config.transcribe, "meta_asr_api_key", None),
            ):
                if secret:
                    error_str = error_str.replace(str(secret), "***")
            error_str = " ".join(error_str.split())[:500]
            log.error(
                "Transcription failed for %s (msg_id=%s): %s",
                filename,
                msg_id,
                error_str,
            )
            self._manifest.mark_failed(msg_id, "transcription", error_str)
            return TranscriptionResult(
                msg_id=msg_id,
                filename=filename,
                success=False,
                error=error_str,
                elapsed_seconds=elapsed,
                model=result_model,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _output_filename(audio_path: Path, title: str | None) -> str:
        """
        Derive the output ``.txt`` filename.

        Uses the sanitised title when available; falls back to the audio file
        stem.  (Uniqueness against existing files is handled separately by the
        caller via :func:`src.audio.unique_filepath`.)
        """
        if title:
            safe = re.sub(r'[<>:"/\\|?*]', "_", title)
            safe = re.sub(r"[\x00-\x1f\x7f]", "", safe)
            safe = safe.strip(" .")
            if safe:
                return safe + ".txt"
        return audio_path.stem + ".txt"
