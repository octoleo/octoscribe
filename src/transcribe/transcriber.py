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
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.audio import unique_filepath
from src.config import Config, TranscribeConfig
from src.manifest import Manifest
from src.persistence import PeriodicSaver, atomic_write_text
from src.transcribe.backends.base import TranscriptionBackend
from src.transcribe.backends.local_whisper import LocalWhisperBackend
from src.transcribe.backends.openai_backend import OpenAIBackend
from src.transcribe.normalize import normalize_text
from src.transcribe.results import BatchStats, TranscriptionResult

log = logging.getLogger(__name__)

# How many files are processed between periodic manifest saves.
_SAVE_INTERVAL: int = 5


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
    ) -> None:
        self._config = config
        self._manifest = manifest
        self._backend: TranscriptionBackend = (
            backend if backend is not None
            else self.create_backend(config.transcribe)
        )

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
        pending = self._manifest.pending_transcription()

        if not pending:
            log.info("No files pending transcription.")
            return stats

        tcfg = self._config.transcribe
        audio_dir = self._config.download.audio_dir
        out_dir = tcfg.transcriptions_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "Starting transcription of %d file(s) with backend '%s'",
            len(pending),
            self._backend.name,
        )

        # Local backend cannot be parallelised easily (single GPU).
        use_parallel = self._backend.name == "openai" and tcfg.workers > 1

        if use_parallel:
            self._run_parallel(pending, audio_dir, out_dir, stats)
        else:
            self._run_sequential(pending, audio_dir, out_dir, stats)

        self._manifest.save()
        log.info("%s", stats.summary())
        return stats

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

        audio_path = audio_dir / filename
        if not audio_path.exists():
            log.warning(
                "Audio file not found, skipping msg_id=%s: %s", msg_id, audio_path
            )
            return None

        # Collision-safe output path: if two recordings derive the same name,
        # disambiguate the later one with the message id rather than overwrite.
        desired_name = self._output_filename(audio_path, title)
        output_path = unique_filepath(out_dir, desired_name, msg_id)
        output_filename = output_path.name

        t0 = time.monotonic()
        try:
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

            atomic_write_text(output_path, text)

            self._manifest.mark_transcribed(
                msg_id,
                {
                    "output_file": output_filename,
                    "model": self._backend.name,
                },
            )

            log.info(
                "Transcribed %s -> %s (%.1fs)", filename, output_filename, elapsed
            )
            return TranscriptionResult(
                msg_id=msg_id,
                filename=filename,
                success=True,
                output_file=output_filename,
                text=text,
                elapsed_seconds=elapsed,
                model=self._backend.name,
            )

        except Exception as exc:
            elapsed = time.monotonic() - t0
            error_str = str(exc)
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
                model=self._backend.name,
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
