"""
src/transcribe.py — Verbatim transcription pipeline for OctoScribe.

Strategy pattern: OpenAIBackend and LocalWhisperBackend share the
TranscriptionBackend interface.  Transcriber orchestrates batch processing
against a Manifest, writing .txt files and updating manifest state.

Critical requirement: transcriptions must be VERBATIM — every word exactly
as spoken, nothing added or removed.
"""

from __future__ import annotations

import logging
import os
import random
import re
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import Config, TranscribeConfig
from src.manifest import Manifest

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verbatim transcription prompt — DO NOT ALTER.
# ---------------------------------------------------------------------------

VERBATIM_PROMPT = (
    "Transcribe EXACTLY what is spoken, word for word. "
    "Do NOT add, remove, or change any words. "
    "Do NOT correct grammar or spelling. "
    "Do NOT paraphrase or rephrase anything. "
    "Preserve every repetition exactly as spoken. "
    "Add only standard punctuation and capitalization — nothing else. "
    "Do NOT add headings, labels, speaker names, or any text not spoken."
)

# Pause threshold (seconds) that triggers a paragraph break in local output.
_PARAGRAPH_BREAK_SECONDS: float = 2.0

# How many files are processed between periodic manifest saves.
_SAVE_INTERVAL: int = 5


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------

class TranscriptionBackend(ABC):
    """Abstract base for transcription backends."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        """Transcribe audio file verbatim.  Returns raw text."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier string."""
        ...


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class OpenAIBackend(TranscriptionBackend):
    """
    OpenAI audio transcription using gpt-4o-transcribe (or whisper-1).
    Sends the VERBATIM_PROMPT.  Retries on transient failures.
    """

    # Error patterns that indicate transient (retryable) failures.
    _RETRYABLE_PATTERNS: tuple[str, ...] = (
        "rate limit",
        "rate_limit",
        "429",
        "too many requests",
        "connection",
        "network",
        "dns",
        "unreachable",
        "reset by peer",
        "timeout",
        "timed out",
        "deadline",
        "500",
        "502",
        "503",
        "504",
        "internal server error",
        "bad gateway",
        "service unavailable",
    )

    # Error patterns that indicate permanent (non-retryable) failures.
    _PERMANENT_PATTERNS: tuple[str, ...] = (
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "invalid api key",
        "authentication",
        "invalid audio",
        "unsupported format",
        "corrupt",
        "could not process",
        "invalid file",
    )

    def __init__(self, config: TranscribeConfig) -> None:
        import openai

        self._config = config
        self._client = openai.OpenAI(api_key=config.api_key)

    @property
    def name(self) -> str:
        return "openai"

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe *audio_path* via OpenAI, retrying on transient errors."""
        cfg = self._config
        last_error: Exception | None = None

        for attempt in range(cfg.retry_attempts + 1):
            try:
                with open(audio_path, "rb") as audio_file:
                    result = self._client.audio.transcriptions.create(
                        file=audio_file,
                        model=cfg.model,
                        language=cfg.language,
                        prompt=VERBATIM_PROMPT,
                        response_format="text",
                    )
                # response_format="text" returns a plain string directly.
                return str(result)

            except Exception as exc:
                error_str = str(exc).lower()

                # Permanent failures: raise immediately without retrying.
                if any(pat in error_str for pat in self._PERMANENT_PATTERNS):
                    raise

                # Retryable failures: back off and try again.
                if any(pat in error_str for pat in self._RETRYABLE_PATTERNS):
                    last_error = exc
                    if attempt >= cfg.retry_attempts:
                        break
                    delay = min(
                        cfg.retry_max_delay,
                        cfg.retry_base_delay * (2 ** attempt),
                    )
                    delay *= 0.85 + random.random() * 0.3  # jitter
                    log.warning(
                        "Transient error on attempt %d/%d for %s: %s — "
                        "retrying in %.1fs",
                        attempt + 1,
                        cfg.retry_attempts + 1,
                        audio_path.name,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                # Unknown error: treat as permanent.
                raise

        raise RuntimeError(
            f"Exhausted {cfg.retry_attempts} retries for {audio_path.name}"
        ) from last_error


# ---------------------------------------------------------------------------
# Local Whisper backend
# ---------------------------------------------------------------------------

class LocalWhisperBackend(TranscriptionBackend):
    """
    Local transcription using faster-whisper.
    temperature=0 and condition_on_previous_text=False for maximum faithfulness.
    """

    def __init__(self, config: TranscribeConfig) -> None:
        self._config = config
        self._setup_cuda()

        import faster_whisper  # noqa: PLC0415 — must be after CUDA setup

        self._model = faster_whisper.WhisperModel(
            config.local_model,
            device=config.device,
            compute_type=config.compute_type,
        )

    @property
    def name(self) -> str:
        return "local"

    def _setup_cuda(self) -> None:
        """
        Auto-configure CUDA library paths before faster_whisper is imported.
        Mirrors the logic in the legacy transcribe_local.py.
        """
        try:
            import site  # noqa: PLC0415

            site_packages_dirs: list[str] = list(site.getsitepackages())
            if hasattr(site, "getusersitepackages"):
                site_packages_dirs.append(site.getusersitepackages())

            if hasattr(sys, "prefix"):
                venv_site = Path(sys.prefix) / "lib"
                for pydir in venv_site.glob("python*/site-packages"):
                    site_packages_dirs.append(str(pydir))

            lib_paths: list[str] = []
            for sp_dir in site_packages_dirs:
                sp_path = Path(sp_dir)

                for nvidia_lib in ("cudnn", "cublas"):
                    candidate = sp_path / "nvidia" / nvidia_lib / "lib"
                    if candidate.exists():
                        lib_paths.append(str(candidate))

                ct2_libs = sp_path / "ctranslate2.libs"
                if ct2_libs.exists():
                    lib_paths.append(str(ct2_libs))

            if lib_paths:
                current = os.environ.get("LD_LIBRARY_PATH", "")
                joined = ":".join(lib_paths)
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{joined}:{current}" if current else joined
                )

        except Exception as exc:  # pragma: no cover
            log.warning("Could not auto-configure CUDA paths: %s", exc)

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe *audio_path* locally with faster-whisper."""
        cfg = self._config

        vad_params: dict | None = None
        if cfg.vad_filter:
            vad_params = {
                "min_silence_duration_ms": cfg.vad_min_silence_ms,
                "speech_pad_ms": cfg.vad_speech_pad_ms,
            }

        segments_iter, _info = self._model.transcribe(
            str(audio_path),
            beam_size=cfg.beam_size,
            best_of=cfg.best_of,
            # temperature=0 is MANDATORY for verbatim faithfulness.
            temperature=0,
            language=cfg.language,
            condition_on_previous_text=False,
            vad_filter=cfg.vad_filter,
            vad_parameters=vad_params,
            repetition_penalty=cfg.repetition_penalty,
            word_timestamps=False,
        )

        # Materialise the generator so we can measure timing gaps.
        segments = list(segments_iter)
        return self._format_segments(segments)

    @staticmethod
    def _format_segments(segments: list) -> str:
        """
        Join segments into text.  Insert a blank-line paragraph break
        wherever the gap between consecutive segments exceeds
        _PARAGRAPH_BREAK_SECONDS.
        """
        lines: list[str] = []
        current_paragraph: list[str] = []
        last_end: float = 0.0

        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue

            if current_paragraph and (seg.start - last_end) > _PARAGRAPH_BREAK_SECONDS:
                lines.append(" ".join(current_paragraph))
                lines.append("")
                current_paragraph = []

            current_paragraph.append(text)
            last_end = seg.end

        if current_paragraph:
            lines.append(" ".join(current_paragraph))

        return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Result / statistics dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionResult:
    msg_id: str
    filename: str
    success: bool
    output_file: str = ""
    text: str = ""
    error: str = ""
    elapsed_seconds: float = 0.0
    model: str = ""


@dataclass
class BatchStats:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    total_elapsed_seconds: float = 0.0

    def add(self, result: TranscriptionResult) -> None:
        """Incorporate a single TranscriptionResult into the running totals."""
        self.total += 1
        self.total_elapsed_seconds += result.elapsed_seconds
        if result.success:
            self.succeeded += 1
        else:
            self.failed += 1

    def summary(self) -> str:
        """Return a human-readable one-line summary of the batch."""
        return (
            f"Transcription complete — "
            f"total={self.total} "
            f"succeeded={self.succeeded} "
            f"failed={self.failed} "
            f"skipped={self.skipped} "
            f"elapsed={self.total_elapsed_seconds:.1f}s"
        )


# ---------------------------------------------------------------------------
# Text normalisation (minimal — words are never touched)
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """
    Minimal post-processing that does NOT alter any spoken words:
    - Normalise line endings to \\n
    - Strip trailing whitespace per line
    - Cap consecutive blank lines at 1
    """
    if not text:
        return ""
    # Normalise CR/CRLF.
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace per line.
    out = "\n".join(line.rstrip() for line in out.split("\n"))
    # Cap consecutive blank lines at 1.
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------

class Transcriber:
    """
    Orchestrates batch transcription of pending audio files.

    Reads pending items from Manifest, writes .txt files to
    transcriptions_dir, and updates the Manifest with results.
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
        processed_since_save = 0
        for entry in pending:
            result = self._process_entry(entry, audio_dir, out_dir)
            if result is None:
                stats.skipped += 1
                continue
            stats.add(result)
            processed_since_save += 1
            if processed_since_save >= _SAVE_INTERVAL:
                self._manifest.save()
                processed_since_save = 0

    def _run_parallel(
        self,
        pending: list[dict],
        audio_dir: Path,
        out_dir: Path,
        stats: BatchStats,
    ) -> None:
        tcfg = self._config.transcribe
        processed_since_save = 0

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
                processed_since_save += 1
                if processed_since_save >= _SAVE_INTERVAL:
                    self._manifest.save()
                    processed_since_save = 0

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

        Returns None when the audio file is missing (skip, not failure).
        Returns a TranscriptionResult for both success and failure.
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

        output_filename = self._output_filename(audio_path, title)
        output_path = out_dir / output_filename

        t0 = time.monotonic()
        try:
            raw_text = self._backend.transcribe(audio_path)
            text = _normalize_text(raw_text)
            elapsed = time.monotonic() - t0

            self._atomic_write(output_path, text)

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
        Derive the output .txt filename.

        Uses the sanitised title when available; falls back to the audio
        file stem.
        """
        if title:
            safe = re.sub(r'[<>:"/\\|?*]', "_", title)
            safe = re.sub(r"[\x00-\x1f\x7f]", "", safe)
            safe = safe.strip(" .")
            if safe:
                return safe + ".txt"
        return audio_path.stem + ".txt"

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        """Write *text* to *path* atomically (tmp → rename)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp_path.write_text(text, encoding="utf-8", newline="\n")
            os.replace(tmp_path, path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
