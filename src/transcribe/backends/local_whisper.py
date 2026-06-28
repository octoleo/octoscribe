"""
src/transcribe/backends/local_whisper.py — Local Faster-Whisper backend.

Runs the ``large-v3`` Whisper model locally via CTranslate2.  Decoding settings
are pinned for maximum faithfulness: ``temperature=0`` and
``condition_on_previous_text=False`` so the model never "improves" or
hallucinates beyond what was spoken.

CUDA library discovery is handled before ``faster_whisper`` is imported, because
the import itself triggers native library loading that needs the correct
``LD_LIBRARY_PATH``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from src.config import TranscribeConfig
from src.transcribe.backends.base import TranscriptionBackend

log = logging.getLogger(__name__)

#: Pause (seconds) between segments that triggers a paragraph break in output.
_PARAGRAPH_BREAK_SECONDS: float = 2.0


class LocalWhisperBackend(TranscriptionBackend):
    """
    Local transcription using faster-whisper.

    ``temperature=0`` and ``condition_on_previous_text=False`` are mandatory for
    verbatim faithfulness and must not be changed.
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

        Scans site-packages for the NVIDIA cuDNN/cuBLAS and CTranslate2 shared
        libraries shipped as wheels and prepends them to ``LD_LIBRARY_PATH`` so
        the native runtime can find them.  Any failure here is non-fatal: it is
        logged and the import is attempted regardless.
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
        Join segments into text, inserting a blank-line paragraph break wherever
        the gap between consecutive segments exceeds
        :data:`_PARAGRAPH_BREAK_SECONDS`.
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
