"""
src/transcribe — Verbatim transcription pipeline for OctoScribe.

This package replaces the former single ``src/transcribe.py`` module, split by
responsibility:

* :mod:`src.transcribe.prompt`            — the verbatim instruction string.
* :mod:`src.transcribe.normalize`         — whitespace-only text normalisation.
* :mod:`src.transcribe.results`           — result/statistics value objects.
* :mod:`src.transcribe.chunking`          — deterministic chunk/seam policy.
* :mod:`src.transcribe.audio_chunks`      — ffmpeg extraction and hashing.
* :mod:`src.transcribe.backends`          — the Strategy interface + backends:
    * :mod:`~src.transcribe.backends.base`          — the interface.
    * :mod:`~src.transcribe.backends.retry`         — retry policy + classifier.
    * :mod:`~src.transcribe.backends.openai_backend`
    * :mod:`~src.transcribe.backends.local_whisper`
* :mod:`src.transcribe.transcriber`       — the batch orchestrator + factory.

Design: backends share the :class:`TranscriptionBackend` Strategy; the
:class:`Transcriber` orchestrates batch processing against a
:class:`~src.manifest.Manifest`, writing ``.txt`` files and updating manifest
state.

Critical requirement: transcriptions must be VERBATIM — every word exactly as
spoken, nothing added or removed.

Everything previously importable from ``src/transcribe.py`` is re-exported here
(including the private ``_normalize_text``) so existing imports keep working.
"""

from __future__ import annotations

from src.transcribe.audio_chunks import (
    AudioToolError,
    FFmpegAudioTools,
    MaterializedChunk,
)
from src.transcribe.backends.base import TranscriptionBackend
from src.transcribe.backends.local_whisper import LocalWhisperBackend
from src.transcribe.backends.meta_backend import MetaASRBackend
from src.transcribe.backends.openai_backend import OpenAIBackend
from src.transcribe.backends.registry import (
    CANONICAL_PROVIDERS,
    build_backend_registry,
    create_backend_registry,
    provider_model_name,
)
from src.transcribe.backends.xai_backend import XAIBackend
from src.transcribe.chunking import (
    ChunkMetadata,
    OverlapAlignment,
    StitchResult,
    find_overlap_alignment,
    plan_chunks,
    stitch_transcripts,
    stitch_with_alignment,
)
from src.transcribe.normalize import _normalize_text, normalize_text
from src.transcribe.prompt import VERBATIM_PROMPT
from src.transcribe.results import BatchStats, TranscriptionResult
from src.transcribe.transcriber import Transcriber

__all__ = [
    "VERBATIM_PROMPT",
    "TranscriptionBackend",
    "OpenAIBackend",
    "XAIBackend",
    "MetaASRBackend",
    "LocalWhisperBackend",
    "CANONICAL_PROVIDERS",
    "build_backend_registry",
    "create_backend_registry",
    "provider_model_name",
    "AudioToolError",
    "FFmpegAudioTools",
    "MaterializedChunk",
    "ChunkMetadata",
    "OverlapAlignment",
    "StitchResult",
    "find_overlap_alignment",
    "plan_chunks",
    "stitch_transcripts",
    "stitch_with_alignment",
    "BatchStats",
    "TranscriptionResult",
    "Transcriber",
    "normalize_text",
    "_normalize_text",
]
