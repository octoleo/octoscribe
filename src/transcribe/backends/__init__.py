"""
src/transcribe/backends — Transcription backend strategies.

Re-exports the backend interface, the two concrete implementations, and the
retry primitives so callers can import them from one cohesive place.
"""

from __future__ import annotations

from src.transcribe.backends.base import TranscriptionBackend
from src.transcribe.backends.local_whisper import LocalWhisperBackend
from src.transcribe.backends.openai_backend import OpenAIBackend
from src.transcribe.backends.retry import ErrorClassifier, RetryPolicy

__all__ = [
    "TranscriptionBackend",
    "OpenAIBackend",
    "LocalWhisperBackend",
    "RetryPolicy",
    "ErrorClassifier",
]
