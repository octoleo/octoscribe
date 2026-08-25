"""
src/transcribe/backends — Transcription backend strategies.

Re-exports the backend interface, concrete implementations, ordered registry,
and retry primitives so callers can import them from one cohesive place.
"""

from __future__ import annotations

from src.transcribe.backends.base import TranscriptionBackend
from src.transcribe.backends.local_whisper import LocalWhisperBackend
from src.transcribe.backends.meta_backend import MetaASRBackend
from src.transcribe.backends.openai_backend import OpenAIBackend
from src.transcribe.backends.registry import (
    CANONICAL_PROVIDERS,
    BackendFactory,
    build_backend_registry,
    create_backend_registry,
    provider_model_name,
)
from src.transcribe.backends.retry import ErrorClassifier, RetryPolicy
from src.transcribe.backends.xai_backend import XAIBackend

__all__ = [
    "TranscriptionBackend",
    "OpenAIBackend",
    "XAIBackend",
    "MetaASRBackend",
    "LocalWhisperBackend",
    "CANONICAL_PROVIDERS",
    "BackendFactory",
    "build_backend_registry",
    "create_backend_registry",
    "provider_model_name",
    "RetryPolicy",
    "ErrorClassifier",
]
