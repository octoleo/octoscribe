"""
src/transcribe/backends/base.py — The transcription backend interface.

Defines the single, narrow contract every backend must satisfy.  Keeping the
interface tiny (one method plus an identifier) is a deliberate
Interface-Segregation choice: the orchestrator depends only on the ability to
turn an audio path into text, and nothing more.  New backends (cloud or local)
slot in by implementing this Strategy without the orchestrator changing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TranscriptionBackend(ABC):
    """Abstract Strategy for turning an audio file into verbatim text."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        """Transcribe the audio at *audio_path* verbatim and return raw text."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, stable identifier for this backend (e.g. ``"openai"``)."""
        ...
