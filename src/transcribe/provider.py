"""Structured, provider-neutral transcription evidence.

Backends may still return ``str`` for compatibility.  The ensemble coerces
those values into :class:`ProviderTranscript`; timestamp-capable providers can
return the structured value directly.  Neither path normalizes or rewrites the
provider's text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.transcribe.backends.base import TranscriptionBackend


@dataclass(frozen=True, slots=True)
class TimedWord:
    """One provider word with optional timing and confidence evidence."""

    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    confidence: float | None = None
    speaker: int | str | None = None

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("timed word text cannot be empty")
        if self.start_seconds is not None and self.start_seconds < 0:
            raise ValueError("word start cannot be negative")
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("word end cannot precede its start")


@dataclass(frozen=True, slots=True)
class ProviderTranscript:
    """Immutable output of one provider listening pass."""

    provider: str
    model: str
    text: str
    words: tuple[TimedWord, ...] = ()
    language: str | None = None
    duration_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider name cannot be empty")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("provider transcript cannot be empty")


def coerce_provider_transcript(
    value: str | ProviderTranscript,
    *,
    provider: str,
    model: str = "",
) -> ProviderTranscript:
    """Convert a legacy string result without changing its characters."""
    if isinstance(value, ProviderTranscript):
        if value.provider != provider:
            raise ValueError(
                f"backend {provider!r} returned evidence for {value.provider!r}"
            )
        return value
    if not isinstance(value, str):
        raise TypeError(
            f"backend {provider!r} returned {type(value).__name__}, expected text"
        )
    return ProviderTranscript(provider=provider, model=model, text=value)


def run_backend(
    backend: TranscriptionBackend,
    audio_path: Path,
    *,
    model: str = "",
    provider: str | None = None,
) -> ProviderTranscript:
    """Run a backend through its detailed or legacy compatibility surface."""
    canonical_provider = provider or backend.name
    detailed = getattr(backend, "transcribe_detailed", None)
    value = detailed(audio_path) if callable(detailed) else backend.transcribe(audio_path)
    if isinstance(value, ProviderTranscript) and value.provider != canonical_provider:
        # Registry aliases (notably legacy ``local`` -> canonical ``whisper``)
        # must not leak into provenance.  Preserve every provider field except
        # the configured identity used by the ensemble.
        return ProviderTranscript(
            provider=canonical_provider,
            model=value.model,
            text=value.text,
            words=value.words,
            language=value.language,
            duration_seconds=value.duration_seconds,
            metadata=value.metadata,
        )
    return coerce_provider_transcript(
        value,
        provider=canonical_provider,
        model=model,
    )
