"""Ordered construction of enabled transcription backends.

Configuration parsing canonicalises provider aliases before this layer.  This
module consequently accepts only the stable provenance names ``openai``,
``xai``, ``meta``, and ``whisper``.  Validation is completed before the first
backend is constructed, preventing a partially initialized ensemble.

Concrete backends are imported inside their individual default factories.  An
installation therefore needs an SDK or model runtime only when that provider is
actually enabled.  Tests and embedding applications may inject factories to
avoid all provider side effects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias

from src.config import TranscribeConfig
from src.transcribe.backends.base import TranscriptionBackend

CANONICAL_PROVIDERS: tuple[str, ...] = (
    "openai",
    "xai",
    "meta",
    "whisper",
)

BackendFactory: TypeAlias = Callable[[TranscribeConfig], TranscriptionBackend]


def _openai_factory(config: TranscribeConfig) -> TranscriptionBackend:
    from src.transcribe.backends.openai_backend import OpenAIBackend

    return OpenAIBackend(config)


def _xai_factory(config: TranscribeConfig) -> TranscriptionBackend:
    from src.transcribe.backends.xai_backend import XAIBackend

    return XAIBackend(config)


def _meta_factory(config: TranscribeConfig) -> TranscriptionBackend:
    from src.transcribe.backends.meta_backend import MetaASRBackend

    return MetaASRBackend(config)


def _whisper_factory(config: TranscribeConfig) -> TranscriptionBackend:
    from src.transcribe.backends.local_whisper import LocalWhisperBackend

    return LocalWhisperBackend(config)


_DEFAULT_FACTORIES: Mapping[str, BackendFactory] = {
    "openai": _openai_factory,
    "xai": _xai_factory,
    "meta": _meta_factory,
    "whisper": _whisper_factory,
}


def create_backend_registry(
    config: TranscribeConfig,
    *,
    factories: Mapping[str, BackendFactory] | None = None,
) -> dict[str, TranscriptionBackend]:
    """Instantiate enabled providers in their configured order.

    ``factories`` overlays the defaults by canonical provider name.  Supplying
    factories for tests makes this function entirely independent of SDKs,
    credentials, model downloads, and network access.  Factories for disabled
    providers are never invoked.
    """
    providers = _validated_providers(config)
    injected_factories = _validated_factories(factories)

    registry: dict[str, TranscriptionBackend] = {}
    for provider in providers:
        factory = injected_factories.get(provider, _DEFAULT_FACTORIES[provider])
        registry[provider] = factory(config)
    return registry


def _validated_providers(config: TranscribeConfig) -> tuple[str, ...]:
    raw_providers = getattr(config, "providers", None)
    if raw_providers is None:
        raise ValueError("config.providers must contain at least one provider")
    if isinstance(raw_providers, (str, bytes)) or not isinstance(
        raw_providers, Sequence
    ):
        raise TypeError("config.providers must be an ordered sequence of names")

    providers = tuple(raw_providers)
    if not providers:
        raise ValueError("config.providers must contain at least one provider")
    if any(not isinstance(provider, str) for provider in providers):
        raise TypeError("every configured provider name must be a string")

    unknown = tuple(
        provider for provider in providers if provider not in CANONICAL_PROVIDERS
    )
    if unknown:
        rendered = ", ".join(repr(provider) for provider in dict.fromkeys(unknown))
        raise ValueError(
            f"unknown transcription provider(s): {rendered}; expected canonical "
            f"names {CANONICAL_PROVIDERS!r}"
        )

    seen: set[str] = set()
    duplicates: list[str] = []
    for provider in providers:
        if provider in seen and provider not in duplicates:
            duplicates.append(provider)
        seen.add(provider)
    if duplicates:
        rendered = ", ".join(repr(provider) for provider in duplicates)
        raise ValueError(f"duplicate transcription provider(s): {rendered}")

    primary = getattr(config, "primary_provider", None)
    if not isinstance(primary, str) or not primary:
        raise ValueError("config.primary_provider must name an enabled provider")
    if primary not in providers:
        raise ValueError(
            f"primary provider {primary!r} is not enabled in {providers!r}"
        )
    return providers


def _validated_factories(
    factories: Mapping[str, BackendFactory] | None,
) -> Mapping[str, BackendFactory]:
    if factories is None:
        return {}
    if not isinstance(factories, Mapping):
        raise TypeError("factories must be a mapping keyed by provider name")

    unknown = tuple(name for name in factories if name not in CANONICAL_PROVIDERS)
    if unknown:
        rendered = ", ".join(repr(name) for name in unknown)
        raise ValueError(f"factory mapping contains unknown provider(s): {rendered}")
    for name, factory in factories.items():
        if not callable(factory):
            raise TypeError(f"factory for provider {name!r} must be callable")
    return factories


def provider_model_name(config: TranscribeConfig, provider: str) -> str:
    """Return the configured provenance model name for ``provider``."""
    if provider not in CANONICAL_PROVIDERS:
        raise ValueError(
            f"unknown transcription provider {provider!r}; expected one of "
            f"{CANONICAL_PROVIDERS!r}"
        )

    if provider == "openai":
        model = getattr(config, "model", None)
    elif provider == "xai":
        model = "xai-stt"
    elif provider == "meta":
        model = getattr(config, "meta_asr_model", None)
    else:
        model = getattr(config, "local_model", None)

    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"model name for provider {provider!r} is not configured")
    return model


# Readable alias for callers that prefer a build-oriented factory name.
build_backend_registry = create_backend_registry


__all__ = [
    "CANONICAL_PROVIDERS",
    "BackendFactory",
    "build_backend_registry",
    "create_backend_registry",
    "provider_model_name",
]
