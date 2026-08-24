"""Offline tests for ordered, lazy transcription backend construction."""

from __future__ import annotations

import importlib
import random
from types import SimpleNamespace

import pytest

from src.transcribe.backends.registry import (
    CANONICAL_PROVIDERS,
    build_backend_registry,
    create_backend_registry,
    provider_model_name,
)


def _config(
    providers=("openai",),
    *,
    primary_provider: str | None = None,
    **overrides,
):
    values = {
        "providers": providers,
        "primary_provider": (
            providers[0] if primary_provider is None and providers else primary_provider
        ),
        "model": "gpt-transcribe",
        "meta_asr_model": "omniASR_LLM_Unlimited_7B_v2",
        "local_model": "large-v3",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _recording_factories(calls: list[tuple[str, object]]):
    def factory(provider: str):
        def construct(config):
            instance = object()
            calls.append((provider, config))
            return instance

        return construct

    return {provider: factory(provider) for provider in CANONICAL_PROVIDERS}


def test_constructs_only_enabled_providers_in_configuration_order() -> None:
    config = _config(("meta", "openai", "whisper"), primary_provider="openai")
    calls: list[tuple[str, object]] = []

    registry = create_backend_registry(
        config, factories=_recording_factories(calls)
    )

    assert list(registry) == ["meta", "openai", "whisper"]
    assert [provider for provider, _ in calls] == ["meta", "openai", "whisper"]
    assert all(received_config is config for _, received_config in calls)
    assert len({id(instance) for instance in registry.values()}) == 3


def test_build_alias_has_the_same_contract() -> None:
    config = _config(("xai",), primary_provider="xai")
    sentinel = object()

    registry = build_backend_registry(
        config, factories={"xai": lambda received: sentinel}
    )

    assert registry == {"xai": sentinel}


def test_factories_for_disabled_providers_are_never_called() -> None:
    config = _config(("openai",))
    calls: list[tuple[str, object]] = []

    registry = create_backend_registry(
        config, factories=_recording_factories(calls)
    )

    assert list(registry) == ["openai"]
    assert [provider for provider, _ in calls] == ["openai"]


@pytest.mark.parametrize(
    ("provider", "module_name", "class_name"),
    [
        (
            "openai",
            "src.transcribe.backends.openai_backend",
            "OpenAIBackend",
        ),
        ("xai", "src.transcribe.backends.xai_backend", "XAIBackend"),
        ("meta", "src.transcribe.backends.meta_backend", "MetaASRBackend"),
        (
            "whisper",
            "src.transcribe.backends.local_whisper",
            "LocalWhisperBackend",
        ),
    ],
)
def test_default_factory_resolves_expected_backend_lazily(
    provider: str,
    module_name: str,
    class_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(module_name)
    sentinel = object()
    received: list[object] = []

    def fake_backend(config):
        received.append(config)
        return sentinel

    monkeypatch.setattr(module, class_name, fake_backend)
    config = _config((provider,), primary_provider=provider)

    assert create_backend_registry(config) == {provider: sentinel}
    assert received == [config]


@pytest.mark.parametrize(
    ("providers", "primary", "message"),
    [
        ((), "openai", "at least one"),
        (("openai", "xai", "openai"), "openai", "duplicate.*openai"),
        (("openai", "claude"), "openai", "unknown.*claude"),
        (("grok",), "grok", "unknown.*grok"),
        (("local",), "local", "unknown.*local"),
        (("openai", "xai"), "meta", "primary provider.*not enabled"),
        (("openai",), "", "primary_provider"),
    ],
)
def test_invalid_provider_configuration_constructs_nothing(
    providers,
    primary,
    message,
) -> None:
    calls: list[tuple[str, object]] = []

    with pytest.raises(ValueError, match=message):
        create_backend_registry(
            _config(providers, primary_provider=primary),
            factories=_recording_factories(calls),
        )

    assert calls == []


@pytest.mark.parametrize("providers", ["openai", b"openai", {"openai"}, 42])
def test_provider_collection_must_be_an_ordered_sequence(providers) -> None:
    with pytest.raises(TypeError, match="ordered sequence"):
        create_backend_registry(
            _config(providers, primary_provider="openai"), factories={}
        )


def test_provider_names_must_be_strings() -> None:
    with pytest.raises(TypeError, match="name must be a string"):
        create_backend_registry(
            _config(("openai", 7), primary_provider="openai"), factories={}
        )


@pytest.mark.parametrize(
    ("factories", "error", "message"),
    [
        ([], TypeError, "must be a mapping"),
        ({"claude": lambda config: object()}, ValueError, "unknown.*claude"),
        ({"openai": None}, TypeError, "must be callable"),
    ],
)
def test_rejects_invalid_factory_mapping_before_construction(
    factories,
    error,
    message,
) -> None:
    with pytest.raises(error, match=message):
        create_backend_registry(_config(), factories=factories)


def test_factory_failure_propagates_and_later_providers_are_not_constructed() -> None:
    config = _config(("openai", "xai"))
    calls: list[str] = []

    def fail(received):
        calls.append("openai")
        raise RuntimeError("provider setup failed")

    def should_not_run(received):
        calls.append("xai")
        return object()

    with pytest.raises(RuntimeError, match="provider setup failed"):
        create_backend_registry(
            config,
            factories={"openai": fail, "xai": should_not_run},
        )

    assert calls == ["openai"]


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("openai", "gpt-transcribe"),
        ("xai", "xai-stt"),
        ("meta", "omniASR_LLM_Unlimited_7B_v2"),
        ("whisper", "large-v3"),
    ],
)
def test_provider_model_name(provider: str, expected: str) -> None:
    assert provider_model_name(_config(), provider) == expected


def test_model_helper_reports_configured_values_without_rewriting() -> None:
    config = _config(
        model="custom/OpenAI model",
        meta_asr_model="meta/model-v2",
        local_model="distil-large-v3",
    )

    assert provider_model_name(config, "openai") == "custom/OpenAI model"
    assert provider_model_name(config, "meta") == "meta/model-v2"
    assert provider_model_name(config, "whisper") == "distil-large-v3"


@pytest.mark.parametrize(
    ("provider", "overrides"),
    [
        ("openai", {"model": ""}),
        ("meta", {"meta_asr_model": None}),
        ("whisper", {"local_model": "   "}),
    ],
)
def test_model_helper_rejects_missing_model(provider, overrides) -> None:
    with pytest.raises(ValueError, match="not configured"):
        provider_model_name(_config(**overrides), provider)


def test_model_helper_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown.*claude"):
        provider_model_name(_config(), "claude")


def test_randomized_provider_orders_are_preserved_exactly() -> None:
    rng = random.Random(0xBAAC)
    for _ in range(100):
        provider_count = rng.randint(1, len(CANONICAL_PROVIDERS))
        providers = tuple(rng.sample(CANONICAL_PROVIDERS, provider_count))
        config = _config(providers)
        calls: list[tuple[str, object]] = []

        registry = create_backend_registry(
            config, factories=_recording_factories(calls)
        )

        assert tuple(registry) == providers
        assert tuple(provider for provider, _ in calls) == providers
