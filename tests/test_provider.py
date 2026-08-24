from __future__ import annotations

from pathlib import Path

import pytest

from src.transcribe.provider import (
    ProviderTranscript,
    TimedWord,
    coerce_provider_transcript,
    run_backend,
)


class _Legacy:
    name = "legacy"

    def transcribe(self, path: Path) -> str:
        return "  Exact provider surface.\n"


class _Detailed(_Legacy):
    name = "detailed"

    def transcribe_detailed(self, path: Path) -> ProviderTranscript:
        return ProviderTranscript(provider=self.name, model="m", text="Detailed.")


def test_legacy_text_is_preserved_character_for_character(tmp_path: Path) -> None:
    result = run_backend(_Legacy(), tmp_path / "audio.wav", model="old")  # type: ignore[arg-type]
    assert result.text == "  Exact provider surface.\n"
    assert result.provider == "legacy"
    assert result.model == "old"


def test_detailed_result_is_used_without_calling_legacy(tmp_path: Path) -> None:
    result = run_backend(_Detailed(), tmp_path / "audio.wav")  # type: ignore[arg-type]
    assert result.text == "Detailed."
    assert result.model == "m"


def test_registry_alias_is_canonicalised_without_changing_text(tmp_path: Path) -> None:
    result = run_backend(  # type: ignore[arg-type]
        _Legacy(),
        tmp_path / "audio.wav",
        model="large-v3",
        provider="whisper",
    )
    assert result.provider == "whisper"
    assert result.model == "large-v3"
    assert result.text == "  Exact provider surface.\n"


def test_rejects_provider_identity_mismatch() -> None:
    evidence = ProviderTranscript(provider="xai", model="", text="Amen")
    with pytest.raises(ValueError, match="returned evidence"):
        coerce_provider_transcript(evidence, provider="openai")


@pytest.mark.parametrize("value", ["", " \n "])
def test_rejects_empty_transcript(value: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        ProviderTranscript(provider="openai", model="m", text=value)


def test_timed_word_validates_bounds() -> None:
    with pytest.raises(ValueError, match="precede"):
        TimedWord("word", start_seconds=2, end_seconds=1)
    assert TimedWord("word", start_seconds=1, end_seconds=2).text == "word"
