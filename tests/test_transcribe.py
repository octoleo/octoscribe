"""
tests/test_transcribe.py — Comprehensive pytest tests for src/transcribe.py.

Run with:
    pytest tests/test_transcribe.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure the project root is on the path so both `src.*` imports and the
# direct `from src.transcribe import …` style work.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcribe import (
    VERBATIM_PROMPT,
    BatchStats,
    LocalWhisperBackend,
    OpenAIBackend,
    TranscriptionBackend,
    TranscriptionResult,
    Transcriber,
    _normalize_text,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_transcribe_config(tmp_path: Path, **overrides):
    """Build a minimal TranscribeConfig-like namespace for tests."""
    defaults = dict(
        backend="openai",
        model="gpt-transcribe",
        language="en",
        workers=1,
        retry_attempts=2,
        retry_base_delay=0.01,
        retry_max_delay=0.05,
        api_key="sk-test",
        local_model="large-v3",
        device="cpu",
        compute_type="int8",
        beam_size=5,
        best_of=5,
        repetition_penalty=1.1,
        vad_filter=False,
        vad_min_silence_ms=500,
        vad_speech_pad_ms=400,
        transcriptions_dir=tmp_path / "transcriptions",
        manifest_file=tmp_path / "manifest.json",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_download_config(tmp_path: Path):
    return SimpleNamespace(
        audio_dir=tmp_path / "audio",
        manifest_file=tmp_path / "manifest.json",
    )


def _make_config(tmp_path: Path, **transcribe_overrides):
    """Build a minimal Config-like namespace."""
    return SimpleNamespace(
        transcribe=_make_transcribe_config(tmp_path, **transcribe_overrides),
        download=_make_download_config(tmp_path),
    )


def _make_manifest(pending: list[dict] | None = None) -> MagicMock:
    """Return a mock Manifest with configurable pending_transcription()."""
    m = MagicMock()
    m.pending_transcription.return_value = pending or []
    return m


def _make_audio_file(tmp_path: Path, filename: str = "sermon.ogg") -> Path:
    """Create a zero-byte audio file in tmp_path/audio/."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    p = audio_dir / filename
    p.write_bytes(b"FAKE_AUDIO")
    return p


# ---------------------------------------------------------------------------
# 1. TranscriptionBackend is abstract — cannot instantiate directly
# ---------------------------------------------------------------------------

def test_transcription_backend_is_abstract():
    with pytest.raises(TypeError):
        TranscriptionBackend()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Helper: build an OpenAIBackend with the openai module fully mocked out.
# ---------------------------------------------------------------------------

def _make_openai_backend(cfg, mock_client: MagicMock | None = None) -> OpenAIBackend:
    """
    Construct an OpenAIBackend without the real openai package.

    Injects a mock into sys.modules so the `import openai` inside __init__
    resolves to our mock, then restores the original state on exit.
    """
    mock_openai = MagicMock()
    if mock_client is not None:
        mock_openai.OpenAI.return_value = mock_client
    with patch.dict(sys.modules, {"openai": mock_openai}):
        backend = OpenAIBackend(cfg)
    # Swap in the mock client so transcribe() uses it.
    if mock_client is not None:
        backend._client = mock_client
    return backend


# ---------------------------------------------------------------------------
# 2. OpenAIBackend.name returns "openai"
# ---------------------------------------------------------------------------

def test_openai_backend_name(tmp_path: Path):
    cfg = _make_transcribe_config(tmp_path)
    backend = _make_openai_backend(cfg)
    assert backend.name == "openai"


def test_openai_backend_configures_provider_timeout(tmp_path: Path):
    cfg = _make_transcribe_config(tmp_path, provider_timeout_seconds=42)
    mock_openai = MagicMock()
    with patch.dict(sys.modules, {"openai": mock_openai}):
        OpenAIBackend(cfg)
    mock_openai.OpenAI.assert_called_once_with(
        api_key="sk-test", timeout=42.0, max_retries=0
    )


def test_openai_backend_rejects_diarization_model(tmp_path: Path):
    cfg = _make_transcribe_config(
        tmp_path,
        model="gpt-4o-transcribe-diarize",
    )

    with pytest.raises(ValueError, match="diarization models are not supported"):
        _make_openai_backend(cfg)


# ---------------------------------------------------------------------------
# 3. LocalWhisperBackend.name returns "local"
# ---------------------------------------------------------------------------

def test_local_whisper_backend_name(tmp_path: Path):
    cfg = _make_transcribe_config(tmp_path)
    mock_whisper = MagicMock()
    with patch.dict(sys.modules, {"faster_whisper": mock_whisper}):
        backend = LocalWhisperBackend(cfg)
    assert backend.name == "local"


# ---------------------------------------------------------------------------
# 4. Transcriber.create_backend() returns OpenAIBackend for backend="openai"
# ---------------------------------------------------------------------------

def test_create_backend_openai(tmp_path: Path):
    cfg = _make_transcribe_config(tmp_path, backend="openai")
    mock_openai = MagicMock()
    with patch.dict(sys.modules, {"openai": mock_openai}):
        with patch("src.transcribe.OpenAIBackend.__init__", return_value=None):
            backend = Transcriber.create_backend(cfg)
    assert isinstance(backend, OpenAIBackend)


# ---------------------------------------------------------------------------
# 5. Transcriber.create_backend() returns LocalWhisperBackend for backend="local"
# ---------------------------------------------------------------------------

def test_create_backend_local(tmp_path: Path):
    cfg = _make_transcribe_config(tmp_path, backend="local")
    mock_fw = MagicMock()
    with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
        with patch("src.transcribe.LocalWhisperBackend.__init__", return_value=None):
            backend = Transcriber.create_backend(cfg)
    assert isinstance(backend, LocalWhisperBackend)


# ---------------------------------------------------------------------------
# 6. Transcriber.create_backend() raises ValueError for unknown backend
# ---------------------------------------------------------------------------

def test_create_backend_unknown(tmp_path: Path):
    cfg = _make_transcribe_config(tmp_path, backend="cloud_magic")
    with pytest.raises(ValueError, match="cloud_magic"):
        Transcriber.create_backend(cfg)


# ---------------------------------------------------------------------------
# 7. OpenAIBackend.transcribe() calls OpenAI API with correct model and prompt
# ---------------------------------------------------------------------------

def test_openai_backend_transcribe_calls_api(tmp_path: Path):
    cfg = _make_transcribe_config(tmp_path, model="gpt-4o-transcribe", language="af")
    audio_file = _make_audio_file(tmp_path)

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = SimpleNamespace(
        text="Hello world."
    )

    backend = _make_openai_backend(cfg, mock_client)
    result = backend.transcribe(audio_file)

    assert result == "Hello world."
    mock_client.audio.transcriptions.create.assert_called_once()
    kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-transcribe"
    assert kwargs["language"] == "af"
    assert kwargs["prompt"] == VERBATIM_PROMPT
    assert "response_format" not in kwargs


def test_openai_backend_whisper_uses_supported_text_format(tmp_path: Path):
    cfg = _make_transcribe_config(tmp_path, model="whisper-1", language="en")
    audio_file = _make_audio_file(tmp_path)
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = "Legacy transcript."

    result = _make_openai_backend(cfg, mock_client).transcribe(audio_file)

    assert result == "Legacy transcript."
    kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == "whisper-1"
    assert kwargs["language"] == "en"
    assert kwargs["response_format"] == "text"


def test_openai_backend_uses_current_gpt_transcribe_contract(tmp_path: Path):
    cfg = _make_transcribe_config(tmp_path, model="gpt-transcribe", language="en")
    audio_file = _make_audio_file(tmp_path)
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = SimpleNamespace(
        text="Exact current transcript."
    )

    result = _make_openai_backend(cfg, mock_client).transcribe(audio_file)

    assert result == "Exact current transcript."
    kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-transcribe"
    assert kwargs["extra_body"] == {"languages": ["en"]}
    assert kwargs["prompt"] == VERBATIM_PROMPT
    assert "language" not in kwargs
    assert "response_format" not in kwargs


# ---------------------------------------------------------------------------
# 8. OpenAIBackend.transcribe() retries on rate limit error, then succeeds
# ---------------------------------------------------------------------------

def test_openai_backend_retries_on_rate_limit(tmp_path: Path):
    cfg = _make_transcribe_config(tmp_path, retry_attempts=3, retry_base_delay=0.001, retry_max_delay=0.01)
    audio_file = _make_audio_file(tmp_path)

    rate_limit_exc = RuntimeError("rate limit exceeded — 429")

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.side_effect = [
        rate_limit_exc,
        rate_limit_exc,
        "Transcribed text.",
    ]

    backend = _make_openai_backend(cfg, mock_client)

    with patch("time.sleep"):  # suppress real sleep in tests
        result = backend.transcribe(audio_file)

    assert result == "Transcribed text."
    assert mock_client.audio.transcriptions.create.call_count == 3


# ---------------------------------------------------------------------------
# 9. OpenAIBackend.transcribe() does NOT retry on auth error
# ---------------------------------------------------------------------------

def test_openai_backend_no_retry_on_auth_error(tmp_path: Path):
    cfg = _make_transcribe_config(tmp_path, retry_attempts=3, retry_base_delay=0.001)
    audio_file = _make_audio_file(tmp_path)

    auth_exc = RuntimeError("401 Unauthorized — invalid api key")

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.side_effect = auth_exc

    backend = _make_openai_backend(cfg, mock_client)

    with pytest.raises(RuntimeError, match="invalid api key"):
        backend.transcribe(audio_file)

    # Must NOT have retried.
    assert mock_client.audio.transcriptions.create.call_count == 1


# ---------------------------------------------------------------------------
# 10. OpenAIBackend.transcribe() raises after exhausting retries
# ---------------------------------------------------------------------------

def test_openai_backend_raises_after_exhausted_retries(tmp_path: Path):
    cfg = _make_transcribe_config(tmp_path, retry_attempts=2, retry_base_delay=0.001, retry_max_delay=0.01)
    audio_file = _make_audio_file(tmp_path)

    transient_exc = RuntimeError("503 service unavailable")

    mock_client = MagicMock()
    # Always fail — never succeeds.
    mock_client.audio.transcriptions.create.side_effect = transient_exc

    backend = _make_openai_backend(cfg, mock_client)

    with patch("time.sleep"):
        with pytest.raises(RuntimeError, match="[Ee]xhausted"):
            backend.transcribe(audio_file)

    # retry_attempts=2 means: 1 initial + 2 retries = 3 total calls.
    assert mock_client.audio.transcriptions.create.call_count == 3


# ---------------------------------------------------------------------------
# 11. LocalWhisperBackend.transcribe() calls model.transcribe() with
#     temperature=0 and condition_on_previous_text=False
# ---------------------------------------------------------------------------

def test_local_whisper_backend_transcribe_params(tmp_path: Path):
    cfg = _make_transcribe_config(tmp_path, language="en", beam_size=5, best_of=5,
                                   repetition_penalty=1.1, vad_filter=False)
    audio_file = _make_audio_file(tmp_path)

    mock_segment = SimpleNamespace(text=" Amen. ", start=0.0, end=2.0)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], MagicMock())

    mock_fw = MagicMock()
    mock_fw.WhisperModel.return_value = mock_model

    with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
        backend = LocalWhisperBackend(cfg)

    result = backend.transcribe(audio_file)

    mock_model.transcribe.assert_called_once()
    kwargs = mock_model.transcribe.call_args.kwargs
    assert kwargs["temperature"] == 0
    assert kwargs["condition_on_previous_text"] is False
    assert result == "Amen."


# ---------------------------------------------------------------------------
# 12. Transcriber.run() skips entries with missing audio files
# ---------------------------------------------------------------------------

def test_transcriber_run_skips_missing_audio(tmp_path: Path):
    config = _make_config(tmp_path)
    pending = [{"telegram_msg_id": "1", "filename": "missing.ogg", "title": None}]
    manifest = _make_manifest(pending)

    backend = MagicMock(spec=TranscriptionBackend)
    backend.name = "openai"

    t = Transcriber(config, manifest, backend=backend)
    stats = t.run()

    backend.transcribe.assert_not_called()
    manifest.mark_transcribed.assert_not_called()
    manifest.mark_failed.assert_not_called()
    assert stats.skipped == 1
    assert stats.total == 0


# ---------------------------------------------------------------------------
# 13. Transcriber.run() calls mark_transcribed on success
# ---------------------------------------------------------------------------

def test_transcriber_run_marks_transcribed_on_success(tmp_path: Path):
    config = _make_config(tmp_path)
    audio_file = _make_audio_file(tmp_path, "sermon.ogg")
    pending = [{"telegram_msg_id": "42", "filename": "sermon.ogg", "title": None}]
    manifest = _make_manifest(pending)

    backend = MagicMock(spec=TranscriptionBackend)
    backend.name = "openai"
    backend.transcribe.return_value = "Thus says the Lord."

    t = Transcriber(config, manifest, backend=backend)
    stats = t.run()

    manifest.mark_transcribed.assert_called_once_with(
        "42",
        {
            "output_file": "sermon.txt",
            "model": "openai",
            "transcript_sha256": __import__("hashlib").sha256(
                b"Thus says the Lord."
            ).hexdigest(),
        },
    )
    assert stats.succeeded == 1
    assert stats.failed == 0


# ---------------------------------------------------------------------------
# 14. Transcriber.run() calls mark_failed on backend error
# ---------------------------------------------------------------------------

def test_transcriber_run_marks_failed_on_error(tmp_path: Path):
    config = _make_config(tmp_path)
    _make_audio_file(tmp_path, "broken.ogg")
    pending = [{"telegram_msg_id": "7", "filename": "broken.ogg", "title": None}]
    manifest = _make_manifest(pending)

    backend = MagicMock(spec=TranscriptionBackend)
    backend.name = "openai"
    backend.transcribe.side_effect = RuntimeError("API error")

    t = Transcriber(config, manifest, backend=backend)
    stats = t.run()

    manifest.mark_failed.assert_called_once()
    args = manifest.mark_failed.call_args.args
    assert args[0] == "7"
    assert args[1] == "transcription"
    assert "API error" in args[2]
    assert stats.failed == 1
    assert stats.succeeded == 0


# ---------------------------------------------------------------------------
# 15. Transcriber.run() writes output .txt file with transcription text
# ---------------------------------------------------------------------------

def test_transcriber_run_writes_output_file(tmp_path: Path):
    config = _make_config(tmp_path)
    _make_audio_file(tmp_path, "devotion.ogg")
    pending = [{"telegram_msg_id": "5", "filename": "devotion.ogg", "title": "Morning Devotion"}]
    manifest = _make_manifest(pending)

    backend = MagicMock(spec=TranscriptionBackend)
    backend.name = "openai"
    backend.transcribe.return_value = "Grace and peace to you."

    t = Transcriber(config, manifest, backend=backend)
    t.run()

    out_file = config.transcribe.transcriptions_dir / "Morning Devotion.txt"
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == "Grace and peace to you."


# ---------------------------------------------------------------------------
# 16. Transcriber.run() returns BatchStats with correct counts
# ---------------------------------------------------------------------------

def test_transcriber_run_batch_stats_counts(tmp_path: Path):
    config = _make_config(tmp_path)

    # Two good files, one bad.
    for name in ("a.ogg", "b.ogg", "c.ogg"):
        _make_audio_file(tmp_path, name)
    # Fourth entry has no audio file on disk.
    pending = [
        {"telegram_msg_id": "1", "filename": "a.ogg", "title": None},
        {"telegram_msg_id": "2", "filename": "b.ogg", "title": None},
        {"telegram_msg_id": "3", "filename": "c.ogg", "title": None},
        {"telegram_msg_id": "4", "filename": "missing.ogg", "title": None},
    ]
    manifest = _make_manifest(pending)

    call_count = 0

    def _transcribe(path: Path) -> str:
        nonlocal call_count
        call_count += 1
        if path.name == "b.ogg":
            raise RuntimeError("bad audio")
        return "Text."

    backend = MagicMock(spec=TranscriptionBackend)
    backend.name = "openai"
    backend.transcribe.side_effect = _transcribe

    t = Transcriber(config, manifest, backend=backend)
    stats = t.run()

    assert stats.succeeded == 2   # a.ogg, c.ogg
    assert stats.failed == 1      # b.ogg
    assert stats.skipped == 1     # missing.ogg
    assert stats.total == 3       # total processed (not skipped)


# ---------------------------------------------------------------------------
# 17. BatchStats.summary() returns a non-empty string
# ---------------------------------------------------------------------------

def test_batch_stats_summary():
    s = BatchStats(total=10, succeeded=8, failed=2, skipped=1, total_elapsed_seconds=42.5)
    summary = s.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "10" in summary
    assert "8" in summary
    assert "2" in summary


# ---------------------------------------------------------------------------
# 18. Text normalisation: preserves words, normalises whitespace only
# ---------------------------------------------------------------------------

def test_normalize_text_preserves_words():
    raw = "  Thus  says \r\nthe Lord.  \n\n\n\nAmen.  "
    out = _normalize_text(raw)
    # Words must be intact.
    assert "Thus  says" in out
    assert "the Lord." in out
    assert "Amen." in out


def test_normalize_text_caps_blank_lines():
    raw = "Line one.\n\n\n\n\nLine two."
    out = _normalize_text(raw)
    assert "\n\n\n" not in out
    assert "Line one." in out
    assert "Line two." in out


def test_normalize_text_strips_trailing_whitespace():
    raw = "Verse one.   \nVerse two.  "
    out = _normalize_text(raw)
    for line in out.splitlines():
        assert line == line.rstrip()


# ---------------------------------------------------------------------------
# 19. VERBATIM_PROMPT is defined and non-empty
# ---------------------------------------------------------------------------

def test_verbatim_prompt_defined_and_non_empty():
    assert isinstance(VERBATIM_PROMPT, str)
    assert len(VERBATIM_PROMPT.strip()) > 0
    # Core requirement keywords must be present.
    assert "EXACTLY" in VERBATIM_PROMPT
    assert "word for word" in VERBATIM_PROMPT
