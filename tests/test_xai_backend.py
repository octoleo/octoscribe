"""Unit tests for the offline, injectable xAI/Grok STT backend."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.transcribe.backends.xai_backend import XAIBackend


class _FakeResponse:
    """Small urllib-compatible response that records resource cleanup."""

    def __init__(self, payload: bytes, *, status: int = 200) -> None:
        self._payload = payload
        self.status = status
        self.closed = False

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:
        self.closed = True


class _RecordingTransport:
    """Callable transport returning queued responses or exceptions."""

    def __init__(self, *results: _FakeResponse | Exception) -> None:
        self._results = list(results)
        self.calls: list[tuple[object, float]] = []

    def __call__(self, request, *, timeout: float):
        self.calls.append((request, timeout))
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _OpenerClient:
    """Opener-shaped injectable client used to exercise that integration seam."""

    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls = []

    def open(self, request, *, timeout: float):
        self.calls.append((request, timeout))
        return self.response


def _config(**overrides):
    values = {
        "retry_attempts": 2,
        "retry_base_delay": 0.001,
        "retry_max_delay": 0.01,
        "language": "en",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _audio(tmp_path: Path, name: str = "sermon.ogg") -> Path:
    path = tmp_path / name
    path.write_bytes(b"ORIGINAL-AUDIO-BYTES")
    return path


def _json_response(text: str, *, status: int = 200) -> _FakeResponse:
    return _FakeResponse(
        json.dumps({"text": text, "duration": 1.25, "words": []}).encode(),
        status=status,
    )


def test_name_is_stable_provider_identifier() -> None:
    backend = XAIBackend(_config(), api_key="xai-test", transport=lambda *a, **k: None)
    assert backend.name == "xai"


def test_requires_xai_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="XAI_API_KEY"):
        XAIBackend(_config())


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://api.x.ai/v1/stt",
        "https://attacker.example/v1/stt",
        "https://api.x.ai/v1/other",
        "https://user:secret@api.x.ai/v1/stt",
    ],
)
def test_rejects_endpoint_that_could_exfiltrate_key(endpoint: str) -> None:
    with pytest.raises(ValueError, match="exactly"):
        XAIBackend(
            _config(),
            api_key="xai-test",
            endpoint=endpoint,
            transport=lambda *args, **kwargs: None,
        )


def test_posts_fidelity_safe_multipart_and_preserves_raw_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-from-environment")
    raw_text = "  In the beginning—exactly as spoken.  \n"
    response = _json_response(raw_text)
    transport = _RecordingTransport(response)
    audio = _audio(tmp_path, 'sermon "one".ogg')

    result = XAIBackend(_config(), transport=transport, timeout=42).transcribe(audio)

    assert result == raw_text
    assert response.closed is True
    assert len(transport.calls) == 1
    request, timeout = transport.calls[0]
    assert request.full_url == "https://api.x.ai/v1/stt"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer xai-from-environment"
    assert request.get_header("Accept") == "application/json"
    assert request.get_header("Content-type").startswith(
        "multipart/form-data; boundary="
    )
    assert timeout == 42

    body = request.data
    assert b'name="format"\r\n\r\nfalse\r\n' in body
    assert b'name="filler_words"\r\n\r\ntrue\r\n' in body
    assert b'name="language"\r\n\r\nen\r\n' in body
    assert b'name="file"' in body
    assert b"ORIGINAL-AUDIO-BYTES" in body
    assert b'filename="sermon \\"one\\".ogg"' in body
    # xAI requires the file field after all option fields.
    assert body.index(b'name="format"') < body.index(b'name="file"')
    assert body.index(b'name="filler_words"') < body.index(b'name="file"')


def test_accepts_an_opener_style_injected_client(tmp_path: Path) -> None:
    response = _json_response("Amen.")
    client = _OpenerClient(response)

    result = XAIBackend(_config(), api_key="xai-test", transport=client).transcribe(
        _audio(tmp_path)
    )

    assert result == "Amen."
    assert len(client.calls) == 1
    assert response.closed is True


def test_detailed_result_preserves_word_timing_evidence(tmp_path: Path) -> None:
    payload = {
        "text": "Do not fear.",
        "language": "en",
        "duration": 1.5,
        "words": [
            {"word": "Do", "start": 0.1, "end": 0.3, "confidence": 0.9},
            {"word": "not", "start": 0.31, "end": 0.6, "speaker": 0},
        ],
    }
    response = _FakeResponse(json.dumps(payload).encode())
    backend = XAIBackend(
        _config(), api_key="xai-test", transport=_RecordingTransport(response)
    )

    result = backend.transcribe_detailed(_audio(tmp_path))

    assert result.text == "Do not fear."
    assert result.language == "en"
    assert result.duration_seconds == 1.5
    assert [word.text for word in result.words] == ["Do", "not"]
    assert result.words[0].confidence == 0.9
    assert result.words[1].speaker == 0


def test_retries_transient_network_failure_with_fresh_request(tmp_path: Path) -> None:
    transport = _RecordingTransport(
        urllib.error.URLError("connection timed out"),
        _json_response("Recovered transcript."),
    )
    backend = XAIBackend(
        _config(retry_attempts=1), api_key="xai-test", transport=transport
    )

    with patch("time.sleep"):
        result = backend.transcribe(_audio(tmp_path))

    assert result == "Recovered transcript."
    assert len(transport.calls) == 2
    assert transport.calls[0][0] is not transport.calls[1][0]
    assert b"ORIGINAL-AUDIO-BYTES" in transport.calls[0][0].data
    assert b"ORIGINAL-AUDIO-BYTES" in transport.calls[1][0].data


def test_http_auth_failure_is_not_retried_and_response_is_closed(
    tmp_path: Path,
) -> None:
    response = _FakeResponse(b'{"error":"invalid key"}', status=401)
    transport = _RecordingTransport(response)
    backend = XAIBackend(
        _config(retry_attempts=3), api_key="xai-test", transport=transport
    )

    with pytest.raises(RuntimeError, match="HTTP 401"):
        backend.transcribe(_audio(tmp_path))

    assert len(transport.calls) == 1
    assert response.closed is True


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not json", "malformed JSON"),
        (b"[]", "JSON object"),
        (b"{}", "missing the 'text' field"),
        (b'{"text": 42}', "must be a string"),
        (b'{"text": "  \\n  "}', "empty transcript"),
        (b"\xff", "non-UTF-8"),
    ],
)
def test_rejects_malformed_or_empty_responses_and_closes_them(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    response = _FakeResponse(payload)
    backend = XAIBackend(
        _config(), api_key="xai-test", transport=_RecordingTransport(response)
    )

    with pytest.raises(RuntimeError, match=message):
        backend.transcribe(_audio(tmp_path))

    assert response.closed is True


@pytest.mark.parametrize("kind", ["missing", "empty", "directory"])
def test_rejects_invalid_audio_before_opening_network(
    tmp_path: Path, kind: str
) -> None:
    transport = _RecordingTransport(_json_response("must not be reached"))
    backend = XAIBackend(_config(), api_key="xai-test", transport=transport)

    path = tmp_path / "input.ogg"
    if kind == "empty":
        path.touch()
    elif kind == "directory":
        path.mkdir()

    with pytest.raises((FileNotFoundError, ValueError), match="Audio"):
        backend.transcribe(path)

    assert transport.calls == []
