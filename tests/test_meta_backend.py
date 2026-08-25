"""Offline unit tests for the explicit local Meta ASR HTTP backend."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.transcribe.backends.meta_backend import MetaASRBackend


class _FakeResponse:
    """Small urllib-compatible response with observable cleanup."""

    def __init__(
        self,
        payload: bytes,
        *,
        status: int = 200,
        content_type: str | None = "application/json",
    ) -> None:
        self._payload = payload
        self.status = status
        self.headers = (
            {"Content-Type": content_type} if content_type is not None else {}
        )
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
    """Opener-style client covering the second injection form."""

    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls = []

    def open(self, request, *, timeout: float):
        self.calls.append((request, timeout))
        return self.response


def _config(**overrides):
    values = {
        "meta_asr_url": "http://127.0.0.1:9000",
        "meta_asr_model": "omniASR_LLM_Unlimited_7B_v2",
        "meta_asr_api_key": None,
        "meta_asr_language": "eng_Latn",
        "language": "en",
        "provider_timeout_seconds": 321.0,
        "retry_attempts": 2,
        "retry_base_delay": 0.001,
        "retry_max_delay": 0.01,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _audio(tmp_path: Path, name: str = "sermon.wav") -> Path:
    path = tmp_path / name
    path.write_bytes(b"META-ASR-AUDIO")
    return path


def _json_response(text: str, *, status: int = 200) -> _FakeResponse:
    payload = json.dumps({"text": text, "language": "en"}).encode()
    return _FakeResponse(payload, status=status)


def test_name_is_stable_provider_identifier() -> None:
    backend = MetaASRBackend(
        _config(), transport=lambda *args, **kwargs: None
    )
    assert backend.name == "meta"


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (
            "http://127.0.0.1:9000",
            "http://127.0.0.1:9000/v1/audio/transcriptions",
        ),
        (
            "http://127.0.0.1:9000/v1",
            "http://127.0.0.1:9000/v1/audio/transcriptions",
        ),
        (
            "https://host.local/asr",
            "https://host.local/asr/v1/audio/transcriptions",
        ),
        (
            "https://host.local/v1/audio/transcriptions/",
            "https://host.local/v1/audio/transcriptions",
        ),
    ],
)
def test_accepts_base_or_complete_endpoint(configured: str, expected: str) -> None:
    backend = MetaASRBackend(
        _config(meta_asr_url=configured),
        transport=lambda *args, **kwargs: None,
    )
    assert backend._endpoint == expected


def test_rejects_cleartext_remote_endpoint() -> None:
    with pytest.raises(ValueError, match="loopback"):
        MetaASRBackend(
            _config(meta_asr_url="http://host.local/asr"),
            transport=lambda *args, **kwargs: None,
        )


def test_rejects_query_string_that_could_leak_credentials() -> None:
    with pytest.raises(ValueError, match="query string"):
        MetaASRBackend(
            _config(
                meta_asr_url=(
                    "https://host.local/v1/audio/transcriptions?token=secret"
                )
            ),
            transport=lambda *args, **kwargs: None,
        )


def test_missing_url_never_falls_back_to_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

    with pytest.raises(ValueError, match="will not implicitly call Llama or Ollama"):
        MetaASRBackend(_config(meta_asr_url=None))


@pytest.mark.parametrize("url", ["localhost:9000", "file:///tmp/asr", "ftp://host"])
def test_rejects_non_http_endpoint(url: str) -> None:
    with pytest.raises(ValueError, match="absolute http"):
        MetaASRBackend(_config(meta_asr_url=url))


def test_posts_required_fields_and_preserves_json_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("META_ASR_API_KEY", "environment-secret")
    raw_text = "  Every word remains here.  \n"
    response = _json_response(raw_text)
    transport = _RecordingTransport(response)
    backend = MetaASRBackend(
        _config(meta_asr_api_key="configured-secret"),
        transport=transport,
        timeout=42,
    )

    result = backend.transcribe(_audio(tmp_path, 'sermon "one".wav'))

    assert result == raw_text
    assert response.closed is True
    assert len(transport.calls) == 1
    request, timeout = transport.calls[0]
    assert request.full_url == "http://127.0.0.1:9000/v1/audio/transcriptions"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer environment-secret"
    assert timeout == 42

    body = request.data
    assert (
        b'name="model"\r\n\r\nomniASR_LLM_Unlimited_7B_v2\r\n' in body
    )
    assert b'name="language"\r\n\r\neng_Latn\r\n' in body
    assert b'name="file"' in body
    assert b"META-ASR-AUDIO" in body
    assert b'filename="sermon \\"one\\".wav"' in body


def test_plain_text_response_and_optional_auth_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("META_ASR_API_KEY", raising=False)
    raw_text = '{This is literal plain transcript text.}\n'
    response = _FakeResponse(
        raw_text.encode(), content_type="text/plain; charset=utf-8"
    )
    transport = _RecordingTransport(response)

    result = MetaASRBackend(_config(), transport=transport).transcribe(
        _audio(tmp_path)
    )

    assert result == raw_text
    request, timeout = transport.calls[0]
    assert request.get_header("Authorization") is None
    assert timeout == 321.0
    assert response.closed is True


def test_uses_configured_api_key_when_environment_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("META_ASR_API_KEY", raising=False)
    transport = _RecordingTransport(_json_response("Amen."))
    backend = MetaASRBackend(
        _config(meta_asr_api_key="configured-secret"), transport=transport
    )

    assert backend.transcribe(_audio(tmp_path)) == "Amen."
    assert (
        transport.calls[0][0].get_header("Authorization")
        == "Bearer configured-secret"
    )


def test_accepts_opener_style_injected_client(tmp_path: Path) -> None:
    response = _FakeResponse(b"Exact plain text.", content_type="text/plain")
    client = _OpenerClient(response)

    result = MetaASRBackend(_config(), transport=client).transcribe(
        _audio(tmp_path)
    )

    assert result == "Exact plain text."
    assert len(client.calls) == 1
    assert response.closed is True


def test_retries_transient_network_failure_with_fresh_request(tmp_path: Path) -> None:
    transport = _RecordingTransport(
        urllib.error.URLError("connection timed out"),
        _json_response("Recovered transcript."),
    )
    backend = MetaASRBackend(
        _config(retry_attempts=1), transport=transport
    )

    with patch("time.sleep"):
        result = backend.transcribe(_audio(tmp_path))

    assert result == "Recovered transcript."
    assert len(transport.calls) == 2
    assert transport.calls[0][0] is not transport.calls[1][0]
    assert b"META-ASR-AUDIO" in transport.calls[0][0].data
    assert b"META-ASR-AUDIO" in transport.calls[1][0].data


def test_http_auth_failure_is_not_retried_and_response_is_closed(
    tmp_path: Path,
) -> None:
    response = _FakeResponse(b'{"error":"invalid key"}', status=401)
    transport = _RecordingTransport(response)
    backend = MetaASRBackend(
        _config(retry_attempts=3), transport=transport
    )

    with pytest.raises(RuntimeError, match="HTTP 401"):
        backend.transcribe(_audio(tmp_path))

    assert len(transport.calls) == 1
    assert response.closed is True


@pytest.mark.parametrize(
    ("payload", "content_type", "message"),
    [
        (b"not json", "application/json", "malformed JSON"),
        (b"[]", "application/json", "must be an object"),
        (b"{}", "application/json", "missing the 'text' field"),
        (b'{"text": 42}', "application/json", "must be a string"),
        (b'{"text": "  \\n  "}', "application/json", "empty transcript"),
        (b"   \n", "text/plain", "empty transcript"),
        (b"\xff", "text/plain", "non-UTF-8"),
    ],
)
def test_rejects_malformed_or_empty_responses_and_closes_them(
    tmp_path: Path,
    payload: bytes,
    content_type: str,
    message: str,
) -> None:
    response = _FakeResponse(payload, content_type=content_type)
    backend = MetaASRBackend(
        _config(), transport=_RecordingTransport(response)
    )

    with pytest.raises(RuntimeError, match=message):
        backend.transcribe(_audio(tmp_path))

    assert response.closed is True


def test_json_is_detected_when_content_type_is_missing(tmp_path: Path) -> None:
    response = _FakeResponse(
        b'{"text":"Detected JSON."}', content_type=None
    )
    backend = MetaASRBackend(
        _config(), transport=_RecordingTransport(response)
    )

    assert backend.transcribe(_audio(tmp_path)) == "Detected JSON."


@pytest.mark.parametrize("kind", ["missing", "empty", "directory"])
def test_rejects_invalid_audio_before_opening_network(
    tmp_path: Path, kind: str
) -> None:
    transport = _RecordingTransport(_json_response("must not be reached"))
    backend = MetaASRBackend(_config(), transport=transport)

    path = tmp_path / "input.wav"
    if kind == "empty":
        path.touch()
    elif kind == "directory":
        path.mkdir()

    with pytest.raises((FileNotFoundError, ValueError), match="Audio"):
        backend.transcribe(path)

    assert transport.calls == []
