"""
src/transcribe/backends/xai_backend.py — xAI/Grok transcription backend.

Calls xAI's dedicated Speech-to-Text endpoint without a general-purpose text
cleanup pass.  The request explicitly disables inverse text normalization and
retains filler words so the returned transcript is as close as possible to the
spoken audio.  Response text is validated, but deliberately returned character-
for-character as decoded from the API's JSON string: this backend never strips,
wraps, or otherwise rewrites it.

The HTTP transport is injectable to keep unit tests deterministic and entirely
offline. Production uses urllib with redirects disabled, avoiding both another
runtime dependency and cross-origin credential/audio forwarding.
"""

from __future__ import annotations

import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from src.config import TranscribeConfig
from src.transcribe.backends.base import TranscriptionBackend
from src.transcribe.backends.retry import RetryPolicy
from src.transcribe.provider import ProviderTranscript, TimedWord

_XAI_STT_ENDPOINT = "https://api.x.ai/v1/stt"
_DEFAULT_TIMEOUT_SECONDS = 300.0
_MAX_FILE_BYTES = 500_000_000


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent bearer credentials and audio bodies crossing redirect origins."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _secure_urlopen(request, *, timeout):
    return urllib.request.build_opener(_NoRedirectHandler()).open(
        request, timeout=timeout
    )


class _Response(Protocol):
    """Minimal response surface consumed from urllib or an injected opener."""

    def read(self) -> bytes: ...

    def close(self) -> None: ...


_Transport = Callable[..., _Response] | object


class XAIBackend(TranscriptionBackend):
    """Transcribe audio with xAI/Grok STT using bounded transient retries."""

    def __init__(
        self,
        config: TranscribeConfig,
        *,
        transport: _Transport | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """
        Create an xAI backend.

        ``api_key`` and ``transport`` are explicit seams for unit tests.  In
        normal use, credentials come only from ``XAI_API_KEY`` and HTTP is
        performed by a redirect-blocking urllib transport.
        """
        resolved_key = (
            api_key
            if api_key is not None
            else getattr(config, "xai_api_key", None) or os.environ.get("XAI_API_KEY")
        )
        resolved_timeout = (
            timeout
            if timeout is not None
            else float(
                getattr(config, "provider_timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)
            )
        )
        if not resolved_key or not resolved_key.strip():
            raise ValueError(
                "XAI_API_KEY is required when the xAI transcription backend is enabled"
            )
        if resolved_timeout <= 0:
            raise ValueError("xAI transcription timeout must be greater than zero")

        self._config = config
        self._api_key = resolved_key.strip()
        self._transport = transport or _secure_urlopen
        self._endpoint = self._validated_endpoint(
            endpoint or getattr(config, "xai_base_url", _XAI_STT_ENDPOINT)
        )
        self._timeout = resolved_timeout
        self._retry = RetryPolicy(
            attempts=config.retry_attempts,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
        )

    @property
    def name(self) -> str:
        """Return the stable provider identifier recorded in provenance."""
        return "xai"

    @staticmethod
    def _validated_endpoint(value: str) -> str:
        """Pin xAI credentials and sermon audio to xAI's TLS endpoint."""
        parsed = urllib.parse.urlsplit(str(value).strip())
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.x.ai"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/") != "/v1/stt"
        ):
            raise ValueError(
                "xAI STT endpoint must be exactly https://api.x.ai/v1/stt"
            )
        return "https://api.x.ai/v1/stt"

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe *audio_path* and return xAI's transcript unchanged."""
        return self.transcribe_detailed(audio_path).text

    def transcribe_detailed(self, audio_path: Path) -> ProviderTranscript:
        """Return text plus xAI word timestamps as comparison evidence."""
        path = Path(audio_path)
        self._validate_audio_path(path)

        def _attempt() -> ProviderTranscript:
            request = self._build_request(path)
            payload = self._send(request)
            document = self._parse_document(payload)
            text = document["text"]
            assert isinstance(text, str)
            words: list[TimedWord] = []
            raw_words = document.get("words")
            if isinstance(raw_words, list):
                for item in raw_words:
                    if not isinstance(item, dict):
                        continue
                    surface = item.get("word", item.get("text"))
                    if not isinstance(surface, str) or not surface:
                        continue
                    words.append(
                        TimedWord(
                            text=surface,
                            start_seconds=self._optional_float(
                                item.get("start", item.get("start_time"))
                            ),
                            end_seconds=self._optional_float(
                                item.get("end", item.get("end_time"))
                            ),
                            confidence=self._optional_float(item.get("confidence")),
                            speaker=item.get("speaker"),
                        )
                    )
            return ProviderTranscript(
                provider=self.name,
                model="xai-stt",
                text=text,
                words=tuple(words),
                language=(
                    str(document["language"])
                    if document.get("language") is not None
                    else None
                ),
                duration_seconds=self._optional_float(document.get("duration")),
                metadata={"word_count": len(words)},
            )

        return self._retry.run(_attempt, label=path.name)

    @staticmethod
    def _validate_audio_path(audio_path: Path) -> None:
        """Reject missing, non-regular, empty, or over-limit inputs locally."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        if not audio_path.is_file():
            raise ValueError(f"Audio path is not a regular file: {audio_path}")

        size = audio_path.stat().st_size
        if size <= 0:
            raise ValueError(f"Audio file is empty: {audio_path}")
        if size > _MAX_FILE_BYTES:
            raise ValueError(
                f"Audio file exceeds xAI's 500 MB upload limit: {audio_path}"
            )

    def _build_request(self, audio_path: Path) -> urllib.request.Request:
        """Build one multipart request, opening and closing the audio afresh."""
        boundary = f"octoscribe-{uuid.uuid4().hex}"
        body = self._multipart_body(audio_path, boundary)
        return urllib.request.Request(
            self._endpoint,
            data=body,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "OctoScribe/xAI-STT",
            },
            method="POST",
        )

    def _multipart_body(self, audio_path: Path, boundary: str) -> bytes:
        """Encode fidelity options followed by the required final file part."""
        marker = boundary.encode("ascii")
        chunks: list[bytes] = []

        fields = [("format", "false"), ("filler_words", "true")]
        language = str(getattr(self._config, "language", "")).strip()
        if language:
            fields.append(("language", language))
        for name, value in fields:
            chunks.extend(
                (
                    b"--" + marker + b"\r\n",
                    f'Content-Disposition: form-data; name="{name}"\r\n'.encode(
                        "ascii"
                    ),
                    b"\r\n",
                    value.encode("ascii") + b"\r\n",
                )
            )

        # The xAI API requires the file field to be the final multipart field.
        filename = XAIBackend._safe_header_value(audio_path.name)
        content_type = (
            mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
        )
        with audio_path.open("rb") as audio_file:
            audio_bytes = audio_file.read()

        if not audio_bytes:
            # Protect against a file being truncated after the initial stat.
            raise ValueError(f"Audio file became empty while reading: {audio_path}")
        if len(audio_bytes) > _MAX_FILE_BYTES:
            raise ValueError(
                f"Audio file exceeds xAI's 500 MB upload limit: {audio_path}"
            )

        chunks.extend(
            (
                b"--" + marker + b"\r\n",
                (
                    'Content-Disposition: form-data; name="file"; '
                    f'filename="{filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n".encode("ascii"),
                b"\r\n",
                audio_bytes,
                b"\r\n",
                b"--" + marker + b"--\r\n",
            )
        )
        return b"".join(chunks)

    @staticmethod
    def _safe_header_value(value: str) -> str:
        """Prevent a local filename from injecting multipart headers."""
        return (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\r", "_")
            .replace("\n", "_")
        )

    def _send(self, request: urllib.request.Request) -> bytes:
        """Execute one HTTP attempt and always close the response object."""
        try:
            response = self._open(request)
        except urllib.error.HTTPError as exc:
            try:
                exc.close()
            finally:
                raise RuntimeError(
                    f"xAI STT request failed with HTTP {exc.code}: {exc.reason}"
                ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"xAI STT network error: {exc.reason}") from exc

        try:
            status = self._response_status(response)
            payload = response.read()
        finally:
            response.close()

        if status is not None and not 200 <= status < 300:
            raise RuntimeError(f"xAI STT request failed with HTTP {status}")
        if not isinstance(payload, bytes):
            raise RuntimeError("xAI STT returned a non-binary HTTP response body")
        return payload

    def _open(self, request: urllib.request.Request) -> _Response:
        """Invoke either a callable transport or an opener-style client."""
        if callable(self._transport):
            return self._transport(request, timeout=self._timeout)

        opener = getattr(self._transport, "open", None)
        if callable(opener):
            return opener(request, timeout=self._timeout)
        raise TypeError(
            "xAI transport must be callable or provide open(request, timeout=)"
        )

    @staticmethod
    def _response_status(response: _Response) -> int | None:
        """Read an HTTP status from urllib and lightweight test responses."""
        status = getattr(response, "status", None)
        if status is None:
            getcode = getattr(response, "getcode", None)
            status = getcode() if callable(getcode) else None
        return int(status) if status is not None else None

    @staticmethod
    def _parse_transcript(payload: bytes) -> str:
        """Validate the documented response shape without altering ``text``."""
        document = XAIBackend._parse_document(payload)
        transcript = document["text"]
        assert isinstance(transcript, str)
        return transcript

    @staticmethod
    def _parse_document(payload: bytes) -> dict[str, object]:
        """Validate and return xAI's JSON response object."""
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("xAI STT returned a non-UTF-8 response") from exc

        try:
            document = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise RuntimeError("xAI STT returned malformed JSON") from exc

        if not isinstance(document, dict):
            raise RuntimeError("xAI STT response must be a JSON object")
        if "text" not in document:
            raise RuntimeError("xAI STT response is missing the 'text' field")

        transcript = document["text"]
        if not isinstance(transcript, str):
            raise RuntimeError("xAI STT response field 'text' must be a string")
        if not transcript.strip():
            raise RuntimeError("xAI STT returned an empty transcript")

        # Do not strip or normalize: the provider's raw text is evidence.
        return document

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
