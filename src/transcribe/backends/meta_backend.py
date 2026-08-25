"""
src/transcribe/backends/meta_backend.py — Explicit local Meta ASR backend.

This adapter targets a user-configured, OpenAI-compatible audio transcription
server hosting a Meta speech-recognition model.  It intentionally has no
implicit localhost, Llama, or Ollama fallback: ``meta_asr_url`` must identify an
actual ASR service before the backend can be constructed.

The response may be the standard OpenAI-style ``{"text": ...}`` JSON object or
plain UTF-8 text. In either case the transcript is validated for emptiness and
then returned character-for-character without cleanup or normalization.
Redirects are blocked so optional bearer tokens and audio cannot be forwarded
to a different origin.
"""

from __future__ import annotations

import json
import ipaddress
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

_TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"
_DEFAULT_TIMEOUT_SECONDS = 900.0


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent optional bearer tokens and sermon audio crossing redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _secure_urlopen(request, *, timeout):
    return urllib.request.build_opener(_NoRedirectHandler()).open(
        request, timeout=timeout
    )


class _Response(Protocol):
    """Minimal urllib response surface used by production and test clients."""

    def read(self) -> bytes: ...

    def close(self) -> None: ...


_Transport = Callable[..., _Response] | object


class MetaASRBackend(TranscriptionBackend):
    """Transcribe with an explicitly configured OpenAI-compatible Meta ASR."""

    def __init__(
        self,
        config: TranscribeConfig,
        *,
        transport: _Transport | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Create the backend without probing or assuming any local service."""
        configured_url = getattr(config, "meta_asr_url", None)
        if not configured_url or not str(configured_url).strip():
            raise ValueError(
                "meta_asr_url is required for Meta ASR; OctoScribe will not "
                "implicitly call Llama or Ollama"
            )

        configured_model = getattr(config, "meta_asr_model", None)
        if not configured_model or not str(configured_model).strip():
            raise ValueError("meta_asr_model is required for Meta ASR")

        language = getattr(
            config,
            "meta_asr_language",
            getattr(config, "language", None),
        )
        if not language or not str(language).strip():
            raise ValueError("language is required for Meta ASR")

        configured_key = getattr(config, "meta_asr_api_key", None)
        resolved_key = (
            api_key
            if api_key is not None
            else os.environ.get("META_ASR_API_KEY") or configured_key
        )
        resolved_timeout = (
            timeout
            if timeout is not None
            else getattr(
                config, "provider_timeout_seconds", _DEFAULT_TIMEOUT_SECONDS
            )
        )
        if resolved_timeout is None or float(resolved_timeout) <= 0:
            raise ValueError("Meta ASR timeout must be greater than zero")

        self._config = config
        self._endpoint = self._resolve_endpoint(str(configured_url))
        self._model = str(configured_model).strip()
        self._language = str(language).strip()
        self._api_key = str(resolved_key).strip() if resolved_key else None
        self._timeout = float(resolved_timeout)
        self._transport = transport or _secure_urlopen
        self._retry = RetryPolicy(
            attempts=config.retry_attempts,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
        )

    @property
    def name(self) -> str:
        """Return the stable provider identifier recorded in provenance."""
        return "meta"

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe *audio_path* without altering the provider's text."""
        path = Path(audio_path)
        self._validate_audio_path(path)

        def _attempt() -> str:
            request = self._build_request(path)
            payload, content_type = self._send(request)
            return self._parse_transcript(payload, content_type=content_type)

        return self._retry.run(_attempt, label=path.name)

    @staticmethod
    def _resolve_endpoint(configured_url: str) -> str:
        """Accept a base URL, ``.../v1``, or the complete transcription URL."""
        raw_url = configured_url.strip()
        parsed = urllib.parse.urlsplit(raw_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "meta_asr_url must be an absolute http:// or https:// URL"
            )
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("meta_asr_url must not contain embedded credentials")
        if parsed.scheme == "http":
            hostname = parsed.hostname or ""
            try:
                loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                loopback = hostname.casefold() == "localhost"
            if not loopback:
                raise ValueError(
                    "unencrypted Meta ASR URLs are allowed only on loopback; "
                    "use https:// for remote endpoints"
                )
        if parsed.fragment:
            raise ValueError("meta_asr_url must not contain a URL fragment")
        if parsed.query:
            raise ValueError(
                "meta_asr_url must not contain a query string; put credentials "
                "in META_ASR_API_KEY"
            )

        path = parsed.path.rstrip("/")
        if path.endswith(_TRANSCRIPTIONS_PATH):
            resolved_path = path
        elif path.endswith("/v1"):
            resolved_path = f"{path}/audio/transcriptions"
        else:
            resolved_path = f"{path}{_TRANSCRIPTIONS_PATH}"

        return urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, resolved_path, "", "")
        )

    @staticmethod
    def _validate_audio_path(audio_path: Path) -> None:
        """Reject missing, non-regular, or empty audio before HTTP begins."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        if not audio_path.is_file():
            raise ValueError(f"Audio path is not a regular file: {audio_path}")
        if audio_path.stat().st_size <= 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

    def _build_request(self, audio_path: Path) -> urllib.request.Request:
        """Build one fresh multipart request for a single retry attempt."""
        boundary = f"octoscribe-meta-{uuid.uuid4().hex}"
        body = self._multipart_body(audio_path, boundary)
        headers = {
            "Accept": "application/json, text/plain",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "OctoScribe/Meta-ASR",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        return urllib.request.Request(
            self._endpoint,
            data=body,
            headers=headers,
            method="POST",
        )

    def _multipart_body(self, audio_path: Path, boundary: str) -> bytes:
        """Encode model, language, and the audio file as multipart data."""
        marker = boundary.encode("ascii")
        chunks: list[bytes] = []

        for name, value in (
            ("model", self._model),
            ("language", self._language),
        ):
            chunks.extend(
                (
                    b"--" + marker + b"\r\n",
                    f'Content-Disposition: form-data; name="{name}"\r\n'.encode(
                        "ascii"
                    ),
                    b"\r\n",
                    value.encode("utf-8") + b"\r\n",
                )
            )

        filename = self._safe_header_value(audio_path.name)
        content_type = (
            mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
        )
        with audio_path.open("rb") as audio_file:
            audio_bytes = audio_file.read()
        if not audio_bytes:
            raise ValueError(f"Audio file became empty while reading: {audio_path}")

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

    def _send(
        self, request: urllib.request.Request
    ) -> tuple[bytes, str | None]:
        """Execute one HTTP attempt and close its response in every case."""
        try:
            response = self._open(request)
        except urllib.error.HTTPError as exc:
            try:
                exc.close()
            finally:
                raise RuntimeError(
                    f"Meta ASR request failed with HTTP {exc.code}: {exc.reason}"
                ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Meta ASR network error: {exc.reason}") from exc

        try:
            status = self._response_status(response)
            content_type = self._response_content_type(response)
            payload = response.read()
        finally:
            response.close()

        if status is not None and not 200 <= status < 300:
            raise RuntimeError(f"Meta ASR request failed with HTTP {status}")
        if not isinstance(payload, bytes):
            raise RuntimeError("Meta ASR returned a non-binary HTTP response body")
        return payload, content_type

    def _open(self, request: urllib.request.Request) -> _Response:
        """Invoke either a callable transport or an opener-style client."""
        if callable(self._transport):
            return self._transport(request, timeout=self._timeout)

        opener = getattr(self._transport, "open", None)
        if callable(opener):
            return opener(request, timeout=self._timeout)
        raise TypeError(
            "Meta ASR transport must be callable or provide open(request, timeout=)"
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
    def _response_content_type(response: _Response) -> str | None:
        """Read Content-Type without requiring a specific response class."""
        headers = getattr(response, "headers", None)
        if headers is None:
            return None
        getter = getattr(headers, "get", None)
        value = getter("Content-Type") if callable(getter) else None
        return str(value) if value is not None else None

    @staticmethod
    def _parse_transcript(payload: bytes, *, content_type: str | None) -> str:
        """Parse JSON or plain text while preserving the transcript exactly."""
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("Meta ASR returned a non-UTF-8 response") from exc

        media_type = (content_type or "").split(";", 1)[0].strip().lower()
        explicitly_json = media_type == "application/json" or media_type.endswith(
            "+json"
        )
        explicitly_text = media_type == "text/plain"

        if explicitly_text:
            transcript = decoded
        elif explicitly_json or decoded.lstrip().startswith("{"):
            transcript = MetaASRBackend._text_from_json(decoded)
        else:
            transcript = decoded

        if not transcript.strip():
            raise RuntimeError("Meta ASR returned an empty transcript")
        return transcript

    @staticmethod
    def _text_from_json(decoded: str) -> str:
        """Extract and validate the OpenAI-compatible JSON ``text`` field."""
        try:
            document = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Meta ASR returned malformed JSON") from exc

        if not isinstance(document, dict):
            raise RuntimeError("Meta ASR JSON response must be an object")
        if "text" not in document:
            raise RuntimeError("Meta ASR JSON response is missing the 'text' field")

        transcript = document["text"]
        if not isinstance(transcript, str):
            raise RuntimeError("Meta ASR JSON field 'text' must be a string")
        return transcript
