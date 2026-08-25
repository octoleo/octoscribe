"""
src/transcribe/backends/openai_backend.py — OpenAI transcription backend.

Wraps the OpenAI audio transcription API (recommended: ``gpt-transcribe``)
and always sends the :data:`~src.transcribe.prompt.VERBATIM_PROMPT`.  All retry
and error-classification concerns are delegated to
:class:`~src.transcribe.backends.retry.RetryPolicy`, leaving this class with a
single job: perform one API call.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import TranscribeConfig
from src.transcribe.backends.base import TranscriptionBackend
from src.transcribe.backends.retry import RetryPolicy
from src.transcribe.prompt import VERBATIM_PROMPT

log = logging.getLogger(__name__)


class OpenAIBackend(TranscriptionBackend):
    """
    Transcribe audio via the OpenAI API, retrying transient failures.

    The OpenAI client is created lazily in ``__init__`` (the ``openai`` package
    is imported there) so the dependency is only required when this backend is
    actually selected.
    """

    def __init__(self, config: TranscribeConfig) -> None:
        if str(config.model).casefold().startswith("gpt-4o-transcribe-diarize"):
            raise ValueError(
                "OpenAI diarization models are not supported by OctoScribe's "
                "verbatim chunk pipeline; select gpt-transcribe instead"
            )
        import openai

        self._config = config
        self._client = openai.OpenAI(
            api_key=config.api_key,
            timeout=float(getattr(config, "provider_timeout_seconds", 900.0)),
            # RetryPolicy below is the single, auditable owner of retries.
            # Leaving SDK retries enabled would silently multiply paid uploads.
            max_retries=0,
        )
        # Resilience is the policy's responsibility, not ours.
        self._retry = RetryPolicy(
            attempts=config.retry_attempts,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
        )

    @property
    def name(self) -> str:
        return "openai"

    def transcribe(self, audio_path: Path) -> str:
        """
        Transcribe *audio_path* via OpenAI, retrying on transient errors.

        Each attempt opens the file freshly (a retried request needs a fresh,
        rewound stream) and requests plain-text output so the API returns the
        transcript directly.
        """
        cfg = self._config
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"Audio file is empty: {path}")
        if path.stat().st_size > 25 * 1024 * 1024:
            raise ValueError(
                "OpenAI transcription files must be 25 MB or smaller; "
                "run the OctoScribe chunking pipeline for long recordings"
            )

        def _attempt() -> str:
            with open(path, "rb") as audio_file:
                request: dict[str, object] = {
                    "file": audio_file,
                    "model": cfg.model,
                    "prompt": VERBATIM_PROMPT,
                    # Use the API's lowest-randomness setting for every
                    # supported OpenAI transcription model.  The service may
                    # still raise temperature automatically when log-
                    # probability thresholds are reached, so this reduces
                    # variability without claiming perfect determinism.
                    "temperature": 0,
                }
                if cfg.model == "gpt-transcribe":
                    # The current API accepts a ranked language list.  Do not
                    # request text response format: the SDK returns an object
                    # whose .text field is the exact provider transcript.
                    if cfg.language:
                        # languages is a current API field that the Python SDK
                        # accepts through its forward-compatible extra body.
                        request["extra_body"] = {"languages": [cfg.language]}
                elif cfg.model == "whisper-1":
                    # whisper-1 supports the plain-text response format.
                    request["language"] = cfg.language
                    request["response_format"] = "text"
                else:
                    # gpt-4o transcription models support JSON, not the
                    # whisper-1-only text response format.  Leaving the format
                    # unset selects the API's JSON default, which the parser
                    # below handles without altering its text field.
                    request["language"] = cfg.language
                result = self._client.audio.transcriptions.create(**request)

            if isinstance(result, str):
                transcript = result
            elif isinstance(result, dict):
                transcript = result.get("text")
            else:
                transcript = getattr(result, "text", None)
            if not isinstance(transcript, str) or not transcript.strip():
                raise RuntimeError("OpenAI returned an empty or malformed transcript")
            return transcript

        return self._retry.run(_attempt, label=path.name)
