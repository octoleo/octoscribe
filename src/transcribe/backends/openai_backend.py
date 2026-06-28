"""
src/transcribe/backends/openai_backend.py — OpenAI transcription backend.

Wraps the OpenAI audio transcription API (``gpt-4o-transcribe`` or ``whisper-1``)
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
        import openai

        self._config = config
        self._client = openai.OpenAI(api_key=config.api_key)
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

        def _attempt() -> str:
            with open(audio_path, "rb") as audio_file:
                result = self._client.audio.transcriptions.create(
                    file=audio_file,
                    model=cfg.model,
                    language=cfg.language,
                    prompt=VERBATIM_PROMPT,
                    response_format="text",
                )
            # response_format="text" returns a plain string directly.
            return str(result)

        return self._retry.run(_attempt, label=audio_path.name)
