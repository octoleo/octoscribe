"""
tests/test_retry.py — Tests for src/transcribe/backends/retry.py.

The retry policy and error classifier were extracted from OpenAIBackend so the
resilience behaviour could be reasoned about and tested in isolation.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.transcribe.backends.retry import ErrorClassifier, RetryPolicy


# ---------------------------------------------------------------------------
# ErrorClassifier
# ---------------------------------------------------------------------------

class TestErrorClassifier:
    def setup_method(self) -> None:
        self.c = ErrorClassifier()

    @pytest.mark.parametrize("message", [
        "rate limit exceeded",
        "HTTP 429 Too Many Requests",
        "connection reset by peer",
        "request timed out",
        "503 service unavailable",
        "502 bad gateway",
    ])
    def test_retryable_messages(self, message: str) -> None:
        assert self.c.is_retryable(RuntimeError(message)) is True

    @pytest.mark.parametrize("message", [
        "401 Unauthorized",
        "invalid api key",
        "403 forbidden",
        "unsupported format",
        "could not process file",
    ])
    def test_permanent_messages(self, message: str) -> None:
        assert self.c.is_permanent(RuntimeError(message)) is True

    def test_unknown_message_is_neither(self) -> None:
        exc = RuntimeError("some totally novel error")
        assert self.c.is_permanent(exc) is False
        assert self.c.is_retryable(exc) is False


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------

class TestRetryPolicy:
    def _policy(self, attempts: int = 2) -> RetryPolicy:
        return RetryPolicy(attempts=attempts, base_delay=0.001, max_delay=0.01)

    def test_returns_on_first_success_without_sleeping(self) -> None:
        calls = []

        def op() -> str:
            calls.append(1)
            return "ok"

        with patch("time.sleep") as sleep:
            result = self._policy().run(op, label="x")

        assert result == "ok"
        assert len(calls) == 1
        sleep.assert_not_called()

    def test_retries_transient_then_succeeds(self) -> None:
        seq = [RuntimeError("503 service unavailable"), RuntimeError("timeout"), "done"]

        def op():
            item = seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with patch("time.sleep"):
            result = self._policy(attempts=3).run(op, label="x")

        assert result == "done"

    def test_permanent_error_raises_immediately(self) -> None:
        calls = []

        def op() -> str:
            calls.append(1)
            raise RuntimeError("401 invalid api key")

        with patch("time.sleep") as sleep:
            with pytest.raises(RuntimeError, match="invalid api key"):
                self._policy(attempts=3).run(op, label="x")

        assert len(calls) == 1
        sleep.assert_not_called()

    def test_unknown_error_raises_immediately(self) -> None:
        calls = []

        def op() -> str:
            calls.append(1)
            raise RuntimeError("mystery failure")

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="mystery failure"):
                self._policy(attempts=3).run(op, label="x")

        assert len(calls) == 1

    def test_exhausts_retries_and_raises_with_chained_cause(self) -> None:
        calls = []

        def op() -> str:
            calls.append(1)
            raise RuntimeError("503 service unavailable")

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="[Ee]xhausted") as exc_info:
                self._policy(attempts=2).run(op, label="sermon.ogg")

        # 1 initial attempt + 2 retries.
        assert len(calls) == 3
        assert "sermon.ogg" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)
