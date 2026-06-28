"""
src/transcribe/backends/retry.py — Reusable retry policy and error classifier.

Network transcription is the least reliable part of the pipeline, so its
resilience logic deserves to be a first-class, independently testable unit
rather than something buried inside an API call.  This module separates two
concerns that were previously entangled in ``OpenAIBackend.transcribe``:

* :class:`ErrorClassifier` decides whether a raised exception is *transient*
  (worth retrying) or *permanent* (pointless to retry).
* :class:`RetryPolicy` owns the retry loop itself: exponential backoff with
  jitter, the attempt budget, and the final "exhausted" error.

A backend supplies an *operation* (a zero-argument callable that performs one
attempt) and the policy takes care of the rest.  This is a Single-Responsibility
split that also makes the backoff behaviour easy to unit-test in isolation.

The classifier uses substring matching on the lower-cased exception text.  This
is intentionally conservative: a truly *unknown* error is treated as permanent
(re-raised immediately) so we never hammer an endpoint over an error we do not
understand.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, Optional, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

# Error fragments that indicate a transient (retryable) failure.
RETRYABLE_PATTERNS: tuple[str, ...] = (
    "rate limit",
    "rate_limit",
    "429",
    "too many requests",
    "connection",
    "network",
    "dns",
    "unreachable",
    "reset by peer",
    "timeout",
    "timed out",
    "deadline",
    "500",
    "502",
    "503",
    "504",
    "internal server error",
    "bad gateway",
    "service unavailable",
)

# Error fragments that indicate a permanent (non-retryable) failure.
PERMANENT_PATTERNS: tuple[str, ...] = (
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "invalid api key",
    "authentication",
    "invalid audio",
    "unsupported format",
    "corrupt",
    "could not process",
    "invalid file",
)


class ErrorClassifier:
    """
    Classifies exceptions as permanent or transient by inspecting their text.

    ``permanent`` is checked first so that, for example, a ``401`` error is
    never mistaken for retryable even if its message also mentions a
    "connection".
    """

    def __init__(
        self,
        retryable: tuple[str, ...] = RETRYABLE_PATTERNS,
        permanent: tuple[str, ...] = PERMANENT_PATTERNS,
    ) -> None:
        self._retryable = retryable
        self._permanent = permanent

    def is_permanent(self, exc: Exception) -> bool:
        """Return ``True`` if *exc* should never be retried."""
        text = str(exc).lower()
        return any(pat in text for pat in self._permanent)

    def is_retryable(self, exc: Exception) -> bool:
        """Return ``True`` if *exc* looks transient and is worth retrying."""
        text = str(exc).lower()
        return any(pat in text for pat in self._retryable)


class RetryPolicy:
    """
    Runs an operation with bounded exponential-backoff retries.

    The attempt budget is ``attempts`` *retries* on top of the initial try, so
    the operation runs at most ``attempts + 1`` times.  Backoff grows as
    ``base_delay * 2**n`` (capped at ``max_delay``) with +/-15% jitter to avoid
    thundering-herd retries against a recovering service.
    """

    def __init__(
        self,
        attempts: int,
        base_delay: float,
        max_delay: float,
        classifier: Optional[ErrorClassifier] = None,
    ) -> None:
        self._attempts = attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._classifier = classifier or ErrorClassifier()

    def run(self, operation: Callable[[], T], *, label: str) -> T:
        """
        Execute *operation*, retrying transient failures up to the budget.

        Parameters
        ----------
        operation:
            Zero-argument callable performing exactly one attempt.
        label:
            Human-readable identifier (e.g. a filename) used in log messages
            and the final exhaustion error.

        Behaviour
        ---------
        * A *permanent* error is re-raised immediately, unchanged.
        * An *unknown* error (matching neither list) is also re-raised
          immediately — we do not retry failures we cannot reason about.
        * A *transient* error triggers backoff-and-retry until the budget is
          exhausted, after which a :class:`RuntimeError` is raised, chained to
          the last transient error.
        """
        last_error: Exception | None = None

        for attempt in range(self._attempts + 1):
            try:
                return operation()
            except Exception as exc:  # noqa: BLE001 — classification decides handling
                # Permanent failures: raise immediately without retrying.
                if self._classifier.is_permanent(exc):
                    raise

                # Transient failures: back off and try again (budget permitting).
                if self._classifier.is_retryable(exc):
                    last_error = exc
                    if attempt >= self._attempts:
                        break
                    delay = self._backoff_delay(attempt)
                    log.warning(
                        "Transient error on attempt %d/%d for %s: %s — "
                        "retrying in %.1fs",
                        attempt + 1,
                        self._attempts + 1,
                        label,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                # Unknown error: treat as permanent.
                raise

        raise RuntimeError(
            f"Exhausted {self._attempts} retries for {label}"
        ) from last_error

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff for *attempt*, capped at ``max_delay``, with jitter."""
        delay = min(self._max_delay, self._base_delay * (2 ** attempt))
        return delay * (0.85 + random.random() * 0.3)  # +/-15% jitter
