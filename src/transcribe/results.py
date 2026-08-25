"""
src/transcribe/results.py — Result and statistics value objects.

These dataclasses are the transcriber's vocabulary for reporting outcomes:
:class:`TranscriptionResult` describes one file, and :class:`BatchStats`
aggregates a whole run.  Keeping them separate from the orchestration logic
makes the orchestrator easier to read and lets callers (the CLI, tests) depend
on a small, stable reporting shape.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    """Outcome of transcribing a single audio file (success or failure)."""

    msg_id: str
    filename: str
    success: bool
    output_file: str = ""
    text: str = ""
    error: str = ""
    elapsed_seconds: float = 0.0
    model: str = ""
    quality_state: str = ""
    evidence_report: str = ""
    unresolved_discrepancies: int = 0


@dataclass
class BatchStats:
    """Running totals across a single transcription batch."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    total_elapsed_seconds: float = 0.0
    completed_with_warnings: int = 0

    def add(self, result: TranscriptionResult) -> None:
        """Incorporate a single :class:`TranscriptionResult` into the totals."""
        self.total += 1
        self.total_elapsed_seconds += result.elapsed_seconds
        if result.success:
            self.succeeded += 1
            if result.quality_state == "completed_with_warnings":
                self.completed_with_warnings += 1
        else:
            self.failed += 1

    def summary(self) -> str:
        """Return a human-readable one-line summary of the batch."""
        return (
            f"Transcription complete — "
            f"total={self.total} "
            f"succeeded={self.succeeded} "
            f"failed={self.failed} "
            f"skipped={self.skipped} "
            f"completed_with_warnings={self.completed_with_warnings} "
            f"elapsed={self.total_elapsed_seconds:.1f}s"
        )
