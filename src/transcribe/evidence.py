"""Immutable, deterministic provenance records for transcription evidence.

The schema is intentionally strict.  It accepts audio/chunk identity,
provider/model identity, attempt numbers, raw transcript text, comparison
counts, and a final quality state.  It has no catch-all metadata mapping in
which credentials, authorization headers, or provider request objects could be
accidentally persisted.

Candidate files are append-only by identity: writing byte-identical JSON is an
idempotent success, while trying to reuse the same provider/model/attempt for
different evidence raises :class:`EvidenceConflictError`.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.audio import sha256_file
from src.persistence import atomic_write_text
from src.transcribe.consensus import QualityState


SCHEMA_VERSION = "1.2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class EvidenceError(RuntimeError):
    """Base class for evidence persistence failures."""


class HashMismatchError(EvidenceError):
    """Raised when recorded content identity differs from the actual file."""


class EvidenceConflictError(EvidenceError):
    """Raised when immutable evidence would be replaced or removed."""


def _validate_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lower-case SHA-256 hex digest")


def _validate_label(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if _CONTROL_RE.search(value):
        raise ValueError(f"{field_name} must not contain control characters")


def _validate_seconds(value: float, field_name: str, *, positive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    if value != value or value in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} must be finite")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """Return stable UTF-8 JSON text with a final newline."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _transcript_sha256(raw_transcript: str) -> str:
    return hashlib.sha256(raw_transcript.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AudioEvidence:
    """Identity and duration of the immutable source recording."""

    path: str
    sha256: str
    duration_seconds: float

    def __post_init__(self) -> None:
        _validate_label(self.path, "audio path")
        _validate_sha256(self.sha256, "audio sha256")
        _validate_seconds(
            self.duration_seconds,
            "audio duration_seconds",
            positive=False,
        )

    @classmethod
    def from_file(
        cls,
        file_path: Path,
        duration_seconds: float,
        *,
        recorded_path: str | Path | None = None,
    ) -> "AudioEvidence":
        """Build evidence from an actual file and optional logical path."""
        file_path = Path(file_path)
        logical_path = file_path if recorded_path is None else recorded_path
        return cls(
            path=str(logical_path),
            sha256=sha256_file(file_path),
            duration_seconds=duration_seconds,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible audio object."""
        return {
            "duration_seconds": self.duration_seconds,
            "path": self.path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TimedWordEvidence:
    """Optional provider word timing retained without altering its text."""

    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    confidence: float | None = None
    speaker: int | str | None = None

    def __post_init__(self) -> None:
        _validate_label(self.text, "timed word text")
        for field_name in ("start_seconds", "end_seconds"):
            value = getattr(self, field_name)
            if value is not None:
                _validate_seconds(value, f"timed word {field_name}", positive=False)
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("timed word end cannot precede its start")
        if self.confidence is not None:
            _validate_seconds(
                self.confidence,
                "timed word confidence",
                positive=False,
            )
            if self.confidence > 1:
                raise ValueError("timed word confidence must not exceed one")
        if isinstance(self.speaker, bool) or not isinstance(
            self.speaker, (int, str, type(None))
        ):
            raise TypeError("timed word speaker must be an integer, string, or None")
        if isinstance(self.speaker, str):
            _validate_label(self.speaker, "timed word speaker")

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "end_seconds": self.end_seconds,
            "speaker": self.speaker,
            "start_seconds": self.start_seconds,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class ProviderAttemptEvidence:
    """One provider's unedited output for one chunk and attempt number."""

    provider: str
    model: str
    attempt: int
    raw_transcript: str
    words: tuple[TimedWordEvidence, ...] = ()
    language: str | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        _validate_label(self.provider, "provider")
        _validate_label(self.model, "model")
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ValueError("attempt must be a positive integer")
        if not isinstance(self.raw_transcript, str):
            raise TypeError("raw_transcript must be a string")
        if not isinstance(self.words, tuple) or not all(
            isinstance(word, TimedWordEvidence) for word in self.words
        ):
            raise TypeError("words must contain TimedWordEvidence values")
        if self.language is not None:
            _validate_label(self.language, "provider language")
        if self.duration_seconds is not None:
            _validate_seconds(
                self.duration_seconds,
                "provider duration_seconds",
                positive=False,
            )

    @property
    def identity(self) -> tuple[str, str, int]:
        """Return the immutable provider/model/attempt identity."""
        return (self.provider, self.model, self.attempt)

    @property
    def transcript_sha256(self) -> str:
        """Return the byte identity of the UTF-8 raw transcript."""
        return _transcript_sha256(self.raw_transcript)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible provider attempt object."""
        return {
            "attempt": self.attempt,
            "model": self.model,
            "provider": self.provider,
            "raw_transcript": self.raw_transcript,
            "transcript_sha256": self.transcript_sha256,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "words": [word.to_dict() for word in self.words],
        }


@dataclass(frozen=True, slots=True)
class ChunkEvidence:
    """Audio boundaries, identity, and provider attempts for one chunk."""

    index: int
    path: str
    start_seconds: float
    end_seconds: float
    sha256: str
    attempts: tuple[ProviderAttemptEvidence, ...] = ()
    canonical_provider: str | None = None
    canonical_attempt: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ValueError("chunk index must be a non-negative integer")
        _validate_label(self.path, "chunk path")
        _validate_seconds(self.start_seconds, "chunk start_seconds", positive=False)
        _validate_seconds(self.end_seconds, "chunk end_seconds", positive=True)
        if self.end_seconds <= self.start_seconds:
            raise ValueError("chunk end_seconds must follow start_seconds")
        _validate_sha256(self.sha256, "chunk sha256")
        if not isinstance(self.attempts, tuple):
            raise TypeError("chunk attempts must be a tuple")
        identities: set[tuple[str, str, int]] = set()
        for attempt in self.attempts:
            if not isinstance(attempt, ProviderAttemptEvidence):
                raise TypeError("chunk attempts must contain provider evidence")
            if attempt.identity in identities:
                raise ValueError(
                    "a chunk cannot contain duplicate provider/model/attempt identities"
                )
            identities.add(attempt.identity)
        if (self.canonical_provider is None) != (self.canonical_attempt is None):
            raise ValueError(
                "canonical_provider and canonical_attempt must be set together"
            )
        if self.canonical_provider is not None:
            _validate_label(self.canonical_provider, "canonical_provider")
            if self.canonical_attempt is None or self.canonical_attempt < 1:
                raise ValueError("canonical_attempt must be a positive integer")
            if not any(
                attempt.provider == self.canonical_provider
                and attempt.attempt == self.canonical_attempt
                for attempt in self.attempts
            ):
                raise ValueError("canonical selection must reference a recorded attempt")

    @classmethod
    def from_file(
        cls,
        file_path: Path,
        *,
        index: int,
        start_seconds: float,
        end_seconds: float,
        attempts: tuple[ProviderAttemptEvidence, ...] = (),
        recorded_path: str | Path | None = None,
    ) -> "ChunkEvidence":
        """Build chunk evidence from the exact bytes sent to providers."""
        file_path = Path(file_path)
        logical_path = file_path if recorded_path is None else recorded_path
        return cls(
            index=index,
            path=str(logical_path),
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            sha256=sha256_file(file_path),
            attempts=attempts,
        )

    def to_dict(self, *, include_attempts: bool = True) -> dict[str, Any]:
        """Return the JSON-compatible chunk object."""
        payload: dict[str, Any] = {
            "end_seconds": self.end_seconds,
            "index": self.index,
            "path": self.path,
            "sha256": self.sha256,
            "start_seconds": self.start_seconds,
        }
        if include_attempts:
            payload["attempts"] = [
                attempt.to_dict()
                for attempt in sorted(
                    self.attempts,
                    key=lambda item: item.identity,
                )
            ]
        payload["canonical"] = (
            {
                "attempt": self.canonical_attempt,
                "provider": self.canonical_provider,
            }
            if self.canonical_provider is not None
            else None
        )
        return payload


@dataclass(frozen=True, slots=True)
class DiscrepancySummary:
    """Exact token spans retained so each warning can be located precisely."""

    kind: str
    priority: str
    left_start: int
    left_end: int
    right_start: int
    right_end: int
    left_tokens: tuple[str, ...] = ()
    right_tokens: tuple[str, ...] = ()
    critical_categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"addition", "deletion", "substitution"}:
            raise ValueError("unsupported discrepancy kind")
        if self.priority not in {"standard", "critical"}:
            raise ValueError("unsupported discrepancy priority")
        for start_name, end_name in (
            ("left_start", "left_end"),
            ("right_start", "right_end"),
        ):
            start = getattr(self, start_name)
            end = getattr(self, end_name)
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or start < 0
                or end < start
            ):
                raise ValueError("discrepancy spans require 0 <= start <= end")
        for field_name in ("left_tokens", "right_tokens"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or not all(
                isinstance(value, str) and not _CONTROL_RE.search(value)
                for value in values
            ):
                raise TypeError(f"{field_name} must be a tuple of safe strings")
        allowed_categories = {"negation", "number", "scripture_reference"}
        if (
            not isinstance(self.critical_categories, tuple)
            or tuple(sorted(set(self.critical_categories)))
            != self.critical_categories
            or not set(self.critical_categories).issubset(allowed_categories)
        ):
            raise ValueError(
                "critical_categories must be a sorted unique tuple of known values"
            )
        if self.priority == "standard" and self.critical_categories:
            raise ValueError("standard discrepancies cannot have critical categories")

    def to_dict(self) -> dict[str, Any]:
        return {
            "critical_categories": list(self.critical_categories),
            "kind": self.kind,
            "left": {
                "end": self.left_end,
                "start": self.left_start,
                "tokens": list(self.left_tokens),
            },
            "priority": self.priority,
            "right": {
                "end": self.right_end,
                "start": self.right_start,
                "tokens": list(self.right_tokens),
            },
        }


@dataclass(frozen=True, slots=True)
class ComparisonSummary:
    """Non-generative counts describing one pairwise transcript comparison."""

    chunk_index: int | None
    left_provider: str
    left_model: str
    left_attempt: int
    right_provider: str
    right_model: str
    right_attempt: int
    agrees: bool
    discrepancy_count: int
    critical_discrepancy_count: int = 0
    additions: int = 0
    deletions: int = 0
    substitutions: int = 0
    stage: str = "comparison"
    pass_index: int = 0
    discrepancies: tuple[DiscrepancySummary, ...] = ()

    def __post_init__(self) -> None:
        if self.chunk_index is not None and (
            isinstance(self.chunk_index, bool)
            or not isinstance(self.chunk_index, int)
            or self.chunk_index < 0
        ):
            raise ValueError("comparison chunk_index must be non-negative or None")
        for field_name in (
            "left_provider",
            "left_model",
            "right_provider",
            "right_model",
        ):
            _validate_label(getattr(self, field_name), field_name)
        for field_name in ("left_attempt", "right_attempt"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(self.agrees, bool):
            raise TypeError("agrees must be a boolean")
        for field_name in (
            "discrepancy_count",
            "critical_discrepancy_count",
            "additions",
            "deletions",
            "substitutions",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        detailed_count = self.additions + self.deletions + self.substitutions
        if detailed_count not in {0, self.discrepancy_count}:
            raise ValueError(
                "addition/deletion/substitution counts must sum to discrepancy_count"
            )
        if self.critical_discrepancy_count > self.discrepancy_count:
            raise ValueError(
                "critical_discrepancy_count cannot exceed discrepancy_count"
            )
        if self.agrees != (self.discrepancy_count == 0):
            raise ValueError(
                "agrees must be true exactly when discrepancy_count is zero"
            )
        _validate_label(self.stage, "comparison stage")
        if (
            isinstance(self.pass_index, bool)
            or not isinstance(self.pass_index, int)
            or self.pass_index < 0
        ):
            raise ValueError("comparison pass_index must be non-negative")
        if not isinstance(self.discrepancies, tuple) or not all(
            isinstance(item, DiscrepancySummary) for item in self.discrepancies
        ):
            raise TypeError("discrepancies must contain DiscrepancySummary values")
        if self.discrepancies:
            if len(self.discrepancies) != self.discrepancy_count:
                raise ValueError(
                    "detailed discrepancies must match discrepancy_count"
                )
            kinds = [item.kind for item in self.discrepancies]
            if (
                kinds.count("addition") != self.additions
                or kinds.count("deletion") != self.deletions
                or kinds.count("substitution") != self.substitutions
            ):
                raise ValueError("detailed discrepancy kinds must match counts")
            critical = sum(item.priority == "critical" for item in self.discrepancies)
            if critical != self.critical_discrepancy_count:
                raise ValueError("detailed critical discrepancies must match count")

    @property
    def sort_key(self) -> tuple[Any, ...]:
        """Return deterministic report ordering fields."""
        return (
            -1 if self.chunk_index is None else self.chunk_index,
            self.pass_index,
            self.left_provider,
            self.left_model,
            self.left_attempt,
            self.right_provider,
            self.right_model,
            self.right_attempt,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible comparison summary."""
        return {
            "additions": self.additions,
            "agrees": self.agrees,
            "chunk_index": self.chunk_index,
            "critical_discrepancy_count": self.critical_discrepancy_count,
            "deletions": self.deletions,
            "discrepancy_count": self.discrepancy_count,
            "discrepancies": [item.to_dict() for item in self.discrepancies],
            "left": {
                "attempt": self.left_attempt,
                "model": self.left_model,
                "provider": self.left_provider,
            },
            "pass_index": self.pass_index,
            "right": {
                "attempt": self.right_attempt,
                "model": self.right_model,
                "provider": self.right_provider,
            },
            "substitutions": self.substitutions,
            "stage": self.stage,
        }


@dataclass(frozen=True, slots=True)
class ProviderFailureSummary:
    """Bounded provider failure retained in the immutable aggregate report."""

    chunk_index: int
    provider: str
    attempt: int
    role: str
    error: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.chunk_index, bool)
            or not isinstance(self.chunk_index, int)
            or self.chunk_index < 0
        ):
            raise ValueError("failure chunk_index must be non-negative")
        _validate_label(self.provider, "failure provider")
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ValueError("failure attempt must be positive")
        _validate_label(self.role, "failure role")
        _validate_label(self.error, "failure error")
        if len(self.error) > 500:
            raise ValueError("failure error must be at most 500 characters")

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (self.chunk_index, self.provider, self.attempt, self.role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "chunk_index": self.chunk_index,
            "error": self.error,
            "provider": self.provider,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class SeamSummary:
    """Deterministic, non-generative alignment result for adjacent chunks."""

    left_chunk: int
    right_chunk: int
    aligned: bool
    duplicate_tokens_removed: int = 0
    exact_matches: int = 0
    edits: int = 0
    similarity: float | None = None

    def __post_init__(self) -> None:
        if self.left_chunk < 0 or self.right_chunk != self.left_chunk + 1:
            raise ValueError("a seam must join adjacent non-negative chunk indices")
        for name in ("duplicate_tokens_removed", "exact_matches", "edits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.similarity is not None and not 0 <= self.similarity <= 1:
            raise ValueError("seam similarity must be between zero and one")
        if not self.aligned and any(
            (self.duplicate_tokens_removed, self.exact_matches, self.edits)
        ):
            raise ValueError("an unaligned seam cannot claim removed or matched tokens")

    def to_dict(self) -> dict[str, Any]:
        return {
            "aligned": self.aligned,
            "duplicate_tokens_removed": self.duplicate_tokens_removed,
            "edits": self.edits,
            "exact_matches": self.exact_matches,
            "left_chunk": self.left_chunk,
            "right_chunk": self.right_chunk,
            "similarity": self.similarity,
        }


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Strict schema for one persisted provider candidate."""

    audio: AudioEvidence
    chunk: ChunkEvidence
    attempt: ProviderAttemptEvidence
    run_id: str = "legacy"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence schema {self.schema_version!r}")
        _validate_label(self.run_id, "run_id")
        if self.chunk.end_seconds > self.audio.duration_seconds:
            raise ValueError("chunk boundaries exceed the source audio duration")
        matching = [
            item
            for item in self.chunk.attempts
            if item.identity == self.attempt.identity
        ]
        if matching and matching[0] != self.attempt:
            raise EvidenceConflictError(
                "chunk already contains different bytes for this attempt"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a complete JSON-compatible candidate payload."""
        chunk_payload = self.chunk.to_dict(include_attempts=False)
        # Canonical selection is a run-level decision, not an attribute of one
        # provider's immutable raw candidate.  Excluding it also makes an
        # eagerly persisted candidate byte-identical to the final report pass.
        chunk_payload.pop("canonical", None)
        return {
            "attempt": self.attempt.to_dict(),
            "audio": self.audio.to_dict(),
            "chunk": chunk_payload,
            "kind": "transcription_candidate",
            "run_id": self.run_id,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class EvidenceReport:
    """Aggregate evidence and final, truthful quality state for one audio file."""

    audio: AudioEvidence
    chunks: tuple[ChunkEvidence, ...]
    comparisons: tuple[ComparisonSummary, ...]
    final_quality_state: QualityState
    seams: tuple[SeamSummary, ...] = ()
    failures: tuple[ProviderFailureSummary, ...] = ()
    primary_provider: str | None = None
    final_transcript_sha256: str | None = None
    audio_revision: str | None = None
    audio_repository_branch: str | None = None
    run_id: str = "legacy"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence schema {self.schema_version!r}")
        _validate_label(self.run_id, "run_id")
        if self.primary_provider is not None:
            _validate_label(self.primary_provider, "primary_provider")
        if self.final_transcript_sha256 is not None:
            _validate_sha256(
                self.final_transcript_sha256,
                "final transcript sha256",
            )
        if self.audio_revision is not None and _GIT_REVISION_RE.fullmatch(
            self.audio_revision
        ) is None:
            raise ValueError("audio_revision must be a 40- or 64-character Git hash")
        if self.audio_repository_branch is not None:
            _validate_label(
                self.audio_repository_branch,
                "audio_repository_branch",
            )
        if not isinstance(self.chunks, tuple):
            raise TypeError("report chunks must be a tuple")
        if not isinstance(self.comparisons, tuple):
            raise TypeError("report comparisons must be a tuple")
        if not isinstance(self.final_quality_state, QualityState):
            raise TypeError("final_quality_state must be a QualityState")
        if not isinstance(self.seams, tuple) or not all(
            isinstance(seam, SeamSummary) for seam in self.seams
        ):
            raise TypeError("report seams must contain SeamSummary values")
        if not isinstance(self.failures, tuple) or not all(
            isinstance(failure, ProviderFailureSummary)
            for failure in self.failures
        ):
            raise TypeError("report failures must contain ProviderFailureSummary values")

        chunk_indices: set[int] = set()
        for chunk in self.chunks:
            if not isinstance(chunk, ChunkEvidence):
                raise TypeError("report chunks must contain ChunkEvidence")
            if chunk.index in chunk_indices:
                raise ValueError("report chunk indices must be unique")
            if chunk.end_seconds > self.audio.duration_seconds:
                raise ValueError("chunk boundaries exceed the source audio duration")
            chunk_indices.add(chunk.index)
        if chunk_indices and chunk_indices != set(range(len(chunk_indices))):
            raise ValueError("report chunk indices must be contiguous from zero")

        for comparison in self.comparisons:
            if not isinstance(comparison, ComparisonSummary):
                raise TypeError("report comparisons must contain ComparisonSummary")
            if (
                comparison.chunk_index is not None
                and comparison.chunk_index not in chunk_indices
            ):
                raise ValueError("comparison refers to an unknown chunk index")
        for failure in self.failures:
            if failure.chunk_index not in chunk_indices:
                raise ValueError("failure refers to an unknown chunk index")
        expected_seams = {
            (index, index + 1) for index in range(max(0, len(chunk_indices) - 1))
        }
        actual_seams = {(seam.left_chunk, seam.right_chunk) for seam in self.seams}
        if actual_seams and actual_seams != expected_seams:
            raise ValueError("report seams must cover each adjacent chunk exactly once")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible report payload."""
        return {
            "audio": self.audio.to_dict(),
            "audio_repository": {
                "branch": self.audio_repository_branch,
                "revision": self.audio_revision,
            },
            "chunks": [
                chunk.to_dict()
                for chunk in sorted(self.chunks, key=lambda item: item.index)
            ],
            "comparisons": [
                comparison.to_dict()
                for comparison in sorted(
                    self.comparisons,
                    key=lambda item: item.sort_key,
                )
            ],
            "final_quality_state": self.final_quality_state.value,
            "final_transcript_sha256": self.final_transcript_sha256,
            "failures": [
                failure.to_dict()
                for failure in sorted(
                    self.failures,
                    key=lambda item: item.sort_key,
                )
            ],
            "kind": "transcription_evidence_report",
            "primary_provider": self.primary_provider,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "seams": [
                seam.to_dict()
                for seam in sorted(
                    self.seams,
                    key=lambda item: (item.left_chunk, item.right_chunk),
                )
            ],
        }


def _safe_component(value: str, *, maximum: int = 48) -> str:
    """Return an ASCII filename component with no path semantics."""
    canonical = unicodedata.normalize("NFKD", value)
    ascii_value = canonical.encode("ascii", "ignore").decode("ascii")
    safe = re.sub(r"[^A-Za-z0-9]+", "-", ascii_value).strip("-").lower()
    safe = safe[:maximum].rstrip("-")
    return safe or "unnamed"


def _verify_file(path: Path, expected_sha256: str, description: str) -> None:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise HashMismatchError(
            f"{description} SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def _attempts_from_report_payload(
    payload: Mapping[str, Any],
) -> dict[tuple[int, str, str, int], Mapping[str, Any]]:
    attempts: dict[tuple[int, str, str, int], Mapping[str, Any]] = {}
    for chunk in payload.get("chunks", []):
        chunk_index = chunk["index"]
        for attempt in chunk.get("attempts", []):
            key = (
                chunk_index,
                attempt["provider"],
                attempt["model"],
                attempt["attempt"],
            )
            attempts[key] = attempt
    return attempts


def _chunk_metadata_from_report_payload(
    payload: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    chunks: dict[int, Mapping[str, Any]] = {}
    for chunk in payload.get("chunks", []):
        chunks[chunk["index"]] = {
            key: value for key, value in chunk.items() if key != "attempts"
        }
    return chunks


class EvidenceStore:
    """Atomically persist candidate and aggregate report JSON evidence."""

    def __init__(self, artifact_dir: Path, report_dir: Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.report_dir = Path(report_dir)
        self._lock = threading.RLock()

    def candidate_path(
        self,
        audio: AudioEvidence,
        chunk: ChunkEvidence,
        attempt: ProviderAttemptEvidence,
        *,
        run_id: str = "legacy",
    ) -> Path:
        """Return the deterministic, path-safe candidate destination."""
        source = _safe_component(Path(audio.path).stem)
        provider = _safe_component(attempt.provider, maximum=32)
        model = _safe_component(attempt.model, maximum=40)
        identity_material = json.dumps(
            attempt.identity,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        identity = hashlib.sha256(identity_material).hexdigest()[:12]
        filename = (
            f"{source}-{audio.sha256[:12]}-chunk-{chunk.index:04d}-"
            f"run-{_safe_component(run_id, maximum=16)}-"
            f"{provider}-{model}-attempt-{attempt.attempt:02d}-"
            f"{identity}.candidate.json"
        )
        return self.artifact_dir / filename

    def report_path(
        self, audio: AudioEvidence, *, run_id: str = "legacy"
    ) -> Path:
        """Return the deterministic, path-safe aggregate report destination."""
        source = _safe_component(Path(audio.path).stem)
        return self.report_dir / (
            f"{source}-{audio.sha256[:16]}-"
            f"run-{_safe_component(run_id, maximum=16)}.evidence.json"
        )

    def write_candidate(
        self,
        audio: AudioEvidence,
        chunk: ChunkEvidence,
        attempt: ProviderAttemptEvidence,
        *,
        audio_file: Path | None = None,
        chunk_file: Path | None = None,
        run_id: str = "legacy",
    ) -> Path:
        """Atomically persist one candidate without permitting replacement."""
        candidate = CandidateEvidence(
            audio=audio,
            chunk=chunk,
            attempt=attempt,
            run_id=run_id,
        )
        _verify_file(
            Path(audio.path) if audio_file is None else Path(audio_file),
            audio.sha256,
            "source audio",
        )
        _verify_file(
            Path(chunk.path) if chunk_file is None else Path(chunk_file),
            chunk.sha256,
            f"chunk {chunk.index}",
        )
        path = self.candidate_path(audio, chunk, attempt, run_id=run_id)
        text = _canonical_json(candidate.to_dict())
        encoded = text.encode("utf-8")

        with self._lock:
            if path.exists():
                if path.read_bytes() == encoded:
                    return path
                raise EvidenceConflictError(
                    "refusing to overwrite a provider attempt with different bytes"
                )
            atomic_write_text(path, text)
        return path

    def write_report(
        self,
        report: EvidenceReport,
        *,
        audio_file: Path | None = None,
        chunk_files: Mapping[int, Path] | None = None,
    ) -> Path:
        """Verify source bytes and atomically persist an aggregate report."""
        _verify_file(
            Path(report.audio.path) if audio_file is None else Path(audio_file),
            report.audio.sha256,
            "source audio",
        )
        resolved_chunk_files = chunk_files or {}
        for chunk in report.chunks:
            _verify_file(
                Path(resolved_chunk_files.get(chunk.index, chunk.path)),
                chunk.sha256,
                f"chunk {chunk.index}",
            )

        path = self.report_path(report.audio, run_id=report.run_id)
        payload = report.to_dict()
        text = _canonical_json(payload)
        encoded = text.encode("utf-8")
        with self._lock:
            if path.exists():
                existing_bytes = path.read_bytes()
                if existing_bytes == encoded:
                    return path
                self._validate_report_update(existing_bytes, payload)
            atomic_write_text(path, text)
        return path

    @staticmethod
    def _validate_report_update(
        existing_bytes: bytes,
        new_payload: Mapping[str, Any],
    ) -> None:
        """Reject aggregate updates that replace or remove prior evidence."""
        try:
            existing = json.loads(existing_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvidenceConflictError(
                "existing report is not valid UTF-8 JSON; refusing replacement"
            ) from exc
        if not isinstance(existing, dict):
            raise EvidenceConflictError(
                "existing report is not a JSON object; refusing replacement"
            )
        if existing.get("schema_version") != new_payload.get("schema_version"):
            raise EvidenceConflictError("refusing to replace a different schema")
        if existing.get("run_id") != new_payload.get("run_id"):
            raise EvidenceConflictError("refusing to replace a different run")
        if existing.get("audio") != new_payload.get("audio"):
            raise EvidenceConflictError("refusing to replace source audio evidence")
        for immutable_field in (
            "comparisons",
            "failures",
            "seams",
            "final_quality_state",
            "final_transcript_sha256",
            "primary_provider",
            "audio_repository",
        ):
            if existing.get(immutable_field) != new_payload.get(immutable_field):
                raise EvidenceConflictError(
                    f"refusing to replace immutable report field {immutable_field}"
                )

        old_chunks = _chunk_metadata_from_report_payload(existing)
        new_chunks = _chunk_metadata_from_report_payload(new_payload)
        for index, old_chunk in old_chunks.items():
            if index not in new_chunks or new_chunks[index] != old_chunk:
                raise EvidenceConflictError(
                    f"refusing to replace or remove chunk {index} evidence"
                )

        old_attempts = _attempts_from_report_payload(existing)
        new_attempts = _attempts_from_report_payload(new_payload)
        for identity, old_attempt in old_attempts.items():
            if identity not in new_attempts:
                raise EvidenceConflictError(
                    "refusing to remove an existing provider attempt"
                )
            if new_attempts[identity] != old_attempt:
                raise EvidenceConflictError(
                    "refusing to overwrite a provider attempt with different bytes"
                )


__all__ = [
    "SCHEMA_VERSION",
    "AudioEvidence",
    "CandidateEvidence",
    "ChunkEvidence",
    "ComparisonSummary",
    "DiscrepancySummary",
    "EvidenceConflictError",
    "EvidenceError",
    "EvidenceReport",
    "EvidenceStore",
    "HashMismatchError",
    "ProviderAttemptEvidence",
    "ProviderFailureSummary",
    "SeamSummary",
    "TimedWordEvidence",
]
