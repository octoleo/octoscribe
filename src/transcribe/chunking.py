"""Deterministic planning and stitching for long-form transcription.

The planner is deliberately independent of audio tooling.  A caller probes the
recording duration and (optionally) silence intervals, then injects those
millisecond values here.  That keeps policy deterministic and makes it possible
to test every boundary without ffmpeg or a model runtime.

Stitching is similarly non-generative.  It aligns normalized token sequences at
the seam, discards only a duplicated prefix from the continuation, and retains
the primary transcript verbatim.  It never asks a language model to rewrite or
"smooth" a seam.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

DEFAULT_TARGET_CORE_MS = 8 * 60 * 1_000
DEFAULT_OVERLAP_MS = 12_000
DEFAULT_HARD_MAX_MS = 10 * 60 * 1_000
DEFAULT_SILENCE_SEARCH_MS = 60_000

SilenceTimestamp: TypeAlias = int | tuple[int, int]


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """Immutable timing metadata for one transcription request.

    ``core_*`` partitions the recording exactly: adjacent cores meet at the
    same millisecond and therefore cannot create a gap.  ``context_*`` is the
    audio actually sent to a recognizer and includes half of the configured
    overlap on either side of an interior core.
    """

    index: int
    core_start_ms: int
    core_end_ms: int
    context_start_ms: int
    context_end_ms: int

    def __post_init__(self) -> None:
        values = (
            self.index,
            self.core_start_ms,
            self.core_end_ms,
            self.context_start_ms,
            self.context_end_ms,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise TypeError("chunk indices and timestamps must be integers")
        if self.index < 0:
            raise ValueError("chunk index must be non-negative")
        if self.context_start_ms < 0:
            raise ValueError("context start must be non-negative")
        if self.core_start_ms < self.context_start_ms:
            raise ValueError("context must contain the core start")
        if self.core_end_ms <= self.core_start_ms:
            raise ValueError("a chunk core must have positive duration")
        if self.context_end_ms < self.core_end_ms:
            raise ValueError("context must contain the core end")

    @property
    def core_duration_ms(self) -> int:
        """Duration of the non-overlapping ownership region."""
        return self.core_end_ms - self.core_start_ms

    @property
    def context_duration_ms(self) -> int:
        """Duration of audio sent to the transcription backend."""
        return self.context_end_ms - self.context_start_ms


def plan_chunks(
    duration_ms: int,
    silence_timestamps_ms: Iterable[SilenceTimestamp] = (),
    *,
    target_core_ms: int = DEFAULT_TARGET_CORE_MS,
    overlap_ms: int = DEFAULT_OVERLAP_MS,
    hard_max_ms: int = DEFAULT_HARD_MAX_MS,
    silence_search_ms: int = DEFAULT_SILENCE_SEARCH_MS,
) -> tuple[ChunkMetadata, ...]:
    """Plan deterministic, silence-aware chunks for a recording.

    Silence values may be point timestamps (``1234``) or inclusive intervals
    (``(1200, 1400)``).  For each target boundary the closest point in a silence
    interval inside ``silence_search_ms`` is selected; equal-distance ties go
    to the earlier point.  When no eligible silence exists, the exact target is
    used.

    ``overlap_ms`` means total overlap between adjacent context windows.  It is
    split deterministically around their shared core boundary.  The common
    10--15 second range is recommended but not enforced, which also permits
    small synthetic units in callers and tests.  Both core and context windows
    are guaranteed not to exceed ``hard_max_ms``.
    """
    _validate_plan_arguments(
        duration_ms=duration_ms,
        target_core_ms=target_core_ms,
        overlap_ms=overlap_ms,
        hard_max_ms=hard_max_ms,
        silence_search_ms=silence_search_ms,
    )
    if duration_ms == 0:
        return ()

    silences = _normalize_silences(silence_timestamps_ms, duration_ms)
    leading_context_ms = overlap_ms // 2
    trailing_context_ms = overlap_ms - leading_context_ms

    # A recording already below the hard request limit needs no seam at all.
    if duration_ms <= hard_max_ms:
        return (
            ChunkMetadata(
                index=0,
                core_start_ms=0,
                core_end_ms=duration_ms,
                context_start_ms=0,
                context_end_ms=duration_ms,
            ),
        )

    cores: list[tuple[int, int]] = []
    core_start = 0
    while core_start < duration_ms:
        # A final core has only leading context, so it may be slightly longer
        # than an interior core while the actual request remains under the cap.
        final_core_limit = hard_max_ms - leading_context_ms
        remaining_ms = duration_ms - core_start
        if remaining_ms <= final_core_limit:
            cores.append((core_start, duration_ms))
            break

        context_cost_ms = trailing_context_ms if not cores else overlap_ms
        maximum_core_ms = hard_max_ms - context_cost_ms
        ideal_boundary = core_start + target_core_ms
        maximum_boundary = core_start + maximum_core_ms
        boundary = _nearest_silence_boundary(
            ideal_ms=ideal_boundary,
            lower_ms=max(core_start + 1, ideal_boundary - silence_search_ms),
            upper_ms=min(maximum_boundary, ideal_boundary + silence_search_ms),
            silences=silences,
        )
        if boundary is None:
            boundary = min(ideal_boundary, maximum_boundary)

        cores.append((core_start, boundary))
        core_start = boundary

    chunks: list[ChunkMetadata] = []
    last_index = len(cores) - 1
    for index, (core_start, core_end) in enumerate(cores):
        context_start = (
            core_start if index == 0 else core_start - leading_context_ms
        )
        context_end = (
            core_end if index == last_index else core_end + trailing_context_ms
        )
        chunk = ChunkMetadata(
            index=index,
            core_start_ms=core_start,
            core_end_ms=core_end,
            context_start_ms=context_start,
            context_end_ms=context_end,
        )
        # This is an internal invariant, kept explicit so policy changes fail
        # loudly instead of quietly sending an over-limit request.
        if chunk.context_duration_ms > hard_max_ms:
            raise AssertionError("planned context exceeds the hard maximum")
        chunks.append(chunk)

    return tuple(chunks)


def _validate_plan_arguments(
    *,
    duration_ms: int,
    target_core_ms: int,
    overlap_ms: int,
    hard_max_ms: int,
    silence_search_ms: int,
) -> None:
    named_values = {
        "duration_ms": duration_ms,
        "target_core_ms": target_core_ms,
        "overlap_ms": overlap_ms,
        "hard_max_ms": hard_max_ms,
        "silence_search_ms": silence_search_ms,
    }
    for name, value in named_values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if duration_ms < 0:
        raise ValueError("duration_ms must be non-negative")
    if target_core_ms <= 0:
        raise ValueError("target_core_ms must be positive")
    if overlap_ms < 0:
        raise ValueError("overlap_ms must be non-negative")
    if hard_max_ms <= 0:
        raise ValueError("hard_max_ms must be positive")
    if silence_search_ms < 0:
        raise ValueError("silence_search_ms must be non-negative")
    if overlap_ms >= hard_max_ms:
        raise ValueError("overlap_ms must be smaller than hard_max_ms")
    if target_core_ms + overlap_ms > hard_max_ms:
        raise ValueError(
            "target_core_ms plus overlap_ms must not exceed hard_max_ms"
        )


def _normalize_silences(
    values: Iterable[SilenceTimestamp], duration_ms: int
) -> tuple[tuple[int, int], ...]:
    normalized: set[tuple[int, int]] = set()
    for value in values:
        if isinstance(value, bool):
            raise TypeError("silence timestamps must be integers, not booleans")
        if isinstance(value, int):
            start = end = value
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 2:
                raise ValueError("silence intervals must contain exactly two values")
            start, end = value
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
            ):
                raise TypeError("silence interval endpoints must be integers")
        else:
            raise TypeError("silence timestamps must be integers or pairs")

        if end < start:
            raise ValueError("silence interval end precedes its start")
        if end < 0 or start > duration_ms:
            continue
        normalized.add((max(0, start), min(duration_ms, end)))
    return tuple(sorted(normalized))


def _nearest_silence_boundary(
    *,
    ideal_ms: int,
    lower_ms: int,
    upper_ms: int,
    silences: Sequence[tuple[int, int]],
) -> int | None:
    if lower_ms > upper_ms:
        return None

    candidates: list[int] = []
    for silence_start, silence_end in silences:
        eligible_start = max(silence_start, lower_ms)
        eligible_end = min(silence_end, upper_ms)
        if eligible_start > eligible_end:
            continue
        candidates.append(min(max(ideal_ms, eligible_start), eligible_end))

    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (abs(candidate - ideal_ms), candidate))


@dataclass(frozen=True, slots=True)
class OverlapAlignment:
    """Diagnostics for a normalized-token alignment at a transcript seam."""

    primary_start_token: int
    primary_tokens: int
    continuation_tokens: int
    exact_matches: int
    edits: int
    similarity: float


@dataclass(frozen=True, slots=True)
class StitchResult:
    """A stitched transcript and the alignment that justified its deletion."""

    text: str
    alignment: OverlapAlignment | None

    @property
    def duplicate_tokens_removed(self) -> int:
        """Number of normalized continuation tokens removed at the seam."""
        return 0 if self.alignment is None else self.alignment.continuation_tokens


@dataclass(frozen=True, slots=True)
class _SurfaceToken:
    normalized: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _AlignmentState:
    score: int
    exact_matches: int
    edits: int
    primary_start: int


_NON_WHITESPACE_RE = re.compile(r"\S+")


def _surface_tokens(text: str) -> tuple[_SurfaceToken, ...]:
    tokens: list[_SurfaceToken] = []
    for match in _NON_WHITESPACE_RE.finditer(text):
        normalized = _normalize_token(match.group())
        if normalized:
            tokens.append(
                _SurfaceToken(
                    normalized=normalized,
                    start=match.start(),
                    end=match.end(),
                )
            )
    return tuple(tokens)


def _normalize_token(token: str) -> str:
    canonical = unicodedata.normalize("NFKC", token).casefold()
    return "".join(
        character
        for character in canonical
        if unicodedata.category(character)[0] in {"L", "N"}
    )


def find_overlap_alignment(
    primary: str,
    continuation: str,
    *,
    max_alignment_tokens: int = 96,
    min_matching_tokens: int = 6,
    min_similarity: float = 1.0,
) -> OverlapAlignment | None:
    """Align a primary suffix to a continuation prefix.

    This is a bounded semi-global token alignment: skipping old tokens at the
    beginning of ``primary`` is free, while substitutions and insertions or
    deletions inside the overlap count as edits.  The alignment must end at the
    final primary token and begin at the first continuation token.  It therefore
    cannot delete text from the middle of the continuation.
    """
    if isinstance(max_alignment_tokens, bool) or not isinstance(
        max_alignment_tokens, int
    ):
        raise TypeError("max_alignment_tokens must be an integer")
    if max_alignment_tokens <= 0:
        raise ValueError("max_alignment_tokens must be positive")
    if isinstance(min_matching_tokens, bool) or not isinstance(
        min_matching_tokens, int
    ):
        raise TypeError("min_matching_tokens must be an integer")
    if min_matching_tokens <= 0:
        raise ValueError("min_matching_tokens must be positive")
    if isinstance(min_similarity, bool) or not isinstance(min_similarity, (int, float)):
        raise TypeError("min_similarity must be numeric")
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between zero and one")

    all_primary_tokens = _surface_tokens(primary)
    all_continuation_tokens = _surface_tokens(continuation)
    primary_offset = max(0, len(all_primary_tokens) - max_alignment_tokens)
    primary_tokens = all_primary_tokens[primary_offset:]
    continuation_tokens = all_continuation_tokens[:max_alignment_tokens]
    if not primary_tokens or not continuation_tokens:
        return None

    primary_values = tuple(token.normalized for token in primary_tokens)
    continuation_values = tuple(token.normalized for token in continuation_tokens)
    rows = len(primary_values) + 1
    columns = len(continuation_values) + 1
    matrix: list[list[_AlignmentState | None]] = [
        [None] * columns for _ in range(rows)
    ]

    # Free starts in the primary make every possible primary suffix eligible.
    for row in range(rows):
        matrix[row][0] = _AlignmentState(0, 0, 0, row)

    for row in range(rows):
        for column in range(1, columns):
            choices: list[_AlignmentState] = []
            if row > 0:
                diagonal = matrix[row - 1][column - 1]
                assert diagonal is not None
                is_match = (
                    primary_values[row - 1] == continuation_values[column - 1]
                )
                choices.append(
                    _AlignmentState(
                        score=diagonal.score + (3 if is_match else -2),
                        exact_matches=diagonal.exact_matches + int(is_match),
                        edits=diagonal.edits + int(not is_match),
                        primary_start=diagonal.primary_start,
                    )
                )

                deletion = matrix[row - 1][column]
                assert deletion is not None
                choices.append(
                    _AlignmentState(
                        score=deletion.score - 2,
                        exact_matches=deletion.exact_matches,
                        edits=deletion.edits + 1,
                        primary_start=deletion.primary_start,
                    )
                )

            insertion = matrix[row][column - 1]
            assert insertion is not None
            choices.append(
                _AlignmentState(
                    score=insertion.score - 2,
                    exact_matches=insertion.exact_matches,
                    edits=insertion.edits + 1,
                    primary_start=insertion.primary_start,
                )
            )
            matrix[row][column] = max(choices, key=_alignment_state_rank)

    accepted: list[tuple[tuple[float | int, ...], OverlapAlignment]] = []
    final_row = len(primary_values)
    for continuation_count in range(1, columns):
        state = matrix[final_row][continuation_count]
        assert state is not None
        primary_count = final_row - state.primary_start
        span = max(primary_count, continuation_count)
        if span <= 0:
            continue
        similarity = 1.0 - (state.edits / span)
        if state.exact_matches < min_matching_tokens or similarity < min_similarity:
            continue
        alignment = OverlapAlignment(
            primary_start_token=primary_offset + state.primary_start,
            primary_tokens=primary_count,
            continuation_tokens=continuation_count,
            exact_matches=state.exact_matches,
            edits=state.edits,
            similarity=similarity,
        )
        rank: tuple[float | int, ...] = (
            state.score,
            state.exact_matches,
            similarity,
            span,
            continuation_count,
            -state.primary_start,
        )
        accepted.append((rank, alignment))

    if not accepted:
        return None
    return max(accepted, key=lambda candidate: candidate[0])[1]


def _alignment_state_rank(state: _AlignmentState) -> tuple[int, int, int, int]:
    return (
        state.score,
        state.exact_matches,
        -state.edits,
        -state.primary_start,
    )


def stitch_with_alignment(
    primary: str,
    continuation: str,
    *,
    max_alignment_tokens: int = 96,
    min_matching_tokens: int = 6,
    min_similarity: float = 1.0,
) -> StitchResult:
    """Stitch two transcript surfaces and return alignment diagnostics."""
    if not primary:
        return StitchResult(continuation, None)
    if not continuation:
        return StitchResult(primary, None)

    alignment = find_overlap_alignment(
        primary,
        continuation,
        max_alignment_tokens=max_alignment_tokens,
        min_matching_tokens=min_matching_tokens,
        min_similarity=min_similarity,
    )
    if alignment is None:
        return StitchResult(_join_surfaces(primary, continuation), None)

    continuation_tokens = _surface_tokens(continuation)
    cut_at = continuation_tokens[alignment.continuation_tokens - 1].end
    remainder = continuation[cut_at:]
    return StitchResult(_join_surfaces(primary, remainder), alignment)


def stitch_transcripts(
    primary: str,
    continuation: str,
    *,
    max_alignment_tokens: int = 96,
    min_matching_tokens: int = 6,
    min_similarity: float = 1.0,
) -> str:
    """Return a deterministic stitched transcript string.

    The returned value always starts with ``primary`` byte-for-byte.  Only an
    aligned duplicate prefix of ``continuation`` may be omitted.
    """
    return stitch_with_alignment(
        primary,
        continuation,
        max_alignment_tokens=max_alignment_tokens,
        min_matching_tokens=min_matching_tokens,
        min_similarity=min_similarity,
    ).text


def _join_surfaces(primary: str, continuation: str) -> str:
    if not primary:
        return continuation
    if not continuation:
        return primary
    if primary[-1].isspace() or continuation[0].isspace():
        return primary + continuation
    return primary + " " + continuation


__all__ = [
    "DEFAULT_TARGET_CORE_MS",
    "DEFAULT_OVERLAP_MS",
    "DEFAULT_HARD_MAX_MS",
    "DEFAULT_SILENCE_SEARCH_MS",
    "ChunkMetadata",
    "OverlapAlignment",
    "StitchResult",
    "find_overlap_alignment",
    "plan_chunks",
    "stitch_transcripts",
    "stitch_with_alignment",
]
