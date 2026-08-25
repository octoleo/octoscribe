"""Tests for deterministic long-recording chunk planning and seam stitching."""

from __future__ import annotations

import random
from dataclasses import FrozenInstanceError

import pytest

from src.transcribe.chunking import (
    DEFAULT_HARD_MAX_MS,
    DEFAULT_OVERLAP_MS,
    ChunkMetadata,
    find_overlap_alignment,
    plan_chunks,
    stitch_transcripts,
    stitch_with_alignment,
)


def _toy_plan(duration_ms: int, silences=(), **overrides):
    settings = {
        "target_core_ms": 100,
        "overlap_ms": 20,
        "hard_max_ms": 150,
        "silence_search_ms": 30,
    }
    settings.update(overrides)
    return plan_chunks(duration_ms, silences, **settings)


def _assert_plan_invariants(
    chunks: tuple[ChunkMetadata, ...],
    duration_ms: int,
    overlap_ms: int,
    hard_max_ms: int,
) -> None:
    if duration_ms == 0:
        assert chunks == ()
        return

    assert chunks[0].core_start_ms == 0
    assert chunks[0].context_start_ms == 0
    assert chunks[-1].core_end_ms == duration_ms
    assert chunks[-1].context_end_ms == duration_ms
    assert tuple(chunk.index for chunk in chunks) == tuple(range(len(chunks)))

    for chunk in chunks:
        assert 0 <= chunk.context_start_ms <= chunk.core_start_ms
        assert chunk.core_start_ms < chunk.core_end_ms <= chunk.context_end_ms
        assert chunk.context_end_ms <= duration_ms
        assert chunk.core_duration_ms <= hard_max_ms
        assert chunk.context_duration_ms <= hard_max_ms

    for left, right in zip(chunks, chunks[1:]):
        # Cores form a lossless partition: neither gaps nor double ownership.
        assert left.core_end_ms == right.core_start_ms
        # Context windows share exactly the requested evidence around the seam.
        assert left.context_end_ms - right.context_start_ms == overlap_ms


class TestChunkPlanning:
    def test_zero_duration_has_no_chunks(self):
        assert plan_chunks(0, [0]) == ()

    def test_recording_below_hard_limit_needs_no_seam(self):
        chunks = plan_chunks(DEFAULT_HARD_MAX_MS - 1, [480_000])
        assert chunks == (
            ChunkMetadata(
                index=0,
                core_start_ms=0,
                core_end_ms=DEFAULT_HARD_MAX_MS - 1,
                context_start_ms=0,
                context_end_ms=DEFAULT_HARD_MAX_MS - 1,
            ),
        )

    def test_default_plan_targets_eight_minutes_and_twelve_second_overlap(self):
        chunks = plan_chunks(20 * 60_000)

        assert [(chunk.core_start_ms, chunk.core_end_ms) for chunk in chunks] == [
            (0, 480_000),
            (480_000, 960_000),
            (960_000, 1_200_000),
        ]
        _assert_plan_invariants(
            chunks,
            duration_ms=20 * 60_000,
            overlap_ms=DEFAULT_OVERLAP_MS,
            hard_max_ms=DEFAULT_HARD_MAX_MS,
        )

    def test_ninety_minute_sermon_is_gapless_and_request_bounded(self):
        """The longest expected recording remains lossless and API-safe."""
        duration_ms = 90 * 60_000

        chunks = plan_chunks(duration_ms)

        assert len(chunks) == 12
        _assert_plan_invariants(
            chunks,
            duration_ms=duration_ms,
            overlap_ms=DEFAULT_OVERLAP_MS,
            hard_max_ms=DEFAULT_HARD_MAX_MS,
        )

    def test_selects_nearest_silence_and_breaks_ties_earlier(self):
        nearest = _toy_plan(300, [85, 110])
        tied = _toy_plan(300, [110, 90])

        assert nearest[0].core_end_ms == 110
        assert tied[0].core_end_ms == 90

    def test_silence_interval_uses_nearest_point_inside_interval(self):
        containing_target = _toy_plan(300, [(80, 120)])
        after_target = _toy_plan(300, [(105, 120)])

        assert containing_target[0].core_end_ms == 100
        assert after_target[0].core_end_ms == 105

    def test_ignores_silence_outside_search_window_and_hard_cap(self):
        chunks = _toy_plan(
            300,
            [69, 131, 1_000],
            hard_max_ms=130,
            silence_search_ms=30,
        )

        assert chunks[0].core_end_ms == 100
        assert all(chunk.context_duration_ms <= 130 for chunk in chunks)

    def test_unsorted_duplicate_and_out_of_recording_silences_are_deterministic(self):
        first = _toy_plan(400, [205, -50, 95, 205, 900, (90, 100)])
        second = _toy_plan(400, [(90, 100), 900, 205, -50])

        assert first == second
        assert first[0].core_end_ms == 100

    def test_odd_overlap_is_split_without_losing_a_millisecond(self):
        chunks = _toy_plan(400, overlap_ms=11)

        _assert_plan_invariants(chunks, 400, overlap_ms=11, hard_max_ms=150)
        assert chunks[0].context_end_ms - chunks[0].core_end_ms == 6
        assert chunks[1].core_start_ms - chunks[1].context_start_ms == 5

    def test_metadata_is_immutable(self):
        chunk = _toy_plan(50)[0]

        with pytest.raises(FrozenInstanceError):
            chunk.core_end_ms = 40  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("field", "value", "error"),
        [
            ("duration_ms", -1, ValueError),
            ("duration_ms", 1.5, TypeError),
            ("target_core_ms", 0, ValueError),
            ("overlap_ms", -1, ValueError),
            ("overlap_ms", 150, ValueError),
            ("hard_max_ms", 0, ValueError),
            ("silence_search_ms", -1, ValueError),
        ],
    )
    def test_rejects_invalid_plan_configuration(self, field, value, error):
        settings = {
            "duration_ms": 300,
            "target_core_ms": 100,
            "overlap_ms": 20,
            "hard_max_ms": 150,
            "silence_search_ms": 30,
        }
        settings[field] = value

        with pytest.raises(error):
            plan_chunks(**settings)

    @pytest.mark.parametrize(
        "silences",
        [
            [(20,)],
            [(30, 20)],
            [(10, "20")],
            [True],
            [object()],
        ],
    )
    def test_rejects_malformed_silence_data(self, silences):
        with pytest.raises((TypeError, ValueError)):
            _toy_plan(300, silences)

    def test_randomized_plans_are_gapless_bounded_and_deterministic(self):
        rng = random.Random(0x0C70)
        for _ in range(250):
            duration_ms = rng.randint(0, 8_000)
            overlap_ms = rng.randint(0, 30)
            hard_max_ms = rng.randint(overlap_ms + 70, overlap_ms + 180)
            target_core_ms = rng.randint(20, hard_max_ms - overlap_ms)
            search_ms = rng.randint(0, 50)
            silence_points = [
                rng.randint(-100, duration_ms + 100)
                for _ in range(rng.randint(0, 30))
            ]
            kwargs = {
                "target_core_ms": target_core_ms,
                "overlap_ms": overlap_ms,
                "hard_max_ms": hard_max_ms,
                "silence_search_ms": search_ms,
            }

            chunks = plan_chunks(duration_ms, silence_points, **kwargs)

            assert chunks == plan_chunks(
                duration_ms, reversed(silence_points), **kwargs
            )
            _assert_plan_invariants(
                chunks,
                duration_ms=duration_ms,
                overlap_ms=overlap_ms,
                hard_max_ms=hard_max_ms,
            )


class TestTranscriptStitching:
    def test_exact_normalized_overlap_keeps_primary_surface_verbatim(self):
        primary = "He said, JESUS is Lord."
        continuation = "Jesus is lord. And we rejoiced."

        result = stitch_with_alignment(
            primary, continuation, min_matching_tokens=2
        )

        assert result.text == "He said, JESUS is Lord. And we rejoiced."
        assert result.text.startswith(primary)
        assert result.duplicate_tokens_removed == 3
        assert result.alignment is not None
        assert result.alignment.similarity == 1.0

    def test_default_seam_does_not_delete_disputed_substitution(self):
        primary = "one two three four five"
        continuation = "one two altered four five six"

        result = stitch_with_alignment(primary, continuation)

        assert result.text == (
            "one two three four five one two altered four five six"
        )
        assert result.alignment is None

        # Fuzzy alignment remains available for diagnostics, but production's
        # fidelity-first default never removes a disputed word.
        fuzzy = stitch_with_alignment(
            primary,
            continuation,
            min_similarity=0.75,
            min_matching_tokens=2,
        )
        assert fuzzy.text == "one two three four five six"
        assert fuzzy.text.startswith(primary)
        assert fuzzy.alignment is not None
        assert fuzzy.alignment.edits == 1
        assert fuzzy.alignment.exact_matches == 4

    def test_default_seam_does_not_delete_disputed_insertion(self):
        primary = "alpha beta gamma delta"
        continuation = "alpha beta extra gamma delta epsilon"

        assert stitch_transcripts(
            primary, continuation, min_matching_tokens=2
        ) == (
            "alpha beta gamma delta alpha beta extra gamma delta epsilon"
        )

    def test_non_overlap_is_appended_without_deleting_words(self):
        primary = "This is the primary transcript."
        continuation = "Completely unrelated opening words."

        result = stitch_with_alignment(primary, continuation)

        assert result.text == f"{primary} {continuation}"
        assert result.alignment is None
        assert result.duplicate_tokens_removed == 0

    def test_single_common_word_is_not_enough_evidence_by_default(self):
        primary = "The sermon ends with amen"
        continuation = "amen We now sing"

        assert stitch_transcripts(
            primary, continuation, min_matching_tokens=2
        ) == (
            "The sermon ends with amen amen We now sing"
        )
        assert stitch_transcripts(
            primary, continuation, min_matching_tokens=1
        ) == "The sermon ends with amen We now sing"

    def test_unicode_compatibility_case_and_punctuation_are_normalized(self):
        primary = "We proclaim ＦＡＩＴＨ, Straße!"
        continuation = "faith STRASSE -- without fear."

        assert stitch_transcripts(
            primary, continuation, min_matching_tokens=2
        ) == (
            "We proclaim ＦＡＩＴＨ, Straße! -- without fear."
        )

    def test_primary_and_remainder_whitespace_are_not_normalized(self):
        primary = "Intro  JESUS saves.\n"
        continuation = "jesus saves.   Next  line."

        result = stitch_transcripts(
            primary, continuation, min_matching_tokens=2
        )

        assert result == "Intro  JESUS saves.\n   Next  line."
        assert result[: len(primary)] == primary

    @pytest.mark.parametrize(
        ("primary", "continuation", "expected"),
        [
            ("", "continuation only", "continuation only"),
            ("primary only", "", "primary only"),
            ("", "", ""),
            ("primary ", "new", "primary new"),
            ("primary", "\nnew", "primary\nnew"),
        ],
    )
    def test_empty_inputs_and_join_whitespace(self, primary, continuation, expected):
        assert stitch_transcripts(primary, continuation) == expected

    def test_short_repeated_phrase_is_preserved_as_ambiguous(self):
        primary = "opening repeated phrase repeated phrase"
        continuation = "repeated phrase repeated phrase closing"

        result = stitch_with_alignment(primary, continuation)

        assert result.text == (
            "opening repeated phrase repeated phrase "
            "repeated phrase repeated phrase closing"
        )
        assert result.alignment is None

    def test_alignment_is_bounded_to_configured_token_window(self):
        primary = "zero one two three"
        continuation = "zero one two three after"

        assert find_overlap_alignment(
            primary,
            continuation,
            max_alignment_tokens=2,
            min_matching_tokens=4,
        ) is None
        assert find_overlap_alignment(
            primary,
            continuation,
            max_alignment_tokens=4,
            min_matching_tokens=4,
        ) is not None

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"max_alignment_tokens": 0}, ValueError),
            ({"max_alignment_tokens": 1.5}, TypeError),
            ({"min_matching_tokens": 0}, ValueError),
            ({"min_matching_tokens": 1.5}, TypeError),
            ({"min_similarity": -0.1}, ValueError),
            ({"min_similarity": 1.1}, ValueError),
            ({"min_similarity": "high"}, TypeError),
        ],
    )
    def test_rejects_invalid_alignment_configuration(self, kwargs, error):
        with pytest.raises(error):
            find_overlap_alignment("one two", "one two", **kwargs)

    def test_randomized_exact_overlaps_preserve_primary_surface(self):
        rng = random.Random(0x5EAD)
        vocabulary = [f"token{index}" for index in range(200)]
        for _ in range(200):
            prefix = rng.sample(vocabulary[:80], rng.randint(0, 12))
            overlap = rng.sample(vocabulary[80:140], rng.randint(6, 20))
            tail = rng.sample(vocabulary[140:], rng.randint(1, 12))
            overlap_surface = [
                word.upper() + ("," if index % 2 else ".")
                for index, word in enumerate(overlap)
            ]
            primary = " ".join(prefix + overlap_surface)
            continuation = " ".join(overlap + tail)

            first = stitch_transcripts(primary, continuation)
            second = stitch_transcripts(primary, continuation)

            assert first == second
            assert first.startswith(primary)
            assert first == primary + " " + " ".join(tail)
