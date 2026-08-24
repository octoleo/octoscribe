"""Deterministic tests for evidence-preserving transcript comparison."""

from __future__ import annotations

from dataclasses import fields

import pytest

from src.transcribe.consensus import (
    CriticalCategory,
    DiscrepancyKind,
    DiscrepancyPriority,
    QualityState,
    ResolutionAction,
    ResolutionProgress,
    TokenSpan,
    compare_transcripts,
    next_resolution_decision,
    quality_state_for,
)


def test_agreement_ignores_case_punctuation_and_whitespace() -> None:
    first = "  In the BEGINNING,\nGod created the heavens.  "
    second = "in the beginning god created the heavens"

    report = compare_transcripts((first, second), labels=("openai", "local"))

    assert report.all_agree
    assert report.quality_state is QualityState.CROSS_CHECKED
    assert report.discrepancies == ()
    assert report.transcripts[0].label == "openai"
    assert report.transcripts[0].original == first
    assert report.transcripts[1].original == second


def test_apostrophe_punctuation_does_not_create_a_difference() -> None:
    report = compare_transcripts(("Don't lose heart.", "dont lose heart"))

    assert report.all_agree


def test_addition_surfaces_half_open_token_spans_and_original_spelling() -> None:
    report = compare_transcripts(("Jesus wept", "And JESUS wept"))

    discrepancy = report.comparisons[0].discrepancies[0]
    assert discrepancy.kind is DiscrepancyKind.ADDITION
    assert discrepancy.left_span == TokenSpan(0, 0)
    assert discrepancy.right_span == TokenSpan(0, 1)
    assert discrepancy.left_tokens == ()
    assert discrepancy.right_tokens == ("And",)
    assert discrepancy.priority is DiscrepancyPriority.STANDARD


def test_deletion_surfaces_half_open_token_spans() -> None:
    report = compare_transcripts(("And Jesus wept", "Jesus wept"))

    discrepancy = report.comparisons[0].discrepancies[0]
    assert discrepancy.kind is DiscrepancyKind.DELETION
    assert discrepancy.left_span == TokenSpan(0, 1)
    assert discrepancy.right_span == TokenSpan(0, 0)
    assert discrepancy.left_tokens == ("And",)
    assert discrepancy.right_tokens == ()


def test_substitution_surfaces_both_original_token_spans() -> None:
    report = compare_transcripts(("God is faithful", "God was faithful"))

    discrepancy = report.comparisons[0].discrepancies[0]
    assert discrepancy.kind is DiscrepancyKind.SUBSTITUTION
    assert discrepancy.left_span == TokenSpan(1, 2)
    assert discrepancy.right_span == TokenSpan(1, 2)
    assert discrepancy.left_tokens == ("is",)
    assert discrepancy.right_tokens == ("was",)
    assert report.quality_state is QualityState.NEEDS_REVIEW


def test_inserted_negation_is_elevated_as_critical() -> None:
    report = compare_transcripts(("God is able", "God is not able"))

    discrepancy = report.comparisons[0].discrepancies[0]
    assert discrepancy.kind is DiscrepancyKind.ADDITION
    assert discrepancy.is_critical
    assert discrepancy.priority is DiscrepancyPriority.CRITICAL
    hit_summaries = [
        (hit.transcript_index, hit.category, hit.original_tokens)
        for hit in discrepancy.critical_hits
    ]
    assert hit_summaries == [
        (1, CriticalCategory.NEGATION, ("not",)),
    ]


def test_changed_scripture_verse_elevates_number_and_reference() -> None:
    report = compare_transcripts(("Read John 3:16 today", "Read John 3:17 today"))

    discrepancy = report.comparisons[0].discrepancies[0]
    categories = {hit.category for hit in discrepancy.critical_hits}
    assert discrepancy.kind is DiscrepancyKind.SUBSTITUTION
    assert discrepancy.is_critical
    assert categories == {
        CriticalCategory.NUMBER,
        CriticalCategory.SCRIPTURE_REFERENCE,
    }
    scripture_hits = [
        hit
        for hit in discrepancy.critical_hits
        if hit.category is CriticalCategory.SCRIPTURE_REFERENCE
    ]
    assert [hit.original_tokens for hit in scripture_hits] == [
        ("John", "3", "16"),
        ("John", "3", "17"),
    ]


def test_book_name_substitution_is_recognized_as_scripture_critical() -> None:
    report = compare_transcripts(("John 3:16", "Luke 3:16"))

    discrepancy = report.comparisons[0].discrepancies[0]
    assert discrepancy.is_critical
    assert [
        hit.category for hit in discrepancy.critical_hits
    ] == [
        CriticalCategory.SCRIPTURE_REFERENCE,
        CriticalCategory.SCRIPTURE_REFERENCE,
    ]


def test_spelled_number_is_elevated() -> None:
    report = compare_transcripts(("He called twelve", "He called eleven"))

    discrepancy = report.comparisons[0].discrepancies[0]
    assert discrepancy.is_critical
    assert [hit.category for hit in discrepancy.critical_hits] == [
        CriticalCategory.NUMBER,
        CriticalCategory.NUMBER,
    ]


def test_three_transcripts_compare_every_pair_without_majority_rewrite() -> None:
    originals = (
        "The Lord is good.",
        "the lord is good",
        "The Lord was good.",
    )
    report = compare_transcripts(originals, labels=("first", "second", "third"))

    assert [(item.left_index, item.right_index) for item in report.comparisons] == [
        (0, 1),
        (0, 2),
        (1, 2),
    ]
    assert [item.agrees for item in report.comparisons] == [True, False, False]
    assert tuple(item.original for item in report.transcripts) == originals
    assert report.quality_state is QualityState.NEEDS_REVIEW
    assert "consensus_text" not in {field.name for field in fields(report)}
    assert not hasattr(report, "consensus_text")


def test_repeated_words_align_deterministically_without_autojunk() -> None:
    repeated = " ".join(["amen"] * 250)
    changed = " ".join(["amen"] * 125 + ["truly"] + ["amen"] * 125)

    report = compare_transcripts((repeated, changed))

    discrepancy = report.comparisons[0].discrepancies[0]
    assert discrepancy.kind is DiscrepancyKind.ADDITION
    assert discrepancy.right_tokens == ("truly",)
    assert discrepancy.right_span == TokenSpan(125, 126)


@pytest.mark.parametrize(
    ("count", "all_agree", "human_verified", "expected"),
    [
        (1, None, False, QualityState.MACHINE_TRANSCRIBED),
        (2, True, False, QualityState.CROSS_CHECKED),
        (3, False, False, QualityState.NEEDS_REVIEW),
        (1, None, True, QualityState.HUMAN_VERIFIED),
        (3, False, True, QualityState.HUMAN_VERIFIED),
    ],
)
def test_quality_state_is_conservative(
    count: int,
    all_agree: bool | None,
    human_verified: bool,
    expected: QualityState,
) -> None:
    assert quality_state_for(
        count,
        all_agree=all_agree,
        human_verified=human_verified,
    ) is expected


def test_resolution_accepts_cross_checked_evidence_without_a_retry() -> None:
    report = compare_transcripts(("Grace and peace", "grace, and peace."))

    decision = next_resolution_decision(report, arbiter_available=True)

    assert decision.action is ResolutionAction.ACCEPT
    assert decision.quality_state is QualityState.CROSS_CHECKED


def test_resolution_accepts_primary_supported_two_of_three_after_arbitration() -> None:
    report = compare_transcripts(
        ("one", "two", "one"), labels=("primary", "checker", "arbiter")
    )
    progress = ResolutionProgress(retry_completed=True, arbiter_completed=True)

    decision = next_resolution_decision(
        report,
        progress,
        arbiter_available=True,
        primary_label="primary",
    )

    assert decision.action is ResolutionAction.ACCEPT
    assert decision.quality_state is QualityState.CROSS_CHECKED


def test_resolution_rejects_non_primary_two_of_three_after_arbitration() -> None:
    report = compare_transcripts(
        ("one", "two", "two"), labels=("primary", "checker", "arbiter")
    )
    progress = ResolutionProgress(retry_completed=True, arbiter_completed=True)

    decision = next_resolution_decision(
        report,
        progress,
        arbiter_available=True,
        primary_label="primary",
    )

    assert decision.action is ResolutionAction.REQUIRE_HUMAN_REVIEW
    assert decision.quality_state is QualityState.NEEDS_REVIEW


def test_resolution_allows_exactly_one_retry_then_one_arbiter() -> None:
    report = compare_transcripts(("Jesus is Lord", "Jesus was Lord"))
    progress = ResolutionProgress()

    retry = next_resolution_decision(report, progress, arbiter_available=True)
    assert retry.action is ResolutionAction.RETRY
    progress = progress.after(retry.action)

    arbiter = next_resolution_decision(report, progress, arbiter_available=True)
    assert arbiter.action is ResolutionAction.ARBITRATE
    progress = progress.after(arbiter.action)

    exhausted = next_resolution_decision(report, progress, arbiter_available=True)
    assert exhausted.action is ResolutionAction.REQUIRE_HUMAN_REVIEW
    assert exhausted.quality_state is QualityState.NEEDS_REVIEW
    with pytest.raises(ValueError, match="retry"):
        progress.after(ResolutionAction.RETRY)
    with pytest.raises(ValueError, match="arbiter"):
        progress.after(ResolutionAction.ARBITRATE)


def test_resolution_skips_unavailable_arbiter_after_retry() -> None:
    report = compare_transcripts(("Jesus is Lord", "Jesus was Lord"))
    progress = ResolutionProgress(retry_completed=True)

    decision = next_resolution_decision(report, progress, arbiter_available=False)

    assert decision.action is ResolutionAction.REQUIRE_HUMAN_REVIEW


def test_human_verification_is_explicit_and_terminal() -> None:
    report = compare_transcripts(("Jesus is Lord", "Jesus was Lord"))
    progress = ResolutionProgress(retry_completed=True).with_human_verification()

    decision = next_resolution_decision(report, progress, arbiter_available=True)

    assert decision.action is ResolutionAction.ACCEPT
    assert decision.quality_state is QualityState.HUMAN_VERIFIED


def test_invalid_inputs_fail_before_comparison() -> None:
    with pytest.raises(ValueError, match="two or three"):
        compare_transcripts(("only one",))
    with pytest.raises(ValueError, match="two or three"):
        compare_transcripts(("one", "two", "three", "four"))
    with pytest.raises(TypeError, match="sequence"):
        compare_transcripts("not a sequence of transcripts")
    with pytest.raises(ValueError, match="labels"):
        compare_transcripts(("one", "two"), labels=("only-one",))
    with pytest.raises(ValueError, match="one, two, or three"):
        quality_state_for(0)
    with pytest.raises(ValueError, match="follow"):
        ResolutionProgress(arbiter_completed=True)
