from __future__ import annotations

import json
from pathlib import Path

import pytest

from octoscribe import build_overrides, build_parser, cmd_verify
from src.transcript_compare import (
    ComparisonInputError,
    compare_transcript_directories,
    compare_word_sequences,
    comparison_output_lines,
    spoken_words,
    validate_max_word_error_rate,
)


def test_spoken_words_ignore_only_case_punctuation_and_whitespace() -> None:
    assert spoken_words("  DON'T fear; Faith—comes.\n") == (
        "dont",
        "fear",
        "faith",
        "comes",
    )
    assert spoken_words("word") != spoken_words("world")


def test_alignment_reports_every_addition_deletion_and_substitution() -> None:
    result = compare_word_sequences(
        ("alpha", "bravo", "charlie", "delta"),
        ("alpha", "beta", "charlie", "echo", "foxtrot"),
    )
    assert result["exact_spoken_word_match"] is False
    assert result["substitution_count"] == 2
    assert result["deletion_count"] == 0
    assert result["addition_count"] == 1
    assert [item["operation"] for item in result["differences"]] == [
        "substitution",
        "substitution",
        "addition",
    ]


def test_directory_comparison_writes_json_and_fails_on_word_change(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated"
    references = tmp_path / "references"
    reports = tmp_path / "comparisons"
    generated.mkdir()
    references.mkdir()
    (generated / "sermon.txt").write_text(
        "Grace and peace today.", encoding="utf-8"
    )
    (references / "sermon.txt").write_text(
        "Grace and truth today!", encoding="utf-8"
    )

    summary = compare_transcript_directories(generated, references, reports)

    assert summary["success"] is False
    report = json.loads(
        (reports / "sermon.txt.comparison.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "mismatch"
    assert report["differences"] == [
        {
            "operation": "substitution",
            "reference_index": 2,
            "generated_index": 2,
            "reference_word": "truth",
            "generated_word": "peace",
        }
    ]
    assert json.loads((reports / "summary.json").read_text())["success"] is False


def test_missing_reference_is_reported_but_never_claimed_as_success(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "sermon.txt").write_text("Faith comes by hearing.")

    strict = compare_transcript_directories(
        generated,
        tmp_path / "references",
        tmp_path / "strict-reports",
    )
    bootstrap = compare_transcript_directories(
        generated,
        tmp_path / "references",
        tmp_path / "bootstrap-reports",
        reference_required=False,
    )

    assert strict["success"] is False
    assert bootstrap["success"] is False
    assert bootstrap["counts"]["missing_reference_transcript"] == 1
    assert bootstrap["exact_spoken_word_match"] is False


def test_default_verification_is_exact_and_tolerance_retains_every_diff(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated"
    references = tmp_path / "references"
    generated.mkdir()
    references.mkdir()
    (references / "sermon.txt").write_text(
        "alpha beta gamma delta", encoding="utf-8"
    )
    (generated / "sermon.txt").write_text(
        "alpha beta theta delta", encoding="utf-8"
    )

    strict = compare_transcript_directories(
        generated, references, tmp_path / "strict"
    )
    tolerated = compare_transcript_directories(
        generated,
        references,
        tmp_path / "tolerated",
        max_word_error_rate=0.25,
    )

    assert strict["success"] is False
    assert strict["comparisons"][0]["status"] == "mismatch"
    assert tolerated["success"] is True
    assert tolerated["exact_spoken_word_match"] is False
    report = tolerated["comparisons"][0]
    assert report["status"] == "mismatch_within_tolerance"
    assert report["exact_spoken_word_match"] is False
    assert report["word_error_count"] == 1
    assert report["word_error_rate"] == 0.25
    assert report["differences"] == strict["comparisons"][0]["differences"]
    output = "\n".join(comparison_output_lines(tolerated))
    assert "MISMATCH WITHIN TOLERANCE sermon.txt" in output
    assert "substitutions=1 deletions=0 additions=0 errors=1" in output
    assert "word_error_rate=0.25000000" in output
    assert "SUBSTITUTION reference[2]='gamma' generated[2]='theta'" in output


def test_verification_fails_when_word_error_rate_exceeds_threshold(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated"
    references = tmp_path / "references"
    generated.mkdir()
    references.mkdir()
    (references / "sermon.txt").write_text("alpha beta gamma delta")
    (generated / "sermon.txt").write_text("alpha beta theta delta")

    summary = compare_transcript_directories(
        generated,
        references,
        tmp_path / "reports",
        max_word_error_rate=0.249,
    )

    assert summary["success"] is False
    assert summary["comparisons"][0]["status"] == "mismatch"


@pytest.mark.parametrize(
    ("reference", "generated"),
    [
        ("chapter 12 ends", "chapter 13 ends"),
        ("do not fear", "do now fear"),
        ("we cannot leave", "we can leave"),
    ],
)
def test_numeric_and_negation_differences_never_pass_tolerance(
    tmp_path: Path,
    reference: str,
    generated: str,
) -> None:
    generated_dir = tmp_path / "generated"
    reference_dir = tmp_path / "references"
    generated_dir.mkdir()
    reference_dir.mkdir()
    (reference_dir / "sermon.txt").write_text(reference)
    (generated_dir / "sermon.txt").write_text(generated)

    summary = compare_transcript_directories(
        generated_dir,
        reference_dir,
        tmp_path / "reports",
        max_word_error_rate=1,
    )

    report = summary["comparisons"][0]
    assert summary["success"] is False
    assert report["status"] == "mismatch"
    assert report["protected_difference_count"] >= 1
    assert report["differences"]


@pytest.mark.parametrize("value", ["", "nope", "nan", "inf", "-0.1", "1.1"])
def test_invalid_word_error_rate_is_rejected(value: str) -> None:
    with pytest.raises(ComparisonInputError, match="number from 0 to 1"):
        validate_max_word_error_rate(value)


def test_verify_cli_threshold_precedence_is_cli_then_environment_then_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MAX_WORD_ERROR_RATE", raising=False)
    assert build_parser().parse_args(["verify"]).max_word_error_rate == 0

    monkeypatch.setenv("MAX_WORD_ERROR_RATE", "0.0025")
    assert build_parser().parse_args(["verify"]).max_word_error_rate == 0.0025
    assert (
        build_parser()
        .parse_args(["verify", "--max-word-error-rate", "0.001"])
        .max_word_error_rate
        == 0.001
    )


def test_capture_reference_never_overwrites_existing_text(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    references = tmp_path / "references"
    generated.mkdir()
    (generated / "sermon.txt").write_text("Original words.")

    summary = compare_transcript_directories(
        generated,
        references,
        tmp_path / "reports",
        capture_reference=True,
    )
    assert summary["success"] is True
    assert (references / "sermon.txt").read_text() == "Original words."

    with pytest.raises(ComparisonInputError, match="refuses to overwrite"):
        compare_transcript_directories(
            generated,
            references,
            tmp_path / "reports-2",
            capture_reference=True,
        )


def test_three_direct_paths_resolve_from_cwd_and_derive_evidence_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(
        [
            "--audio-path",
            "audio-in",
            "--transcript-path",
            "text/generated",
            "--manifest-path",
            "text/index.json",
            "verify",
        ]
    )
    overrides = build_overrides(args)
    assert overrides["paths__audio_dir"] == tmp_path / "audio-in"
    assert overrides["paths__transcriptions_dir"] == tmp_path / "text/generated"
    assert overrides["paths__manifest_file"] == tmp_path / "text/index.json"
    assert overrides["paths__reference_dir"] == tmp_path / "text/reference-transcripts"
    assert overrides["paths__comparison_reports_dir"] == tmp_path / "text/comparison-reports"


def test_verify_command_prints_mismatch_and_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    generated = tmp_path / "generated"
    references = tmp_path / "references"
    generated.mkdir()
    references.mkdir()
    (generated / "sermon.txt").write_text("one changed three")
    (references / "sermon.txt").write_text("one two three")
    config = type(
        "ConfigFixture",
        (),
        {
            "transcribe": type(
                "TranscribeFixture",
                (),
                {
                    "transcriptions_dir": generated,
                    "reference_dir": references,
                    "comparison_reports_dir": tmp_path / "reports",
                },
            )()
        },
    )()
    args = type(
        "Args",
        (),
        {
            "reference_dir": None,
            "comparison_reports_dir": None,
            "reference_required": True,
            "capture_reference": False,
            "max_word_error_rate": 0,
        },
    )()

    with pytest.raises(SystemExit, match="1"):
        cmd_verify(args, config)
    assert "MISMATCH sermon.txt" in capsys.readouterr().out
