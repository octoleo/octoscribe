from __future__ import annotations

import json
from pathlib import Path

import pytest

from octoscribe import build_overrides, build_parser, cmd_verify
from src.transcript_compare import (
    ComparisonInputError,
    compare_transcript_directories,
    compare_word_sequences,
    spoken_words,
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


def test_missing_reference_is_only_allowed_in_explicit_bootstrap_mode(
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
    assert bootstrap["success"] is True
    assert bootstrap["counts"]["missing_reference_transcript"] == 1


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
        },
    )()

    with pytest.raises(SystemExit, match="1"):
        cmd_verify(args, config)
    assert "MISMATCH sermon.txt" in capsys.readouterr().out
