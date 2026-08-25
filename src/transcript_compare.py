"""Word-for-word comparison of generated transcripts and references.

The comparison deliberately ignores presentation differences only: Unicode
case, punctuation, and whitespace.  It never rewrites a generated transcript.
Every spoken-word addition, deletion, and substitution is retained in a JSON
report and can also be rendered to the command's standard output.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
_APOSTROPHES = frozenset({"'", "’", "ʼ", "＇"})


class ComparisonInputError(ValueError):
    """Raised when transcript comparison inputs are missing or unsafe."""


def _is_word_character(character: str) -> bool:
    category = unicodedata.category(character)
    return category[0] in {"L", "M", "N"}


def spoken_words(text: str) -> tuple[str, ...]:
    """Return case-folded spoken words with punctuation/whitespace ignored.

    Apostrophes inside a word are removed, so ``don't`` and ``dont`` compare
    equally.  Other punctuation acts as a word boundary, preventing adjacent
    words from being silently joined.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    characters: list[str] = []
    for index, character in enumerate(normalized):
        if _is_word_character(character):
            characters.append(character)
            continue
        previous_is_word = index > 0 and _is_word_character(normalized[index - 1])
        next_is_word = (
            index + 1 < len(normalized)
            and _is_word_character(normalized[index + 1])
        )
        if character in _APOSTROPHES and previous_is_word and next_is_word:
            continue
        characters.append(" ")
    return tuple("".join(characters).split())


def _difference(
    operation: str,
    *,
    reference_index: int | None,
    generated_index: int | None,
    reference_word: str | None,
    generated_word: str | None,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "reference_index": reference_index,
        "generated_index": generated_index,
        "reference_word": reference_word,
        "generated_word": generated_word,
    }


def compare_word_sequences(
    reference_words: tuple[str, ...], generated_words: tuple[str, ...]
) -> dict[str, Any]:
    """Align two word sequences and describe every spoken-word difference."""
    differences: list[dict[str, Any]] = []
    matches = 0
    substitutions = 0
    deletions = 0
    additions = 0
    matcher = SequenceMatcher(
        None,
        reference_words,
        generated_words,
        autojunk=False,
    )
    for operation, ref_start, ref_end, gen_start, gen_end in matcher.get_opcodes():
        if operation == "equal":
            matches += ref_end - ref_start
            continue
        if operation == "delete":
            for ref_index in range(ref_start, ref_end):
                deletions += 1
                differences.append(
                    _difference(
                        "deletion",
                        reference_index=ref_index,
                        generated_index=None,
                        reference_word=reference_words[ref_index],
                        generated_word=None,
                    )
                )
            continue
        if operation == "insert":
            for gen_index in range(gen_start, gen_end):
                additions += 1
                differences.append(
                    _difference(
                        "addition",
                        reference_index=None,
                        generated_index=gen_index,
                        reference_word=None,
                        generated_word=generated_words[gen_index],
                    )
                )
            continue

        paired = min(ref_end - ref_start, gen_end - gen_start)
        for offset in range(paired):
            substitutions += 1
            ref_index = ref_start + offset
            gen_index = gen_start + offset
            differences.append(
                _difference(
                    "substitution",
                    reference_index=ref_index,
                    generated_index=gen_index,
                    reference_word=reference_words[ref_index],
                    generated_word=generated_words[gen_index],
                )
            )
        for ref_index in range(ref_start + paired, ref_end):
            deletions += 1
            differences.append(
                _difference(
                    "deletion",
                    reference_index=ref_index,
                    generated_index=None,
                    reference_word=reference_words[ref_index],
                    generated_word=None,
                )
            )
        for gen_index in range(gen_start + paired, gen_end):
            additions += 1
            differences.append(
                _difference(
                    "addition",
                    reference_index=None,
                    generated_index=gen_index,
                    reference_word=None,
                    generated_word=generated_words[gen_index],
                )
            )

    errors = substitutions + deletions + additions
    error_rate = errors / len(reference_words) if reference_words else (
        0.0 if not errors else None
    )
    return {
        "exact_spoken_word_match": errors == 0,
        "reference_word_count": len(reference_words),
        "generated_word_count": len(generated_words),
        "matching_word_count": matches,
        "substitution_count": substitutions,
        "deletion_count": deletions,
        "addition_count": additions,
        "word_error_count": errors,
        "word_error_rate": error_rate,
        "differences": differences,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transcript_files(root: Path, *, required: bool) -> dict[Path, Path]:
    if not root.exists():
        if required:
            raise ComparisonInputError(f"transcript directory does not exist: {root}")
        return {}
    if not root.is_dir() or root.is_symlink():
        raise ComparisonInputError(f"transcript path is not a safe directory: {root}")
    result: dict[Path, Path] = {}
    for path in sorted(root.rglob("*.txt")):
        if path.is_symlink() or not path.is_file():
            raise ComparisonInputError(f"transcript is not a safe regular file: {path}")
        result[path.relative_to(root)] = path
    return result


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ComparisonInputError(f"cannot read UTF-8 transcript {path}: {exc}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _report_path(reports_dir: Path, relative: Path) -> Path:
    return reports_dir / relative.parent / f"{relative.name}.comparison.json"


def _base_report(relative: Path, status: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "transcript_reference_comparison",
        "status": status,
        "transcript": relative.as_posix(),
        "normalization": {
            "unicode": "NFKC",
            "case": "casefolded",
            "punctuation": "ignored",
            "whitespace": "ignored",
        },
    }


def _capture_references(
    generated_files: dict[Path, Path],
    reference_dir: Path,
    reports_dir: Path,
) -> list[dict[str, Any]]:
    existing = _transcript_files(reference_dir, required=False)
    if existing:
        raise ComparisonInputError(
            "capture_reference refuses to overwrite existing reference transcripts"
        )
    comparisons: list[dict[str, Any]] = []
    for relative, generated_path in generated_files.items():
        reference_path = reference_dir / relative
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = reference_path.with_name(f".{reference_path.name}.tmp")
        shutil.copyfile(generated_path, temporary)
        temporary.replace(reference_path)
        report = _base_report(relative, "reference_captured")
        report.update(
            {
                "generated_sha256": _sha256(generated_path),
                "reference_sha256": _sha256(reference_path),
                "exact_spoken_word_match": True,
            }
        )
        _write_json(_report_path(reports_dir, relative), report)
        comparisons.append(report)
    return comparisons


def compare_transcript_directories(
    generated_dir: Path,
    reference_dir: Path,
    reports_dir: Path,
    *,
    reference_required: bool = True,
    capture_reference: bool = False,
) -> dict[str, Any]:
    """Compare or capture every generated ``.txt`` transcript by relative path.

    Missing references fail by default.  ``reference_required=False`` is a
    bootstrap mode that records missing-reference reports without claiming a
    successful match.  ``capture_reference=True`` copies generated text to an
    initially empty reference directory and is intended only for deliberate
    manual baseline capture.
    """
    generated_dir = Path(generated_dir).resolve()
    reference_dir = Path(reference_dir).resolve()
    reports_dir = Path(reports_dir).resolve()
    if generated_dir == reference_dir:
        raise ComparisonInputError("generated and reference directories must differ")
    if reports_dir in {generated_dir, reference_dir}:
        raise ComparisonInputError("comparison reports require a separate directory")

    generated_files = _transcript_files(generated_dir, required=True)
    if not generated_files:
        raise ComparisonInputError(
            f"no generated .txt transcripts found in {generated_dir}"
        )
    reports_dir.mkdir(parents=True, exist_ok=True)

    if capture_reference:
        comparisons = _capture_references(
            generated_files,
            reference_dir,
            reports_dir,
        )
    else:
        reference_files = _transcript_files(reference_dir, required=False)
        comparisons = []
        for relative in sorted(set(generated_files) | set(reference_files)):
            generated_path = generated_files.get(relative)
            reference_path = reference_files.get(relative)
            if generated_path is None:
                report = _base_report(relative, "missing_generated_transcript")
                report["reference_sha256"] = _sha256(reference_path)  # type: ignore[arg-type]
            elif reference_path is None:
                report = _base_report(relative, "missing_reference_transcript")
                report["generated_sha256"] = _sha256(generated_path)
            else:
                generated_text = _read_text(generated_path)
                reference_text = _read_text(reference_path)
                details = compare_word_sequences(
                    spoken_words(reference_text),
                    spoken_words(generated_text),
                )
                report = _base_report(
                    relative,
                    "match" if details["exact_spoken_word_match"] else "mismatch",
                )
                report.update(
                    {
                        "reference_sha256": _sha256(reference_path),
                        "generated_sha256": _sha256(generated_path),
                        **details,
                    }
                )
            _write_json(_report_path(reports_dir, relative), report)
            comparisons.append(report)

    counts = {
        status: sum(1 for report in comparisons if report["status"] == status)
        for status in (
            "match",
            "mismatch",
            "missing_reference_transcript",
            "missing_generated_transcript",
            "reference_captured",
        )
    }
    success = (
        counts["mismatch"] == 0
        and counts["missing_generated_transcript"] == 0
        and (
            counts["missing_reference_transcript"] == 0
            or not reference_required
        )
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "transcript_reference_comparison_summary",
        "generated_dir": str(generated_dir),
        "reference_dir": str(reference_dir),
        "reports_dir": str(reports_dir),
        "reference_required": reference_required,
        "capture_reference": capture_reference,
        "success": success,
        "comparison_count": len(comparisons),
        "counts": counts,
        "comparisons": comparisons,
    }
    _write_json(reports_dir / "summary.json", summary)
    return summary


def comparison_output_lines(summary: dict[str, Any]) -> Iterable[str]:
    """Render complete, deterministic stdout for one comparison run."""
    for report in summary["comparisons"]:
        transcript = report["transcript"]
        status = report["status"]
        if status == "match":
            yield (
                f"MATCH {transcript}: "
                f"{report['reference_word_count']} spoken words"
            )
        elif status == "mismatch":
            yield (
                f"MISMATCH {transcript}: substitutions="
                f"{report['substitution_count']} deletions={report['deletion_count']} "
                f"additions={report['addition_count']}"
            )
            for difference in report["differences"]:
                yield (
                    f"  {difference['operation'].upper()} "
                    f"reference[{difference['reference_index']}]="
                    f"{difference['reference_word']!r} generated["
                    f"{difference['generated_index']}]="
                    f"{difference['generated_word']!r}"
                )
        elif status == "missing_reference_transcript":
            yield f"MISSING REFERENCE {transcript}"
        elif status == "missing_generated_transcript":
            yield f"MISSING GENERATED TRANSCRIPT {transcript}"
        else:
            yield f"CAPTURED REFERENCE {transcript}"
    counts = summary["counts"]
    yield (
        "Comparison summary: "
        f"success={str(summary['success']).lower()} "
        f"matches={counts['match']} mismatches={counts['mismatch']} "
        f"missing_references={counts['missing_reference_transcript']} "
        f"missing_generated={counts['missing_generated_transcript']} "
        f"captured={counts['reference_captured']} "
        f"reports={summary['reports_dir']}"
    )


__all__ = [
    "ComparisonInputError",
    "compare_transcript_directories",
    "compare_word_sequences",
    "comparison_output_lines",
    "spoken_words",
]
