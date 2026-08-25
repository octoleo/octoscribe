"""Word-for-word comparison of generated transcripts and references.

The comparison deliberately ignores presentation differences only: Unicode
case, punctuation, whitespace, and unambiguous contraction spelling.  It never
rewrites a generated transcript.  Every spoken-word addition, deletion, and
substitution is retained in a JSON report and can also be rendered to the
command's standard output.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
_APOSTROPHES = frozenset({"'", "’", "ʼ", "＇"})
_UNAMBIGUOUS_CONTRACTIONS: dict[str, tuple[str, ...]] = {
    "i'm": ("i", "am"),
    "you're": ("you", "are"),
    "we're": ("we", "are"),
    "they're": ("they", "are"),
    "i've": ("i", "have"),
    "you've": ("you", "have"),
    "we've": ("we", "have"),
    "they've": ("they", "have"),
    "i'll": ("i", "will"),
    "you'll": ("you", "will"),
    "he'll": ("he", "will"),
    "she'll": ("she", "will"),
    "it'll": ("it", "will"),
    "we'll": ("we", "will"),
    "they'll": ("they", "will"),
    "aren't": ("are", "not"),
    "can't": ("can", "not"),
    "couldn't": ("could", "not"),
    "didn't": ("did", "not"),
    "doesn't": ("does", "not"),
    "don't": ("do", "not"),
    "hadn't": ("had", "not"),
    "hasn't": ("has", "not"),
    "haven't": ("have", "not"),
    "isn't": ("is", "not"),
    "mightn't": ("might", "not"),
    "mustn't": ("must", "not"),
    "needn't": ("need", "not"),
    "shan't": ("shall", "not"),
    "shouldn't": ("should", "not"),
    "wasn't": ("was", "not"),
    "weren't": ("were", "not"),
    "won't": ("will", "not"),
    "wouldn't": ("would", "not"),
    "could've": ("could", "have"),
    "might've": ("might", "have"),
    "must've": ("must", "have"),
    "should've": ("should", "have"),
    "would've": ("would", "have"),
}
# ASR punctuation can be inconsistent.  Accept an apostrophe-free spelling
# only when it cannot collide with a common English word (for example,
# ``youre`` is safe, while ``were`` could mean either "we're" or "were").
_UNMARKED_UNAMBIGUOUS_CONTRACTIONS: dict[str, tuple[str, ...]] = {
    contraction.replace("'", ""): expansion
    for contraction, expansion in _UNAMBIGUOUS_CONTRACTIONS.items()
    if contraction.replace("'", "")
    not in {"cant", "hell", "ill", "shell", "were", "well", "wont"}
}
_PROTECTED_NEGATIONS = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "nothing",
        "neither",
        "nor",
        "without",
        "dont",
        "doesnt",
        "didnt",
        "isnt",
        "wasnt",
        "werent",
        "cant",
        "cannot",
        "wont",
        "wouldnt",
        "shouldnt",
        "couldnt",
        "arent",
        "hadnt",
        "hasnt",
        "havent",
        "mightnt",
        "mustnt",
        "neednt",
        "shant",
        "aint",
    }
)


class ComparisonInputError(ValueError):
    """Raised when transcript comparison inputs are missing or unsafe."""


def validate_max_word_error_rate(value: float | str) -> float:
    """Return a finite verification threshold in the inclusive range 0..1."""
    if isinstance(value, bool):
        raise ComparisonInputError("max_word_error_rate must be a number from 0 to 1")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonInputError(
            "max_word_error_rate must be a number from 0 to 1"
        ) from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ComparisonInputError("max_word_error_rate must be a number from 0 to 1")
    return result


def _is_word_character(character: str) -> bool:
    category = unicodedata.category(character)
    return category[0] in {"L", "M", "N"}


def spoken_words(text: str) -> tuple[str, ...]:
    """Return case-folded spoken words with punctuation/whitespace ignored.

    Unambiguous contractions are expanded, so ``you're`` and ``you are``
    compare equally.  Ambiguous contractions such as ``he's`` are deliberately
    left as one word.  Apostrophes in all remaining words are ignored.  Other
    punctuation acts as a word boundary, preventing adjacent words from being
    silently joined.
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
            # Preserve an internal apostrophe until contraction expansion has
            # had a chance to distinguish ``we're`` from the ordinary word
            # ``were``.  Remaining apostrophes are removed below.
            characters.append("'")
            continue
        characters.append(" ")

    words: list[str] = []
    for token in "".join(characters).split():
        expansion = _UNAMBIGUOUS_CONTRACTIONS.get(token)
        collapsed = token.replace("'", "")
        if expansion is None:
            expansion = _UNMARKED_UNAMBIGUOUS_CONTRACTIONS.get(collapsed)
        if expansion is not None:
            words.extend(expansion)
        else:
            words.append(collapsed)
    return tuple(words)


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


def _is_protected_word(word: str | None) -> bool:
    return bool(
        word
        and (
            word in _PROTECTED_NEGATIONS
            or any(character.isdigit() for character in word)
        )
    )


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
    protected_differences = sum(
        1
        for difference in differences
        if _is_protected_word(difference["reference_word"])
        or _is_protected_word(difference["generated_word"])
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
        "protected_difference_count": protected_differences,
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
            "unambiguous_contractions": "expanded",
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
                "reference_word_count": len(spoken_words(_read_text(reference_path))),
                "generated_word_count": len(spoken_words(_read_text(generated_path))),
                "matching_word_count": len(spoken_words(_read_text(reference_path))),
                "substitution_count": 0,
                "deletion_count": 0,
                "addition_count": 0,
                "word_error_count": 0,
                "word_error_rate": 0.0,
                "protected_difference_count": 0,
                "differences": [],
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
    max_word_error_rate: float | str = 0.0,
) -> dict[str, Any]:
    """Compare or capture every generated ``.txt`` transcript by relative path.

    Missing references never produce a successful verification.
    ``reference_required=False`` is retained as bootstrap provenance in the
    summary, but does not turn missing evidence into a pass.
    ``capture_reference=True`` copies generated text to an initially empty
    reference directory and is intended only for deliberate manual baseline
    capture.
    """
    generated_dir = Path(generated_dir).resolve()
    reference_dir = Path(reference_dir).resolve()
    reports_dir = Path(reports_dir).resolve()
    max_word_error_rate = validate_max_word_error_rate(max_word_error_rate)
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
                report["exact_spoken_word_match"] = False
                report["max_word_error_rate"] = max_word_error_rate
                report["protected_difference_count"] = 0
                report["differences"] = []
            elif reference_path is None:
                report = _base_report(relative, "missing_reference_transcript")
                report["generated_sha256"] = _sha256(generated_path)
                report["exact_spoken_word_match"] = False
                report["max_word_error_rate"] = max_word_error_rate
                report["protected_difference_count"] = 0
                report["differences"] = []
            else:
                generated_text = _read_text(generated_path)
                reference_text = _read_text(reference_path)
                details = compare_word_sequences(
                    spoken_words(reference_text),
                    spoken_words(generated_text),
                )
                exact = details["exact_spoken_word_match"]
                within_tolerance = bool(
                    details["word_error_rate"] is not None
                    and details["word_error_rate"] <= max_word_error_rate
                    and details["protected_difference_count"] == 0
                )
                if exact:
                    status = "exact_match"
                elif within_tolerance:
                    status = "mismatch_within_tolerance"
                else:
                    status = "mismatch"
                report = _base_report(relative, status)
                report.update(
                    {
                        "reference_sha256": _sha256(reference_path),
                        "generated_sha256": _sha256(generated_path),
                        "max_word_error_rate": max_word_error_rate,
                        "within_tolerance": within_tolerance,
                        **details,
                    }
                )
            _write_json(_report_path(reports_dir, relative), report)
            comparisons.append(report)

    counts = {
        status: sum(1 for report in comparisons if report["status"] == status)
        for status in (
            "exact_match",
            "mismatch_within_tolerance",
            "mismatch",
            "missing_reference_transcript",
            "missing_generated_transcript",
            "reference_captured",
        )
    }
    success = (
        counts["mismatch"] == 0
        and counts["missing_reference_transcript"] == 0
        and counts["missing_generated_transcript"] == 0
    )
    exact_spoken_word_match = bool(comparisons) and all(
        report.get("exact_spoken_word_match", False) for report in comparisons
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "transcript_reference_comparison_summary",
        "generated_dir": str(generated_dir),
        "reference_dir": str(reference_dir),
        "reports_dir": str(reports_dir),
        "reference_required": reference_required,
        "capture_reference": capture_reference,
        "max_word_error_rate": max_word_error_rate,
        "exact_spoken_word_match": exact_spoken_word_match,
        "protected_difference_count": sum(
            report.get("protected_difference_count", 0) for report in comparisons
        ),
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
        if status in {"exact_match", "mismatch_within_tolerance", "mismatch"}:
            rate = report["word_error_rate"]
            rendered_rate = "undefined" if rate is None else f"{rate:.8f}"
            if status == "exact_match":
                heading = "EXACT MATCH"
            elif status == "mismatch_within_tolerance":
                heading = "MISMATCH WITHIN TOLERANCE"
            else:
                heading = "MISMATCH"
            yield (
                f"{heading} {transcript}: reference_words="
                f"{report['reference_word_count']} generated_words="
                f"{report['generated_word_count']} substitutions="
                f"{report['substitution_count']} deletions={report['deletion_count']} "
                f"additions={report['addition_count']} errors="
                f"{report['word_error_count']} word_error_rate={rendered_rate} "
                f"max_word_error_rate={summary['max_word_error_rate']:.8f} "
                f"protected_differences={report['protected_difference_count']}"
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
        f"exact_spoken_word_match={str(summary['exact_spoken_word_match']).lower()} "
        f"exact_matches={counts['exact_match']} "
        f"within_tolerance={counts['mismatch_within_tolerance']} "
        f"mismatches={counts['mismatch']} "
        f"missing_references={counts['missing_reference_transcript']} "
        f"missing_generated={counts['missing_generated_transcript']} "
        f"protected_differences={summary['protected_difference_count']} "
        f"max_word_error_rate={summary['max_word_error_rate']:.8f} "
        f"captured={counts['reference_captured']} "
        f"reports={summary['reports_dir']}"
    )


__all__ = [
    "ComparisonInputError",
    "compare_transcript_directories",
    "compare_word_sequences",
    "comparison_output_lines",
    "spoken_words",
    "validate_max_word_error_rate",
]
