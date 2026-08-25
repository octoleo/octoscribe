#!/usr/bin/env python3
"""Prepare and verify the paid post-merge OpenAI integration fixture.

This helper deliberately uses only the Python standard library.  It runs before
the action has installed OctoScribe's dependencies, validates the exact owner-
supplied Telegram evidence, and copies that evidence into an isolated runner
workspace.  After transcription it checks the complete manifest/evidence
contract and records a content snapshot.  A second invocation compares that
snapshot byte-for-byte, proving that an idempotent rerun did not create or
replace any durable artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


EXPECTED: dict[str, dict[str, Any]] = {
    "856": {
        "date": "2024-11-25",
        "downloaded": True,
        "duration": 1412,
        "duration_formatted": "23:32",
        "extension": ".ogg",
        "filename": "1 Timothy 15-6.ogg",
        "hash": (
            "7b2bc9c89b96cc528cd3f63a88e63710"
            "e2128516dae1079f4fabf8414cd8b060"
        ),
        "original_filename": "record.ogg",
        "performer": "Family Devotions",
        "telegram_msg_id": 856,
        "title": "1 Timothy 1:5-6",
        "container_duration_ms": 1_411_093,
        "expected_chunks": 3,
    },
    "990": {
        "date": "2025-02-06",
        "downloaded": True,
        "duration": 1850,
        "duration_formatted": "30:50",
        "extension": ".ogg",
        "filename": "1 John 17-8.ogg",
        "hash": (
            "698056c50804e1033b1c68adcbf7d4064"
            "c32f112aaa6aba23f0fbe524472849a"
        ),
        "original_filename": "record.ogg",
        "performer": "Family Devotions",
        "telegram_msg_id": 990,
        "title": "1 John 1:7-8",
        "container_duration_ms": 1_849_253,
        "expected_chunks": 4,
    },
}

HISTORICAL_FIELDS = (
    "date",
    "downloaded",
    "duration",
    "duration_formatted",
    "extension",
    "filename",
    "hash",
    "original_filename",
    "performer",
    "telegram_msg_id",
    "title",
)
ALLOWED_SINGLE_PROVIDER_STATES = {"machine_transcribed", "needs_review"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssertionError(f"invalid JSON object at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"expected a JSON object at {path}")
    return payload


def _regular_file(path: Path, description: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise AssertionError(f"{description} is not a safe regular file: {path}")


def _within(root: Path, relative: str, description: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in relative:
        raise AssertionError(f"unsafe {description} path: {relative!r}")
    result = root / candidate
    _regular_file(result, description)
    try:
        result.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise AssertionError(f"{description} escapes its workspace: {result}") from exc
    return result


def validate_source_fixture(root: Path) -> dict[str, dict[str, Any]]:
    """Validate immutable OGG bytes and the historical Telegram metadata."""
    root = root.resolve(strict=True)
    manifest_path = root / "manifest.json"
    _regular_file(manifest_path, "fixture manifest")
    manifest = _json_object(manifest_path)
    if set(manifest) != set(EXPECTED):
        raise AssertionError(
            f"fixture IDs changed: expected {sorted(EXPECTED)}, got {sorted(manifest)}"
        )

    for message_id, expected in EXPECTED.items():
        entry = manifest[message_id]
        if not isinstance(entry, dict):
            raise AssertionError(f"manifest entry {message_id} is not an object")
        for field in HISTORICAL_FIELDS:
            if entry.get(field) != expected[field]:
                raise AssertionError(
                    f"fixture {message_id} field {field!r} changed: "
                    f"expected {expected[field]!r}, got {entry.get(field)!r}"
                )
        audio_path = root / "audio" / expected["filename"]
        _regular_file(audio_path, f"fixture audio {message_id}")
        actual_hash = _sha256(audio_path)
        if actual_hash != expected["hash"]:
            raise AssertionError(
                f"fixture audio {message_id} SHA-256 changed: {actual_hash}"
            )
    return manifest


def prepare_workspace(fixture_root: Path, workspace: Path) -> None:
    """Copy the verified fixture into a new isolated evidence workspace."""
    validate_source_fixture(fixture_root)
    if workspace.exists() and any(workspace.iterdir()):
        raise AssertionError(f"refusing to replace non-empty workspace: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "audio").mkdir()
    shutil.copy2(fixture_root / "manifest.json", workspace / "manifest.json")
    for expected in EXPECTED.values():
        name = str(expected["filename"])
        shutil.copy2(fixture_root / "audio" / name, workspace / "audio" / name)
    validate_source_fixture(workspace)


def _validate_output_locations(
    workspace: Path,
    *,
    audio_dir: Path | None,
    manifest_file: Path | None,
    transcriptions_dir: Path | None,
    candidates_dir: Path | None,
    reports_dir: Path | None,
) -> None:
    supplied = {
        "audio_dir": (audio_dir, workspace / "audio"),
        "manifest_file": (manifest_file, workspace / "manifest.json"),
        "transcriptions_dir": (transcriptions_dir, workspace / "transcriptions"),
        "candidates_dir": (candidates_dir, workspace / "candidates"),
        "reports_dir": (reports_dir, workspace / "reports"),
    }
    for name, (actual, expected) in supplied.items():
        if actual is not None and actual.resolve() != expected.resolve():
            raise AssertionError(
                f"action output {name} points to {actual}, expected {expected}"
            )


def _validate_report_chunks(
    report: dict[str, Any], expected: dict[str, Any]
) -> tuple[set[tuple[str, int, str, str, int]], bool]:
    chunks = report.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != expected["expected_chunks"]:
        raise AssertionError(
            f"{expected['filename']} did not use the expected deterministic "
            f"chunk count ({expected['expected_chunks']})"
        )

    candidate_identities: set[tuple[str, int, str, str, int]] = set()
    previous_end: float | None = None
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or chunk.get("index") != index:
            raise AssertionError("report chunks are not contiguous and ordered")
        start = chunk.get("start_seconds")
        end = chunk.get("end_seconds")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise AssertionError(f"chunk {index} has invalid time boundaries")
        if start < 0 or end <= start or end - start > 600.01:
            raise AssertionError(f"chunk {index} violates the ten-minute hard limit")
        if index == 0 and start != 0:
            raise AssertionError("the first chunk does not begin at zero")
        if previous_end is not None:
            overlap = previous_end - start
            if abs(overlap - 12.0) > 0.01:
                raise AssertionError(
                    f"chunk {index} has {overlap:.3f}s overlap instead of 12s"
                )
        previous_end = float(end)

        chunk_hash = chunk.get("sha256")
        if not isinstance(chunk_hash, str) or len(chunk_hash) != 64:
            raise AssertionError(f"chunk {index} has no SHA-256 evidence")
        attempts = chunk.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != 1:
            raise AssertionError(f"chunk {index} must contain one OpenAI attempt")
        attempt = attempts[0]
        if not isinstance(attempt, dict):
            raise AssertionError(f"chunk {index} attempt is not an object")
        if (
            attempt.get("provider") != "openai"
            or attempt.get("model") != "gpt-transcribe"
            or attempt.get("attempt") != 1
        ):
            raise AssertionError(f"chunk {index} has unexpected provider evidence")
        raw = attempt.get("raw_transcript")
        if not isinstance(raw, str) or not raw.strip():
            raise AssertionError(f"chunk {index} has an empty raw provider transcript")
        expected_attempt_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if attempt.get("transcript_sha256") != expected_attempt_hash:
            raise AssertionError(f"chunk {index} raw transcript hash does not match")
        if chunk.get("canonical") != {"provider": "openai", "attempt": 1}:
            raise AssertionError(f"chunk {index} does not retain canonical selection")
        candidate_identities.add(
            (expected["hash"], index, "openai", "gpt-transcribe", 1)
        )

    if previous_end is None or abs(
        previous_end - expected["container_duration_ms"] / 1000
    ) > 0.01:
        raise AssertionError("final chunk does not reach the end of the recording")

    seams = report.get("seams")
    if not isinstance(seams, list) or len(seams) != len(chunks) - 1:
        raise AssertionError("report does not retain every deterministic seam")
    all_seams_aligned = True
    for index, seam in enumerate(seams):
        if not isinstance(seam, dict) or (
            seam.get("left_chunk"), seam.get("right_chunk")
        ) != (index, index + 1):
            raise AssertionError("seam evidence is incomplete or out of order")
        aligned = seam.get("aligned")
        if not isinstance(aligned, bool):
            raise AssertionError(
                f"seam {index}->{index + 1} has no boolean alignment decision"
            )
        all_seams_aligned = all_seams_aligned and aligned
    return candidate_identities, all_seams_aligned


def _require_release_quality_state(
    message_id: str, state: Any, *, all_seams_aligned: bool
) -> None:
    """Require the published state to truthfully reflect recorded seam evidence."""
    expected_state = (
        "machine_transcribed" if all_seams_aligned else "needs_review"
    )
    if state != expected_state:
        raise AssertionError(
            f"fixture {message_id} has all_seams_aligned={all_seams_aligned}; "
            f"expected {expected_state!r}, got {state!r}"
        )


def _validate_candidate_files(
    workspace: Path,
    expected_identities: set[tuple[str, int, str, str, int]],
) -> None:
    paths = sorted((workspace / "candidates").glob("*.candidate.json"))
    actual_identities: set[tuple[str, int, str, str, int]] = set()
    for path in paths:
        payload = _json_object(path)
        if payload.get("kind") != "transcription_candidate":
            raise AssertionError(f"candidate has incorrect kind: {path}")
        audio = payload.get("audio")
        chunk = payload.get("chunk")
        attempt = payload.get("attempt")
        if not all(isinstance(item, dict) for item in (audio, chunk, attempt)):
            raise AssertionError(f"candidate has an incomplete schema: {path}")
        raw = attempt.get("raw_transcript")
        if not isinstance(raw, str) or not raw.strip():
            raise AssertionError(f"candidate has no raw transcript: {path}")
        if attempt.get("transcript_sha256") != hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest():
            raise AssertionError(f"candidate transcript hash mismatch: {path}")
        actual_identities.add(
            (
                str(audio.get("sha256")),
                int(chunk.get("index")),
                str(attempt.get("provider")),
                str(attempt.get("model")),
                int(attempt.get("attempt")),
            )
        )
    if actual_identities != expected_identities or len(paths) != len(
        expected_identities
    ):
        raise AssertionError(
            "candidate evidence does not correspond one-to-one with report chunks"
        )


def validate_transcribed_workspace(
    workspace: Path,
    *,
    revision: str,
    branch: str,
    audio_dir: Path | None = None,
    manifest_file: Path | None = None,
    transcriptions_dir: Path | None = None,
    candidates_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> None:
    """Validate transcript publication, chunk evidence, and provenance links."""
    workspace = workspace.resolve(strict=True)
    _validate_output_locations(
        workspace,
        audio_dir=audio_dir,
        manifest_file=manifest_file,
        transcriptions_dir=transcriptions_dir,
        candidates_dir=candidates_dir,
        reports_dir=reports_dir,
    )
    manifest = validate_source_fixture(workspace)
    all_candidate_identities: set[tuple[str, int, str, str, int]] = set()
    report_paths: set[Path] = set()
    transcript_paths: set[Path] = set()

    for message_id, expected in EXPECTED.items():
        entry = manifest[message_id]
        if "failed_stage" in entry:
            raise AssertionError(f"fixture {message_id} records a failed stage")
        transcription = entry.get("transcription")
        if not isinstance(transcription, dict):
            raise AssertionError(f"fixture {message_id} was not transcribed")
        state = transcription.get("status")
        if state not in ALLOWED_SINGLE_PROVIDER_STATES:
            raise AssertionError(
                f"fixture {message_id} has invalid single-provider state {state!r}"
            )
        expected_manifest_fields = {
            "model": "gpt-transcribe",
            "providers": ["openai"],
            "models": {"openai": "gpt-transcribe"},
            "primary_provider": "openai",
            "audio_sha256": expected["hash"],
            "audio_revision": revision,
            "audio_repository_branch": branch,
            "unresolved_discrepancies": 0,
        }
        for field, value in expected_manifest_fields.items():
            if transcription.get(field) != value:
                raise AssertionError(
                    f"fixture {message_id} transcription field {field!r}: "
                    f"expected {value!r}, got {transcription.get(field)!r}"
                )
        if transcription.get("quality_state") != state:
            raise AssertionError(f"fixture {message_id} quality state is inconsistent")
        if abs(
            int(transcription.get("duration_ms", -1))
            - int(expected["container_duration_ms"])
        ) > 2:
            raise AssertionError(f"fixture {message_id} duration evidence changed")

        output_file = str(transcription.get("output_file", ""))
        in_review_directory = Path(output_file).parts[:1] == ("needs-review",)
        if in_review_directory != (state == "needs_review"):
            raise AssertionError(
                f"fixture {message_id} output path does not match state {state!r}"
            )
        transcript_path = _within(
            workspace / "transcriptions", output_file, "published transcript"
        )
        transcript_paths.add(transcript_path)
        if transcription.get("output_path") != (
            Path("transcriptions") / output_file
        ).as_posix():
            raise AssertionError(f"fixture {message_id} output link is incorrect")
        if transcription.get("audio_path") != (
            Path("audio") / str(expected["filename"])
        ).as_posix():
            raise AssertionError(f"fixture {message_id} audio link is incorrect")
        transcript_bytes = transcript_path.read_bytes()
        try:
            transcript = transcript_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AssertionError("published transcript is not UTF-8") from exc
        if "\x00" in transcript or len(transcript.split()) < 300:
            raise AssertionError(
                f"fixture {message_id} transcript is implausibly short or malformed"
            )
        transcript_hash = hashlib.sha256(transcript_bytes).hexdigest()
        if transcription.get("transcript_sha256") != transcript_hash:
            raise AssertionError(f"fixture {message_id} transcript hash does not match")

        report_relative = str(transcription.get("evidence_report", ""))
        report_path = _within(workspace, report_relative, "evidence report")
        report_paths.add(report_path)
        report = _json_object(report_path)
        if report.get("kind") != "transcription_evidence_report":
            raise AssertionError(f"fixture {message_id} report kind is invalid")
        if report.get("final_quality_state") != state:
            raise AssertionError(f"fixture {message_id} report quality is inconsistent")
        if report.get("final_transcript_sha256") != transcript_hash:
            raise AssertionError(f"fixture {message_id} final report hash does not match")
        if report.get("primary_provider") != "openai":
            raise AssertionError(f"fixture {message_id} report primary is not OpenAI")
        if report.get("audio_repository") != {
            "revision": revision,
            "branch": branch,
        }:
            raise AssertionError(f"fixture {message_id} source provenance changed")
        audio = report.get("audio")
        if not isinstance(audio, dict) or (
            audio.get("path") != (Path("audio") / expected["filename"]).as_posix()
            or audio.get("sha256") != expected["hash"]
        ):
            raise AssertionError(f"fixture {message_id} report audio identity changed")
        if report.get("comparisons") != [] or report.get("failures") != []:
            raise AssertionError(
                f"fixture {message_id} contains unexpected comparison or failure evidence"
            )
        candidate_identities, all_seams_aligned = _validate_report_chunks(
            report, expected
        )
        _require_release_quality_state(
            message_id, state, all_seams_aligned=all_seams_aligned
        )
        all_candidate_identities.update(candidate_identities)

    actual_reports = set((workspace / "reports").glob("*.evidence.json"))
    actual_transcripts = {
        path
        for path in (workspace / "transcriptions").rglob("*.txt")
        if path.is_file()
    }
    if actual_reports != report_paths:
        raise AssertionError("workspace contains missing or unlinked evidence reports")
    if actual_transcripts != transcript_paths:
        raise AssertionError("workspace contains missing or unlinked transcripts")
    _validate_candidate_files(workspace, all_candidate_identities)
    if list(workspace.rglob(".git")):
        raise AssertionError("OctoScribe created repository metadata")


def content_snapshot(workspace: Path) -> dict[str, dict[str, Any]]:
    """Return hashes and sizes of every persistent file in the workspace."""
    snapshot: dict[str, dict[str, Any]] = {}
    for path in sorted(workspace.rglob("*")):
        if path.is_symlink():
            raise AssertionError(f"workspace contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(workspace).as_posix()
        snapshot[relative] = {"sha256": _sha256(path), "size": path.stat().st_size}
    return snapshot


def write_snapshot(workspace: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(content_snapshot(workspace), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def assert_idempotent(workspace: Path, snapshot_path: Path) -> None:
    expected = _json_object(snapshot_path)
    actual = content_snapshot(workspace)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        added = sorted(set(actual) - set(expected))
        changed = sorted(
            path
            for path in set(actual) & set(expected)
            if actual[path] != expected[path]
        )
        raise AssertionError(
            "idempotent rerun changed persistent evidence: "
            f"missing={missing}, added={added}, changed={changed}"
        )


def _add_verification_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--manifest-file", type=Path)
    parser.add_argument("--transcriptions-dir", type=Path)
    parser.add_argument("--candidates-dir", type=Path)
    parser.add_argument("--reports-dir", type=Path)


def _verify_from_arguments(args: argparse.Namespace) -> None:
    validate_transcribed_workspace(
        args.workspace,
        revision=args.revision,
        branch=args.branch,
        audio_dir=args.audio_dir,
        manifest_file=args.manifest_file,
        transcriptions_dir=args.transcriptions_dir,
        candidates_dir=args.candidates_dir,
        reports_dir=args.reports_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--fixture-root", type=Path, required=True)
    prepare.add_argument("--workspace", type=Path, required=True)
    verify = commands.add_parser("verify")
    _add_verification_arguments(verify)
    verify.add_argument("--snapshot", type=Path, required=True)
    idempotent = commands.add_parser("assert-idempotent")
    _add_verification_arguments(idempotent)
    idempotent.add_argument("--snapshot", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepare_workspace(args.fixture_root, args.workspace)
            print(f"Prepared verified Telegram fixture at {args.workspace}")
        elif args.command == "verify":
            _verify_from_arguments(args)
            write_snapshot(args.workspace, args.snapshot)
            print(f"Verified OpenAI evidence and wrote snapshot {args.snapshot}")
        else:
            _verify_from_arguments(args)
            assert_idempotent(args.workspace, args.snapshot)
            print("Verified idempotent rerun: every persistent byte is unchanged")
    except (AssertionError, OSError, ValueError, TypeError) as exc:
        print(f"OpenAI live verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
