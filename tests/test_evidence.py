"""Tests for immutable and append-only transcription evidence persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest

from src.transcribe.consensus import QualityState
from src.transcribe.evidence import (
    SCHEMA_VERSION,
    AudioEvidence,
    ChunkEvidence,
    ComparisonSummary,
    EvidenceConflictError,
    EvidenceReport,
    EvidenceStore,
    HashMismatchError,
    ProviderAttemptEvidence,
    SeamSummary,
    TimedWordEvidence,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _evidence_files(tmp_path: Path) -> tuple[Path, Path]:
    audio = tmp_path / "audio" / "Sunday Sermon.mp3"
    chunk = tmp_path / "working" / "chunk-0000.flac"
    audio.parent.mkdir()
    chunk.parent.mkdir()
    audio.write_bytes(b"complete immutable sermon audio")
    chunk.write_bytes(b"exact chunk bytes sent to provider")
    return audio, chunk


def _attempt(
    *,
    provider: str = "openai",
    model: str = "gpt-transcribe",
    attempt: int = 1,
    transcript: str = "Jesus said, I am the way.",
) -> ProviderAttemptEvidence:
    return ProviderAttemptEvidence(
        provider=provider,
        model=model,
        attempt=attempt,
        raw_transcript=transcript,
    )


def _objects(
    tmp_path: Path,
    *,
    attempts: tuple[ProviderAttemptEvidence, ...] | None = None,
) -> tuple[Path, Path, AudioEvidence, ChunkEvidence]:
    audio_file, chunk_file = _evidence_files(tmp_path)
    audio = AudioEvidence.from_file(audio_file, duration_seconds=120.5)
    chunk = ChunkEvidence.from_file(
        chunk_file,
        index=0,
        start_seconds=0.0,
        end_seconds=120.5,
        attempts=(_attempt(),) if attempts is None else attempts,
    )
    return audio_file, chunk_file, audio, chunk


def _summary(*, agrees: bool = False) -> ComparisonSummary:
    return ComparisonSummary(
        chunk_index=0,
        left_provider="openai",
        left_model="gpt-transcribe",
        left_attempt=1,
        right_provider="local",
        right_model="large-v3",
        right_attempt=1,
        agrees=agrees,
        discrepancy_count=0 if agrees else 2,
        critical_discrepancy_count=0 if agrees else 1,
        additions=0 if agrees else 1,
        deletions=0,
        substitutions=0 if agrees else 1,
    )


def _report(
    audio: AudioEvidence,
    chunk: ChunkEvidence,
    *,
    quality: QualityState = QualityState.COMPLETED_WITH_WARNINGS,
) -> EvidenceReport:
    return EvidenceReport(
        audio=audio,
        chunks=(chunk,),
        comparisons=(_summary(),),
        final_quality_state=quality,
    )


def test_provider_attempt_and_chunk_evidence_are_immutable(tmp_path: Path) -> None:
    _, _, _, chunk = _objects(tmp_path)
    attempt = chunk.attempts[0]

    with pytest.raises(FrozenInstanceError):
        attempt.raw_transcript = "rewritten"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        chunk.sha256 = "0" * 64  # type: ignore[misc]


def test_candidate_records_required_provenance_and_raw_text(tmp_path: Path) -> None:
    audio_file, chunk_file, audio, chunk = _objects(tmp_path)
    store = EvidenceStore(tmp_path / "artifacts", tmp_path / "reports")

    path = store.write_candidate(
        audio,
        chunk,
        chunk.attempts[0],
        audio_file=audio_file,
        chunk_file=chunk_file,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["kind"] == "transcription_candidate"
    assert payload["audio"] == {
        "duration_seconds": 120.5,
        "path": str(audio_file),
        "sha256": _sha256(audio_file.read_bytes()),
    }
    assert payload["chunk"]["start_seconds"] == 0.0
    assert payload["chunk"]["end_seconds"] == 120.5
    assert payload["chunk"]["sha256"] == _sha256(chunk_file.read_bytes())
    assert payload["attempt"]["provider"] == "openai"
    assert payload["attempt"]["model"] == "gpt-transcribe"
    assert payload["attempt"]["attempt"] == 1
    assert payload["attempt"]["raw_transcript"] == (
        "Jesus said, I am the way."
    )
    assert payload["attempt"]["transcript_sha256"] == _sha256(
        b"Jesus said, I am the way."
    )


def test_candidate_retains_provider_word_timing_evidence(tmp_path: Path) -> None:
    timed = ProviderAttemptEvidence(
        provider="xai",
        model="xai-stt",
        attempt=1,
        raw_transcript="In the beginning",
        words=(
            TimedWordEvidence(
                text="beginning",
                start_seconds=0.4,
                end_seconds=0.9,
                confidence=0.98,
                speaker=0,
            ),
        ),
        language="en",
        duration_seconds=1.2,
    )
    audio_file, chunk_file, audio, chunk = _objects(
        tmp_path, attempts=(timed,)
    )
    path = EvidenceStore(
        tmp_path / "artifacts", tmp_path / "reports"
    ).write_candidate(
        audio,
        chunk,
        timed,
        audio_file=audio_file,
        chunk_file=chunk_file,
    )

    attempt = json.loads(path.read_text())["attempt"]
    assert attempt["language"] == "en"
    assert attempt["duration_seconds"] == 1.2
    assert attempt["words"] == [
        {
            "confidence": 0.98,
            "end_seconds": 0.9,
            "speaker": 0,
            "start_seconds": 0.4,
            "text": "beginning",
        }
    ]


def test_report_records_chunks_attempts_comparisons_and_quality(tmp_path: Path) -> None:
    audio_file, chunk_file, audio, chunk = _objects(
        tmp_path,
        attempts=(
            _attempt(provider="local", model="large-v3"),
            _attempt(),
        ),
    )
    report = _report(audio, chunk)
    store = EvidenceStore(tmp_path / "artifacts", tmp_path / "reports")

    path = store.write_report(
        report,
        audio_file=audio_file,
        chunk_files={0: chunk_file},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["kind"] == "transcription_evidence_report"
    assert payload["final_quality_state"] == "completed_with_warnings"
    assert [attempt["provider"] for attempt in payload["chunks"][0]["attempts"]] == [
        "local",
        "openai",
    ]
    comparison = payload["comparisons"][0]
    assert comparison["discrepancy_count"] == 2
    assert comparison["critical_discrepancy_count"] == 1
    assert comparison["additions"] == 1
    assert comparison["substitutions"] == 1


def test_report_records_non_generative_seam_evidence(tmp_path: Path) -> None:
    audio_file, chunk_file, audio, first = _objects(tmp_path)
    second_file = chunk_file.with_name("chunk-0001.flac")
    second_file.write_bytes(b"second exact chunk")
    second = ChunkEvidence.from_file(
        second_file,
        index=1,
        start_seconds=60,
        end_seconds=120.5,
        attempts=(_attempt(),),
    )
    report = EvidenceReport(
        audio=audio,
        chunks=(first, second),
        comparisons=(),
        final_quality_state=QualityState.MACHINE_TRANSCRIBED,
        seams=(
            SeamSummary(
                0,
                1,
                True,
                duplicate_tokens_removed=4,
                exact_matches=4,
                similarity=1.0,
            ),
        ),
    )
    path = EvidenceStore(
        tmp_path / "artifacts", tmp_path / "reports"
    ).write_report(
        report,
        audio_file=audio_file,
        chunk_files={0: chunk_file, 1: second_file},
    )
    [seam] = json.loads(path.read_text())["seams"]
    assert seam["aligned"] is True
    assert seam["duplicate_tokens_removed"] == 4


def test_json_is_canonical_sorted_and_stable(tmp_path: Path) -> None:
    audio_file, chunk_file, audio, chunk = _objects(tmp_path)
    store = EvidenceStore(tmp_path / "artifacts", tmp_path / "reports")
    path = store.write_report(
        _report(audio, chunk),
        audio_file=audio_file,
        chunk_files={0: chunk_file},
    )

    first = path.read_text(encoding="utf-8")
    payload = json.loads(first)
    expected = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    assert first == expected
    assert store.write_report(
        _report(audio, chunk),
        audio_file=audio_file,
        chunk_files={0: chunk_file},
    ) == path
    assert path.read_text(encoding="utf-8") == first


def test_store_uses_atomic_write_helper_for_both_artifact_types(
    tmp_path: Path,
) -> None:
    audio_file, chunk_file, audio, chunk = _objects(tmp_path)
    store = EvidenceStore(tmp_path / "artifacts", tmp_path / "reports")

    with patch("src.transcribe.evidence.atomic_write_text") as atomic:
        store.write_candidate(
            audio,
            chunk,
            chunk.attempts[0],
            audio_file=audio_file,
            chunk_file=chunk_file,
        )
        store.write_report(
            _report(audio, chunk),
            audio_file=audio_file,
            chunk_files={0: chunk_file},
        )

    assert atomic.call_count == 2
    assert all(call.args[0].suffix == ".json" for call in atomic.call_args_list)


def test_candidate_filename_cannot_escape_artifact_directory(tmp_path: Path) -> None:
    audio_file, chunk_file, audio, chunk = _objects(
        tmp_path,
        attempts=(
            _attempt(
                provider="../../Open AI",
                model="danger:model/../../../secret",
            ),
        ),
    )
    store = EvidenceStore(tmp_path / "artifacts", tmp_path / "reports")

    path = store.write_candidate(
        audio,
        chunk,
        chunk.attempts[0],
        audio_file=audio_file,
        chunk_file=chunk_file,
    )

    assert path.parent == store.artifact_dir
    assert path.name == Path(path.name).name
    assert "/" not in path.name
    assert "\\" not in path.name
    assert ".." not in path.name


def test_audio_hash_mismatch_is_rejected_before_writing(tmp_path: Path) -> None:
    audio_file, chunk_file, audio, chunk = _objects(tmp_path)
    bad_audio = AudioEvidence(
        path=audio.path,
        sha256="0" * 64,
        duration_seconds=audio.duration_seconds,
    )
    store = EvidenceStore(tmp_path / "artifacts", tmp_path / "reports")

    with pytest.raises(HashMismatchError, match="source audio"):
        store.write_candidate(
            bad_audio,
            chunk,
            chunk.attempts[0],
            audio_file=audio_file,
            chunk_file=chunk_file,
        )
    assert not store.artifact_dir.exists()


def test_chunk_hash_mismatch_is_rejected_before_writing(tmp_path: Path) -> None:
    audio_file, chunk_file, audio, chunk = _objects(tmp_path)
    bad_chunk = ChunkEvidence(
        index=chunk.index,
        path=chunk.path,
        start_seconds=chunk.start_seconds,
        end_seconds=chunk.end_seconds,
        sha256="f" * 64,
        attempts=chunk.attempts,
    )
    store = EvidenceStore(tmp_path / "artifacts", tmp_path / "reports")

    with pytest.raises(HashMismatchError, match="chunk 0"):
        store.write_candidate(
            audio,
            bad_chunk,
            bad_chunk.attempts[0],
            audio_file=audio_file,
            chunk_file=chunk_file,
        )
    assert not store.artifact_dir.exists()


def test_duplicate_candidate_is_idempotent_only_when_byte_identical(
    tmp_path: Path,
) -> None:
    audio_file, chunk_file, audio, chunk = _objects(tmp_path)
    store = EvidenceStore(tmp_path / "artifacts", tmp_path / "reports")
    path = store.write_candidate(
        audio,
        chunk,
        chunk.attempts[0],
        audio_file=audio_file,
        chunk_file=chunk_file,
    )
    original_bytes = path.read_bytes()

    with patch("src.transcribe.evidence.atomic_write_text") as atomic:
        second = store.write_candidate(
            audio,
            chunk,
            chunk.attempts[0],
            audio_file=audio_file,
            chunk_file=chunk_file,
        )

    assert second == path
    assert path.read_bytes() == original_bytes
    atomic.assert_not_called()


def test_duplicate_attempt_with_different_transcript_is_rejected(
    tmp_path: Path,
) -> None:
    audio_file, chunk_file, audio, chunk = _objects(tmp_path)
    store = EvidenceStore(tmp_path / "artifacts", tmp_path / "reports")
    path = store.write_candidate(
        audio,
        chunk,
        chunk.attempts[0],
        audio_file=audio_file,
        chunk_file=chunk_file,
    )
    original_bytes = path.read_bytes()
    conflicting = _attempt(transcript="A different output for attempt one.")

    with pytest.raises(EvidenceConflictError, match="different bytes"):
        store.write_candidate(
            audio,
            chunk,
            conflicting,
            audio_file=audio_file,
            chunk_file=chunk_file,
        )

    assert path.read_bytes() == original_bytes


def test_distinct_run_ids_preserve_nondeterministic_reruns(tmp_path: Path) -> None:
    audio_file, chunk_file, audio, chunk = _objects(tmp_path)
    store = EvidenceStore(tmp_path / "artifacts", tmp_path / "reports")
    first = store.write_candidate(
        audio,
        chunk,
        chunk.attempts[0],
        audio_file=audio_file,
        chunk_file=chunk_file,
        run_id="run-one",
    )
    changed = _attempt(transcript="A different but preserved second run.")
    changed_chunk = ChunkEvidence(
        index=chunk.index,
        path=chunk.path,
        start_seconds=chunk.start_seconds,
        end_seconds=chunk.end_seconds,
        sha256=chunk.sha256,
        attempts=(changed,),
    )
    second = store.write_candidate(
        audio,
        changed_chunk,
        changed,
        audio_file=audio_file,
        chunk_file=chunk_file,
        run_id="run-two",
    )
    assert first != second
    assert first.exists() and second.exists()


def test_report_update_cannot_replace_existing_attempt(tmp_path: Path) -> None:
    audio_file, chunk_file, audio, chunk = _objects(tmp_path)
    store = EvidenceStore(tmp_path / "artifacts", tmp_path / "reports")
    path = store.write_report(
        _report(audio, chunk),
        audio_file=audio_file,
        chunk_files={0: chunk_file},
    )
    original_bytes = path.read_bytes()
    changed_attempt = _attempt(transcript="Changed evidence")
    changed_chunk = ChunkEvidence(
        index=chunk.index,
        path=chunk.path,
        start_seconds=chunk.start_seconds,
        end_seconds=chunk.end_seconds,
        sha256=chunk.sha256,
        attempts=(changed_attempt,),
    )

    with pytest.raises(EvidenceConflictError, match="overwrite"):
        store.write_report(
            _report(audio, changed_chunk),
            audio_file=audio_file,
            chunk_files={0: chunk_file},
        )

    assert path.read_bytes() == original_bytes


def test_report_can_grow_without_replacing_prior_attempts(tmp_path: Path) -> None:
    audio_file, chunk_file, audio, chunk = _objects(tmp_path)
    store = EvidenceStore(tmp_path / "artifacts", tmp_path / "reports")
    store.write_report(
        _report(audio, chunk),
        audio_file=audio_file,
        chunk_files={0: chunk_file},
    )
    expanded_chunk = ChunkEvidence(
        index=chunk.index,
        path=chunk.path,
        start_seconds=chunk.start_seconds,
        end_seconds=chunk.end_seconds,
        sha256=chunk.sha256,
        attempts=(
            *chunk.attempts,
            _attempt(provider="local", model="large-v3"),
        ),
    )

    path = store.write_report(
        _report(audio, expanded_chunk),
        audio_file=audio_file,
        chunk_files={0: chunk_file},
    )

    attempts = json.loads(path.read_text())["chunks"][0]["attempts"]
    assert {item["provider"] for item in attempts} == {"openai", "local"}


def test_serialized_schema_has_no_secret_or_request_metadata_keys(
    tmp_path: Path,
) -> None:
    _, _, audio, chunk = _objects(tmp_path)
    payload = _report(audio, chunk).to_dict()

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key.casefold()
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    serialized_keys = set(keys(payload))
    assert "api_key" not in serialized_keys
    assert "authorization" not in serialized_keys
    assert "access_token" not in serialized_keys
    assert "request_headers" not in serialized_keys


@pytest.mark.parametrize("bad_hash", ["", "abc", "A" * 64, "g" * 64])
def test_hash_fields_require_canonical_sha256(bad_hash: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        AudioEvidence("audio.mp3", bad_hash, 1.0)


def test_invalid_attempt_boundaries_and_comparison_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _attempt(attempt=0)
    with pytest.raises(ValueError, match="follow"):
        ChunkEvidence(
            index=0,
            path="chunk.flac",
            start_seconds=5.0,
            end_seconds=5.0,
            sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="sum"):
        ComparisonSummary(
            chunk_index=None,
            left_provider="a",
            left_model="one",
            left_attempt=1,
            right_provider="b",
            right_model="two",
            right_attempt=1,
            agrees=False,
            discrepancy_count=2,
            additions=1,
        )


def test_report_rejects_unknown_comparison_chunk(tmp_path: Path) -> None:
    _, _, audio, chunk = _objects(tmp_path)
    comparison = ComparisonSummary(
        chunk_index=9,
        left_provider="a",
        left_model="one",
        left_attempt=1,
        right_provider="b",
        right_model="two",
        right_attempt=1,
        agrees=True,
        discrepancy_count=0,
    )

    with pytest.raises(ValueError, match="unknown chunk"):
        EvidenceReport(
            audio=audio,
            chunks=(chunk,),
            comparisons=(comparison,),
            final_quality_state=QualityState.CROSS_CHECKED,
        )
