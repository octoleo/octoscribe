from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.audio import sha256_file
from src.transcribe.audio_chunks import MaterializedChunk
from src.transcribe.chunking import ChunkMetadata
from src.transcribe.consensus import QualityState
from src.transcribe.ensemble import EnsembleEngine
from src.transcribe.evidence import EvidenceStore


class _Backend:
    def __init__(self, name: str, values) -> None:
        self.name = name
        self.values = list(values)
        self.calls = 0

    def transcribe(self, path: Path) -> str:
        value = self.values[self.calls]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return value


class _AudioTools:
    def __init__(self, duration_ms=1000, chunk_count=1) -> None:
        self.duration_ms = duration_ms
        self.chunk_count = chunk_count

    def probe_duration_ms(self, path: Path) -> int:
        return self.duration_ms

    def detect_silences(self, path: Path, **kwargs):
        return ()

    def materialize(self, path: Path, plan, output_dir: Path, **kwargs):
        results = []
        for index in range(self.chunk_count):
            chunk_path = output_dir / f"chunk-{index}.wav"
            chunk_path.write_bytes(f"CHUNK-{index}".encode())
            metadata = ChunkMetadata(index, index * 100, (index + 1) * 100,
                                     index * 100, (index + 1) * 100)
            results.append(
                MaterializedChunk(
                    metadata, chunk_path, sha256_file(chunk_path), chunk_path.stat().st_size
                )
            )
        return tuple(results)


def _config(providers=("openai",), **overrides):
    values = dict(
        providers=providers,
        primary_provider=providers[0],
        workers=3,
        chunk_target_seconds=480,
        chunk_overlap_seconds=12,
        chunk_max_seconds=600,
        silence_search_seconds=45,
        silence_threshold_db=-35.0,
        silence_min_ms=500,
        max_chunk_megabytes=24,
        disagreement_retry_limit=1,
        arbitration_limit=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "sermon.mp3"
    path.write_bytes(b"IMMUTABLE ORIGINAL")
    return path


def test_one_provider_is_machine_transcribed(tmp_path: Path) -> None:
    backend = _Backend("openai", ["Exact text."])
    outcome = EnsembleEngine(
        _config(), {"openai": backend}, audio_tools=_AudioTools()
    ).transcribe(_audio(tmp_path))
    assert outcome.text == "Exact text."
    assert outcome.quality_state is QualityState.MACHINE_TRANSCRIBED
    assert backend.calls == 1


def test_two_providers_ignore_punctuation_case_for_agreement(tmp_path: Path) -> None:
    backends = {
        "openai": _Backend("openai", ["Jesus is Lord."]),
        "xai": _Backend("xai", ["JESUS is lord"]),
    }
    outcome = EnsembleEngine(
        _config(("openai", "xai")), backends, audio_tools=_AudioTools()
    ).transcribe(_audio(tmp_path))
    assert outcome.text == "Jesus is Lord."
    assert outcome.quality_state is QualityState.CROSS_CHECKED
    assert all(backend.calls == 1 for backend in backends.values())


def test_disagreement_retries_once_and_uses_retried_primary(tmp_path: Path) -> None:
    backends = {
        "openai": _Backend("openai", ["Do fear.", "Do not fear."]),
        "xai": _Backend("xai", ["Do not fear.", "Do not fear"]),
    }
    outcome = EnsembleEngine(
        _config(("openai", "xai")), backends, audio_tools=_AudioTools()
    ).transcribe(_audio(tmp_path))
    assert outcome.text == "Do not fear."
    assert outcome.quality_state is QualityState.CROSS_CHECKED
    assert [backend.calls for backend in backends.values()] == [2, 2]


def test_third_provider_runs_only_after_retry_and_loop_stops(tmp_path: Path) -> None:
    backends = {
        "openai": _Backend("openai", ["one", "one"]),
        "xai": _Backend("xai", ["two", "two"]),
        "meta": _Backend("meta", ["one"]),
    }
    outcome = EnsembleEngine(
        _config(("openai", "xai", "meta")), backends, audio_tools=_AudioTools()
    ).transcribe(_audio(tmp_path))
    assert outcome.text == "one"
    assert outcome.quality_state is QualityState.CROSS_CHECKED
    assert [backends[name].calls for name in ("openai", "xai", "meta")] == [2, 2, 1]
    assert outcome.unresolved_discrepancies == 0


def test_third_provider_cannot_outvote_or_rewrite_primary(tmp_path: Path) -> None:
    backends = {
        "openai": _Backend("openai", ["one", "one"]),
        "xai": _Backend("xai", ["two", "two"]),
        "meta": _Backend("meta", ["two"]),
    }
    outcome = EnsembleEngine(
        _config(("openai", "xai", "meta")), backends, audio_tools=_AudioTools()
    ).transcribe(_audio(tmp_path))

    assert outcome.text == "one"
    assert outcome.quality_state is QualityState.NEEDS_REVIEW
    assert [backends[name].calls for name in ("openai", "xai", "meta")] == [
        2,
        2,
        1,
    ]
    assert outcome.unresolved_discrepancies > 0


def test_persistent_two_provider_disagreement_has_a_hard_stop(tmp_path: Path) -> None:
    backends = {
        "openai": _Backend("openai", ["one", "one"]),
        "xai": _Backend("xai", ["two", "two"]),
    }
    outcome = EnsembleEngine(
        _config(("openai", "xai")), backends, audio_tools=_AudioTools()
    ).transcribe(_audio(tmp_path))

    assert outcome.text == "one"
    assert outcome.quality_state is QualityState.NEEDS_REVIEW
    assert [backend.calls for backend in backends.values()] == [2, 2]
    assert [item.stage for item in outcome.chunks[0].comparison_history] == [
        "initial",
        "discrepancy_retry",
    ]


def test_third_provider_replaces_unavailable_checker(tmp_path: Path) -> None:
    backends = {
        "openai": _Backend("openai", ["Primary evidence.", "Primary evidence."]),
        "xai": _Backend(
            "xai", [RuntimeError("offline"), RuntimeError("still offline")]
        ),
        "meta": _Backend("meta", ["primary evidence"]),
    }
    store = EvidenceStore(tmp_path / "candidates", tmp_path / "reports")
    outcome = EnsembleEngine(
        _config(("openai", "xai", "meta")),
        backends,
        audio_tools=_AudioTools(),
        model_names={"openai": "gpt-transcribe", "xai": "xai-stt", "meta": "omni"},
        evidence_store=store,
    ).transcribe(_audio(tmp_path))

    assert outcome.text == "Primary evidence."
    assert outcome.quality_state is QualityState.CROSS_CHECKED
    assert [backends[name].calls for name in ("openai", "xai", "meta")] == [1, 1, 1]
    assert [item.stage for item in outcome.chunks[0].comparison_history] == [
        "fallback_checker"
    ]
    payload = json.loads(outcome.evidence_report_path.read_text())
    assert payload["failures"] == [
        {
            "attempt": 1,
            "chunk_index": 0,
            "error": "RuntimeError: offline",
            "provider": "xai",
            "role": "initial",
        }
    ]


def test_availability_retry_can_recover_without_third_provider(tmp_path: Path) -> None:
    backends = {
        "openai": _Backend("openai", ["Primary evidence.", "Primary evidence."]),
        "xai": _Backend("xai", [RuntimeError("offline"), "primary evidence"]),
    }
    outcome = EnsembleEngine(
        _config(("openai", "xai")), backends, audio_tools=_AudioTools()
    ).transcribe(_audio(tmp_path))

    assert outcome.quality_state is QualityState.CROSS_CHECKED
    assert [backends[name].calls for name in ("openai", "xai")] == [2, 2]
    assert [item.stage for item in outcome.chunks[0].comparison_history] == [
        "availability_retry"
    ]


def test_fallback_checker_disagreement_retries_only_active_pair(
    tmp_path: Path,
) -> None:
    backends = {
        "openai": _Backend("openai", ["one", "one"]),
        "xai": _Backend("xai", [RuntimeError("offline")]),
        "meta": _Backend("meta", ["two", "two"]),
    }
    outcome = EnsembleEngine(
        _config(("openai", "xai", "meta")), backends, audio_tools=_AudioTools()
    ).transcribe(_audio(tmp_path))

    assert outcome.text == "one"
    assert outcome.quality_state is QualityState.NEEDS_REVIEW
    assert [backends[name].calls for name in ("openai", "xai", "meta")] == [2, 1, 2]
    assert [item.stage for item in outcome.chunks[0].comparison_history] == [
        "fallback_checker",
        "discrepancy_retry",
    ]


def test_failed_fallback_is_not_invoked_again_as_arbiter(tmp_path: Path) -> None:
    backends = {
        "openai": _Backend("openai", ["one", "one"]),
        "xai": _Backend("xai", [RuntimeError("offline"), "two"]),
        "meta": _Backend("meta", [RuntimeError("also offline")]),
    }
    outcome = EnsembleEngine(
        _config(("openai", "xai", "meta")), backends, audio_tools=_AudioTools()
    ).transcribe(_audio(tmp_path))

    assert outcome.text == "one"
    assert outcome.quality_state is QualityState.NEEDS_REVIEW
    assert [backends[name].calls for name in ("openai", "xai", "meta")] == [2, 2, 1]
    assert [item.stage for item in outcome.chunks[0].comparison_history] == [
        "availability_retry"
    ]
    assert [
        (item.provider, item.attempt, item.role)
        for item in outcome.chunks[0].failures
    ] == [
        ("xai", 1, "initial"),
        ("meta", 1, "fallback_checker"),
    ]


def test_secondary_outage_preserves_primary_but_downgrades(tmp_path: Path) -> None:
    backends = {
        "openai": _Backend("openai", ["Primary evidence.", "Primary evidence."]),
        "xai": _Backend(
            "xai", [RuntimeError("offline"), RuntimeError("still offline")]
        ),
    }
    outcome = EnsembleEngine(
        _config(("openai", "xai")), backends, audio_tools=_AudioTools()
    ).transcribe(_audio(tmp_path))
    assert outcome.text == "Primary evidence."
    assert outcome.quality_state is QualityState.NEEDS_REVIEW
    assert "offline" in outcome.chunks[0].failures[0].error


def test_primary_failure_is_fatal(tmp_path: Path) -> None:
    backends = {"openai": _Backend("openai", [RuntimeError("bad key")])}
    with pytest.raises(RuntimeError, match="primary provider.*bad key"):
        EnsembleEngine(
            _config(), backends, audio_tools=_AudioTools()
        ).transcribe(_audio(tmp_path))


def test_exact_overlap_is_removed_without_rewriting_primary(tmp_path: Path) -> None:
    backend = _Backend(
        "openai",
        [
            "Opening now Jesus Christ alone is our risen Lord.",
            "Jesus Christ alone is our risen Lord. Closing",
        ],
    )
    outcome = EnsembleEngine(
        _config(chunk_overlap_seconds=12),
        {"openai": backend},
        audio_tools=_AudioTools(duration_ms=1_200_000, chunk_count=2),
    ).transcribe(_audio(tmp_path))
    assert outcome.text == "Opening now Jesus Christ alone is our risen Lord. Closing"
    assert outcome.quality_state is QualityState.MACHINE_TRANSCRIBED
    assert outcome.seams[0].alignment is not None


def test_unresolved_seam_is_needs_review(tmp_path: Path) -> None:
    backend = _Backend("openai", ["First ending.", "Unrelated opening."])
    outcome = EnsembleEngine(
        _config(),
        {"openai": backend},
        audio_tools=_AudioTools(duration_ms=1_200_000, chunk_count=2),
    ).transcribe(_audio(tmp_path))
    assert outcome.quality_state is QualityState.NEEDS_REVIEW
    assert outcome.seams[0].alignment is None


def test_audio_hash_mismatch_refuses_changed_evidence(tmp_path: Path) -> None:
    path = _audio(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        EnsembleEngine(
            _config(), {"openai": _Backend("openai", ["unused"])},
            audio_tools=_AudioTools(),
        ).transcribe(path, expected_sha256="0" * 64)


def test_persists_raw_candidates_and_quality_report_before_chunks_are_removed(
    tmp_path: Path,
) -> None:
    audio = _audio(tmp_path)
    store = EvidenceStore(tmp_path / "candidates", tmp_path / "reports")
    engine = EnsembleEngine(
        _config(("openai", "xai")),
        {
            "openai": _Backend("openai", ["Jesus is Lord."]),
            "xai": _Backend("xai", ["Jesus is lord"]),
        },
        audio_tools=_AudioTools(),
        model_names={"openai": "gpt-transcribe", "xai": "xai-stt"},
        evidence_store=store,
    )
    outcome = engine.transcribe(
        audio,
        logical_audio_path="audio/Sunday Sermon.mp3",
        audio_revision="a" * 40,
        audio_repository_branch="main",
    )
    assert outcome.evidence_report_path is not None
    assert outcome.evidence_report_path.exists()
    assert len(outcome.candidate_paths) == 2
    assert all(path.exists() for path in outcome.candidate_paths)
    payload = outcome.evidence_report_path.read_text(encoding="utf-8")
    assert '"final_quality_state": "cross_checked"' in payload
    assert '"raw_transcript": "Jesus is Lord."' in payload
    assert '"path": "audio/Sunday Sermon.mp3"' in payload
    assert '"revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"' in payload


def test_comparison_history_references_exact_mixed_retry_attempts(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "candidates", tmp_path / "reports")
    engine = EnsembleEngine(
        _config(("openai", "xai")),
        {
            "openai": _Backend("openai", ["wrong", "canonical words"]),
            "xai": _Backend("xai", ["other words", RuntimeError("retry outage")]),
        },
        audio_tools=_AudioTools(),
        model_names={"openai": "gpt-transcribe", "xai": "xai-stt"},
        evidence_store=store,
    )

    outcome = engine.transcribe(_audio(tmp_path))
    payload = json.loads(outcome.evidence_report_path.read_text())
    comparisons = payload["comparisons"]

    assert [
        (
            item["pass_index"],
            item["stage"],
            item["left"]["attempt"],
            item["right"]["attempt"],
        )
        for item in comparisons
    ] == [
        (0, "initial", 1, 1),
        (1, "discrepancy_retry", 2, 1),
    ]
    assert comparisons[0]["discrepancies"]
    assert comparisons[0]["discrepancies"][0]["left"]["tokens"]
    assert payload["chunks"][0]["canonical"] == {
        "provider": "openai",
        "attempt": 2,
    }


def test_paid_candidates_survive_a_later_primary_failure(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "candidates", tmp_path / "reports")
    engine = EnsembleEngine(
        _config(),
        {"openai": _Backend("openai", ["first chunk", RuntimeError("offline")])},
        audio_tools=_AudioTools(duration_ms=1_200_000, chunk_count=2),
        model_names={"openai": "gpt-transcribe"},
        evidence_store=store,
    )

    with pytest.raises(RuntimeError, match="primary provider"):
        engine.transcribe(_audio(tmp_path))

    candidates = list((tmp_path / "candidates").glob("*.candidate.json"))
    assert len(candidates) == 1
    assert "first chunk" in candidates[0].read_text()
    assert not list((tmp_path / "reports").glob("*.evidence.json"))
