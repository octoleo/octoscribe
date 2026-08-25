from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.transcribe.consensus import QualityState
from src.transcribe.transcriber import Transcriber


def test_transcriber_records_truthful_ensemble_state_and_provenance(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "sermon.mp3"
    audio.write_bytes(b"AUDIO")
    out_dir = tmp_path / "transcriptions"
    report = tmp_path / "reports" / "sermon.evidence.json"

    transcribe = SimpleNamespace(
        providers=("openai", "xai"),
        primary_provider="openai",
        transcriptions_dir=out_dir,
        artifacts_dir=tmp_path / "candidates",
        reports_dir=tmp_path / "reports",
        workers=2,
    )
    config = SimpleNamespace(
        transcribe=transcribe,
        download=SimpleNamespace(audio_dir=audio_dir),
    )
    manifest = MagicMock()
    manifest.pending_transcription.return_value = [
        {
            "telegram_msg_id": "42",
            "filename": "sermon.mp3",
            "title": None,
            "hash": "not-a-full-hash",
        }
    ]
    backend = MagicMock()
    backend.name = "openai"
    engine = MagicMock()
    engine.transcribe.return_value = SimpleNamespace(
        text="Do not fear.",
        quality_state=QualityState.COMPLETED_WITH_WARNINGS,
        audio_sha256="a" * 64,
        duration_ms=1234,
        evidence_report_path=report,
        unresolved_discrepancies=1,
        seams=(SimpleNamespace(left_chunk=0, right_chunk=1, alignment=None),),
        provider_failures=(
            SimpleNamespace(
                provider="xai",
                attempt=1,
                role="checker",
                error="temporary outage",
            ),
        ),
    )

    with (
        patch(
            "src.transcribe.transcriber.create_backend_registry",
            return_value={"openai": backend, "xai": MagicMock()},
        ),
        patch(
            "src.transcribe.transcriber.provider_model_name",
            side_effect=lambda cfg, provider: f"{provider}-model",
        ),
        patch("src.transcribe.transcriber.EnsembleEngine", return_value=engine),
    ):
        result = Transcriber(config, manifest).run()

    assert result.succeeded == 1
    assert result.completed_with_warnings == 1
    assert (out_dir / "sermon.txt").read_text() == "Do not fear."
    assert not (out_dir / "needs-review").exists()
    audio_hash = hashlib.sha256(b"AUDIO").hexdigest()
    engine.transcribe.assert_called_once_with(
        audio,
        expected_sha256=audio_hash,
        logical_audio_path=Path("audio") / "sermon.mp3",
    )
    manifest.mark_transcribed.assert_called_once_with(
        "42",
        {
            "output_file": "sermon.txt",
            "output_path": "transcriptions/sermon.txt",
            "audio_path": "audio/sermon.mp3",
            "model": "openai-model",
            "transcript_sha256": hashlib.sha256(b"Do not fear.").hexdigest(),
            "quality_state": "completed_with_warnings",
            "providers": ["openai", "xai"],
            "models": {
                "openai": "openai-model",
                "xai": "xai-model",
            },
            "primary_provider": "openai",
            "audio_sha256": "a" * 64,
            "duration_ms": 1234,
            "evidence_report": "reports/sermon.evidence.json",
            "unresolved_discrepancies": 1,
            "provider_failures": [
                {
                    "provider": "xai",
                    "attempt": 1,
                    "role": "checker",
                    "error": "temporary outage",
                }
            ],
            "integrity_warnings": [
                {"kind": "provider_disagreement", "count": 1},
                {"kind": "unaligned_seam", "left_chunk": 0, "right_chunk": 1},
                {
                    "kind": "provider_failure",
                    "provider": "xai",
                    "attempt": 1,
                    "role": "checker",
                    "error": "temporary outage",
                },
            ],
        },
    )
