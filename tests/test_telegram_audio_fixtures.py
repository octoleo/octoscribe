"""Integrity and real-decoder checks for owner-supplied Telegram OGG files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.audio import sha256_file
from src.manifest import Manifest
from src.transcribe.audio_chunks import FFmpegAudioTools
from src.transcribe.backends.base import TranscriptionBackend
from src.transcribe.chunking import (
    DEFAULT_HARD_MAX_MS,
    DEFAULT_OVERLAP_MS,
    DEFAULT_TARGET_CORE_MS,
    ChunkMetadata,
    plan_chunks,
)
from src.transcribe.transcriber import Transcriber


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "telegram"
EXPECTED = {
    "856": {
        "filename": "1 Timothy 15-6.ogg",
        "title": "1 Timothy 1:5-6",
        "performer": "Family Devotions",
        "hash": "7b2bc9c89b96cc528cd3f63a88e63710e2128516dae1079f4fabf8414cd8b060",
        "duration": 1412,
        "container_duration_ms": 1_411_093,
        "planned_chunks": 3,
    },
    "990": {
        "filename": "1 John 17-8.ogg",
        "title": "1 John 1:7-8",
        "performer": "Family Devotions",
        "hash": "698056c50804e1033b1c68adcbf7d4064c32f112aaa6aba23f0fbe524472849a",
        "duration": 1850,
        "container_duration_ms": 1_849_253,
        "planned_chunks": 4,
    },
}


def test_fixture_manifest_preserves_telegram_contract_and_pending_state() -> None:
    raw = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert set(raw) == set(EXPECTED)
    for message_id, expected in EXPECTED.items():
        entry = raw[message_id]
        assert entry["telegram_msg_id"] == int(message_id)
        assert entry["downloaded"] is True
        assert entry["extension"] == ".ogg"
        assert entry["original_filename"] == "record.ogg"
        for field in ("filename", "title", "performer", "hash", "duration"):
            assert entry[field] == expected[field]
        assert sha256_file(FIXTURE_ROOT / "audio" / entry["filename"]) == entry["hash"]

    manifest = Manifest(FIXTURE_ROOT / "manifest.json")
    assert len(manifest.pending_transcription()) == 2


@pytest.mark.parametrize("message_id", sorted(EXPECTED))
def test_real_ffmpeg_decodes_owner_supplied_telegram_ogg(
    message_id: str, tmp_path: Path
) -> None:
    expected = EXPECTED[message_id]
    source = FIXTURE_ROOT / "audio" / str(expected["filename"])
    source_hash = sha256_file(source)
    tools = FFmpegAudioTools()
    duration_ms = tools.probe_duration_ms(source)
    assert duration_ms == pytest.approx(expected["container_duration_ms"], abs=2)

    silences = tools.detect_silences(
        source,
        duration_ms=duration_ms,
    )
    planner_options = {
        "target_core_ms": DEFAULT_TARGET_CORE_MS,
        "overlap_ms": DEFAULT_OVERLAP_MS,
        "hard_max_ms": DEFAULT_HARD_MAX_MS,
        "silence_search_ms": 45_000,
    }
    planned = plan_chunks(duration_ms, silences, **planner_options)
    assert planned == plan_chunks(
        duration_ms,
        reversed(silences),
        **planner_options,
    )
    assert len(planned) == expected["planned_chunks"]
    assert planned[0].core_start_ms == planned[0].context_start_ms == 0
    assert planned[-1].core_end_ms == planned[-1].context_end_ms == duration_ms
    assert all(
        chunk.context_duration_ms <= DEFAULT_HARD_MAX_MS for chunk in planned
    )
    for left, right in zip(planned, planned[1:]):
        assert left.core_end_ms == right.core_start_ms
        assert left.context_end_ms - right.context_start_ms == DEFAULT_OVERLAP_MS

    # Full-duration decoding above proves that silence discovery works across
    # the complete owner-supplied recording. Keep derived WAV materialization
    # short so normal CI validates the real OGG decoder without writing about
    # 100 MiB of temporary PCM on every run.
    probe = ChunkMetadata(
        index=0,
        core_start_ms=0,
        core_end_ms=2_000,
        context_start_ms=0,
        context_end_ms=2_000,
    )
    [chunk] = tools.materialize(source, [probe], tmp_path / message_id)

    assert sha256_file(source) == source_hash == expected["hash"]
    assert chunk.metadata is probe
    assert chunk.path.is_file()
    assert chunk.path.stat().st_size > 0
    assert chunk.sha256 == sha256_file(chunk.path)


def test_real_fixture_manifest_transcribes_once_and_then_resumes(
    tmp_path: Path,
) -> None:
    """The production manifest/output contract prevents repeat API work."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes((FIXTURE_ROOT / "manifest.json").read_bytes())
    out_dir = tmp_path / "transcriptions"
    config = SimpleNamespace(
        transcribe=SimpleNamespace(
            providers=(),
            transcriptions_dir=out_dir,
            workers=1,
            api_key=None,
            xai_api_key=None,
            meta_asr_api_key=None,
        ),
        download=SimpleNamespace(audio_dir=FIXTURE_ROOT / "audio"),
    )
    manifest = Manifest(manifest_path)
    backend = MagicMock(spec=TranscriptionBackend)
    backend.name = "fixture-backend"
    backend.transcribe.side_effect = lambda path: f"Transcript for {path.stem}."

    first = Transcriber(config, manifest, backend=backend).run()

    assert first.succeeded == 2
    assert backend.transcribe.call_count == 2
    first_state = manifest.all_entries()
    output_hashes: dict[str, str] = {}
    for message_id, entry in first_state.items():
        transcription = entry["transcription"]
        output = out_dir / transcription["output_file"]
        assert output.is_file()
        assert transcription["audio_path"] == f"audio/{entry['filename']}"
        assert transcription["output_path"] == (
            f"transcriptions/{transcription['output_file']}"
        )
        output_hashes[message_id] = hashlib.sha256(output.read_bytes()).hexdigest()
        assert transcription["transcript_sha256"] == output_hashes[message_id]

    second = Transcriber(config, manifest, backend=backend).run()

    assert second.total == 0
    assert backend.transcribe.call_count == 2
    assert manifest.all_entries() == first_state
    for message_id, entry in manifest.all_entries().items():
        output = out_dir / entry["transcription"]["output_file"]
        assert hashlib.sha256(output.read_bytes()).hexdigest() == output_hashes[message_id]
