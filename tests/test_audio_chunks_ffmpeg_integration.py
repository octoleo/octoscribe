"""Real-tool integration coverage for deterministic ffmpeg chunk extraction."""

from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from src.audio import sha256_file
from src.transcribe.audio_chunks import FFmpegAudioTools
from src.transcribe.chunking import ChunkMetadata


def _require_audio_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        pytest.skip("real audio integration requires " + " and ".join(missing))


def _write_test_wav(path: Path) -> None:
    """Write tone, digital silence, then tone as mono 16 kHz PCM."""
    sample_rate = 16_000
    amplitude = 12_000
    frames = bytearray()
    for frame_index in range(3 * sample_rate):
        second = frame_index / sample_rate
        sample = (
            0
            if 1.0 <= second < 2.0
            else int(amplitude * math.sin(2 * math.pi * 440 * second))
        )
        frames.extend(struct.pack("<h", sample))

    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames)


def test_real_ffmpeg_probe_silence_and_materialization_are_deterministic(
    tmp_path: Path,
) -> None:
    _require_audio_tools()
    source = tmp_path / "generated-source.wav"
    _write_test_wav(source)
    tools = FFmpegAudioTools()

    duration_ms = tools.probe_duration_ms(source)
    assert duration_ms == pytest.approx(3_000, abs=2)

    silences = tools.detect_silences(
        source,
        threshold_db=-40,
        min_silence_ms=500,
        duration_ms=duration_ms,
    )
    assert len(silences) == 1
    silence_start_ms, silence_end_ms = silences[0]
    assert silence_start_ms == pytest.approx(1_000, abs=10)
    assert silence_end_ms == pytest.approx(2_000, abs=10)

    metadata = ChunkMetadata(
        index=0,
        core_start_ms=750,
        core_end_ms=2_250,
        context_start_ms=500,
        context_end_ms=2_500,
    )
    [first] = tools.materialize(source, [metadata], tmp_path / "first")
    [second] = tools.materialize(source, [metadata], tmp_path / "second")

    assert first.metadata is metadata
    assert first.sha256 == sha256_file(first.path)
    assert first.sha256 == second.sha256
    assert first.size_bytes == first.path.stat().st_size
    assert first.size_bytes == second.size_bytes
    assert not list(tmp_path.rglob("*.partial.wav"))

    with wave.open(str(first.path), "rb") as extracted:
        assert extracted.getnchannels() == 1
        assert extracted.getsampwidth() == 2
        assert extracted.getframerate() == 16_000
        assert extracted.getnframes() == pytest.approx(32_000, abs=16)

    assert tools.probe_duration_ms(first.path) == pytest.approx(2_000, abs=2)
