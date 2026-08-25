from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.audio import sha256_file
from src.transcribe.audio_chunks import AudioToolError, FFmpegAudioTools
from src.transcribe.chunking import ChunkMetadata


class _Runner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.duration = "1200.125\n"
        self.silence = ""
        self.fail = False
        self.chunk_bytes = b"RIFF-DERIVED-WAV"

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if self.fail:
            return subprocess.CompletedProcess(command, 1, "", "failure")
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, self.duration, "")
        if "silencedetect=" in " ".join(command):
            return subprocess.CompletedProcess(command, 0, "", self.silence)
        Path(command[-1]).write_bytes(self.chunk_bytes)
        return subprocess.CompletedProcess(command, 0, "", "")


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "sermon.mp3"
    path.write_bytes(b"ORIGINAL")
    return path


def test_probe_duration_uses_ffprobe_and_rounds_ms(tmp_path: Path) -> None:
    runner = _Runner()
    tools = FFmpegAudioTools(runner)
    assert tools.probe_duration_ms(_audio(tmp_path)) == 1_200_125
    assert runner.commands[0][0] == "ffprobe"


def test_probe_rejects_invalid_duration(tmp_path: Path) -> None:
    runner = _Runner()
    runner.duration = "not-a-number"
    with pytest.raises(AudioToolError, match="invalid duration"):
        FFmpegAudioTools(runner).probe_duration_ms(_audio(tmp_path))


def test_detect_silence_intervals_and_closes_trailing_interval(tmp_path: Path) -> None:
    runner = _Runner()
    runner.silence = (
        "[silencedetect] silence_start: 10.25\n"
        "[silencedetect] silence_end: 11.5 | silence_duration: 1.25\n"
        "[silencedetect] silence_start: 19\n"
    )
    result = FFmpegAudioTools(runner).detect_silences(
        _audio(tmp_path), duration_ms=20_000
    )
    assert result == ((10_250, 11_500), (19_000, 20_000))


def test_materialize_uses_exact_context_window_and_hashes_bytes(tmp_path: Path) -> None:
    runner = _Runner()
    chunk = ChunkMetadata(0, 12_000, 20_000, 10_000, 22_000)
    [result] = FFmpegAudioTools(runner).materialize(
        _audio(tmp_path), [chunk], tmp_path / "chunks"
    )
    command = runner.commands[-1]
    assert command[command.index("-ss") + 1] == "10.000"
    assert command[command.index("-t") + 1] == "12.000"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert result.path.read_bytes() == runner.chunk_bytes
    assert result.sha256 == sha256_file(result.path)
    assert not any(p.name.endswith("partial.wav") for p in result.path.parent.iterdir())


def test_materialize_rejects_oversize_and_removes_partial(tmp_path: Path) -> None:
    runner = _Runner()
    runner.chunk_bytes = b"too large"
    chunk = ChunkMetadata(0, 0, 1000, 0, 1000)
    output = tmp_path / "chunks"
    with pytest.raises(AudioToolError, match="above limit"):
        FFmpegAudioTools(runner).materialize(
            _audio(tmp_path), [chunk], output, max_bytes=2
        )
    assert not list(output.iterdir())


def test_tool_failure_has_actionable_error(tmp_path: Path) -> None:
    runner = _Runner()
    runner.fail = True
    with pytest.raises(AudioToolError, match="ffprobe failed: failure"):
        FFmpegAudioTools(runner).probe_duration_ms(_audio(tmp_path))


def test_missing_and_empty_audio_rejected_before_tool(tmp_path: Path) -> None:
    runner = _Runner()
    with pytest.raises(FileNotFoundError):
        FFmpegAudioTools(runner).probe_duration_ms(tmp_path / "missing.mp3")
    empty = tmp_path / "empty.mp3"
    empty.touch()
    with pytest.raises(AudioToolError, match="empty"):
        FFmpegAudioTools(runner).probe_duration_ms(empty)
    assert runner.commands == []
