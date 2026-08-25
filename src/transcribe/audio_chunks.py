"""ffmpeg-backed extraction of deterministic ASR chunks.

The policy for boundaries lives in :mod:`src.transcribe.chunking`; this module
only probes audio, detects silence, and materializes the exact context windows
as mono 16 kHz PCM WAV.  Every derived chunk is hashed before use so all
providers can be proven to have received identical bytes.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.audio import sha256_file
from src.transcribe.chunking import ChunkMetadata


class AudioToolError(RuntimeError):
    """Raised when probing or deterministic chunk extraction fails."""


@dataclass(frozen=True, slots=True)
class MaterializedChunk:
    """One context window on disk plus its immutable content identity."""

    metadata: ChunkMetadata
    path: Path
    sha256: str
    size_bytes: int


_Runner = Callable[..., subprocess.CompletedProcess[str]]
_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)")


class FFmpegAudioTools:
    """Small, injectable wrapper around ffprobe and ffmpeg."""

    def __init__(
        self,
        runner: _Runner = subprocess.run,
        *,
        timeout_seconds: float = 900.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("audio tool timeout must be positive")
        self._runner = runner
        self._timeout_seconds = timeout_seconds

    def probe_duration_ms(self, audio_path: Path) -> int:
        """Return rounded duration in milliseconds using ffprobe."""
        path = self._validate_source(audio_path)
        result = self._run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
        )
        try:
            seconds = float(result.stdout.strip())
        except (TypeError, ValueError) as exc:
            raise AudioToolError("ffprobe returned an invalid duration") from exc
        if seconds <= 0:
            raise AudioToolError("audio duration must be positive")
        return int(round(seconds * 1000))

    def detect_silences(
        self,
        audio_path: Path,
        *,
        threshold_db: float = -35.0,
        min_silence_ms: int = 500,
        duration_ms: int | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """Return complete silence intervals reported by ffmpeg."""
        path = self._validate_source(audio_path)
        if min_silence_ms <= 0:
            raise ValueError("min_silence_ms must be positive")
        result = self._run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-i",
                str(path),
                "-af",
                f"silencedetect=noise={threshold_db:g}dB:d={min_silence_ms / 1000:g}",
                "-f",
                "null",
                "-",
            ],
            allow_stderr=True,
        )
        starts = [float(value) for value in _SILENCE_START_RE.findall(result.stderr)]
        ends = [float(value) for value in _SILENCE_END_RE.findall(result.stderr)]
        intervals: list[tuple[int, int]] = []
        for index, start in enumerate(starts):
            if index < len(ends):
                end = ends[index]
            elif duration_ms is not None:
                end = duration_ms / 1000
            else:
                break
            start_ms = max(0, int(round(start * 1000)))
            end_ms = max(start_ms, int(round(end * 1000)))
            intervals.append((start_ms, end_ms))
        return tuple(intervals)

    def materialize(
        self,
        audio_path: Path,
        chunks: Sequence[ChunkMetadata],
        output_dir: Path,
        *,
        max_bytes: int = 24 * 1024 * 1024,
    ) -> tuple[MaterializedChunk, ...]:
        """Extract and hash all context windows as deterministic WAV files."""
        source = self._validate_source(audio_path)
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[MaterializedChunk] = []
        for chunk in chunks:
            final_path = output_dir / f"chunk-{chunk.index:04d}.wav"
            partial_path = output_dir / f".chunk-{chunk.index:04d}.partial.wav"
            start_seconds = chunk.context_start_ms / 1000
            duration_seconds = chunk.context_duration_ms / 1000
            try:
                self._run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        "-y",
                        "-ss",
                        f"{start_seconds:.3f}",
                        "-i",
                        str(source),
                        "-t",
                        f"{duration_seconds:.3f}",
                        "-map",
                        "0:a:0",
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-c:a",
                        "pcm_s16le",
                        str(partial_path),
                    ]
                )
                if not partial_path.is_file() or partial_path.stat().st_size <= 0:
                    raise AudioToolError(
                        f"ffmpeg produced no audio for chunk {chunk.index}"
                    )
                size = partial_path.stat().st_size
                if size > max_bytes:
                    raise AudioToolError(
                        f"chunk {chunk.index} is {size} bytes, above limit {max_bytes}"
                    )
                os.replace(partial_path, final_path)
            finally:
                if partial_path.exists():
                    partial_path.unlink()
            results.append(
                MaterializedChunk(
                    metadata=chunk,
                    path=final_path,
                    sha256=sha256_file(final_path),
                    size_bytes=final_path.stat().st_size,
                )
            )
        return tuple(results)

    @staticmethod
    def _validate_source(audio_path: Path) -> Path:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"audio file does not exist: {path}")
        if path.stat().st_size <= 0:
            raise AudioToolError(f"audio file is empty: {path}")
        return path

    def _run(
        self,
        command: list[str],
        *,
        allow_stderr: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = self._runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=self._timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise AudioToolError(
                f"required audio tool {command[0]!r} is not installed"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioToolError(
                f"{command[0]} exceeded the {self._timeout_seconds:g}s timeout"
            ) from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise AudioToolError(f"{command[0]} failed: {detail}")
        if result.stderr and not allow_stderr and command[0] == "ffprobe":
            # ffprobe may emit harmless warnings; successful stdout remains
            # authoritative, so do not reject it.
            pass
        return result
