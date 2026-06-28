"""
src/config/models.py — Typed configuration value objects.

Each dataclass models one cohesive slice of OctoScribe's configuration.  They
hold *validated, resolved* values only: parsing, defaulting, env/INI precedence
and validation all happen in :mod:`src.config.loader`.  Keeping these as plain,
logic-free data containers is a deliberate Single-Responsibility split — code
that consumes configuration depends on these stable shapes, not on how the
values were assembled.

The root :class:`~src.config.root.Config` aggregates one instance of each of
these.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SourceConfig:
    """Audio source selection: a Telegram group or a local folder."""

    mode: str                 # "telegram" | "folder"
    folder: Optional[Path]    # local folder to import from when mode == "folder"
    recursive: bool           # scan the folder recursively for audio files


@dataclass
class TelegramConfig:
    """Telegram client credentials and target group."""

    api_id: Optional[int]   # required in telegram mode; may be None in folder mode
    api_hash: str
    phone: str
    group: str          # INI [telegram] group
    session_dir: Path   # where to store .session files


@dataclass
class DownloadConfig:
    """Audio download behaviour."""

    workers: int
    resume: bool
    deduplicate: bool
    audio_dir: Path
    manifest_file: Path


@dataclass
class TranscribeConfig:
    """Transcription pipeline settings (both OpenAI and local Whisper)."""

    backend: str                 # "openai" | "local"
    model: str                   # e.g. "gpt-4o-transcribe"
    language: str                # e.g. "en"
    workers: int
    retry_attempts: int
    retry_base_delay: float
    retry_max_delay: float
    api_key: Optional[str]       # from env OPENAI_API_KEY

    # Local Whisper options
    local_model: str             # e.g. "large-v3"
    device: str                  # "cuda" | "cpu"
    compute_type: str            # e.g. "int8_float16"
    beam_size: int
    best_of: int
    repetition_penalty: float
    vad_filter: bool
    vad_min_silence_ms: int
    vad_speech_pad_ms: int

    transcriptions_dir: Path
    manifest_file: Path


@dataclass
class DataRepoConfig:
    """Git data-repository settings."""

    url: Optional[str]   # from env DATA_REPO_URL
    path: Path           # local clone path
    branch: str
    auto_push: bool
