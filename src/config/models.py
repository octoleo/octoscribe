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
    model: str                   # e.g. "gpt-transcribe"
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

    # Fidelity-first ensemble options.  These fields are defaulted so callers
    # constructing the legacy configuration shape continue to work unchanged.
    # ``providers`` is ordered: the first item is the canonical transcript
    # source unless ``primary_provider`` explicitly selects another enabled
    # provider.
    providers: tuple[str, ...] = ()
    primary_provider: str = "openai"
    xai_api_key: Optional[str] = None
    xai_base_url: str = "https://api.x.ai/v1/stt"
    meta_asr_url: Optional[str] = None
    meta_asr_api_key: Optional[str] = None
    meta_asr_model: str = "omniASR_LLM_Unlimited_7B_v2"
    meta_asr_language: str = "eng_Latn"
    provider_timeout_seconds: float = 900.0

    # Long-recording policy.  WAV chunks are mono 16 kHz PCM; a ten-minute
    # hard maximum remains below OpenAI's documented 25 MB request limit.
    chunk_target_seconds: int = 480
    chunk_max_seconds: int = 600
    chunk_overlap_seconds: int = 12
    silence_search_seconds: int = 45
    silence_threshold_db: float = -35.0
    silence_min_ms: int = 500
    max_chunk_megabytes: int = 24

    # The disagreement loop is intentionally finite.  No provider may keep a
    # sermon in an automatic retry cycle forever.
    disagreement_retry_limit: int = 1
    arbitration_limit: int = 1
    artifacts_dir: Optional[Path] = None
    reports_dir: Optional[Path] = None


@dataclass
class DataRepoConfig:
    """Filesystem workspace supplied by the calling process or workflow.

    OctoScribe deliberately knows only the resolved local path.  Cloning,
    committing, and publishing that path are responsibilities of the caller.
    """

    path: Path
