"""
tests/conftest.py — Shared pytest fixtures for the OctoScribe test suite.

Provides:
  - tmp_data_dir     A temporary directory with audio/ and transcriptions/
  - sample_config    A fully populated Config with fake credentials
  - tmp_manifest     A fresh empty Manifest backed by a temp file
  - populated_manifest  A Manifest with 3 downloaded entries (2 transcribed)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    Config,
    DataRepoConfig,
    DownloadConfig,
    TelegramConfig,
    TranscribeConfig,
)
from src.manifest import Manifest


# ---------------------------------------------------------------------------
# Directory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """A temporary data repository directory with required subdirs."""
    (tmp_path / "audio").mkdir()
    (tmp_path / "transcriptions").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config(tmp_data_dir: Path) -> Config:
    """
    A Config with fake credentials and tmp_path-based directories.

    All required fields are set to fake-but-valid values.
    All Path fields point to subdirectories of tmp_data_dir.
    No real network or filesystem side-effects are triggered by construction.
    """
    session_dir = tmp_data_dir / ".session"
    session_dir.mkdir(exist_ok=True)

    telegram = TelegramConfig(
        api_id=12345,
        api_hash="fake_api_hash_abc123",
        phone="+15550001234",
        group="test_group",
        session_dir=session_dir,
    )

    download = DownloadConfig(
        workers=2,
        resume=True,
        deduplicate=True,
        audio_dir=tmp_data_dir / "audio",
        manifest_file=tmp_data_dir / "manifest.json",
    )

    transcribe = TranscribeConfig(
        backend="openai",
        model="gpt-4o-transcribe",
        language="en",
        workers=2,
        retry_attempts=1,
        retry_base_delay=0.1,
        retry_max_delay=1.0,
        api_key="sk-test-fakekey",
        local_model="large-v3",
        device="cpu",
        compute_type="int8",
        beam_size=5,
        best_of=5,
        repetition_penalty=1.1,
        vad_filter=True,
        vad_min_silence_ms=500,
        vad_speech_pad_ms=400,
        transcriptions_dir=tmp_data_dir / "transcriptions",
        manifest_file=tmp_data_dir / "manifest.json",
    )

    data_repo = DataRepoConfig(
        url=None,
        path=tmp_data_dir,
        branch="main",
        auto_push=False,
    )

    return Config(
        telegram=telegram,
        download=download,
        transcribe=transcribe,
        data_repo=data_repo,
        ini_path=tmp_data_dir / "octoscribe.ini",
    )


# ---------------------------------------------------------------------------
# Manifest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_manifest(tmp_data_dir: Path) -> Manifest:
    """A fresh empty Manifest backed by a temp file."""
    return Manifest(tmp_data_dir / "manifest.json")


@pytest.fixture
def populated_manifest(tmp_manifest: Manifest) -> Manifest:
    """
    A Manifest with 3 downloaded entries (2 transcribed, 1 pending).

    Entry 1001 — downloaded + transcribed
    Entry 1002 — downloaded + transcribed
    Entry 1003 — downloaded, pending transcription
    """
    tmp_manifest.mark_downloaded(
        "1001",
        {
            "filename": "sermon1.mp3",
            "title": "Sermon One",
            "date": "2024-01-01",
        },
    )
    tmp_manifest.mark_downloaded(
        "1002",
        {
            "filename": "sermon2.mp3",
            "title": "Sermon Two",
            "date": "2024-01-02",
        },
    )
    tmp_manifest.mark_downloaded(
        "1003",
        {
            "filename": "sermon3.mp3",
            "title": "Sermon Three",
            "date": "2024-01-03",
        },
    )
    tmp_manifest.mark_transcribed(
        "1001",
        {"output_file": "sermon1.txt", "model": "openai"},
    )
    tmp_manifest.mark_transcribed(
        "1002",
        {"output_file": "sermon2.txt", "model": "openai"},
    )
    return tmp_manifest
