from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from octoscribe import cmd_transcribe
from src.repository import DataRepoError, EvidenceRepositories


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        audio_repo=SimpleNamespace(branch="main"),
        download=SimpleNamespace(
            audio_dir=tmp_path / "audio",
            manifest_file=tmp_path / "text" / "manifest.json",
        ),
        transcribe=SimpleNamespace(
            providers=("openai",),
            backend="openai",
        ),
    )


def test_standalone_transcribe_requires_committed_audio_revision(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = MagicMock()
    manifest.all_entries.return_value = {
        "1": {"downloaded": True, "filename": "sermon.mp3"}
    }
    repositories = MagicMock(spec=EvidenceRepositories)
    repositories.audio_revision.return_value = "a" * 40
    transcriber = MagicMock()
    transcriber.run.return_value = SimpleNamespace(
        summary=lambda: "ok", failed=0, skipped=0
    )

    with (
        patch("src.manifest.Manifest", return_value=manifest),
        patch("src.repository.EvidenceRepositories", return_value=repositories),
        patch("src.transcribe.Transcriber", return_value=transcriber) as factory,
    ):
        cmd_transcribe(SimpleNamespace(dry_run=False), config)

    repositories.ensure_ready.assert_called_once_with()
    repositories.assert_audio_tracked.assert_called_once_with(
        [config.download.audio_dir / "sermon.mp3"]
    )
    factory.assert_called_once_with(
        config,
        manifest,
        audio_revision="a" * 40,
        audio_repository_branch="main",
    )


def test_standalone_transcribe_stops_when_audio_gate_fails(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = MagicMock()
    manifest.all_entries.return_value = {
        "1": {"downloaded": True, "filename": "sermon.mp3"}
    }
    repositories = MagicMock(spec=EvidenceRepositories)
    repositories.assert_audio_tracked.side_effect = DataRepoError("untracked")

    with (
        patch("src.manifest.Manifest", return_value=manifest),
        patch("src.repository.EvidenceRepositories", return_value=repositories),
        patch("src.transcribe.Transcriber") as factory,
        pytest.raises(SystemExit),
    ):
        cmd_transcribe(SimpleNamespace(dry_run=False), config)

    factory.assert_not_called()
