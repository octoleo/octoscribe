from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from octoscribe import cmd_transcribe
from src.repository import EvidenceWorkspaces


def _config(tmp_path: Path) -> SimpleNamespace:
    audio_root = tmp_path / "audio-workspace"
    text_root = tmp_path / "text-workspace"
    return SimpleNamespace(
        audio_repo=SimpleNamespace(path=audio_root),
        text_repo=SimpleNamespace(path=text_root),
        download=SimpleNamespace(
            audio_dir=audio_root / "audio",
            manifest_file=text_root / "manifest.json",
        ),
        transcribe=SimpleNamespace(
            providers=("openai",),
            backend="openai",
            transcriptions_dir=text_root / "transcriptions",
            artifacts_dir=text_root / "candidates",
            reports_dir=text_root / "reports",
        ),
    )


def test_standalone_transcribe_uses_paths_and_caller_provenance(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = MagicMock()
    workspaces = MagicMock(spec=EvidenceWorkspaces)
    transcriber = MagicMock()
    transcriber.run.return_value = SimpleNamespace(
        summary=lambda: "ok", failed=0, skipped=0
    )
    args = SimpleNamespace(
        dry_run=False,
        audio_revision="a" * 40,
        audio_repository_branch="sermon-audio",
    )

    with (
        patch("src.manifest.Manifest", return_value=manifest),
        patch("src.repository.EvidenceWorkspaces", return_value=workspaces),
        patch("src.transcribe.Transcriber", return_value=transcriber) as factory,
    ):
        cmd_transcribe(args, config)

    workspaces.ensure_ready.assert_called_once_with()
    factory.assert_called_once_with(
        config,
        manifest,
        audio_revision="a" * 40,
        audio_repository_branch="sermon-audio",
    )


def test_standalone_transcribe_allows_provenance_to_be_omitted(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = MagicMock()
    transcriber = MagicMock()
    transcriber.run.return_value = SimpleNamespace(
        summary=lambda: "ok", failed=0, skipped=0
    )

    with (
        patch("src.manifest.Manifest", return_value=manifest),
        patch("src.repository.EvidenceWorkspaces"),
        patch("src.transcribe.Transcriber", return_value=transcriber) as factory,
    ):
        cmd_transcribe(SimpleNamespace(dry_run=False), config)

    factory.assert_called_once_with(
        config,
        manifest,
        audio_revision=None,
        audio_repository_branch=None,
    )
