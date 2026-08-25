"""Regression tests for the strict filesystem-only runtime boundary."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import octoscribe
from octoscribe import build_parser, cmd_download, cmd_run


def _config(tmp_path: Path) -> SimpleNamespace:
    audio_root = tmp_path / "audio-workspace"
    text_root = tmp_path / "text-workspace"
    return SimpleNamespace(
        source=SimpleNamespace(mode="folder", folder=tmp_path / "incoming"),
        telegram=SimpleNamespace(group=""),
        audio_repo=SimpleNamespace(path=audio_root),
        text_repo=SimpleNamespace(path=text_root),
        download=SimpleNamespace(
            audio_dir=audio_root / "audio",
            manifest_file=text_root / "manifest.json",
        ),
        transcribe=SimpleNamespace(
            providers=("openai",),
            backend="openai",
            model="gpt-transcribe",
            transcriptions_dir=text_root / "transcriptions",
            artifacts_dir=text_root / "candidates",
            reports_dir=text_root / "reports",
        ),
    )


def test_cli_exposes_no_source_control_command_or_push_flag() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["sync"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--no-push"])


def test_cli_module_does_not_import_process_execution() -> None:
    tree = ast.parse(inspect.getsource(octoscribe))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "subprocess" not in imported_modules


def test_download_flushes_manifest_then_fails_on_partial_source_failure(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = MagicMock()
    stats = SimpleNamespace(failed=1, summary=lambda: "failed=1")

    with (
        patch("src.manifest.Manifest", return_value=manifest),
        patch("src.repository.EvidenceWorkspaces"),
        patch("octoscribe.acquire_audio", return_value=stats),
        pytest.raises(SystemExit),
    ):
        cmd_download(SimpleNamespace(), config)

    manifest.save.assert_called_once_with()


def test_run_finishes_good_transcriptions_before_reporting_source_failures(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = MagicMock()
    source_stats = SimpleNamespace(failed=1, summary=lambda: "failed=1")
    transcribe_stats = SimpleNamespace(
        failed=0,
        skipped=0,
        summary=lambda: "transcribed=1",
    )
    transcriber = MagicMock()
    transcriber.run.return_value = transcribe_stats
    args = SimpleNamespace(
        dry_run=False,
        audio_revision="b" * 40,
        audio_repository_branch="audio-main",
    )

    with (
        patch("src.manifest.Manifest", return_value=manifest),
        patch("src.repository.EvidenceWorkspaces"),
        patch("octoscribe.acquire_audio", return_value=source_stats),
        patch("src.transcribe.Transcriber", return_value=transcriber) as factory,
        pytest.raises(SystemExit),
    ):
        cmd_run(args, config)

    manifest.save.assert_called_once_with()
    transcriber.run.assert_called_once_with()
    factory.assert_called_once_with(
        config,
        manifest,
        audio_revision="b" * 40,
        audio_repository_branch="audio-main",
    )
