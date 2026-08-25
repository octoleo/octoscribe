"""Tests for caller-owned filesystem workspaces."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

from src.config import DataRepoConfig
from src.repository import EvidenceWorkspaces


def _config(tmp_path: Path, *, split: bool = True) -> SimpleNamespace:
    audio_root = tmp_path / "audio-workspace"
    text_root = tmp_path / "text-workspace" if split else audio_root
    return SimpleNamespace(
        audio_repo=DataRepoConfig(path=audio_root),
        text_repo=DataRepoConfig(path=text_root),
        download=SimpleNamespace(
            audio_dir=audio_root / "audio",
            manifest_file=text_root / "manifest.json",
        ),
        transcribe=SimpleNamespace(
            transcriptions_dir=text_root / "transcriptions",
            artifacts_dir=text_root / "candidates",
            reports_dir=text_root / "reports",
        ),
    )


def test_data_workspace_config_contains_only_a_path(tmp_path: Path) -> None:
    config = DataRepoConfig(path=tmp_path)
    assert config.path == tmp_path
    assert set(config.__dataclass_fields__) == {"path"}


def test_ensure_ready_creates_split_evidence_directories(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workspaces = EvidenceWorkspaces(config)

    workspaces.ensure_ready()

    assert workspaces.is_split
    assert config.download.audio_dir.is_dir()
    assert config.download.manifest_file.parent.is_dir()
    assert config.transcribe.transcriptions_dir.is_dir()
    assert config.transcribe.artifacts_dir.is_dir()
    assert config.transcribe.reports_dir.is_dir()


def test_ensure_ready_supports_legacy_shared_workspace(tmp_path: Path) -> None:
    config = _config(tmp_path, split=False)
    workspaces = EvidenceWorkspaces(config)

    workspaces.ensure_ready()

    assert not workspaces.is_split
    statuses = workspaces.status()
    assert statuses["audio"] is statuses["transcripts"]


def test_status_is_read_only_for_missing_workspaces(tmp_path: Path) -> None:
    config = _config(tmp_path)

    statuses = EvidenceWorkspaces(config).status()

    assert not statuses["audio"].exists
    assert not statuses["transcripts"].exists
    assert not config.audio_repo.path.exists()
    assert not config.text_repo.path.exists()


def test_workspace_module_has_no_source_control_process_boundary() -> None:
    import src.repository as repository

    tree = ast.parse(inspect.getsource(repository))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    call_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "subprocess" not in imported_modules
    assert "run" not in call_names
    assert not hasattr(repository, "DataRepository")
