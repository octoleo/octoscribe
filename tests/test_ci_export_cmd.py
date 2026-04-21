"""
Tests for the `ci-export` subcommand (cmd_ci_export) in octoscribe.py.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from octoscribe import cmd_ci_export, _CI_ENV_MARKERS  # noqa: E402


def _make_config(tmp_path: Path, *, session_exists: bool = True) -> MagicMock:
    cfg = MagicMock()
    cfg.telegram.session_dir = tmp_path
    cfg.telegram.api_id = 12345
    cfg.telegram.api_hash = "testhash"
    cfg.telegram.phone = "+15550001234"
    cfg.telegram.group = "@test_group"
    cfg.transcribe.api_key = "sk-testkey"
    cfg.transcribe.backend = "openai"
    cfg.transcribe.model = "gpt-4o-transcribe"
    cfg.transcribe.language = "en"
    cfg.data_repo.url = "git@github.com:org/data.git"
    cfg.data_repo.branch = "main"

    if session_exists:
        session_file = tmp_path / "octoscribe.session"
        session_file.write_bytes(b"fake-session-bytes")

    return cfg


def _args() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# CI environment guard
# ---------------------------------------------------------------------------

class TestCiGuard:
    @pytest.mark.parametrize("marker", _CI_ENV_MARKERS)
    def test_blocked_when_ci_marker_set(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        marker: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv(marker, "true")
        with pytest.raises(SystemExit):
            cmd_ci_export(_args(), _make_config(tmp_path))

        captured = capsys.readouterr()
        assert "blocked" in captured.err.lower()
        assert marker in captured.err

    def test_allowed_when_no_ci_markers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        for marker in _CI_ENV_MARKERS:
            monkeypatch.delenv(marker, raising=False)

        cmd_ci_export(_args(), _make_config(tmp_path))

        captured = capsys.readouterr()
        assert "OctoScribe CI/CD Export" in captured.out


# ---------------------------------------------------------------------------
# Output content
# ---------------------------------------------------------------------------

class TestCiExportOutput:
    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs):
        for marker in _CI_ENV_MARKERS:
            monkeypatch.delenv(marker, raising=False)
        from io import StringIO
        import sys
        cfg = _make_config(tmp_path, **kwargs)
        cmd_ci_export(_args(), cfg)

    def test_secrets_section_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._run(tmp_path, monkeypatch)
        out = capsys.readouterr().out
        assert "TELEGRAM_API_ID" in out
        assert "TELEGRAM_API_HASH" in out
        assert "TELEGRAM_PHONE" in out
        assert "OPENAI_API_KEY" in out
        assert "TELEGRAM_SESSION_B64" in out

    def test_variables_section_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._run(tmp_path, monkeypatch)
        out = capsys.readouterr().out
        assert "TELEGRAM_GROUP" in out
        assert "TRANSCRIBE_BACKEND" in out
        assert "DATA_REPO_BRANCH" in out

    def test_session_b64_is_valid_base64(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._run(tmp_path, monkeypatch, session_exists=True)
        out = capsys.readouterr().out
        for line in out.splitlines():
            if "TELEGRAM_SESSION_B64" in line:
                b64_val = line.split("=", 1)[1].strip()
                decoded = base64.b64decode(b64_val)
                assert decoded == b"fake-session-bytes"
                break
        else:
            pytest.fail("TELEGRAM_SESSION_B64 line not found in output")

    def test_missing_session_shows_placeholder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._run(tmp_path, monkeypatch, session_exists=False)
        out = capsys.readouterr().out
        assert "not found" in out
