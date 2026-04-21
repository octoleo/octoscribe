"""
Tests for the `session` subcommand (cmd_session) in octoscribe.py.
"""

from __future__ import annotations

import base64
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from octoscribe import cmd_session  # noqa: E402


def _make_config(session_dir: Path) -> MagicMock:
    cfg = MagicMock()
    cfg.telegram.session_dir = session_dir
    return cfg


def _args(action: str | None) -> MagicMock:
    a = MagicMock()
    a.session_action = action
    return a


# ---------------------------------------------------------------------------
# session export
# ---------------------------------------------------------------------------

class TestSessionExport:
    def test_export_prints_base64(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session_file = tmp_path / "octoscribe.session"
        original_bytes = b"fake-sqlite-session-data-xyz"
        session_file.write_bytes(original_bytes)

        cmd_session(_args("export"), _make_config(tmp_path))

        captured = capsys.readouterr()
        decoded = base64.b64decode(captured.out.strip())
        assert decoded == original_bytes

    def test_export_missing_file_exits(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            cmd_session(_args("export"), _make_config(tmp_path))

        captured = capsys.readouterr()
        assert "No session file" in captured.err
        assert "download" in captured.err.lower()

    def test_export_output_has_no_newlines_inside(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session_file = tmp_path / "octoscribe.session"
        session_file.write_bytes(b"x" * 512)

        cmd_session(_args("export"), _make_config(tmp_path))

        captured = capsys.readouterr()
        # strip the trailing newline from print(), but the body must be clean
        assert "\n" not in captured.out.strip()


# ---------------------------------------------------------------------------
# session check
# ---------------------------------------------------------------------------

class TestSessionCheck:
    def test_check_existing_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session_file = tmp_path / "octoscribe.session"
        session_file.write_bytes(b"data" * 100)

        cmd_session(_args("check"), _make_config(tmp_path))

        captured = capsys.readouterr()
        assert str(session_file) in captured.out
        assert "400" in captured.out  # size: 4 * 100

    def test_check_missing_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cmd_session(_args("check"), _make_config(tmp_path))

        captured = capsys.readouterr()
        assert "No session file" in captured.out
        assert "download" in captured.out.lower()

    def test_check_default_when_no_action(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cmd_session(_args(None), _make_config(tmp_path))

        captured = capsys.readouterr()
        assert "No session file" in captured.out
