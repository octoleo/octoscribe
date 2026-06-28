"""
Tests for the audio-source CLI wiring in octoscribe.py.

Covers build_overrides (argument → config-override translation), the
acquire_audio dispatch between the folder and Telegram sources, and the
download/status commands in folder mode.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from octoscribe import acquire_audio, build_overrides, build_parser, cmd_download, cmd_status  # noqa: E402
from src.folder import ImportStats  # noqa: E402
from src.manifest import Manifest  # noqa: E402


# ---------------------------------------------------------------------------
# build_overrides — argument translation
# ---------------------------------------------------------------------------

class TestBuildOverrides:
    def _parse(self, argv: list[str]):
        return build_parser().parse_args(argv)

    def test_folder_shorthand_implies_folder_mode(self) -> None:
        args = self._parse(["download", "--folder", "/data/sermons"])
        overrides = build_overrides(args)
        assert overrides["source__folder"] == "/data/sermons"
        assert overrides["source__mode"] == "folder"

    def test_explicit_source_and_folder(self) -> None:
        args = self._parse(["run", "--source", "folder", "--folder", "/x"])
        overrides = build_overrides(args)
        assert overrides["source__mode"] == "folder"
        assert overrides["source__folder"] == "/x"

    def test_explicit_telegram_source(self) -> None:
        args = self._parse(["download", "--source", "telegram"])
        overrides = build_overrides(args)
        assert overrides["source__mode"] == "telegram"
        assert "source__folder" not in overrides

    def test_no_source_args_yields_no_source_overrides(self) -> None:
        args = self._parse(["download"])
        overrides = build_overrides(args)
        assert "source__mode" not in overrides
        assert "source__folder" not in overrides

    def test_source_args_do_not_disturb_other_overrides(self) -> None:
        args = self._parse(
            ["run", "--group", "@g", "--backend", "local", "--folder", "/f"]
        )
        overrides = build_overrides(args)
        assert overrides["telegram__group"] == "@g"
        assert overrides["transcribe__backend"] == "local"
        assert overrides["source__mode"] == "folder"


# ---------------------------------------------------------------------------
# acquire_audio — source dispatch
# ---------------------------------------------------------------------------

class TestAcquireAudioDispatch:
    def test_folder_mode_uses_folder_importer(
        self, folder_config, tmp_manifest: Manifest, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stats = acquire_audio(folder_config, tmp_manifest)

        assert isinstance(stats, ImportStats)
        assert stats.imported == 2
        out = capsys.readouterr().out
        assert "Importing audio from folder" in out

    def test_telegram_mode_uses_downloader(
        self, sample_config, tmp_manifest: Manifest, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_stats = SimpleNamespace(summary=lambda: "tg-done")

        class _FakeDownloader:
            def __init__(self, config, manifest):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def run(self):
                return fake_stats

        with patch("src.telegram.TelegramDownloader", _FakeDownloader):
            result = acquire_audio(sample_config, tmp_manifest)

        assert result is fake_stats
        out = capsys.readouterr().out
        assert "Downloading audio from Telegram" in out


# ---------------------------------------------------------------------------
# cmd_download — folder mode end-to-end
# ---------------------------------------------------------------------------

class TestCmdDownloadFolder:
    def test_download_imports_from_folder(
        self, folder_config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cmd_download(MagicMock(), folder_config)

        out = capsys.readouterr().out
        assert "Import complete" in out
        copied = {p.name for p in folder_config.download.audio_dir.iterdir()}
        assert copied == {"Sermon One.mp3", "Deep Truth.flac"}

        # The manifest written to disk reflects the import.
        manifest = Manifest(folder_config.download.manifest_file)
        assert len(manifest.all_entries()) == 2

    def test_download_reports_failure_and_exits(self, folder_config) -> None:
        """A broken source folder surfaces as a non-zero exit."""
        folder_config.source.folder = folder_config.source.folder / "missing"
        with pytest.raises(SystemExit):
            cmd_download(MagicMock(), folder_config)


# ---------------------------------------------------------------------------
# cmd_status — shows the active source
# ---------------------------------------------------------------------------

class TestCmdStatusSource:
    def test_status_shows_folder_source(
        self, folder_config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cmd_status(MagicMock(), folder_config)
        out = capsys.readouterr().out
        assert "Source:" in out
        assert "folder" in out
        assert str(folder_config.source.folder) in out

    def test_status_shows_telegram_source(
        self, sample_config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cmd_status(MagicMock(), sample_config)
        out = capsys.readouterr().out
        assert "Source:" in out
        assert "telegram" in out
