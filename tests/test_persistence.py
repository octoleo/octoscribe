"""
tests/test_persistence.py — Tests for src/persistence.py.

Covers the shared atomic-write helpers and the PeriodicSaver, both of which
were extracted to remove duplication across the manifest, transcriber, and
audio sources.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.persistence import (
    DEFAULT_SAVE_INTERVAL,
    PeriodicSaver,
    atomic_write_bytes,
    atomic_write_text,
)


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_write_text_creates_file_with_content(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        atomic_write_text(target, "hello world")
        assert target.read_text(encoding="utf-8") == "hello world"

    def test_write_bytes_creates_file_with_content(self, tmp_path: Path) -> None:
        target = tmp_path / "out.bin"
        atomic_write_bytes(target, b"\x00\x01\x02")
        assert target.read_bytes() == b"\x00\x01\x02"

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "deep" / "out.txt"
        atomic_write_text(target, "x")
        assert target.exists()

    def test_no_tmp_file_remains_after_success(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        atomic_write_text(target, "data")
        assert list(tmp_path.iterdir()) == [target]

    def test_overwrites_existing_file_atomically(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        target.write_text("old")
        atomic_write_text(target, "new")
        assert target.read_text() == "new"

    def test_tmp_file_cleaned_up_on_failure(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        with patch("src.persistence.os.replace", side_effect=OSError("boom")):
            with pytest.raises(OSError, match="boom"):
                atomic_write_text(target, "data")
        # The temporary file must not be left behind, and no target written.
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# PeriodicSaver
# ---------------------------------------------------------------------------

class TestPeriodicSaver:
    def test_saves_on_interval_boundary(self) -> None:
        target = MagicMock()
        saver = PeriodicSaver(target, interval=3)

        assert saver.tick() is False  # 1
        assert saver.tick() is False  # 2
        assert saver.tick() is True   # 3 -> save
        target.save.assert_called_once()

    def test_count_tracks_ticks(self) -> None:
        saver = PeriodicSaver(MagicMock(), interval=5)
        for _ in range(7):
            saver.tick()
        assert saver.count == 7

    def test_multiple_intervals_trigger_multiple_saves(self) -> None:
        target = MagicMock()
        saver = PeriodicSaver(target, interval=2)
        for _ in range(6):
            saver.tick()
        assert target.save.call_count == 3

    def test_default_interval(self) -> None:
        saver = PeriodicSaver(MagicMock())
        assert saver._interval == DEFAULT_SAVE_INTERVAL

    def test_invalid_interval_raises(self) -> None:
        with pytest.raises(ValueError):
            PeriodicSaver(MagicMock(), interval=0)
