"""
tests/test_folder.py — Pytest suite for src/folder.py (FolderImporter).

These tests use only the real filesystem (via tmp_path-based fixtures) and a
real Manifest; no Telegram credentials or libraries are required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.config import Config, SourceConfig
from src.folder import FolderImporter, ImportStats, _probe_duration
from src.manifest import Manifest


# ---------------------------------------------------------------------------
# ImportStats
# ---------------------------------------------------------------------------

class TestImportStats:
    def test_summary_is_non_empty_string(self) -> None:
        stats = ImportStats(imported=2, skipped=1, duplicate=1, failed=0)
        summary = stats.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_includes_counts(self) -> None:
        stats = ImportStats(imported=5, skipped=2, duplicate=1, failed=3)
        s = stats.summary()
        assert "imported=5" in s
        assert "skipped=2" in s
        assert "duplicate=1" in s
        assert "failed=3" in s
        assert "total=11" in s


# ---------------------------------------------------------------------------
# _gather_files
# ---------------------------------------------------------------------------

class TestGatherFiles:
    def test_filters_by_extension(self, sample_audio_folder: Path) -> None:
        files = FolderImporter._gather_files(sample_audio_folder, recursive=True)
        names = {p.name for p in files}
        assert names == {"Sermon One.mp3", "Deep Truth.flac"}
        assert "notes.txt" not in names

    def test_does_not_follow_symlinked_audio_outside_source(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()
        outside = tmp_path / "private.mp3"
        outside.write_bytes(b"PRIVATE")
        (source / "linked.mp3").symlink_to(outside)

        assert FolderImporter._gather_files(source, recursive=True) == []

    def test_recursive_includes_nested(self, sample_audio_folder: Path) -> None:
        files = FolderImporter._gather_files(sample_audio_folder, recursive=True)
        assert any(p.name == "Deep Truth.flac" for p in files)

    def test_non_recursive_excludes_nested(self, sample_audio_folder: Path) -> None:
        files = FolderImporter._gather_files(sample_audio_folder, recursive=False)
        names = {p.name for p in files}
        assert names == {"Sermon One.mp3"}
        assert "Deep Truth.flac" not in names


# ---------------------------------------------------------------------------
# FolderImporter.run — happy path
# ---------------------------------------------------------------------------

class TestFolderImporterRun:
    def test_imports_audio_into_audio_dir(
        self, folder_config: Config, tmp_manifest: Manifest
    ) -> None:
        stats = FolderImporter(folder_config, tmp_manifest).run()

        assert stats.imported == 2
        assert stats.skipped == 0
        assert stats.failed == 0

        copied = {p.name for p in folder_config.download.audio_dir.iterdir()}
        assert copied == {"Sermon One.mp3", "Deep Truth.flac"}

    def test_records_entries_in_manifest(
        self, folder_config: Config, tmp_manifest: Manifest
    ) -> None:
        FolderImporter(folder_config, tmp_manifest).run()

        entries = tmp_manifest.all_entries()
        assert len(entries) == 2
        for entry in entries.values():
            assert entry["downloaded"] is True
            assert entry["source"] == "folder"
            assert entry["hash"]
            assert entry["filename"] in {"Sermon One.mp3", "Deep Truth.flac"}
            assert "source_path" in entry

    def test_manifest_key_is_content_hash(
        self, folder_config: Config, tmp_manifest: Manifest
    ) -> None:
        FolderImporter(folder_config, tmp_manifest).run()
        expected_hash = hashlib.sha256(b"sermon one audio bytes").hexdigest()
        entry = tmp_manifest.get_entry(expected_hash)
        assert entry is not None
        assert entry["filename"] == "Sermon One.mp3"

    def test_imported_files_are_pending_transcription(
        self, folder_config: Config, tmp_manifest: Manifest
    ) -> None:
        FolderImporter(folder_config, tmp_manifest).run()
        pending = tmp_manifest.pending_transcription()
        assert len(pending) == 2

    def test_source_files_are_not_modified(
        self, folder_config: Config, tmp_manifest: Manifest, sample_audio_folder: Path
    ) -> None:
        """The importer copies, it does not move — originals stay put."""
        FolderImporter(folder_config, tmp_manifest).run()
        assert (sample_audio_folder / "Sermon One.mp3").exists()
        assert (sample_audio_folder / "nested" / "Deep Truth.flac").exists()


# ---------------------------------------------------------------------------
# Deduplication and resume
# ---------------------------------------------------------------------------

class TestDedupAndResume:
    def test_duplicate_content_in_same_run(
        self, folder_config: Config, tmp_manifest: Manifest, sample_audio_folder: Path
    ) -> None:
        # Add a second file with identical content to an existing one.
        (sample_audio_folder / "copy of sermon one.mp3").write_bytes(
            b"sermon one audio bytes"
        )
        stats = FolderImporter(folder_config, tmp_manifest).run()

        assert stats.imported == 2  # Sermon One + Deep Truth
        assert stats.duplicate == 1  # the identical copy
        # Only two distinct files land in the audio dir.
        assert len(list(folder_config.download.audio_dir.iterdir())) == 2

    def test_resume_skips_already_imported(
        self, folder_config: Config, tmp_manifest: Manifest
    ) -> None:
        FolderImporter(folder_config, tmp_manifest).run()

        # Second run over the same folder: everything already imported.
        stats = FolderImporter(folder_config, tmp_manifest).run()
        assert stats.imported == 0
        assert stats.skipped == 2

    def test_resume_false_reimports(
        self, folder_config: Config, tmp_manifest: Manifest
    ) -> None:
        FolderImporter(folder_config, tmp_manifest).run()
        folder_config.download.resume = False

        stats = FolderImporter(folder_config, tmp_manifest).run()
        # With resume disabled the files are imported again (overwriting the
        # same manifest keys); none are skipped.
        assert stats.skipped == 0
        assert stats.imported == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestFolderImporterErrors:
    def test_missing_folder_raises(
        self, folder_config: Config, tmp_manifest: Manifest, tmp_path: Path
    ) -> None:
        folder_config.source = SourceConfig(
            mode="folder", folder=tmp_path / "does_not_exist", recursive=True
        )
        with pytest.raises(FileNotFoundError):
            FolderImporter(folder_config, tmp_manifest).run()

    def test_none_folder_raises(
        self, folder_config: Config, tmp_manifest: Manifest
    ) -> None:
        folder_config.source = SourceConfig(mode="folder", folder=None, recursive=True)
        with pytest.raises(ValueError):
            FolderImporter(folder_config, tmp_manifest).run()

    def test_file_instead_of_folder_raises(
        self, folder_config: Config, tmp_manifest: Manifest, tmp_path: Path
    ) -> None:
        a_file = tmp_path / "a_file.mp3"
        a_file.write_bytes(b"x")
        folder_config.source = SourceConfig(mode="folder", folder=a_file, recursive=True)
        with pytest.raises(NotADirectoryError):
            FolderImporter(folder_config, tmp_manifest).run()

    def test_empty_folder_returns_zero_stats(
        self, folder_config: Config, tmp_manifest: Manifest, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        folder_config.source = SourceConfig(mode="folder", folder=empty, recursive=True)
        stats = FolderImporter(folder_config, tmp_manifest).run()
        assert stats.imported == 0
        assert stats.skipped == 0

    def test_unreadable_file_counts_as_failed(
        self, folder_config: Config, tmp_manifest: Manifest
    ) -> None:
        """A hashing error marks the file failed without aborting the run."""
        from unittest.mock import patch

        def _raise(_path):
            raise OSError("cannot read")

        with patch("src.folder.sha256_file", side_effect=_raise):
            stats = FolderImporter(folder_config, tmp_manifest).run()

        assert stats.failed == 2
        assert stats.imported == 0

    def test_copy_failure_counts_as_failed(
        self, folder_config: Config, tmp_manifest: Manifest
    ) -> None:
        """A copy error marks the file failed and records it in the manifest."""
        from unittest.mock import patch

        with patch("src.folder.shutil.copy2", side_effect=OSError("disk full")):
            stats = FolderImporter(folder_config, tmp_manifest).run()

        assert stats.failed == 2
        assert stats.imported == 0
        # Failure was recorded against the content-hash key.
        entries = tmp_manifest.all_entries()
        assert all(e.get("failed_stage") == "import" for e in entries.values())

    def test_copy_hash_mismatch_is_rejected_and_removed(
        self, folder_config: Config, tmp_manifest: Manifest
    ) -> None:
        """A corrupted copy never becomes immutable source evidence."""
        from unittest.mock import patch

        from src.audio import sha256_file as real_sha256_file

        def _hash(path: Path) -> str:
            path = Path(path)
            if path.parent == folder_config.download.audio_dir:
                return "0" * 64
            return real_sha256_file(path)

        with patch("src.folder.sha256_file", side_effect=_hash):
            stats = FolderImporter(folder_config, tmp_manifest).run()

        assert stats.failed == 2
        assert stats.imported == 0
        assert list(folder_config.download.audio_dir.iterdir()) == []
        assert all(
            entry.get("failed_stage") == "import"
            for entry in tmp_manifest.all_entries().values()
        )


# ---------------------------------------------------------------------------
# _probe_duration
# ---------------------------------------------------------------------------

class TestProbeDuration:
    def test_returns_none_for_non_audio(self, tmp_path: Path) -> None:
        f = tmp_path / "fake.mp3"
        f.write_bytes(b"this is not real audio")
        # Best-effort: must never raise, returns None on unreadable formats.
        assert _probe_duration(f) is None
