"""
folder.py — Local folder audio importer for OctoScribe.

An alternative audio source to the Telegram downloader: instead of pulling
audio from a Telegram group, :class:`FolderImporter` scans a local folder for
audio files and registers them in the manifest so the existing transcription
pipeline can process them.  It requires no Telegram credentials and no
Telegram libraries.

Imported files are copied into ``config.download.audio_dir`` (the configured
audio workspace) so the calling workflow can preserve and process them
exactly like Telegram-sourced audio.  Each entry is keyed by its SHA-256
content hash, which gives natural deduplication and resume support.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.audio import (
    AUDIO_EXTENSIONS,
    format_duration,
    sanitize_filename,
    sha256_file,
    unique_filepath,
)
from src.config import Config
from src.manifest import Manifest
from src.persistence import PeriodicSaver

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class ImportStats:
    """Accumulates counters across a single folder-import run."""

    imported: int = 0
    skipped: int = 0
    duplicate: int = 0
    failed: int = 0

    def summary(self) -> str:
        total = self.imported + self.skipped + self.duplicate + self.failed
        return (
            f"Import complete — total={total} "
            f"imported={self.imported} "
            f"skipped={self.skipped} "
            f"duplicate={self.duplicate} "
            f"failed={self.failed}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe_duration(path: Path) -> int | None:
    """
    Best-effort audio duration in whole seconds via mutagen.

    Returns ``None`` on any failure (unreadable file, unknown format, missing
    library) so that duration is purely informational and never blocks an
    import.
    """
    try:
        from mutagen import File as MutagenFile  # local import: optional dependency

        audio = MutagenFile(str(path))
        info = getattr(audio, "info", None) if audio is not None else None
        length = getattr(info, "length", None) if info is not None else None
        if length:
            return int(round(length))
    except Exception:  # pragma: no cover - defensive, never fatal
        return None
    return None


# ---------------------------------------------------------------------------
# Main importer class
# ---------------------------------------------------------------------------

class FolderImporter:
    """
    Imports audio files from a local folder into the data repo and manifest.

    Usage::

        importer = FolderImporter(config, manifest)
        stats = importer.run()
        print(stats.summary())
    """

    def __init__(self, config: Config, manifest: Manifest) -> None:
        self._config = config
        self._manifest = manifest
        # Content hashes imported during this run, used for in-run dedup.
        self._seen_hashes: set[str] = set()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> ImportStats:
        """
        Scan the configured folder and import new audio files.

        Respects ``config.download.resume`` (skip files already imported in a
        previous run) and ``config.download.deduplicate`` (skip files whose
        content matches one already imported this run).  A
        :class:`~src.persistence.PeriodicSaver` flushes the manifest at a fixed
        cadence, and a final authoritative save runs once more at the end.

        Returns an :class:`ImportStats` summary.
        """
        stats = ImportStats()
        cfg = self._config

        folder = cfg.source.folder
        if folder is None:
            raise ValueError(
                "No source folder configured. Set [source] folder in the INI "
                "file or pass --folder PATH."
            )
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Source folder does not exist: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Source folder is not a directory: {folder}")

        files = self._gather_files(folder, recursive=cfg.source.recursive)
        log.info("Found %d audio file(s) in %s", len(files), folder)

        if not files:
            log.info("No audio files found in folder %s", folder)
            return stats

        # Ensure audio output directory exists.
        cfg.download.audio_dir.mkdir(parents=True, exist_ok=True)

        saver = PeriodicSaver(self._manifest)
        for path in files:
            result = self._import_one(path)
            match result:
                case "imported":
                    stats.imported += 1
                    if saver.tick():
                        log.debug("Manifest saved (periodic, %d imported)", saver.count)
                case "skipped":
                    stats.skipped += 1
                case "duplicate":
                    stats.duplicate += 1
                case "failed":
                    stats.failed += 1

        # Final manifest save.
        self._manifest.save()
        log.info(stats.summary())
        return stats

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _gather_files(folder: Path, recursive: bool) -> list[Path]:
        """Return a sorted list of audio files in *folder*."""
        root = folder.resolve()
        if recursive:
            candidates = (p for p in folder.rglob("*") if p.is_file())
        else:
            candidates = (p for p in folder.iterdir() if p.is_file())
        safe: list[Path] = []
        for path in candidates:
            if path.is_symlink() or path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            try:
                path.resolve(strict=True).relative_to(root)
            except (OSError, ValueError):
                continue
            safe.append(path)
        return sorted(safe)

    # ------------------------------------------------------------------
    # Per-file import
    # ------------------------------------------------------------------

    def _import_one(self, path: Path) -> str:
        """
        Import a single audio file.

        Returns one of: ``"imported"``, ``"skipped"``, ``"duplicate"``,
        ``"failed"``.
        """
        cfg = self._config

        try:
            sha256_hex = sha256_file(path)
        except OSError as exc:
            log.warning("Could not read %s: %s", path, exc)
            self._manifest.mark_failed(str(path), "import", str(exc))
            return "failed"

        # Use the content hash as the manifest key: identical content maps to a
        # single entry, giving deduplication and resume for free.
        key = sha256_hex

        # In-run deduplication: another file with identical content was already
        # imported during this run.
        if cfg.download.deduplicate and sha256_hex in self._seen_hashes:
            log.info(
                "Skipping duplicate folder audio: source=%s sha256=%s",
                path,
                sha256_hex,
            )
            return "duplicate"

        # Resume: this content was imported in a previous run and the copied
        # file is still present.
        if cfg.download.resume and self._manifest.is_downloaded(key):
            entry = self._manifest.get_entry(key)
            if entry:
                filename = entry.get("filename")
                if filename and (cfg.download.audio_dir / filename).exists():
                    log.info(
                        "Skipping previously imported audio: source=%s output=%s sha256=%s",
                        path,
                        cfg.download.audio_dir / filename,
                        sha256_hex,
                    )
                    self._seen_hashes.add(sha256_hex)
                    return "skipped"

        ext = path.suffix.lower()
        base = sanitize_filename(path.stem) or sha256_hex[:12]
        target_filename = f"{base}{ext}"
        target_path = unique_filepath(cfg.download.audio_dir, target_filename, sha256_hex[:8])
        actual_filename = target_path.name

        try:
            shutil.copy2(path, target_path)
            copied_hash = sha256_file(target_path)
            if copied_hash != sha256_hex:
                target_path.unlink(missing_ok=True)
                raise OSError(
                    "copied audio failed SHA-256 verification; source evidence "
                    "was not registered"
                )
        except OSError as exc:
            log.warning("Failed to copy %s: %s", path, exc)
            self._manifest.mark_failed(key, "import", str(exc))
            return "failed"

        duration = _probe_duration(target_path)
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            date_str: str | None = mtime.strftime("%Y-%m-%d")
        except OSError:
            date_str = None

        manifest_metadata: dict[str, object] = {
            "filename": actual_filename,
            "title": None,
            "performer": None,
            "date": date_str,
            "duration": duration,
            "duration_formatted": format_duration(duration),
            "extension": ext,
            "hash": sha256_hex,
            "original_filename": path.name,
            "source": "folder",
            "source_path": str(path),
        }
        self._manifest.mark_downloaded(key, manifest_metadata)
        self._seen_hashes.add(sha256_hex)
        log.info(
            "Imported folder audio: source=%s output=%s sha256=%s bytes=%d",
            path,
            target_path,
            sha256_hex,
            target_path.stat().st_size,
        )
        return "imported"
