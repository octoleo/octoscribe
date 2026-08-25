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
import os
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


def _probe_audio_metadata(path: Path) -> dict[str, object]:
    """Return safe embedded tags and technical properties via mutagen.

    Metadata is best-effort and informational: an unreadable or unusual tag
    must never prevent immutable source bytes from being imported. Binary tag
    values (for example embedded cover art) are deliberately excluded from the
    JSON manifest.
    """
    try:
        from mutagen import File as MutagenFile

        audio = MutagenFile(str(path), easy=True)
        if audio is None:
            return {}

        embedded: dict[str, str | list[str]] = {}
        tags = getattr(audio, "tags", None)
        if tags is not None:
            for raw_key, raw_value in tags.items():
                key = str(raw_key).strip().lower()
                if not key or len(key) > 100:
                    continue
                values = (
                    raw_value
                    if isinstance(raw_value, (list, tuple))
                    else (raw_value,)
                )
                rendered = [
                    str(value).strip()[:2_000]
                    for value in values
                    if isinstance(value, (str, int, float)) and str(value).strip()
                ]
                if rendered:
                    embedded[key] = rendered[0] if len(rendered) == 1 else rendered

        def first(*keys: str) -> str | None:
            for key in keys:
                value = embedded.get(key)
                if isinstance(value, list):
                    return value[0] if value else None
                if isinstance(value, str):
                    return value
            return None

        metadata: dict[str, object] = {}
        tag_fields = {
            "title": first("title"),
            "performer": first("artist", "performer", "albumartist"),
            "album": first("album"),
            "album_artist": first("albumartist"),
            "composer": first("composer"),
            "genre": first("genre"),
            "track_number": first("tracknumber"),
            "disc_number": first("discnumber"),
            "copyright": first("copyright"),
            "description": first("description", "comment"),
            "date": first("date", "originaldate", "year"),
        }
        metadata.update({key: value for key, value in tag_fields.items() if value})
        if embedded:
            metadata["embedded_tags"] = embedded

        info = getattr(audio, "info", None)
        if info is not None:
            length = getattr(info, "length", None)
            if isinstance(length, (int, float)) and length > 0:
                metadata["duration"] = int(round(length))
            technical_fields = {
                "bitrate_bps": getattr(info, "bitrate", None),
                "sample_rate_hz": getattr(info, "sample_rate", None),
                "channels": getattr(info, "channels", None),
                "bits_per_sample": getattr(info, "bits_per_sample", None),
            }
            for key, value in technical_fields.items():
                if isinstance(value, (int, float)) and value > 0:
                    metadata[key] = int(value)
            codec = getattr(info, "codec", None) or type(info).__name__
            if codec and codec != "NoneType":
                metadata["codec"] = str(codec)

        mime = getattr(audio, "mime", None)
        if isinstance(mime, (list, tuple)):
            mime_types = [str(value) for value in mime if value]
            if mime_types:
                metadata["mime_types"] = mime_types
        return metadata
    except Exception:  # pragma: no cover - metadata must never block import
        return {}


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
        # file is still present *and* still has the expected bytes.  Existence
        # alone is not sufficient evidence: a truncated or externally changed
        # copy must never be silently accepted as the immutable source.
        if cfg.download.resume and self._manifest.is_downloaded(key):
            entry = self._manifest.get_entry(key)
            if entry:
                filename = entry.get("filename")
                stored_path = self._safe_stored_path(filename)
                if stored_path is not None and (
                    stored_path.exists() or stored_path.is_symlink()
                ):
                    stored_hash: str | None = None
                    stored_error: str | None = None
                    if stored_path.is_symlink():
                        stored_error = "stored audio is a symlink"
                    elif not stored_path.is_file():
                        stored_error = "stored audio is not a regular file"
                    else:
                        try:
                            stored_hash = sha256_file(stored_path)
                        except OSError as exc:
                            stored_error = f"stored audio could not be hashed: {exc}"

                    if stored_hash == sha256_hex:
                        log.info(
                            "Skipping verified previously imported audio: "
                            "source=%s output=%s sha256=%s",
                            path,
                            stored_path,
                            sha256_hex,
                        )
                        # A prior failed repair can leave acquisition failure
                        # markers on disk.  Verified bytes mean that failure is
                        # now resolved, so clear and persist it before skipping.
                        if entry.get("failed_stage") in {"download", "import"}:
                            self._manifest.mark_downloaded(
                                key,
                                {
                                    "filename": stored_path.name,
                                    "hash": sha256_hex,
                                    "size_bytes": stored_path.stat().st_size,
                                },
                            )
                            self._manifest.save()
                        self._seen_hashes.add(sha256_hex)
                        return "skipped"

                    reason = stored_error or (
                        f"stored SHA-256 is {stored_hash}, expected {sha256_hex}"
                    )
                    log.warning(
                        "Previously imported audio failed verification; "
                        "repairing from source: source=%s output=%s reason=%s",
                        path,
                        stored_path,
                        reason,
                    )
                    if self._repair_stored_copy(
                        path, stored_path, expected_sha256=sha256_hex
                    ):
                        self._manifest.mark_downloaded(
                            key,
                            {
                                "filename": stored_path.name,
                                "hash": sha256_hex,
                                "size_bytes": stored_path.stat().st_size,
                            },
                        )
                        # Persist the repaired evidence state immediately.  A
                        # crash cannot leave a stale failure marker behind.
                        self._manifest.save()
                        self._seen_hashes.add(sha256_hex)
                        log.info(
                            "Repaired folder audio: source=%s output=%s sha256=%s",
                            path,
                            stored_path,
                            sha256_hex,
                        )
                        return "imported"

                    # The recorded target itself may be unreplaceable (for
                    # example, a directory created at that path).  Continue to
                    # the normal collision-safe import below so a verified new
                    # copy can supersede it in the manifest.  If the underlying
                    # storage is genuinely unavailable, that copy attempt will
                    # record the concrete failure and a later run may retry.
                    log.warning(
                        "Could not replace invalid stored audio in place; "
                        "attempting a new verified target: source=%s output=%s",
                        path,
                        stored_path,
                    )

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

        embedded_metadata = _probe_audio_metadata(target_path)
        duration = embedded_metadata.pop("duration", None)
        if not isinstance(duration, int):
            duration = _probe_duration(target_path)
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            date_str: str | None = mtime.strftime("%Y-%m-%d")
            modified_at: str | None = mtime.strftime("%Y-%m-%dT%H:%M:%SZ")
        except OSError:
            date_str = None
            modified_at = None

        embedded_title = embedded_metadata.pop("title", None)
        embedded_performer = embedded_metadata.pop("performer", None)
        embedded_date = embedded_metadata.pop("date", None)

        manifest_metadata: dict[str, object] = {
            "filename": actual_filename,
            "title": embedded_title,
            "performer": embedded_performer,
            "date": embedded_date or date_str,
            "duration": duration,
            "duration_formatted": format_duration(duration),
            "extension": ext,
            "hash": sha256_hex,
            "original_filename": path.name,
            "size_bytes": target_path.stat().st_size,
            "source": "folder",
            "source_path": str(path),
            "source_modified_at": modified_at,
            **embedded_metadata,
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

    def _safe_stored_path(self, filename: object) -> Path | None:
        """Resolve a manifest filename to one safe basename in the audio dir."""
        if not isinstance(filename, str) or not filename:
            return None
        relative = Path(filename)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or "/" in filename
            or "\\" in filename
        ):
            log.warning("Ignoring unsafe stored folder-audio filename: %r", filename)
            return None
        return self._config.download.audio_dir / relative

    @staticmethod
    def _repair_stored_copy(
        source: Path,
        target: Path,
        *,
        expected_sha256: str,
    ) -> bool:
        """Atomically replace an invalid stored copy with verified source bytes."""
        temporary = target.with_name(f".{target.name}.{expected_sha256[:12]}.repairing")
        try:
            temporary.unlink(missing_ok=True)
            shutil.copy2(source, temporary)
            if sha256_file(temporary) != expected_sha256:
                raise OSError("replacement copy failed SHA-256 verification")
            os.replace(temporary, target)
            return True
        except OSError as exc:
            log.warning("Failed to repair stored audio %s: %s", target, exc)
            temporary.unlink(missing_ok=True)
            return False
