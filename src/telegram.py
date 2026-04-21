"""
telegram.py — Telegram audio downloader for OctoScribe.

Downloads audio files from a Telegram group, deduplicates by SHA-256 content
hash, resumes from manifest, and records all metadata into the manifest.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from telethon import TelegramClient
from telethon.tl.types import (
    DocumentAttributeAudio,
    DocumentAttributeFilename,
    MessageMediaDocument,
)

from src.config import Config
from src.manifest import Manifest

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS: tuple[str, ...] = (
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus",
)

AUDIO_MIMES: frozenset[str] = frozenset({
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/flac",
    "audio/x-flac",
    "audio/mp4",
    "audio/x-m4a",
    "audio/aac",
    "audio/ogg",
    "application/ogg",
    "audio/opus",
})

MIME_TO_EXT: dict[str, str] = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/ogg": ".ogg",
    "application/ogg": ".ogg",
    "audio/opus": ".opus",
}

_MANIFEST_SAVE_INTERVAL = 10

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioMetadata:
    """Value object for audio file metadata extracted from a Telegram message."""

    msg_id: int
    title: str | None = None
    performer: str | None = None
    duration: int | None = None       # seconds
    extension: str = ".ogg"
    original_filename: str | None = None
    date: str | None = None           # "YYYY-MM-DD"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class DownloadStats:
    """Accumulates counters across a single downloader run."""

    downloaded: int = 0
    skipped: int = 0
    duplicate: int = 0
    failed: int = 0

    def summary(self) -> str:
        total = self.downloaded + self.skipped + self.duplicate + self.failed
        return (
            f"Run complete — total={total} "
            f"downloaded={self.downloaded} "
            f"skipped={self.skipped} "
            f"duplicate={self.duplicate} "
            f"failed={self.failed}"
        )


# ---------------------------------------------------------------------------
# Pure helper functions (module-level so tests can import them directly)
# ---------------------------------------------------------------------------


def sanitize_filename(name: str) -> str:
    """
    Remove characters that are invalid in filenames, trim, and limit length.

    Removes ``<>:"/\\|?*`` and ASCII control characters, strips leading/
    trailing whitespace and dots, then truncates to 200 characters.
    """
    # Remove invalid path characters
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    # Remove control characters (U+0000–U+001F and U+007F)
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    # Strip leading/trailing whitespace and dots
    name = name.strip(" .")
    # Enforce maximum length
    if len(name) > 200:
        name = name[:200]
    return name


def format_duration(seconds: int | None) -> str | None:
    """
    Convert a duration in seconds to a human-readable string.

    Returns ``"M:SS"`` or ``"H:MM:SS"``, or ``None`` if *seconds* is ``None``
    or falsy.
    """
    if not seconds:
        return None
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def get_audio_metadata(msg: object) -> AudioMetadata:
    """
    Extract audio metadata from a Telethon message object.

    Inspects ``DocumentAttributeAudio`` and ``DocumentAttributeFilename``
    attributes as well as the document MIME type.
    """
    # Defaults
    title: str | None = None
    performer: str | None = None
    duration: int | None = None
    extension: str = ".ogg"
    original_filename: str | None = None
    date: str | None = None

    # Date
    raw_date = getattr(msg, "date", None)
    if raw_date is not None:
        date = raw_date.strftime("%Y-%m-%d")

    msg_id: int = getattr(msg, "id", 0)

    media = getattr(msg, "media", None)
    if not isinstance(media, MessageMediaDocument):
        return AudioMetadata(
            msg_id=msg_id,
            title=title,
            performer=performer,
            duration=duration,
            extension=extension,
            original_filename=original_filename,
            date=date,
        )

    document = getattr(media, "document", None)
    if document is None:
        return AudioMetadata(
            msg_id=msg_id,
            title=title,
            performer=performer,
            duration=duration,
            extension=extension,
            original_filename=original_filename,
            date=date,
        )

    # Derive extension from MIME type
    mime = (getattr(document, "mime_type", None) or "").lower()
    extension = MIME_TO_EXT.get(mime, ".ogg")

    # Walk document attributes
    for attr in getattr(document, "attributes", []):
        if isinstance(attr, DocumentAttributeAudio):
            title = getattr(attr, "title", None) or None
            performer = getattr(attr, "performer", None) or None
            raw_dur = getattr(attr, "duration", None)
            duration = int(raw_dur) if raw_dur is not None else None
        elif isinstance(attr, DocumentAttributeFilename):
            fname = getattr(attr, "file_name", None)
            if fname:
                original_filename = fname
                # If we still have the default extension, try to infer from filename
                if extension == ".ogg":
                    _, file_ext = os.path.splitext(fname)
                    if file_ext.lower() in AUDIO_EXTENSIONS:
                        extension = file_ext.lower()

    return AudioMetadata(
        msg_id=msg_id,
        title=title,
        performer=performer,
        duration=duration,
        extension=extension,
        original_filename=original_filename,
        date=date,
    )


def is_audio(msg: object) -> bool:
    """
    Return ``True`` when *msg* carries an audio document.

    Checks for ``DocumentAttributeAudio``, known audio filename extensions,
    and audio MIME types (including ``application/ogg``).
    """
    media = getattr(msg, "media", None)
    if media is None:
        return False
    if not isinstance(media, MessageMediaDocument):
        return False
    document = getattr(media, "document", None)
    if document is None:
        return False

    for attr in getattr(document, "attributes", []):
        if isinstance(attr, DocumentAttributeAudio):
            return True
        if isinstance(attr, DocumentAttributeFilename):
            fname = getattr(attr, "file_name", "") or ""
            if fname.lower().endswith(AUDIO_EXTENSIONS):
                return True

    mime = (getattr(document, "mime_type", None) or "").lower()
    if mime in AUDIO_MIMES or mime.startswith("audio/"):
        return True

    return False


def build_filename(metadata: AudioMetadata) -> str:
    """
    Derive a sanitized filename from *metadata*.

    Priority:
    1. ``title`` — used as-is (sanitized).
    2. ``original_filename`` stem — when the filename is not ``"record.ogg"``.
    3. ``audio_{date}_{msg_id}`` — last resort.

    The appropriate extension from *metadata* is always appended.
    """
    ext = metadata.extension

    if metadata.title:
        base = metadata.title
    elif metadata.original_filename and metadata.original_filename != "record.ogg":
        base, _ = os.path.splitext(metadata.original_filename)
    else:
        base = f"audio_{metadata.date}_{metadata.msg_id}"

    base = sanitize_filename(base)
    return f"{base}{ext}"


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _unique_filepath(directory: Path, filename: str, msg_id: int) -> Path:
    """
    Return a path that does not yet exist in *directory*.

    If *filename* is already taken, the *msg_id* is appended to the stem.
    """
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, ext = os.path.splitext(filename)
    return directory / f"{stem}_{msg_id}{ext}"


# ---------------------------------------------------------------------------
# Main downloader class
# ---------------------------------------------------------------------------


class TelegramDownloader:
    """
    Downloads audio files from a Telegram group.

    Uses the Telethon library. Session files are stored in
    ``config.telegram.session_dir``. Supports deduplication by SHA-256
    content hash and resumption from the manifest.

    Usage::

        async with TelegramDownloader(config, manifest) as dl:
            stats = await dl.run()
            print(stats.summary())
    """

    @staticmethod
    def _restore_session_from_env(session_dir: Path) -> bool:
        """
        Restore a Telegram session file from the TELEGRAM_SESSION_B64 env var.

        If the environment variable is set, its base64 content is decoded and
        written to ``session_dir / "octoscribe.session"``.  This enables
        non-interactive authentication in CI/CD environments (GitHub Actions)
        where interactive phone-code login is not possible.

        Returns True if the session was restored, False if the env var is not set.
        Called automatically by __init__ before TelegramClient is created.
        """
        b64 = os.environ.get("TELEGRAM_SESSION_B64", "").strip()
        if not b64:
            return False
        try:
            session_bytes = base64.b64decode(b64)
        except Exception as exc:
            log.warning("TELEGRAM_SESSION_B64 is set but could not be decoded: %s", exc)
            return False
        session_dir.mkdir(parents=True, exist_ok=True)
        session_file = session_dir / "octoscribe.session"
        session_file.write_bytes(session_bytes)
        log.info("Telegram session restored from TELEGRAM_SESSION_B64 (%d bytes)", len(session_bytes))
        return True

    def __init__(self, config: Config, manifest: Manifest) -> None:
        self._config = config
        self._manifest = manifest
        self._restore_session_from_env(config.telegram.session_dir)
        session_path = str(config.telegram.session_dir / "octoscribe")
        self._client = TelegramClient(
            session_path,
            config.telegram.api_id,
            config.telegram.api_hash,
        )
        self._semaphore = asyncio.Semaphore(config.download.workers)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "TelegramDownloader":
        await self.connect()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Authenticate and connect the Telethon client."""
        self._config.telegram.session_dir.mkdir(parents=True, exist_ok=True)
        await self._client.start(phone=self._config.telegram.phone)
        log.info("Telegram client connected")

    async def disconnect(self) -> None:
        """Gracefully disconnect the Telethon client."""
        await self._client.disconnect()
        log.info("Telegram client disconnected")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> DownloadStats:
        """
        Scan all messages in the configured group and download new audio files.

        Respects ``config.download.resume`` (skip if already downloaded) and
        ``config.download.deduplicate`` (remove duplicate files by hash).
        Saves the manifest every :data:`_MANIFEST_SAVE_INTERVAL` downloads
        and once more at the end.

        Returns a :class:`DownloadStats` summary.
        """
        stats = DownloadStats()
        cfg = self._config

        # Resolve the group entity
        group_raw = cfg.telegram.group.strip()
        if group_raw.lstrip("-").isdigit():
            entity = await self._client.get_entity(int(group_raw))
        else:
            entity = await self._client.get_entity(group_raw)

        log.info("Resolved group: %s", getattr(entity, "title", entity))

        # Collect all audio messages first
        audio_messages: list[object] = []
        offset_id = 0
        total_scanned = 0

        while True:
            batch = await self._client.get_messages(entity, limit=100, offset_id=offset_id)
            if not batch:
                break
            total_scanned += len(batch)
            for msg in batch:
                if is_audio(msg):
                    audio_messages.append(msg)
            offset_id = batch[-1].id
            await asyncio.sleep(0.05)

        log.info(
            "Scan complete: %d messages scanned, %d audio files found",
            total_scanned,
            len(audio_messages),
        )

        if not audio_messages:
            log.info("No audio files found in group.")
            return stats

        # Ensure audio output directory exists
        cfg.download.audio_dir.mkdir(parents=True, exist_ok=True)

        # Build and run download tasks concurrently
        save_counter = 0

        async def _process(msg: object) -> None:
            nonlocal save_counter
            result = await self._download_one(msg)
            match result:
                case "downloaded":
                    stats.downloaded += 1
                case "skipped":
                    stats.skipped += 1
                case "duplicate":
                    stats.duplicate += 1
                case "failed":
                    stats.failed += 1

            if result == "downloaded":
                save_counter += 1
                if save_counter % _MANIFEST_SAVE_INTERVAL == 0:
                    self._manifest.save()
                    log.debug("Manifest saved (periodic, %d downloaded)", save_counter)

        tasks = [_process(msg) for msg in audio_messages]
        await asyncio.gather(*tasks)

        # Final manifest save
        self._manifest.save()
        log.info(stats.summary())
        return stats

    # ------------------------------------------------------------------
    # Per-message download
    # ------------------------------------------------------------------

    async def _download_one(self, msg: object) -> str:
        """
        Download a single audio message.

        Returns one of: ``"downloaded"``, ``"skipped"``, ``"duplicate"``,
        ``"failed"``.
        """
        async with self._semaphore:
            msg_id: int = getattr(msg, "id", 0)
            cfg = self._config

            # Resume: skip if already recorded in manifest AND file exists
            if cfg.download.resume and self._manifest.is_downloaded(msg_id):
                entry = self._manifest.get_entry(msg_id)
                if entry:
                    filename = entry.get("filename")
                    if filename and (cfg.download.audio_dir / filename).exists():
                        log.debug("Skipping msg %d (already downloaded)", msg_id)
                        return "skipped"

            metadata = get_audio_metadata(msg)
            target_filename = build_filename(metadata)
            target_path = _unique_filepath(cfg.download.audio_dir, target_filename, msg_id)
            actual_filename = target_path.name

            try:
                downloaded_path = await self._client.download_media(msg, file=str(target_path))
            except Exception as exc:
                error_str = str(exc)
                log.warning("Download failed for msg %d: %s", msg_id, error_str)
                self._manifest.mark_failed(msg_id, "download", error_str)
                return "failed"

            if not downloaded_path:
                error_str = "download_media returned None"
                log.warning("No file returned for msg %d", msg_id)
                self._manifest.mark_failed(msg_id, "download", error_str)
                return "failed"

            downloaded_path = Path(downloaded_path)

            # Deduplication by SHA-256
            sha256_hex: str | None = None
            if cfg.download.deduplicate:
                sha256_hex = _sha256_file(downloaded_path)
                existing_entries = self._manifest.all_entries()
                for existing_entry in existing_entries.values():
                    if (
                        isinstance(existing_entry, dict)
                        and existing_entry.get("hash") == sha256_hex
                        and existing_entry.get("downloaded")
                    ):
                        downloaded_path.unlink(missing_ok=True)
                        log.debug(
                            "Duplicate detected for msg %d (hash=%s); file removed",
                            msg_id,
                            sha256_hex,
                        )
                        return "duplicate"
            else:
                sha256_hex = _sha256_file(downloaded_path)

            # Record in manifest
            manifest_metadata: dict[str, object] = {
                "filename": actual_filename,
                "title": metadata.title,
                "performer": metadata.performer,
                "date": metadata.date,
                "duration": metadata.duration,
                "duration_formatted": format_duration(metadata.duration),
                "extension": metadata.extension,
                "hash": sha256_hex,
                "original_filename": metadata.original_filename,
            }
            self._manifest.mark_downloaded(msg_id, manifest_metadata)
            log.debug("Downloaded msg %d → %s", msg_id, actual_filename)
            return "downloaded"
