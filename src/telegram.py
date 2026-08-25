"""
telegram.py — Telegram audio downloader for OctoScribe.

Downloads audio files from a Telegram group, deduplicates by SHA-256 content
hash, resumes from manifest, and records all metadata into the manifest.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from telethon import TelegramClient
from telethon.tl.types import (
    DocumentAttributeAudio,
    DocumentAttributeFilename,
    MessageMediaDocument,
)

from src.audio import (
    AUDIO_EXTENSIONS,
    format_duration,
    sanitize_filename,
    sha256_file as _sha256_file,
    unique_filepath as _unique_filepath,
)
from src.config import Config
from src.manifest import Manifest
from src.persistence import PeriodicSaver
from src.telegram_client import (
    resolve_group_entity,
    restore_session_from_env,
    session_base_path,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
#
# ``sanitize_filename`` and ``format_duration`` are re-exported from
# :mod:`src.audio`; they remain importable from this module for backwards
# compatibility.
# ---------------------------------------------------------------------------


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


def _is_sha256(value: object) -> bool:
    """Return whether *value* is a canonical lower-case SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _manifest_audio_path(audio_dir: Path, filename: object) -> Path | None:
    """Resolve a Telegram manifest filename without permitting path escape."""
    if not isinstance(filename, str) or not filename:
        return None
    relative = Path(filename)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != filename:
        return None
    return audio_dir / relative


def _download_metadata(
    metadata: AudioMetadata,
    *,
    filename: str,
    sha256_hex: str,
) -> dict[str, object]:
    """Build the established Telegram manifest payload for one audio item."""
    return {
        "filename": filename,
        "title": metadata.title,
        "performer": metadata.performer,
        "date": metadata.date,
        "duration": metadata.duration,
        "duration_formatted": format_duration(metadata.duration),
        "extension": metadata.extension,
        "hash": sha256_hex,
        "original_filename": metadata.original_filename,
    }


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
        Restore a Telegram session file from the ``TELEGRAM_SESSION_B64`` env var.

        Thin compatibility shim that delegates to
        :func:`src.telegram_client.restore_session_from_env`.  Retained as a
        staticmethod because it is part of this class's established surface and
        is exercised directly by the test suite.  See the shared helper for the
        full contract.
        """
        return restore_session_from_env(session_dir)

    def __init__(self, config: Config, manifest: Manifest) -> None:
        self._config = config
        self._manifest = manifest
        # Restore a CI-provided session (if any) before the client opens the
        # session file, so non-interactive authentication just works.
        restore_session_from_env(config.telegram.session_dir)
        self._client = TelegramClient(
            session_base_path(config.telegram.session_dir),
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
        ``config.download.deduplicate`` (remove duplicate files by hash).  A
        :class:`~src.persistence.PeriodicSaver` flushes the manifest at a fixed
        cadence, with a final authoritative save at the end.

        Returns a :class:`DownloadStats` summary.
        """
        stats = DownloadStats()
        cfg = self._config

        # Resolve the group entity (username, link, or numeric chat ID).
        entity = await resolve_group_entity(self._client, cfg.telegram.group)
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
        saver = PeriodicSaver(self._manifest)

        async def _process(msg: object) -> None:
            result = await self._download_one(msg)
            match result:
                case "downloaded":
                    stats.downloaded += 1
                    if saver.tick():
                        log.debug("Manifest saved (periodic, %d downloaded)", saver.count)
                case "skipped":
                    stats.skipped += 1
                case "duplicate":
                    stats.duplicate += 1
                    if saver.tick():
                        log.debug("Manifest saved (periodic, %d acquired)", saver.count)
                case "failed":
                    stats.failed += 1

        tasks = [_process(msg) for msg in audio_messages]
        await asyncio.gather(*tasks)

        # Final manifest save
        self._manifest.save()
        log.info(stats.summary())
        return stats

    # ------------------------------------------------------------------
    # Per-message download
    # ------------------------------------------------------------------

    def _verified_duplicate(
        self,
        msg_id: int,
        sha256_hex: str,
    ) -> tuple[str, dict[str, Any], Path] | None:
        """Return an existing, on-disk duplicate only after hashing its file."""
        audio_dir = self._config.download.audio_dir
        for existing_id, existing_entry in self._manifest.all_entries().items():
            if str(existing_id) == str(msg_id) or not isinstance(existing_entry, dict):
                continue
            if not existing_entry.get("downloaded"):
                continue
            if existing_entry.get("hash") != sha256_hex:
                continue
            existing_path = _manifest_audio_path(
                audio_dir,
                existing_entry.get("filename"),
            )
            if (
                existing_path is None
                or existing_path.is_symlink()
                or not existing_path.is_file()
            ):
                continue
            try:
                if _sha256_file(existing_path) != sha256_hex:
                    log.warning(
                        "Ignoring stale duplicate candidate: message_id=%s output=%s",
                        existing_id,
                        existing_path,
                    )
                    continue
            except OSError as exc:
                log.warning(
                    "Cannot verify duplicate candidate: message_id=%s output=%s error=%s",
                    existing_id,
                    existing_path,
                    exc,
                )
                continue
            return str(existing_id), existing_entry, existing_path
        return None

    @staticmethod
    def _staging_path(audio_dir: Path, msg_id: int, extension: str) -> Path:
        """Create a private same-filesystem path for an atomic download install."""
        descriptor, raw_path = tempfile.mkstemp(
            dir=audio_dir,
            prefix=f".octoscribe-{msg_id}-",
            suffix=extension,
        )
        os.close(descriptor)
        return Path(raw_path)

    @staticmethod
    def _discard_download(path: Path, staging_path: Path) -> None:
        """Remove only downloader-owned temporary files."""
        try:
            owned_path = path.resolve(strict=False) == staging_path.resolve(strict=False)
        except OSError:
            owned_path = False
        if owned_path:
            path.unlink(missing_ok=True)
        staging_path.unlink(missing_ok=True)

    @staticmethod
    def _install_download(
        downloaded_path: Path,
        target_path: Path,
        *,
        msg_id: int,
        sha256_hex: str,
    ) -> Path | None:
        """
        Atomically install a verified download and preserve conflicting bytes.

        A mismatching pre-existing file is renamed with a non-audio
        ``.corrupt`` suffix before replacement so it is neither lost nor picked
        up by a later folder-source scan.
        """
        if downloaded_path == target_path:
            return None

        target_present = target_path.exists() or target_path.is_symlink()
        quarantine_path: Path | None = None
        if target_present:
            if target_path.is_dir() and not target_path.is_symlink():
                raise IsADirectoryError(str(target_path))

            if not target_path.is_symlink() and target_path.is_file():
                existing_hash = _sha256_file(target_path)
                if existing_hash == sha256_hex:
                    downloaded_path.unlink(missing_ok=True)
                    return None
                quarantine_tag = existing_hash[:12]
            else:
                quarantine_tag = "unsafe-path"

            quarantine_name = (
                f"{target_path.name}.integrity-mismatch-{quarantine_tag}.corrupt"
            )
            quarantine_path = _unique_filepath(
                target_path.parent,
                quarantine_name,
                msg_id,
            )
            target_path.replace(quarantine_path)

        try:
            downloaded_path.replace(target_path)
        except OSError:
            # Best-effort rollback: the prior bytes remain authoritative until
            # the verified replacement has been installed successfully.
            if (
                quarantine_path is not None
                and quarantine_path.exists()
                and not target_path.exists()
            ):
                quarantine_path.replace(target_path)
            raise
        return quarantine_path

    async def _download_one(self, msg: object) -> str:
        """
        Download a single audio message.

        Returns one of: ``"downloaded"``, ``"skipped"``, ``"duplicate"``,
        ``"failed"``.
        """
        async with self._semaphore:
            msg_id: int = getattr(msg, "id", 0)
            cfg = self._config
            resume_entry: dict[str, Any] | None = None
            resume_target: Path | None = None
            expected_hash: str | None = None

            # Resume only after the on-disk bytes match the immutable hash in
            # the manifest.  A mere filename hit is not proof of integrity.
            if cfg.download.resume and self._manifest.is_downloaded(msg_id):
                entry = self._manifest.get_entry(msg_id)
                if entry:
                    resume_entry = entry
                    resume_target = _manifest_audio_path(
                        cfg.download.audio_dir,
                        entry.get("filename"),
                    )
                    recorded_hash = entry.get("hash")
                    expected_hash = recorded_hash if _is_sha256(recorded_hash) else None

                    if (
                        resume_target is not None
                        and expected_hash is not None
                        and not resume_target.is_symlink()
                        and resume_target.is_file()
                    ):
                        try:
                            actual_hash = _sha256_file(resume_target)
                        except OSError as exc:
                            log.warning(
                                "Cannot verify previously downloaded Telegram audio; "
                                "reacquiring: message_id=%d output=%s error=%s",
                                msg_id,
                                resume_target,
                                exc,
                            )
                        else:
                            if actual_hash == expected_hash:
                                log.info(
                                    "Skipping verified Telegram audio: "
                                    "message_id=%d output=%s sha256=%s",
                                    msg_id,
                                    resume_target,
                                    expected_hash,
                                )
                                return "skipped"
                            log.warning(
                                "Telegram audio integrity mismatch; reacquiring: "
                                "message_id=%d output=%s expected_sha256=%s "
                                "actual_sha256=%s",
                                msg_id,
                                resume_target,
                                expected_hash,
                                actual_hash,
                            )
                    else:
                        log.info(
                            "Previously downloaded Telegram audio cannot be verified; "
                            "reacquiring: message_id=%d output=%s recorded_sha256=%s",
                            msg_id,
                            resume_target,
                            recorded_hash,
                        )

            metadata = get_audio_metadata(msg)
            if resume_target is not None:
                target_path = resume_target
            else:
                target_filename = build_filename(metadata)
                target_path = _unique_filepath(
                    cfg.download.audio_dir,
                    target_filename,
                    msg_id,
                )
            staging_path = self._staging_path(
                cfg.download.audio_dir,
                msg_id,
                metadata.extension,
            )

            try:
                downloaded_result = await self._client.download_media(
                    msg,
                    file=str(staging_path),
                )
            except Exception as exc:
                staging_path.unlink(missing_ok=True)
                error_str = str(exc)
                log.warning("Download failed for msg %d: %s", msg_id, error_str)
                self._manifest.mark_failed(msg_id, "download", error_str)
                return "failed"

            if not downloaded_result:
                staging_path.unlink(missing_ok=True)
                error_str = "download_media returned None"
                log.warning("No file returned for msg %d", msg_id)
                self._manifest.mark_failed(msg_id, "download", error_str)
                return "failed"

            downloaded_path = Path(downloaded_result)
            try:
                returned_owned_path = (
                    downloaded_path.resolve(strict=False)
                    == staging_path.resolve(strict=False)
                )
            except OSError:
                returned_owned_path = False
            if not returned_owned_path:
                staging_path.unlink(missing_ok=True)
                error_str = "download_media returned an unexpected path"
                log.warning(
                    "Rejected unexpected download path for msg %d: %s",
                    msg_id,
                    downloaded_path,
                )
                self._manifest.mark_failed(msg_id, "download", error_str)
                return "failed"
            if downloaded_path.is_symlink() or not downloaded_path.is_file():
                self._discard_download(downloaded_path, staging_path)
                error_str = "download_media did not return a regular file"
                log.warning("Invalid file returned for msg %d: %s", msg_id, downloaded_path)
                self._manifest.mark_failed(msg_id, "download", error_str)
                return "failed"

            try:
                sha256_hex = _sha256_file(downloaded_path)
            except OSError as exc:
                self._discard_download(downloaded_path, staging_path)
                error_str = f"cannot hash downloaded audio: {exc}"
                log.warning("Cannot hash download for msg %d: %s", msg_id, exc)
                self._manifest.mark_failed(msg_id, "download", error_str)
                return "failed"

            # Once a historical hash exists, Telegram must reproduce exactly
            # those bytes.  Reject a changed remote object without replacing
            # the old file or rewriting its manifest evidence.
            if expected_hash is not None and sha256_hex != expected_hash:
                self._discard_download(downloaded_path, staging_path)
                error_str = (
                    "downloaded SHA-256 mismatch: "
                    f"expected {expected_hash}, got {sha256_hex}"
                )
                log.error(
                    "Rejected changed Telegram audio: message_id=%d %s",
                    msg_id,
                    error_str,
                )
                self._manifest.mark_failed(msg_id, "download", error_str)
                return "failed"

            # Deduplication by SHA-256
            duplicate = None
            if cfg.download.deduplicate and resume_entry is None:
                duplicate = self._verified_duplicate(msg_id, sha256_hex)
            if duplicate is not None:
                existing_id, existing_entry, existing_path = duplicate
                self._discard_download(downloaded_path, staging_path)
                duplicate_of: object = existing_entry.get(
                    "duplicate_of",
                    existing_entry.get(
                        "telegram_msg_id",
                        int(existing_id) if existing_id.isdigit() else existing_id,
                    ),
                )
                manifest_metadata = _download_metadata(
                    metadata,
                    filename=existing_path.name,
                    sha256_hex=sha256_hex,
                )
                manifest_metadata.update(
                    {
                        "duplicate": True,
                        "duplicate_of": duplicate_of,
                    }
                )
                self._manifest.mark_downloaded(msg_id, manifest_metadata)
                log.info(
                    "Recorded duplicate Telegram audio: message_id=%d "
                    "duplicate_of=%s output=%s sha256=%s",
                    msg_id,
                    duplicate_of,
                    existing_path,
                    sha256_hex,
                )
                return "duplicate"

            try:
                quarantine_path = self._install_download(
                    downloaded_path,
                    target_path,
                    msg_id=msg_id,
                    sha256_hex=sha256_hex,
                )
                if staging_path != downloaded_path:
                    staging_path.unlink(missing_ok=True)
            except OSError as exc:
                self._discard_download(downloaded_path, staging_path)
                error_str = f"cannot safely install downloaded audio: {exc}"
                log.error("Download install failed for msg %d: %s", msg_id, exc)
                self._manifest.mark_failed(msg_id, "download", error_str)
                return "failed"

            if quarantine_path is not None:
                log.warning(
                    "Preserved mismatching Telegram audio before recovery: "
                    "message_id=%d quarantine=%s",
                    msg_id,
                    quarantine_path,
                )

            # Record in manifest
            if resume_entry is not None:
                # Preserve all historical Telegram metadata during recovery;
                # update only the evidence needed to make future resume checks
                # trustworthy.
                manifest_metadata = {
                    "filename": target_path.name,
                    "hash": sha256_hex,
                }
            else:
                manifest_metadata = _download_metadata(
                    metadata,
                    filename=target_path.name,
                    sha256_hex=sha256_hex,
                )
            self._manifest.mark_downloaded(msg_id, manifest_metadata)
            log.info(
                "Downloaded Telegram audio: message_id=%d output=%s sha256=%s bytes=%d",
                msg_id,
                target_path,
                sha256_hex,
                target_path.stat().st_size,
            )
            return "downloaded"
