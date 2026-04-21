"""
manifest.py — Thread-safe manager for manifest.json.

Tracks download and transcription status per Telegram message ID.
The manifest lives in the data repository and is version-controlled.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Manifest:
    """
    Thread-safe manager for manifest.json.

    Tracks download and transcription status per Telegram message ID.
    Lives in the data repository (version-controlled).

    All mutation methods are protected by a threading.Lock. Writes are
    atomic (written to a .tmp file then renamed) to prevent corruption on
    unexpected termination.

    Usage as a context manager saves automatically on exit::

        with Manifest(path) as m:
            m.mark_downloaded(msg_id, metadata)
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        if self._path.exists():
            self.reload()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _str_id(msg_id: str | int) -> str:
        """Normalize a message ID to a string key."""
        return str(msg_id)

    def _now_iso(self) -> str:
        """Return the current UTC time as an ISO-8601 string."""
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ------------------------------------------------------------------
    # Query methods (read-only, no lock required for individual reads
    # because CPython's GIL keeps dict lookups atomic, but we acquire the
    # lock anyway to be correct under all implementations)
    # ------------------------------------------------------------------

    def is_downloaded(self, msg_id: str | int) -> bool:
        """Return True when the entry exists, downloaded==True, and filename is set."""
        with self._lock:
            entry = self._data.get(self._str_id(msg_id))
            if entry is None:
                return False
            return bool(entry.get("downloaded")) and bool(entry.get("filename"))

    def is_transcribed(self, msg_id: str | int) -> bool:
        """Return True when transcription.status == 'completed'."""
        with self._lock:
            entry = self._data.get(self._str_id(msg_id))
            if entry is None:
                return False
            transcription = entry.get("transcription")
            if not isinstance(transcription, dict):
                return False
            return transcription.get("status") == "completed"

    def get_entry(self, msg_id: str | int) -> dict[str, Any] | None:
        """Return a shallow copy of the entry, or None if not present."""
        with self._lock:
            entry = self._data.get(self._str_id(msg_id))
            return dict(entry) if entry is not None else None

    def all_entries(self) -> dict[str, dict[str, Any]]:
        """Return a shallow copy of all entries keyed by string message ID."""
        with self._lock:
            return {k: dict(v) for k, v in self._data.items()}

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    def mark_downloaded(self, msg_id: str | int, metadata: dict[str, Any]) -> None:
        """
        Record that an audio file has been downloaded.

        *metadata* should contain at minimum: filename, title, performer,
        date, duration, duration_seconds, extension, hash.
        """
        key = self._str_id(msg_id)
        with self._lock:
            entry = dict(self._data.get(key, {}))
            entry.update(metadata)
            entry["downloaded"] = True
            entry["telegram_msg_id"] = int(key) if key.isdigit() else key
            self._data[key] = entry

    def mark_transcribed(self, msg_id: str | int, result: dict[str, Any]) -> None:
        """
        Record that an audio file has been transcribed.

        *result* should contain: output_file, model. A completed_at
        timestamp is added automatically if not already present in result.
        The transcription sub-dict is set to status='completed'.
        """
        key = self._str_id(msg_id)
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                entry = {"telegram_msg_id": int(key) if key.isdigit() else key}
            transcription = dict(result)
            transcription["status"] = "completed"
            if "completed_at" not in transcription:
                transcription["completed_at"] = self._now_iso()
            entry["transcription"] = transcription
            self._data[key] = entry

    def mark_failed(self, msg_id: str | int, stage: str, error: str) -> None:
        """
        Record a failure for a given pipeline stage.

        *stage* is typically 'download' or 'transcription'.
        The existing transcription sub-dict (if any) is updated with
        status='failed'; for the download stage a top-level 'error' key is
        also set.
        """
        key = self._str_id(msg_id)
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                entry = {"telegram_msg_id": int(key) if key.isdigit() else key}
            entry["failed_stage"] = stage
            entry["failed_error"] = error
            entry["failed_at"] = self._now_iso()
            if stage == "transcription":
                transcription = dict(entry.get("transcription") or {})
                transcription["status"] = "failed"
                transcription["error"] = error
                entry["transcription"] = transcription
            self._data[key] = entry

    # ------------------------------------------------------------------
    # Pipeline query helpers
    # ------------------------------------------------------------------

    def pending_download(self) -> list[dict[str, Any]]:
        """
        Return entries known but not yet downloaded.

        This handles partial state where an entry was added without a
        completed download step.
        """
        with self._lock:
            return [
                dict(v)
                for v in self._data.values()
                if not (bool(v.get("downloaded")) and bool(v.get("filename")))
            ]

    def pending_transcription(self) -> list[dict[str, Any]]:
        """Return entries that have been downloaded but not yet transcribed."""
        with self._lock:
            result: list[dict[str, Any]] = []
            for entry in self._data.values():
                downloaded = bool(entry.get("downloaded")) and bool(entry.get("filename"))
                if not downloaded:
                    continue
                transcription = entry.get("transcription")
                transcribed = (
                    isinstance(transcription, dict)
                    and transcription.get("status") == "completed"
                )
                if not transcribed:
                    result.append(dict(entry))
            return result

    def stats(self) -> dict[str, int]:
        """Return counts: total, downloaded, transcribed, failed."""
        with self._lock:
            total = len(self._data)
            downloaded = sum(
                1
                for v in self._data.values()
                if bool(v.get("downloaded")) and bool(v.get("filename"))
            )
            transcribed = sum(
                1
                for v in self._data.values()
                if isinstance(v.get("transcription"), dict)
                and v["transcription"].get("status") == "completed"
            )
            failed = sum(1 for v in self._data.values() if "failed_stage" in v)
            return {
                "total": total,
                "downloaded": downloaded,
                "transcribed": transcribed,
                "failed": failed,
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """
        Atomically write the manifest to disk.

        Writes to a .tmp sibling file, then uses os.replace() to
        atomically rename it to the target path. Keys are sorted for
        stable diffs in git.
        """
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(".tmp")
            try:
                content = json.dumps(self._data, indent=2, sort_keys=True, ensure_ascii=False)
                tmp_path.write_text(content, encoding="utf-8")
                os.replace(tmp_path, self._path)
            except Exception:
                # Clean up the temp file if something went wrong before the rename.
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise

    def reload(self) -> None:
        """Re-read the manifest from disk, discarding any unsaved in-memory changes."""
        with self._lock:
            if not self._path.exists():
                self._data = {}
                return
            raw = self._path.read_text(encoding="utf-8")
            loaded = json.loads(raw)
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"manifest.json must contain a JSON object; got {type(loaded).__name__}"
                )
            # Normalise all keys to strings.
            self._data = {str(k): v for k, v in loaded.items()}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Manifest":
        return self

    def __exit__(self, *args: Any) -> None:
        """Save the manifest on context-manager exit (regardless of exceptions)."""
        self.save()
