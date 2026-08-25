"""
manifest.py — Thread-safe manager for manifest.json.

Tracks acquisition and transcription status for Telegram and folder sources.
The manifest lives in the caller-supplied evidence workspace so the calling
workflow can preserve it between runs.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.persistence import atomic_write_text


_TERMINAL_TRANSCRIPTION_STATES = frozenset(
    {
        "completed",  # legacy manifests
        "machine_transcribed",
        "cross_checked",
        "completed_with_warnings",
        "needs_review",  # legacy state; accepted as terminal when loading
        "human_verified",
    }
)

_FAILURE_MARKER_KEYS = ("failed_stage", "failed_error", "failed_at", "error")
_ACQUISITION_FAILURE_STAGES = frozenset({"download", "import"})


class Manifest:
    """
    Thread-safe manager for manifest.json.

    Tracks acquisition and transcription state for every source item.  It
    lives in the caller-supplied text/evidence workspace.

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

    @staticmethod
    def _clear_failure_markers(entry: dict[str, Any]) -> None:
        """Remove the top-level marker for a failure that has been recovered."""
        for key in _FAILURE_MARKER_KEYS:
            entry.pop(key, None)

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
        """Return True once automation has reached a terminal quality state."""
        with self._lock:
            entry = self._data.get(self._str_id(msg_id))
            if entry is None:
                return False
            transcription = entry.get("transcription")
            if not isinstance(transcription, dict):
                return False
            return transcription.get("status") in _TERMINAL_TRANSCRIPTION_STATES

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
            # A successful acquisition retry resolves a previous download or
            # folder-import failure.  Do not hide an unrelated transcription
            # failure merely because the source was seen again.
            failed_stage = entry.get("failed_stage")
            legacy_failure = failed_stage is None and any(
                marker in entry for marker in _FAILURE_MARKER_KEYS
            )
            if failed_stage in _ACQUISITION_FAILURE_STAGES or legacy_failure:
                self._clear_failure_markers(entry)
            self._data[key] = entry

    def record_audio_hash(self, msg_id: str | int, sha256: str) -> None:
        """Backfill or verify the immutable source hash before transcription."""
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError("audio hash must be a lower-case SHA-256 digest")
        key = self._str_id(msg_id)
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                raise KeyError(f"no manifest entry exists for {key}")
            existing = entry.get("hash")
            if isinstance(existing, str) and len(existing) == 64 and existing != sha256:
                raise ValueError(
                    f"audio SHA-256 mismatch for {key}; source evidence changed"
                )
            entry["hash"] = sha256
            self._data[key] = entry

    def mark_transcribed(self, msg_id: str | int, result: dict[str, Any]) -> None:
        """
        Record that an audio file has been transcribed.

        *result* should contain: output_file, output_path, audio_path, and model.
        A completed_at timestamp is added automatically if not already present.
        New callers should provide ``quality_state``.  Legacy callers that do
        not provide it retain status='completed' for backwards compatibility.
        """
        key = self._str_id(msg_id)
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                entry = {"telegram_msg_id": int(key) if key.isdigit() else key}
            transcription = dict(result)
            quality_state = transcription.get("quality_state")
            if quality_state is not None and quality_state not in (
                _TERMINAL_TRANSCRIPTION_STATES - {"completed"}
            ):
                raise ValueError(f"unknown transcription quality state: {quality_state!r}")
            transcription["status"] = quality_state or "completed"
            for marker in _FAILURE_MARKER_KEYS:
                transcription.pop(marker, None)
            if "completed_at" not in transcription:
                transcription["completed_at"] = self._now_iso()
            entry["transcription"] = transcription
            # A terminal transcription supersedes any stale failure on this
            # item, including failures recorded by older manifest versions.
            self._clear_failure_markers(entry)
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
        """Return unique downloaded recordings not yet transcribed.

        Telegram content duplicates remain first-class manifest entries so
        their message IDs are not downloaded again, but only the canonical
        recording is eligible for a paid transcription call.
        """
        with self._lock:
            result: list[dict[str, Any]] = []
            for entry in self._data.values():
                downloaded = bool(entry.get("downloaded")) and bool(entry.get("filename"))
                if not downloaded:
                    continue
                if entry.get("duplicate") is True:
                    continue
                transcription = entry.get("transcription")
                transcribed = (
                    isinstance(transcription, dict)
                    and transcription.get("status") in _TERMINAL_TRANSCRIPTION_STATES
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
                and v["transcription"].get("status") in _TERMINAL_TRANSCRIPTION_STATES
            )
            failed = sum(1 for v in self._data.values() if "failed_stage" in v)
            return {
                "total": total,
                "downloaded": downloaded,
                "transcribed": transcribed,
                "failed": failed,
            }

    def quality_stats(self) -> dict[str, int]:
        """Return counts for truthful quality states without changing stats()."""
        with self._lock:
            result = {
                "machine_transcribed": 0,
                "cross_checked": 0,
                "completed_with_warnings": 0,
                "legacy_needs_review": 0,
                "human_verified": 0,
                "legacy_completed": 0,
            }
            for entry in self._data.values():
                transcription = entry.get("transcription")
                if not isinstance(transcription, dict):
                    continue
                state = transcription.get("status")
                if state == "completed":
                    key = "legacy_completed"
                elif state == "needs_review":
                    key = "legacy_needs_review"
                else:
                    key = state
                if key in result:
                    result[key] += 1
            return result

    def mark_human_verified(
        self,
        msg_id: str | int,
        *,
        reviewer: str | None = None,
    ) -> None:
        """Record explicit human comparison against the source audio."""
        key = self._str_id(msg_id)
        with self._lock:
            entry = self._data.get(key)
            if entry is None or not isinstance(entry.get("transcription"), dict):
                raise KeyError(f"no transcription exists for {key}")
            transcription = dict(entry["transcription"])
            transcription["status"] = "human_verified"
            transcription["quality_state"] = "human_verified"
            transcription["human_verified_at"] = self._now_iso()
            if reviewer:
                transcription["human_verified_by"] = reviewer
            entry["transcription"] = transcription
            self._data[key] = entry

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """
        Atomically write the manifest to disk.

        Serialises the in-memory data with keys sorted (for stable, reviewable
        git diffs) and delegates the crash-safe tmp-write-then-rename to
        :func:`src.persistence.atomic_write_text`, so the manifest and the
        transcript writer share identical durability guarantees.
        """
        with self._lock:
            content = json.dumps(self._data, indent=2, sort_keys=True, ensure_ascii=False)
            atomic_write_text(self._path, content)

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
