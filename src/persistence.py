"""
src/persistence.py — Shared, framework-agnostic persistence utilities.

This module exists to remove duplication that previously lived in several
unrelated modules:

* An *atomic write* (write to a temporary sibling, then ``os.replace``) was
  re-implemented independently in :mod:`src.manifest` and :mod:`src.transcribe`.
* A *periodic-save counter* (save the manifest every N processed items) was
  copy-pasted across :mod:`src.telegram`, :mod:`src.folder` and
  :mod:`src.transcribe`.

Centralising both here gives every caller the same crash-safe write semantics
and the same save cadence, and gives us a single, well-tested place to reason
about durability.  The helpers deliberately depend on nothing beyond the
standard library so they can be reused by any audio source or backend.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default number of processed items between intermediate manifest saves.
#: A periodic flush bounds how much progress is lost if the process is killed
#: mid-run, while still avoiding a disk write after every single item.
DEFAULT_SAVE_INTERVAL: int = 10


# ---------------------------------------------------------------------------
# Atomic file writes
# ---------------------------------------------------------------------------

def atomic_write_bytes(path: Path, data: bytes) -> None:
    """
    Write *data* to *path* atomically.

    The bytes are first written to a temporary sibling file (``<name>.tmp``)
    and then moved into place with :func:`os.replace`, which is atomic on every
    supported platform.  A reader therefore never observes a half-written file:
    it sees either the previous contents or the complete new contents.

    On any failure the temporary file is removed so no ``.tmp`` debris is left
    behind, and the original exception is re-raised unchanged.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_bytes(data)
        os.replace(tmp_path, path)
    except Exception:
        _silent_unlink(tmp_path)
        raise


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """
    Write *text* to *path* atomically using LF (``\\n``) line endings.

    Thin text wrapper around :func:`atomic_write_bytes` so that callers writing
    transcripts and JSON manifests share identical crash-safety guarantees.
    Newlines are written verbatim (``newline=""`` semantics) to keep transcript
    output byte-for-byte reproducible across platforms.
    """
    atomic_write_bytes(path, text.encode(encoding))


def _silent_unlink(path: Path) -> None:
    """Best-effort removal of *path*; never raises (cleanup must not mask errors)."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Periodic saving
# ---------------------------------------------------------------------------

class _Saveable(Protocol):
    """Minimal structural type for anything that can persist itself on demand."""

    def save(self) -> None: ...


class PeriodicSaver:
    """
    Flush a saveable object to disk once every *interval* successful items.

    Long-running download/import/transcription loops call :meth:`tick` after
    each item they successfully process.  Every *interval*-th tick triggers a
    save, bounding the amount of progress that a crash can discard.  The final,
    authoritative save is still the caller's responsibility (typically once more
    at the very end of the loop).

    This replaces three near-identical ``save_counter``/modulo blocks that were
    previously duplicated across the audio sources and the transcriber.

    Example::

        saver = PeriodicSaver(manifest)
        for item in work:
            if process(item) == "ok":
                saver.tick()
        manifest.save()  # final flush
    """

    def __init__(self, target: _Saveable, interval: int = DEFAULT_SAVE_INTERVAL) -> None:
        if interval < 1:
            raise ValueError(f"interval must be >= 1, got {interval!r}")
        self._target = target
        self._interval = interval
        self._count = 0

    @property
    def count(self) -> int:
        """Number of times :meth:`tick` has been called so far."""
        return self._count

    def tick(self) -> bool:
        """
        Register one processed item; save if the interval boundary is reached.

        Returns ``True`` when this tick triggered a save, ``False`` otherwise,
        so callers can log periodic-save events if they wish.
        """
        self._count += 1
        if self._count % self._interval == 0:
            self._target.save()
            return True
        return False
