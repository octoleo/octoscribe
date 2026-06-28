"""
audio.py — Framework-agnostic audio helpers shared across audio sources.

These helpers contain no Telegram/Telethon dependencies so that the local
folder import path can reuse them without requiring any Telegram credentials
or libraries to be present.

Both :mod:`src.telegram` and :mod:`src.folder` import from here.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Recognised audio file extensions (lower-case, leading dot).
AUDIO_EXTENSIONS: tuple[str, ...] = (
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus",
)


# ---------------------------------------------------------------------------
# Pure helper functions
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


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_filepath(directory: Path, filename: str, discriminator: str | int) -> Path:
    """
    Return a path that does not yet exist in *directory*.

    If *filename* is already taken, *discriminator* (e.g. a message ID or a
    short content hash) is appended to the stem to keep the name unique.
    """
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, ext = os.path.splitext(filename)
    return directory / f"{stem}_{discriminator}{ext}"
