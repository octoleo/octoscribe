"""
src/transcribe/normalize.py — Whitespace-only transcript normalisation.

The single rule here is sacred: *never touch a spoken word*.  Normalisation is
limited to making whitespace consistent and deterministic so that transcripts
diff cleanly in git and read uniformly, without altering, adding, or removing
any content the speaker produced.
"""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """
    Apply minimal, word-preserving post-processing to *text*.

    The transformation:

    * normalises CR/CRLF line endings to ``\\n``;
    * strips trailing whitespace from each line; and
    * caps runs of blank lines at a single blank line.

    No spoken word is ever altered, added, or removed — only whitespace is
    touched.  Returns an empty string for empty input.
    """
    if not text:
        return ""
    # Normalise CR/CRLF.
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace per line.
    out = "\n".join(line.rstrip() for line in out.split("\n"))
    # Cap consecutive blank lines at 1.
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# Backwards-compatible private alias: the test suite and historical callers
# import ``_normalize_text``.  Keep both names pointing at one implementation.
_normalize_text = normalize_text
