"""Deterministic, whitespace-only transcript presentation.

OctoScribe must never rewrite the speaker's words. This module therefore
changes whitespace only: it collapses ordinary spacing and places each
provider-punctuated sentence on its own line. The ordered sequence of every
non-whitespace character is preserved exactly.
"""

from __future__ import annotations

import re


_CLOSING_PUNCTUATION = '\"\'\u2019\u201d\u00bb)]}'
_TITLES = frozenset(
    {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "rev.",
        "fr.",
        "sr.",
        "jr.",
        "st.",
    }
)
_CONTINUING_ABBREVIATIONS = frozenset(
    {"e.g.", "i.e.", "cf.", "vs.", "a.m.", "p.m.", "etc."}
)
_NUMBER_PREFIX_ABBREVIATIONS = frozenset(
    {
        # Scripture books commonly abbreviated in sermon transcripts.
        "gen.", "exod.", "lev.", "num.", "deut.", "josh.", "judg.",
        "sam.", "kings.", "chron.", "neh.", "esth.", "ps.", "pss.",
        "prov.", "eccl.", "isa.", "jer.", "lam.", "ezek.", "dan.",
        "hos.", "obad.", "mic.", "nah.", "hab.", "zeph.", "hag.",
        "zech.", "mal.", "matt.", "mark.", "luke.", "john.", "acts.",
        "rom.", "cor.", "gal.", "eph.", "phil.", "col.", "thess.",
        "tim.", "tit.", "philem.", "heb.", "jas.", "pet.", "rev.",
        # Reference labels.
        "ch.", "chs.", "v.", "vv.", "p.", "pp.", "no.", "nos.",
        # Dates.
        "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.",
        "sep.", "sept.", "oct.", "nov.", "dec.",
    }
)
_INITIALS = re.compile(r"^(?:[^\W\d_]\.)+$", re.UNICODE)


def _without_closers(token: str) -> str:
    return token.rstrip(_CLOSING_PUNCTUATION)


def _starts_with_letter_or_number(token: str) -> bool:
    core = token.lstrip('\"\'\u2018\u201c\u00ab([{')
    return bool(core and core[0].isalnum())


def _starts_with_uppercase(token: str) -> bool:
    core = token.lstrip('\"\'\u2018\u201c\u00ab([{')
    return bool(core and core[0].isupper())


def _is_sentence_boundary(token: str, next_token: str | None) -> bool:
    """Return whether whitespace after *token* separates sentences.

    This scanner relies only on punctuation already supplied by the ASR
    provider. It never invents a boundary for unpunctuated speech.
    """
    core = _without_closers(token)
    if not core:
        return False
    if core.endswith(("?", "!")):
        return True
    if not core.endswith("."):
        return False
    if core.endswith("...") or core.endswith("\u2026"):
        return True

    lowered = core.casefold()
    if lowered in _TITLES:
        return False
    if next_token is not None and lowered in _NUMBER_PREFIX_ABBREVIATIONS:
        if next_token.lstrip('\"\'\u2018\u201c\u00ab([{')[:1].isdigit():
            return False
    if next_token is not None and lowered in _CONTINUING_ABBREVIATIONS:
        if _starts_with_letter_or_number(next_token) and not _starts_with_uppercase(
            next_token
        ):
            return False
    if next_token is not None and _INITIALS.fullmatch(core):
        if _starts_with_uppercase(next_token):
            return False
    return True


def normalize_text(text: str) -> str:
    """Return a sentence-per-line transcript without changing its content.

    All input whitespace is made deterministic: one space within a sentence,
    one newline between complete provider-punctuated sentences, and no leading
    or trailing whitespace. Words and punctuation are never added, removed,
    reordered, corrected, or otherwise modified.
    """
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return ""

    pieces: list[str] = []
    for index, token in enumerate(tokens):
        pieces.append(token)
        if index == len(tokens) - 1:
            continue
        next_token = tokens[index + 1]
        pieces.append("\n" if _is_sentence_boundary(token, next_token) else " ")
    return "".join(pieces)


# Backwards-compatible private alias used by historical callers.
_normalize_text = normalize_text
