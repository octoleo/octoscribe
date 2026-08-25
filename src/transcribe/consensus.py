"""Evidence-preserving transcript comparison and bounded resolution policy.

This module deliberately does *not* construct a "best" transcript.  It keeps
every provider's original text unchanged, aligns normalized word views for
comparison, and reports the differences as evidence for a later retry or
independent arbiter.  Exhausted comparison still completes with the primary
transcript unchanged and records warnings alongside it.

Normalization is comparison-only: case, punctuation, and whitespace do not
create discrepancies.  The original strings and the original spelling of
every aligned token remain available on the returned objects.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from enum import Enum
from itertools import combinations
from typing import Sequence


class QualityState(str, Enum):
    """Truthful quality states for a transcript and its evidence."""

    MACHINE_TRANSCRIBED = "machine_transcribed"
    CROSS_CHECKED = "cross_checked"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    HUMAN_VERIFIED = "human_verified"


class DiscrepancyKind(str, Enum):
    """Direction of an edit from the left transcript to the right one."""

    ADDITION = "addition"
    DELETION = "deletion"
    SUBSTITUTION = "substitution"


class DiscrepancyPriority(str, Enum):
    """Whether a discrepancy contains fidelity-sensitive language."""

    STANDARD = "standard"
    CRITICAL = "critical"


class CriticalCategory(str, Enum):
    """Terms whose disagreement deserves elevated review priority."""

    NEGATION = "negation"
    NUMBER = "number"
    SCRIPTURE_REFERENCE = "scripture_reference"


class ResolutionAction(str, Enum):
    """The only actions emitted by the bounded resolution policy."""

    ACCEPT = "accept"
    RETRY = "retry"
    ARBITRATE = "arbitrate"
    COMPLETE_WITH_WARNINGS = "complete_with_warnings"


@dataclass(frozen=True, order=True)
class TokenSpan:
    """A half-open ``[start, end)`` span in a normalized word sequence."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("token spans require 0 <= start <= end")

    @property
    def is_empty(self) -> bool:
        """Return whether the span is an insertion/deletion boundary."""
        return self.start == self.end

    def intersects(self, other: "TokenSpan") -> bool:
        """Return whether two non-empty half-open spans overlap."""
        return (
            not self.is_empty
            and not other.is_empty
            and self.start < other.end
            and other.start < self.end
        )


@dataclass(frozen=True)
class TranscriptToken:
    """One comparison token, tied back to its exact source location."""

    index: int
    original: str
    normalized: str
    character_start: int
    character_end: int


@dataclass(frozen=True)
class TranscriptEvidence:
    """An immutable original transcript plus its comparison-only word view."""

    label: str
    original: str
    tokens: tuple[TranscriptToken, ...]

    @property
    def normalized_words(self) -> tuple[str, ...]:
        """Return normalized words without changing :attr:`original`."""
        return tuple(token.normalized for token in self.tokens)

    def originals_in(self, span: TokenSpan) -> tuple[str, ...]:
        """Return original token spellings covered by *span*."""
        return tuple(token.original for token in self.tokens[span.start : span.end])


@dataclass(frozen=True)
class CriticalHit:
    """A fidelity-sensitive term or reference touched by a discrepancy."""

    transcript_index: int
    category: CriticalCategory
    token_span: TokenSpan
    original_tokens: tuple[str, ...]


@dataclass(frozen=True)
class Discrepancy:
    """One aligned difference between two original transcripts."""

    kind: DiscrepancyKind
    left_span: TokenSpan
    right_span: TokenSpan
    left_tokens: tuple[str, ...]
    right_tokens: tuple[str, ...]
    priority: DiscrepancyPriority
    critical_hits: tuple[CriticalHit, ...] = ()

    @property
    def is_critical(self) -> bool:
        """Return whether this discrepancy needs elevated attention."""
        return self.priority is DiscrepancyPriority.CRITICAL


@dataclass(frozen=True)
class TranscriptComparison:
    """Pairwise alignment result for two entries in a consensus report."""

    left_index: int
    right_index: int
    discrepancies: tuple[Discrepancy, ...]

    @property
    def agrees(self) -> bool:
        """Return whether the normalized word sequences are identical."""
        return not self.discrepancies

    @property
    def has_critical_discrepancy(self) -> bool:
        """Return whether any aligned difference is fidelity-sensitive."""
        return any(item.is_critical for item in self.discrepancies)


@dataclass(frozen=True)
class ConsensusReport:
    """Pairwise evidence for two or three transcripts, never a merged text."""

    transcripts: tuple[TranscriptEvidence, ...]
    comparisons: tuple[TranscriptComparison, ...]
    quality_state: QualityState

    @property
    def all_agree(self) -> bool:
        """Return whether every pair has the same normalized words."""
        return all(comparison.agrees for comparison in self.comparisons)

    @property
    def discrepancies(self) -> tuple[Discrepancy, ...]:
        """Return all pairwise discrepancies in deterministic pair order."""
        return tuple(
            discrepancy
            for comparison in self.comparisons
            for discrepancy in comparison.discrepancies
        )

    @property
    def has_critical_discrepancy(self) -> bool:
        """Return whether any pair touches a critical term or reference."""
        return any(
            comparison.has_critical_discrepancy
            for comparison in self.comparisons
        )

    def has_agreeing_peer(self, label: str) -> bool:
        """Return whether *label* agrees with at least one independent peer."""
        matching = [
            index
            for index, transcript in enumerate(self.transcripts)
            if transcript.label == label
        ]
        if len(matching) != 1:
            raise ValueError(f"report must contain exactly one transcript labelled {label!r}")
        index = matching[0]
        return any(
            comparison.agrees
            and index in {comparison.left_index, comparison.right_index}
            for comparison in self.comparisons
        )


@dataclass(frozen=True)
class ResolutionProgress:
    """Completed passes in a resolution attempt.

    Booleans make the limits structural: there can be no second retry and no
    second arbiter pass.  An arbiter pass is only valid after the retry.
    """

    retry_completed: bool = False
    arbiter_completed: bool = False
    human_verified: bool = False

    def __post_init__(self) -> None:
        if self.arbiter_completed and not self.retry_completed:
            raise ValueError("an arbiter pass may only follow the single retry")

    def after(self, action: ResolutionAction) -> "ResolutionProgress":
        """Return new progress after a policy action, enforcing hard bounds."""
        if action is ResolutionAction.RETRY:
            if self.retry_completed:
                raise ValueError("the single retry has already been completed")
            return replace(self, retry_completed=True)
        if action is ResolutionAction.ARBITRATE:
            if not self.retry_completed:
                raise ValueError("the arbiter pass may only follow the retry")
            if self.arbiter_completed:
                raise ValueError("the single arbiter pass has already been completed")
            return replace(self, arbiter_completed=True)
        return self

    def with_human_verification(self) -> "ResolutionProgress":
        """Return progress explicitly marked as verified against the audio."""
        return replace(self, human_verified=True)


@dataclass(frozen=True)
class ResolutionDecision:
    """A pure next-step decision; it intentionally contains no transcript."""

    action: ResolutionAction
    quality_state: QualityState
    reason: str


# Each match is a word-like unit.  Apostrophes remain inside a token so
# ``don't`` and ``dont`` compare equal after punctuation removal.
_WORD_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)

_NEGATIONS = frozenset(
    {
        "aint",
        "arent",
        "cannot",
        "cant",
        "couldnt",
        "didnt",
        "doesnt",
        "dont",
        "hardly",
        "isnt",
        "mustnt",
        "neither",
        "never",
        "no",
        "nobody",
        "none",
        "nor",
        "not",
        "nothing",
        "nowhere",
        "shouldnt",
        "wasnt",
        "werent",
        "without",
        "wont",
        "wouldnt",
    }
)

_NUMBER_WORDS = frozenset(
    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
    }
)

_UNNUMBERED_BOOK_ALIASES = {
    "genesis": ("gen",),
    "exodus": ("exod", "ex"),
    "leviticus": ("lev",),
    "numbers": ("num",),
    "deuteronomy": ("deut",),
    "joshua": ("josh",),
    "judges": ("judg",),
    "ruth": (),
    "ezra": (),
    "nehemiah": ("neh",),
    "esther": ("esth",),
    "job": (),
    "psalm": ("psalms", "ps", "psa"),
    "proverbs": ("prov",),
    "ecclesiastes": ("eccl", "ecc"),
    "isaiah": ("isa",),
    "jeremiah": ("jer",),
    "lamentations": ("lam",),
    "ezekiel": ("ezek",),
    "daniel": ("dan",),
    "hosea": ("hos",),
    "joel": (),
    "amos": (),
    "obadiah": ("obad",),
    "jonah": ("jon",),
    "micah": ("mic",),
    "nahum": ("nah",),
    "habakkuk": ("hab",),
    "zephaniah": ("zeph",),
    "haggai": ("hag",),
    "zechariah": ("zech",),
    "malachi": ("mal",),
    "matthew": ("matt",),
    "mark": ("mk",),
    "luke": ("lk",),
    "john": ("jn",),
    "acts": ("act",),
    "romans": ("rom",),
    "galatians": ("gal",),
    "ephesians": ("eph",),
    "philippians": ("phil",),
    "colossians": ("col",),
    "titus": ("tit",),
    "philemon": ("philem",),
    "hebrews": ("heb",),
    "james": ("jas",),
    "jude": (),
    "revelation": ("revelations", "rev"),
}

_MULTIWORD_BOOK_ALIASES = {
    ("song", "of", "solomon"),
    ("song", "of", "songs"),
    ("acts", "of", "the", "apostles"),
}

_NUMBERED_BOOK_ALIASES = {
    "samuel": ("sam",),
    "kings": ("kgs",),
    "chronicles": ("chron", "chr"),
    "corinthians": ("cor",),
    "thessalonians": ("thess",),
    "timothy": ("tim",),
    "peter": ("pet",),
    "john": ("jn",),
}

_ORDINAL_PREFIXES = {
    1: ("1", "1st", "first", "i"),
    2: ("2", "2nd", "second", "ii"),
    3: ("3", "3rd", "third", "iii"),
}


def _build_book_aliases() -> tuple[tuple[str, ...], ...]:
    aliases = set(_MULTIWORD_BOOK_ALIASES)
    for name, abbreviations in _UNNUMBERED_BOOK_ALIASES.items():
        aliases.add((name,))
        aliases.update((abbreviation,) for abbreviation in abbreviations)
    for name, abbreviations in _NUMBERED_BOOK_ALIASES.items():
        bases = (name, *abbreviations)
        maximum = 3 if name == "john" else 2
        for ordinal in range(1, maximum + 1):
            for prefix in _ORDINAL_PREFIXES[ordinal]:
                aliases.update((prefix, base) for base in bases)
    return tuple(sorted(aliases, key=lambda item: (-len(item), item)))


_BOOK_ALIASES = _build_book_aliases()
_CHAPTER_MARKERS = frozenset({"chapter", "chapters", "ch"})
_VERSE_MARKERS = frozenset({"verse", "verses", "v", "vv"})
_RANGE_MARKERS = frozenset({"through", "to"})


def _normalize_word(word: str) -> str:
    """Create a case/punctuation-insensitive view of one source word."""
    folded = unicodedata.normalize("NFKC", word).casefold()
    return "".join(
        character
        for character in folded
        if not unicodedata.category(character).startswith("P")
    )


def _tokenize(label: str, original: str) -> TranscriptEvidence:
    tokens = tuple(
        TranscriptToken(
            index=index,
            original=match.group(0),
            normalized=_normalize_word(match.group(0)),
            character_start=match.start(),
            character_end=match.end(),
        )
        for index, match in enumerate(_WORD_RE.finditer(original))
    )
    return TranscriptEvidence(label=label, original=original, tokens=tokens)


def _is_number(word: str) -> bool:
    return any(character.isdigit() for character in word) or word in _NUMBER_WORDS


def _book_alias_length(words: tuple[str, ...], start: int) -> int:
    for alias in _BOOK_ALIASES:
        end = start + len(alias)
        if words[start:end] == alias:
            return len(alias)
    return 0


def _scripture_spans(evidence: TranscriptEvidence) -> tuple[TokenSpan, ...]:
    """Find conservative token spans that look like Scripture references."""
    words = evidence.normalized_words
    spans: list[TokenSpan] = []
    index = 0
    while index < len(words):
        alias_length = _book_alias_length(words, index)
        if not alias_length:
            index += 1
            continue

        cursor = index + alias_length
        if cursor < len(words) and words[cursor] in _CHAPTER_MARKERS:
            cursor += 1
        if cursor >= len(words) or not _is_number(words[cursor]):
            index += 1
            continue

        # Chapter number.
        cursor += 1
        if cursor < len(words) and words[cursor] in _VERSE_MARKERS:
            marker = cursor
            cursor += 1
            if cursor >= len(words) or not _is_number(words[cursor]):
                cursor = marker
            else:
                cursor += 1
        elif cursor < len(words) and _is_number(words[cursor]):
            # Punctuation is intentionally absent from the word view, so the
            # second adjacent number is the verse in forms such as John 3:16.
            cursor += 1

        if (
            cursor + 1 < len(words)
            and words[cursor] in _RANGE_MARKERS
            and _is_number(words[cursor + 1])
        ):
            cursor += 2
        elif cursor < len(words) and _is_number(words[cursor]):
            # Covers punctuation-only ranges such as John 3:16-18.
            cursor += 1

        spans.append(TokenSpan(index, cursor))
        index = cursor
    return tuple(spans)


def _critical_hits(
    evidence: TranscriptEvidence,
    transcript_index: int,
    changed_span: TokenSpan,
) -> tuple[CriticalHit, ...]:
    hits: list[CriticalHit] = []
    for token in evidence.tokens[changed_span.start : changed_span.end]:
        token_span = TokenSpan(token.index, token.index + 1)
        if token.normalized in _NEGATIONS:
            hits.append(
                CriticalHit(
                    transcript_index,
                    CriticalCategory.NEGATION,
                    token_span,
                    (token.original,),
                )
            )
        if _is_number(token.normalized):
            hits.append(
                CriticalHit(
                    transcript_index,
                    CriticalCategory.NUMBER,
                    token_span,
                    (token.original,),
                )
            )

    for scripture_span in _scripture_spans(evidence):
        if changed_span.intersects(scripture_span):
            hits.append(
                CriticalHit(
                    transcript_index,
                    CriticalCategory.SCRIPTURE_REFERENCE,
                    scripture_span,
                    evidence.originals_in(scripture_span),
                )
            )

    return tuple(
        sorted(
            hits,
            key=lambda hit: (
                hit.transcript_index,
                hit.token_span.start,
                hit.token_span.end,
                hit.category.value,
            ),
        )
    )


def _compare_pair(
    transcripts: tuple[TranscriptEvidence, ...],
    left_index: int,
    right_index: int,
) -> TranscriptComparison:
    left = transcripts[left_index]
    right = transcripts[right_index]
    matcher = SequenceMatcher(
        None,
        left.normalized_words,
        right.normalized_words,
        autojunk=False,
    )
    discrepancies: list[Discrepancy] = []
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        kind = {
            "insert": DiscrepancyKind.ADDITION,
            "delete": DiscrepancyKind.DELETION,
            "replace": DiscrepancyKind.SUBSTITUTION,
        }[tag]
        left_span = TokenSpan(left_start, left_end)
        right_span = TokenSpan(right_start, right_end)
        hits = (
            *_critical_hits(left, left_index, left_span),
            *_critical_hits(right, right_index, right_span),
        )
        priority = (
            DiscrepancyPriority.CRITICAL
            if hits
            else DiscrepancyPriority.STANDARD
        )
        discrepancies.append(
            Discrepancy(
                kind=kind,
                left_span=left_span,
                right_span=right_span,
                left_tokens=left.originals_in(left_span),
                right_tokens=right.originals_in(right_span),
                priority=priority,
                critical_hits=hits,
            )
        )
    return TranscriptComparison(left_index, right_index, tuple(discrepancies))


def quality_state_for(
    transcript_count: int,
    *,
    all_agree: bool | None = None,
    human_verified: bool = False,
) -> QualityState:
    """Return a conservative quality state for one to three transcripts."""
    if transcript_count not in {1, 2, 3}:
        raise ValueError("quality state requires one, two, or three transcripts")
    if human_verified:
        return QualityState.HUMAN_VERIFIED
    if transcript_count == 1:
        return QualityState.MACHINE_TRANSCRIBED
    if all_agree is True:
        return QualityState.CROSS_CHECKED
    return QualityState.COMPLETED_WITH_WARNINGS


def compare_transcripts(
    transcripts: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> ConsensusReport:
    """Compare two or three transcripts without selecting or rewriting words.

    Every pair is aligned independently.  This prevents a two-against-one
    majority from silently replacing the third provider's evidence.
    """
    if isinstance(transcripts, (str, bytes)):
        raise TypeError("transcripts must be a sequence of transcript strings")
    if len(transcripts) not in {2, 3}:
        raise ValueError("comparison requires exactly two or three transcripts")
    if not all(isinstance(transcript, str) for transcript in transcripts):
        raise TypeError("every transcript must be a string")

    if labels is None:
        resolved_labels = tuple(
            f"transcript_{index + 1}" for index in range(len(transcripts))
        )
    else:
        if len(labels) != len(transcripts):
            raise ValueError("labels must match the number of transcripts")
        if not all(isinstance(label, str) and label for label in labels):
            raise ValueError("every transcript label must be a non-empty string")
        resolved_labels = tuple(labels)

    evidence = tuple(
        _tokenize(label, original)
        for label, original in zip(resolved_labels, transcripts)
    )
    comparisons = tuple(
        _compare_pair(evidence, left_index, right_index)
        for left_index, right_index in combinations(range(len(evidence)), 2)
    )
    all_agree = all(comparison.agrees for comparison in comparisons)
    return ConsensusReport(
        transcripts=evidence,
        comparisons=comparisons,
        quality_state=quality_state_for(
            len(evidence),
            all_agree=all_agree,
        ),
    )


def next_resolution_decision(
    report: ConsensusReport,
    progress: ResolutionProgress | None = None,
    *,
    arbiter_available: bool = False,
    primary_label: str | None = None,
) -> ResolutionDecision:
    """Choose the next bounded action without calling a provider.

    Disagreement permits one retry, then at most one optional arbiter pass.
    Continued disagreement completes with warnings; the policy never loops,
    never blocks publication, and never manufactures a replacement transcript.
    """
    progress = progress or ResolutionProgress()
    if progress.human_verified:
        return ResolutionDecision(
            ResolutionAction.ACCEPT,
            QualityState.HUMAN_VERIFIED,
            "A human explicitly verified the transcript against the audio.",
        )
    if report.all_agree:
        return ResolutionDecision(
            ResolutionAction.ACCEPT,
            QualityState.CROSS_CHECKED,
            "All normalized word sequences agree.",
        )
    if (
        progress.arbiter_completed
        and primary_label is not None
        and report.has_agreeing_peer(primary_label)
    ):
        return ResolutionDecision(
            ResolutionAction.ACCEPT,
            QualityState.CROSS_CHECKED,
            "The canonical primary has independent support after arbitration.",
        )
    if not progress.retry_completed:
        return ResolutionDecision(
            ResolutionAction.RETRY,
            QualityState.COMPLETED_WITH_WARNINGS,
            "Discrepancies remain; use the single permitted retry.",
        )
    if arbiter_available and not progress.arbiter_completed:
        return ResolutionDecision(
            ResolutionAction.ARBITRATE,
            QualityState.COMPLETED_WITH_WARNINGS,
            "The retry still disagrees; use the single optional arbiter pass.",
        )
    return ResolutionDecision(
        ResolutionAction.COMPLETE_WITH_WARNINGS,
        QualityState.COMPLETED_WITH_WARNINGS,
        "Automated comparison is exhausted; publish the unchanged primary "
        "transcript and preserve all discrepancy evidence as warnings.",
    )


__all__ = [
    "ConsensusReport",
    "CriticalCategory",
    "CriticalHit",
    "Discrepancy",
    "DiscrepancyKind",
    "DiscrepancyPriority",
    "QualityState",
    "ResolutionAction",
    "ResolutionDecision",
    "ResolutionProgress",
    "TokenSpan",
    "TranscriptComparison",
    "TranscriptEvidence",
    "TranscriptToken",
    "compare_transcripts",
    "next_resolution_decision",
    "quality_state_for",
]
