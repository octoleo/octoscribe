"""
src/transcribe/prompt.py — The verbatim transcription instruction.

This single string is the contract that keeps OctoScribe transcripts faithful.
It is isolated in its own module so the wording is easy to find and review, and
so every backend that supports prompting consumes exactly the same text.

Do NOT soften or paraphrase this prompt: the whole point of OctoScribe is a
word-for-word record of what was spoken.
"""

from __future__ import annotations

#: Instruction sent to prompt-aware backends to enforce verbatim output.
VERBATIM_PROMPT: str = (
    "Transcribe EXACTLY what is spoken, word for word. "
    "Do NOT add, remove, or change any words. "
    "Do NOT correct grammar or spelling. "
    "Do NOT paraphrase or rephrase anything. "
    "Preserve every repetition exactly as spoken. "
    "Add only standard punctuation and capitalization — nothing else. "
    "Put each complete sentence on its own line. "
    "Do NOT add headings, labels, speaker names, or any text not spoken."
)
