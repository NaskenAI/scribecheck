"""Sentence-level context shared by the finders: eye context and dosing context."""

from __future__ import annotations

import re
from typing import Literal

from scribecheck.rules.units import find_doses

EyeContext = Literal["eye", "unclear", "none"]

# A full stop ends a sentence unless it sits between digits ("0.5").
_SENTENCE_BREAK = re.compile(r"[;!?\n]|(?<!\d)\.|\.(?!\d)")
_EYE_WORDS = re.compile(
    r"(?<![A-Za-z])(?:eyes?|eye[ \t]*drops?|eyedrops?|ophthalmic)(?![A-Za-z])", re.IGNORECASE
)
_EYE_HINTS = re.compile(r"(?<![A-Za-z])(?:drops?|gtt|ointment)(?![A-Za-z])", re.IGNORECASE)
_DOSING_VERBS = re.compile(
    r"(?<![A-Za-z])(?:take|takes|taking|took|use|uses|using|apply|give|given|insert|"
    r"instil|instill|inhale|spray)(?![A-Za-z])",
    re.IGNORECASE,
)
EYE_ABBREVIATION = re.compile(r"(?<![A-Za-z0-9])(?:OD|OS|OU)(?![A-Za-z0-9])", re.IGNORECASE)


def sentence_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Bounds of the sentence containing text[start:end]."""
    sentence_start = 0
    for found in _SENTENCE_BREAK.finditer(text, 0, start):
        sentence_start = found.end()
    following = _SENTENCE_BREAK.search(text, end)
    return sentence_start, following.start() if following else len(text)


def eye_context(text: str, start: int, end: int) -> EyeContext:
    """Whether the sentence around an OD/OS/OU token at text[start:end] is about eyes.

    Eye words or another eye abbreviation make it "eye". For OD, only OS or OU count,
    because a medication list can repeat OD meaning once daily. Hints such as "drops"
    without an eye word make it "unclear".
    """
    sentence_start, sentence_end = sentence_span(text, start, end)
    around = text[sentence_start:start] + " " + text[end:sentence_end]
    if _EYE_WORDS.search(around):
        return "eye"
    others = {m.group().upper() for m in EYE_ABBREVIATION.finditer(around)}
    if text[start:end].upper() == "OD":
        others.discard("OD")
    if others:
        return "eye"
    return "unclear" if _EYE_HINTS.search(around) else "none"


def has_dosing_context(text: str, start: int, end: int) -> bool:
    """Whether the sentence around text[start:end] has a dose or a dosing verb."""
    sentence_start, sentence_end = sentence_span(text, start, end)
    sentence = text[sentence_start:sentence_end]
    return bool(_DOSING_VERBS.search(sentence) or find_doses(sentence))
