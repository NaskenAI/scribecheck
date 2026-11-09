"""Find a model's quote in the transcript or note, and return evidence that slices back.

Models paraphrase, re-punctuate and re-case quotes, so a quote is matched after
normalising both sides: case-folded, with punctuation and quote marks removed and
whitespace collapsed (a decimal point between digits is kept). An exact normalised match of
whole tokens is accepted when it is at least `MIN_EXACT_CHARS` characters long or a single
clinical abbreviation such as "NKDA" or "BD"; shorter ones ("no", "2") are not evidence.
Otherwise a fuzzy match is tried: a window of whole tokens whose difflib ratio is at least
`FUZZY_THRESHOLD`, for quotes of at least `MIN_CHARS` normalised characters (the
OmissionBench harness's minimum).

Every match, exact or fuzzy, counts only if the quote and the matched span have identical
critical tokens: numbers (number words included), units, negation words (contractions with
or without the apostrophe), side words and drug names the lexicon finds; and only if the
span is not directly preceded by a negation the quote leaves out. So matching may forgive
whitespace, punctuation, case and minor spelling, but never "no chest pain" for "chest
pain", "1.5 mg" for "15 mg", "ten" for "two" or "left knee" for "right knee".

A leading turn marker ("[T3]"), and in a transcript a leading speaker label the transcript
uses, are ignored, as are trailing ellipses; a quote with ellipses in the middle matches
only if every part matches, in order, in the same turn. A transcript quote must sit within
one turn.

The evidence returned is built by `flags.transcript_evidence` / `flags.note_evidence`, so
its text is exactly the source text at its offsets.
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

from scribecheck.flags import NoteEvidence, TranscriptEvidence, note_evidence, transcript_evidence
from scribecheck.note import Note
from scribecheck.rules.lexicon import find_drugs
from scribecheck.transcript import Transcript

FUZZY_THRESHOLD = 0.90
MIN_CHARS = 8
MIN_EXACT_CHARS = 4
ABBREVIATIONS = frozenset(
    {
        "bd",
        "bid",
        "tds",
        "tid",
        "qds",
        "qid",
        "od",
        "prn",
        "nocte",
        "mane",
        "stat",
        "nkda",
        "nka",
        "nkfa",
        "sob",
        "nad",
        "po",
        "iv",
        "im",
        "sc",
        "sl",
        "pr",
        "inh",
    }
)
"""Single short tokens that may verify on their own by exact match."""

_TURN_MARK = re.compile(r"^\s*\[?T\d+\]?\s*")
_LABELLED = re.compile(r"^\s*([^:\n]{1,25}):\s+")
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+", re.UNICODE)
_THOUSANDS = re.compile(r"\d{1,3}(?:,\d{3})+")
_CONTRACTION = re.compile(r"n['’]t\b")
_BARE_CONTRACTION = re.compile(
    r"\b(does|do|did|is|was|were|has|have|had|are|could|would|should|wo|ca|must|need)nt\b"
)
_NEGATIONS = frozenset(
    {
        "no",
        "not",
        "denies",
        "denied",
        "deny",
        "without",
        "never",
        "none",
        "negative",
        "nil",
        "nkda",
        "nka",
        "nkfa",
        "nor",
        "cannot",
    }
)
_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "fifteen": "15",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "hundred": "100",
    "thousand": "1000",
    "half": "0.5",
    "once": "1x",
    "twice": "2x",
    "thrice": "3x",
}
_SIDES = frozenset({"left", "right", "bilateral", "both", "l", "r", "lt", "rt"})
_UNITS = frozenset(
    {
        "mg",
        "mcg",
        "ug",
        "µg",
        "g",
        "gram",
        "grams",
        "kg",
        "ml",
        "l",
        "litre",
        "litres",
        "unit",
        "units",
        "iu",
        "puff",
        "puffs",
        "drop",
        "drops",
        "tab",
        "tabs",
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "mmol",
        "mol",
        "percent",
    }
)


def find_in_transcript(transcript: Transcript, quote: str) -> TranscriptEvidence | None:
    """The span of one turn that `quote` matches, or None."""
    labels = {t.label.casefold() for t in transcript.turns if t.label}
    labels |= {t.speaker.casefold() for t in transcript.turns}
    for candidate in _candidates(quote, labels):
        best: tuple[float, int, int, int] | None = None
        for turn in transcript.turns:
            found = _match(turn.text, candidate)
            if found is not None and (best is None or found[0] > best[0]):
                best = (found[0], turn.number, found[1], found[2])
                if found[0] == 1.0:
                    break
        if best is not None:
            return transcript_evidence(transcript, best[1], best[2], best[3])
    return None


def find_in_note(note: Note, quote: str) -> NoteEvidence | None:
    """The span of the note that `quote` matches, or None."""
    for candidate in _candidates(quote, set()):
        found = _match(note.text, candidate)
        if found is not None:
            return note_evidence(note, found[1], found[2])
    return None


def _candidates(quote: str, labels: set[str]) -> list[str]:
    """The quote, then without a leading turn marker and known speaker label."""
    text = _ELLIPSIS.sub(" … ", quote.strip()).strip(" …")
    variants = [text]
    unmarked = _TURN_MARK.sub("", text, count=1)
    labelled = _LABELLED.match(unmarked)
    if labelled and labelled.group(1).strip().casefold() in labels:
        unmarked = unmarked[labelled.end() :]
    if unmarked and unmarked != text:
        variants.append(unmarked)
    return list(dict.fromkeys(variants))


def _match(source: str, quote: str) -> tuple[float, int, int] | None:
    """Every ellipsis-separated part of `quote`, in order, in `source`: (score, start, end)."""
    parts = [p.strip() for p in quote.split("…") if p.strip()]
    if not parts:
        return None
    score, start, end, pos = 1.0, -1, -1, 0
    for part in parts:
        found = _locate(source[pos:], part)
        if found is None:
            return None
        score = min(score, found[0])
        if start < 0:
            start = pos + found[1]
        end = pos + found[2]
        pos = end
    return score, start, end


def _normalise(text: str) -> tuple[str, list[int]]:
    """Normalised text and, for each of its characters, the offset in `text`."""
    chars: list[str] = []
    index: list[int] = []
    space = True
    for i, ch in enumerate(text):
        if ch.isalnum():
            chars.append(ch.casefold())
            index.append(i)
            space = False
        elif (
            ch in ".,"
            and chars
            and chars[-1].isdigit()
            and i + 1 < len(text)
            and text[i + 1].isdigit()
        ):
            chars.append(".")  # a decimal point (or thousands separator) stays
            index.append(i)
        elif not space and (ch.isspace() or ch in "-/"):
            chars.append(" ")
            index.append(i)
            space = True
    while chars and chars[-1] == " ":
        chars.pop()
        index.pop()
    return "".join(chars), index


def _prepare(text: str) -> str:
    lowered = text.casefold().replace("’", "'")
    lowered = _CONTRACTION.sub(" not", lowered)
    lowered = _BARE_CONTRACTION.sub(r"\1 not", lowered)
    return lowered.replace("cannot", "can not")


def critical_tokens(text: str) -> tuple[Counter[str], ...]:
    """Numbers (and number words), units, negation words, side words and drug names."""
    prepared = _prepare(text)
    tokens = _TOKEN.findall(prepared)
    numbers: Counter[str] = Counter()
    for token in tokens:
        if token[0].isdigit():
            numbers[
                token.replace(",", "") if _THOUSANDS.fullmatch(token) else token.replace(",", ".")
            ] += 1
        elif token in _NUMBER_WORDS:
            numbers[_NUMBER_WORDS[token]] += 1
    units = Counter(t for t in tokens if t in _UNITS)
    negations = Counter(t for t in tokens if t in _NEGATIONS)
    sides = Counter(t for t in tokens if t in _SIDES)
    drugs = Counter(" ".join(sorted(d.generic)) for d in find_drugs(text) if not d.fuzzy)
    return numbers, units, negations, sides, drugs


def _negated_before(source: str, start: int, quote: str) -> bool:
    """The word just before the span is a negation the quote itself does not start with."""
    before = _TOKEN.findall(_prepare(source[max(0, start - 40) : start]))
    if not before or before[-1] not in _NEGATIONS:
        return False
    own = _TOKEN.findall(_prepare(quote))
    return not (own and own[0] in _NEGATIONS)


def _accept(source: str, span: tuple[int, int], quote: str) -> bool:
    return critical_tokens(source[span[0] : span[1]]) == critical_tokens(quote) and not (
        _negated_before(source, span[0], quote)
    )


def _whole_token(s: str, q: str, start: int = 0) -> int:
    """Where `q` occurs in `s` on token boundaries, from `start`, or -1."""
    at = s.find(q, start)
    while at >= 0:
        end = at + len(q)
        if (at == 0 or s[at - 1] == " ") and (end == len(s) or s[end] == " "):
            return at
        at = s.find(q, at + 1)
    return -1


def _locate(source: str, quote: str) -> tuple[float, int, int] | None:
    """(score, start, end) of `quote` in `source`, or None."""
    q, _ = _normalise(quote)
    if not q:
        return None
    s, index = _normalise(source)
    if not s:
        return None
    long_enough = len(q) >= MIN_EXACT_CHARS or q in ABBREVIATIONS
    at = _whole_token(s, q)
    while at >= 0 and long_enough:
        span = (index[at], index[at + len(q) - 1] + 1)
        if _accept(source, span, quote):
            return 1.0, *span
        at = _whole_token(s, q, at + 1)
    if len(q) < MIN_CHARS or len(q) > len(s) * 1.1:
        return None
    best: tuple[float, int, int] | None = None
    width = len(q)
    starts = [0, *(i + 1 for i, c in enumerate(s) if c == " ")]
    matcher = SequenceMatcher(autojunk=False)
    matcher.set_seq2(q)
    for start in starts:
        ends = set()
        for size in (width, int(width * 0.95), int(width * 1.05)):
            end = min(start + size, len(s))
            following = s.find(" ", end)  # grow to the end of the token
            ends.add(len(s) if following < 0 else following)
            previous = s.rfind(" ", start, end)  # or shrink to the last whole token
            if previous > start:
                ends.add(previous)
        for end in ends:
            window = s[start:end]
            if not window.strip():
                continue
            matcher.set_seq1(window)
            if matcher.real_quick_ratio() < FUZZY_THRESHOLD:
                continue
            if matcher.quick_ratio() < FUZZY_THRESHOLD:
                continue
            ratio = matcher.ratio()
            if ratio >= FUZZY_THRESHOLD and (best is None or ratio > best[0]):
                span = (index[start], index[end - 1] + 1)
                if _accept(source, span, quote):
                    best = (ratio, *span)
    return best
