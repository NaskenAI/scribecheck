"""Built-in ConText rules: is a target negated, possible, historical, family or hypothetical?

The caller supplies the target spans (from the lexicons); this module never decides
what counts as a symptom or a drug. A cue applies to a target when it points the
right way (forward cues come before the target, backward cues after it), the two sit
in the same sentence, and the words between them contain no terminator ("but",
"however", ...) and, not counting other targets, number no more than `max_words`. A
colon ends the scope unless it directly follows the cue ("Mother: diabetes"). A comma
ends it too, unless only other targets and list words lie between: with fever, cough and
headache all passed as targets, "no fever, cough or headache" negates all three,
while "No relief, cough persists" leaves cough alone. In "no fever but has a cough"
only fever is negated. A family cue stops at a word that brings the patient back
("he", "she", "I") or a reporting verb, so "mum says he has a fever" is not family.

Pseudo-cues ("no change", "no idea", "prior to") are matched alongside the real
cues so that, keeping the leftmost then longest match, they hide the shorter cue
inside them and are then dropped.

Limits: English only, and the cue lists are short and favour precision. Simple-past
"had" counts as historical even for recent events ("I had a fever last night"), but
present-perfect "I've had" and "has not had" do not.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, NamedTuple

from scribecheck.rules._context import has_sentence_break
from scribecheck.rules._match import Match, non_overlapping

Attribute = Literal["negated", "possible", "historical", "family", "hypothetical"]
Direction = Literal["forward", "backward"]
"""forward: the cue comes before the target; backward: after it."""

DEFAULT_MAX_WORDS = 6


class Cue(Match):
    """A context cue found in the text."""

    attribute: Attribute
    direction: Direction


class TargetContext(Match):
    """A target span with the cues whose scope covers it."""

    cues: tuple[Cue, ...] = ()

    @property
    def attributes(self) -> frozenset[Attribute]:
        return frozenset(cue.attribute for cue in self.cues)

    @property
    def negated(self) -> bool:
        return "negated" in self.attributes

    @property
    def possible(self) -> bool:
        return "possible" in self.attributes

    @property
    def historical(self) -> bool:
        return "historical" in self.attributes

    @property
    def family(self) -> bool:
        return "family" in self.attributes

    @property
    def hypothetical(self) -> bool:
        return "hypothetical" in self.attributes


class _Rule(NamedTuple):
    pattern: re.Pattern[str]
    attribute: Attribute | None  # None for a pseudo-cue
    direction: Direction


def _words(*phrases: str) -> str:
    """Alternation of phrases, longest first, with flexible spacing and apostrophes."""
    ordered = sorted(phrases, key=len, reverse=True)
    escaped = (
        r"\s+".join(re.escape(word).replace("'", "['’]") for word in phrase.split())
        for phrase in ordered
    )
    return rf"(?<![A-Za-z])(?:{'|'.join(escaped)})(?![A-Za-z])"


def _rule(attribute: Attribute | None, direction: Direction, *phrases: str) -> _Rule:
    return _Rule(re.compile(_words(*phrases), re.IGNORECASE), attribute, direction)


_RULES: tuple[_Rule, ...] = (
    _rule(
        None,
        "forward",
        "no change",
        "no increase",
        "no further",
        "not only",
        "not necessarily",
        "no doubt",
        "no idea",
        "no improvement",
        "no better",
        "no worse",
        "no response",
        "no relief",
        "not keen",
        "nil by mouth",
        "prior to",
        "history of present illness",
        "history of presenting illness",
        "history of presenting complaint",
    ),
    _rule(
        "negated",
        "forward",
        "no",
        "not",
        "nil",
        "never",
        "never had",
        "deny",
        "denies",
        "denied",
        "denying",
        "without",
        "negative for",
        "free of",
        "absence of",
        "no sign of",
        "no signs of",
        "no evidence of",
        "ruled out",
        "don't have",
        "doesn't have",
        "didn't have",
        "haven't had",
        "hasn't had",
        "haven't got",
        "hasn't got",
        "haven't noticed",
        "didn't notice",
    ),
    _rule("negated", "backward", "is ruled out", "was ruled out", "not present", "absent"),
    _rule(
        "possible",
        "forward",
        "possible",
        "possibly",
        "probable",
        "probably",
        "likely",
        "suspected",
        "suspicious for",
        "query",
        "questionable",
        "could be",
        "might be",
        "may be",
        "maybe",
        "rule out",
        "can't rule out",
        "cannot rule out",
        "not sure if",
        "not sure whether",
        "suggestive of",
    ),
    _rule(
        "possible",
        "backward",
        "is possible",
        "is likely",
        "is suspected",
        "cannot be excluded",
        "can't be excluded",
        "cannot be ruled out",
        "can't be ruled out",
    ),
    _Rule(re.compile(r"(?:^|(?<=\s))\?(?=[A-Za-z])"), "possible", "forward"),
    _rule(
        "historical",
        "forward",
        "history of",
        "past history of",
        "h/o",
        "hx of",
        "previous",
        "previously",
        "prior",
        "had",
    ),
    _rule("historical", "backward", "years ago", "as a child", "in the past"),
    _Rule(
        re.compile(r"(?<![A-Za-z])in\s+(?:19|20)\d\d(?!\d)", re.IGNORECASE),
        "historical",
        "backward",
    ),
    _rule(
        "family",
        "forward",
        "mother",
        "mum",
        "mom",
        "father",
        "dad",
        "brother",
        "sister",
        "sibling",
        "siblings",
        "son",
        "daughter",
        "grandmother",
        "grandfather",
        "grandma",
        "grandpa",
        "grandad",
        "granddad",
        "aunt",
        "uncle",
        "cousin",
        "family history of",
        "FH of",
        "FH",
    ),
    _rule("family", "backward", "runs in the family", "runs in my family"),
    _rule(
        "hypothetical",
        "forward",
        "if",
        "in case",
        "should you",
        "should it",
        "should the",
        "come back if",
        "return if",
        "watch out for",
        "look out for",
    ),
)

_TERMINATOR = re.compile(
    _words(
        "but",
        "however",
        "although",
        "though",
        "except",
        "apart from",
        "aside from",
        "yet",
        "whereas",
    ),
    re.IGNORECASE,
)
_PERFECT_BEFORE_HAD = re.compile(
    r"(?:(?<![A-Za-z])(?:have|has)(?:\s+(?:you|he|she|they|we|it|i))?|['’](?:ve|s)|n['’]t)"
    r"(?:\s+(?:not|never|ever|also|just|already))?\s+$",
    re.IGNORECASE,
)
_WORD = re.compile(r"[\w'’]+(?:[.\-][\w'’]+)*")
_LIST_WORDS = frozenset({"and", "or", "nor", "any", "a", "an", "no", "the", "your"})
_LABEL_COLON = re.compile(r"[ \t]*:")
_FAMILY_STOP = re.compile(
    _words(
        "he", "she", "him", "her", "I", "I'm", "I've", "me", "you", "patient", "pt",
        "says", "said", "thinks", "thought", "noticed", "reports", "reported",
        "brought", "brings", "mentioned", "worried", "concerned",
    ),
    re.IGNORECASE,
)  # fmt: skip


class _Found(Match):
    attribute: Attribute | None
    direction: Direction


def find_cues(text: str) -> tuple[Cue, ...]:
    """Every context cue in `text`, left to right, without overlaps."""
    return tuple(_cue(found) for found in _find(text) if found.attribute is not None)


def classify(
    text: str,
    targets: Sequence[tuple[int, int]],
    *,
    max_words: int = DEFAULT_MAX_WORDS,
) -> tuple[TargetContext, ...]:
    """Classify each target span in `text`, returning results in the order given.

    Raises ValueError for a span outside `0 <= start < end <= len(text)`.
    """
    for start, end in targets:
        if not 0 <= start < end <= len(text):
            raise ValueError(f"target span ({start}, {end}) is not inside the text")
    found = [f for f in _find(text) if f.attribute is not None]
    results = []
    for start, end in targets:
        cues = tuple(_cue(f) for f in found if _in_scope(text, f, start, end, targets, max_words))
        results.append(TargetContext(start=start, end=end, text=text[start:end], cues=cues))
    return tuple(results)


def _find(text: str) -> tuple[_Found, ...]:
    candidates = []
    for rule in _RULES:
        for match in rule.pattern.finditer(text):
            if rule.attribute == "historical" and match.group().lower() == "had":
                if _PERFECT_BEFORE_HAD.search(text, max(0, match.start() - 40), match.start()):
                    continue
            candidates.append(
                _Found(
                    start=match.start(),
                    end=match.end(),
                    text=match.group(),
                    attribute=rule.attribute,
                    direction=rule.direction,
                )
            )
    return non_overlapping(candidates)


def _in_scope(
    text: str,
    cue: _Found,
    start: int,
    end: int,
    targets: Sequence[tuple[int, int]],
    max_words: int,
) -> bool:
    if cue.direction == "forward":
        if cue.end > start:
            return False
        gap_start, gap_end = cue.end, start
    else:
        if cue.start < end:
            return False
        gap_start, gap_end = end, cue.start
    gap = text[gap_start:gap_end]
    if has_sentence_break(text, gap_start, gap_end) or _TERMINATOR.search(gap):
        return False
    if cue.attribute == "family" and _FAMILY_STOP.search(gap):
        return False
    if cue.direction == "forward":
        # A label colon straight after the cue ("Mother: diabetes", "Denies: fever").
        label = _LABEL_COLON.match(text, gap_start, gap_end)
        if label is not None:
            gap_start = label.end()
    if ":" in text[gap_start:gap_end]:
        return False
    words = _words_outside(text, gap_start, gap_end, targets)
    if "," in text[gap_start:gap_end] and not all(word.lower() in _LIST_WORDS for word in words):
        return False
    return len(words) <= max_words


def _words_outside(
    text: str, start: int, end: int, targets: Sequence[tuple[int, int]]
) -> list[str]:
    """Words in text[start:end] that are not part of any target."""
    rest = list(text[start:end])
    for target_start, target_end in targets:
        for index in range(max(start, target_start), min(end, target_end)):
            rest[index - start] = " "
    return _WORD.findall("".join(rest))


def _cue(found: _Found) -> Cue:
    assert found.attribute is not None
    return Cue(
        start=found.start,
        end=found.end,
        text=found.text,
        attribute=found.attribute,
        direction=found.direction,
    )
