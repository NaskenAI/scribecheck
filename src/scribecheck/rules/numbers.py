"""Numbers written as digits, words or fractions.

"five hundred" = 500, "twenty-five" = 25, "one and a half" = 1.5, "1,000" = 1000,
"½" = 0.5. "half" counts only in "half a/an ..." (0.5), "a half" and "one half".
Spoken decimals count ("zero point five" = 0.5, "point five" = 0.5). Ambiguous uses
are skipped: digits glued to a preceding letter ("B12", "HbA1c"), digits joined to
another number by "/" or a comma that is not a thousands separator ("1/2", "1,5"),
unclear spoken decimals ("one point twenty-five") and a bare "one" used as a pronoun
("one of the tablets", "which one").
"""

from __future__ import annotations

import re
from decimal import Decimal

from scribecheck.rules._match import Match, non_overlapping


class NumberMatch(Match):
    value: Decimal


_SMALL = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_FRACTION_WORDS = {"half": Decimal("0.5"), "quarter": Decimal("0.25")}
_VULGAR = {"½": Decimal("0.5"), "¼": Decimal("0.25"), "¾": Decimal("0.75")}
_ONE_DETERMINERS = frozenset({"the", "this", "that", "which", "any", "no", "every", "each"})
# "point" after one of these is the noun ("at that point two nurses came").
_POINT_NOUN_CONTEXT = _ONE_DETERMINERS | {"a", "some", "what", "at", "to", "on"}

_DIGITS = re.compile(
    r"(?<![A-Za-z0-9_.,/])"
    r"(?:(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?(?P<frac>[½¼¾])?|(?P<alone>[½¼¾]))"
    r"(?!\d|[,/]\d)"
)
_WORD = re.compile(r"(?<![A-Za-z])[A-Za-z]+(?:-[A-Za-z]+)*(?![A-Za-z])")
_AND_FRACTION = re.compile(r"\s+and\s+a\s+(half|quarter)(?![A-Za-z])", re.IGNORECASE)
_DIGIT_WORDS = {word: value for word, value in _SMALL.items() if value < 10}
# After "point": single digit words ("point two five") or digits ("point 5").
_DECIMAL_TAIL = re.compile(
    r"(?P<words>(?:[ \t]+(?:" + "|".join(_DIGIT_WORDS) + r"))+)(?![A-Za-z-])"
    r"|[ \t]+(?P<digits>\d+)(?![\d,./])",
    re.IGNORECASE,
)


class _Token:
    __slots__ = ("word", "start", "end")

    def __init__(self, word: str, start: int, end: int) -> None:
        self.word = word
        self.start = start
        self.end = end


def _small_value(word: str) -> tuple[int, bool] | None:
    """Value of a number word below 100, and whether it is a bare multiple of ten."""
    if word in _SMALL:
        return _SMALL[word], False
    if word in _TENS:
        return _TENS[word], True
    tens, _, unit = word.partition("-")
    if tens in _TENS and unit in _SMALL and 1 <= _SMALL[unit] <= 9:
        return _TENS[tens] + _SMALL[unit], False
    return None


def _cardinal(text: str, tokens: list[_Token], i: int) -> tuple[int, int] | None:
    """Parse a cardinal number starting at token i; return (value, index after it)."""

    def adjacent(j: int) -> bool:
        return j < len(tokens) and not text[tokens[j - 1].end : tokens[j].start].strip()

    total = 0
    current = 0
    last: str | None = None  # "small", "tens", "hundred", "thousand"
    has_hundred = False
    j = i
    if tokens[i].word == "a" and adjacent(i + 1) and tokens[i + 1].word in ("hundred", "thousand"):
        current, last, j = 1, "small", i + 1
    while j < len(tokens):
        if j > i and not adjacent(j):
            break
        word = tokens[j].word
        if word == "and":
            following = _small_value(tokens[j + 1].word) if adjacent(j + 1) else None
            if last in ("hundred", "thousand") and following is not None:
                j += 1
                continue
            break
        small = _small_value(word)
        if small is not None:
            value, is_tens = small
            if last in (None, "hundred", "thousand"):
                current += value
                last = "tens" if is_tens else "small"
            elif last == "tens" and value < 10 and word in _SMALL:
                current += value
                last = "small"
            else:
                break
        elif word == "hundred" and last in ("small", "tens") and not has_hundred and current < 100:
            current *= 100
            has_hundred = True
            last = "hundred"
        elif word == "thousand" and last in ("small", "tens", "hundred") and total == 0:
            total = current * 1000
            current = 0
            has_hundred = False
            last = "thousand"
        else:
            break
        j += 1
    if last is None:
        return None
    return total + current, j


def find_numbers(text: str) -> tuple[NumberMatch, ...]:
    """Find every number in `text`, left to right, without overlaps."""
    candidates: list[NumberMatch] = []

    for match in _DIGITS.finditer(text):
        if match.group("alone"):
            value = _VULGAR[match.group("alone")]
        else:
            value = Decimal(match.group("int").replace(",", "") + (match.group("dec") or ""))
            if match.group("frac"):
                value += _VULGAR[match.group("frac")]
        candidates.append(_with_and_fraction(text, match.start(), match.end(), value))

    tokens = [_Token(m.group().lower(), m.start(), m.end()) for m in _WORD.finditer(text)]
    for i, token in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        joined = nxt is not None and not text[token.end : nxt.start].strip()
        if (
            token.word in ("a", "one")
            and joined
            and nxt is not None
            and nxt.word in _FRACTION_WORDS
        ):
            if token.word == "one" and nxt.word != "half":
                continue
            candidates.append(_number(text, token.start, nxt.end, _FRACTION_WORDS[nxt.word]))
            continue
        if token.word == "half" and joined and nxt is not None and nxt.word in ("a", "an"):
            candidates.append(_number(text, token.start, token.end, Decimal("0.5")))
            continue
        parsed = _cardinal(text, tokens, i)
        if parsed is None:
            continue
        value_int, after = parsed
        end = tokens[after - 1].end
        if token.word == "one" and after == i + 1 and _is_pronoun_one(text, tokens, i):
            continue
        candidates.append(_with_and_fraction(text, token.start, end, Decimal(value_int)))

    for i, token in enumerate(tokens):
        if token.word == "point":
            previous = tokens[i - 1] if i > 0 else None
            noun = (
                previous is not None
                and not text[previous.end : token.start].strip()
                and previous.word in _POINT_NOUN_CONTEXT
            )
            candidates = _apply_spoken_decimal(text, token, candidates, noun=noun)
    return non_overlapping(candidates)


def parse_number(text: str) -> Decimal | None:
    """The value of `text` if the whole string (ignoring outer spaces) is one number."""
    stripped = text.strip()
    matches = find_numbers(stripped)
    if len(matches) == 1 and matches[0].start == 0 and matches[0].end == len(stripped):
        return matches[0].value
    return None


def _number(text: str, start: int, end: int, value: Decimal) -> NumberMatch:
    return NumberMatch(start=start, end=end, text=text[start:end], value=value)


def _with_and_fraction(text: str, start: int, end: int, value: Decimal) -> NumberMatch:
    extra = _AND_FRACTION.match(text, end)
    if extra:
        value += _FRACTION_WORDS[extra.group(1).lower()]
        end = extra.end()
    return _number(text, start, end, value)


def _apply_spoken_decimal(
    text: str, point: _Token, candidates: list[NumberMatch], *, noun: bool
) -> list[NumberMatch]:
    """Join "<whole> point <digits>" into one number; drop numbers around an unclear one.

    When `noun` is set ("that point"), "point" is not a decimal point and nothing changes.
    """
    if noun:
        return candidates
    before = [
        c for c in candidates if c.end <= point.start and not text[c.end : point.start].strip()
    ]
    after = [
        c for c in candidates if c.start >= point.end and not text[point.end : c.start].strip()
    ]
    tail = _DECIMAL_TAIL.match(text, point.end)
    if tail is None:
        unclear = {id(c) for c in before + after}
        return [c for c in candidates if id(c) not in unclear]
    if tail.group("digits"):
        digits = tail.group("digits")
    else:
        digits = "".join(str(_DIGIT_WORDS[w.lower()]) for w in tail.group("words").split())
    whole = min(before, key=lambda c: c.start, default=None)
    if whole is not None and whole.value != whole.value.to_integral_value():
        whole = None
    start = whole.start if whole is not None else point.start
    whole_part = int(whole.value) if whole is not None else 0
    value = Decimal(f"{whole_part}.{digits}")
    return [*candidates, _number(text, start, tail.end(), value)]


def _is_pronoun_one(text: str, tokens: list[_Token], i: int) -> bool:
    before = tokens[i - 1] if i > 0 else None
    after = tokens[i + 1] if i + 1 < len(tokens) else None
    if before and not text[before.end : tokens[i].start].strip():
        if before.word in _ONE_DETERMINERS:
            return True
    return bool(after and not text[tokens[i].end : after.start].strip() and after.word == "of")
