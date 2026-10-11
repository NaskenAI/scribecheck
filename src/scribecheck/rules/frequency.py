"""Dosing frequency: times per period, interval, as-needed flag and time of day.

Covers UK and US abbreviations (OD, BD/BID, TDS/TID, QDS/QID, mane, nocte, QHS,
PRN, qNh), phrases ("twice a day", "every 8 hours", "once weekly"), and Indian
prescribing forms: HS, SOS and the morning-afternoon-night pattern ("1-0-1").
Timing phrases ("at night", "in the morning") count only in a sentence with a dose
or a dosing verb, since they often describe symptoms. OD is once daily only outside
eye context; near eye words or OS/OU it is the right eye (see side.py), and near
"drops" without an eye word it is unclear and gives no match. Ranges ("every 4-6
hours") and durations ("for five days") give no match.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from decimal import Decimal
from typing import Literal

from scribecheck.rules._context import eye_context, has_dosing_context
from scribecheck.rules._match import Comparison, Match, non_overlapping

Period = Literal["daily", "weekly"]
Timing = Literal["morning", "afternoon", "night"]


class FrequencyMatch(Match):
    times_per_period: int | None = None
    period: Period | None = None
    interval_hours: Decimal | None = None
    as_needed: bool = False
    timing: tuple[Timing, ...] = ()


_Fields = dict[str, object]

_ABBREVIATIONS: dict[str, _Fields] = {
    "od": {"times_per_period": 1, "period": "daily"},
    "daily": {"times_per_period": 1, "period": "daily"},
    "bd": {"times_per_period": 2, "period": "daily"},
    "bid": {"times_per_period": 2, "period": "daily"},
    "tds": {"times_per_period": 3, "period": "daily"},
    "tid": {"times_per_period": 3, "period": "daily"},
    "qds": {"times_per_period": 4, "period": "daily"},
    "qid": {"times_per_period": 4, "period": "daily"},
    "mane": {"times_per_period": 1, "period": "daily", "timing": ("morning",)},
    "nocte": {"times_per_period": 1, "period": "daily", "timing": ("night",)},
    "qhs": {"times_per_period": 1, "period": "daily", "timing": ("night",)},
    "hs": {"times_per_period": 1, "period": "daily", "timing": ("night",)},
    "prn": {"as_needed": True},
    "sos": {"as_needed": True},
    "weekly": {"times_per_period": 1, "period": "weekly"},
}
_COUNT_WORDS = {
    "once": 1,
    "twice": 2,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "eight": 8,
    "twelve": 12,
    "twenty-four": 24,
}
_B = r"(?<![A-Za-z0-9])"  # word start
_E = r"(?![A-Za-z0-9])"  # word end
_COUNT = r"(?P<count>once|twice|\d+\s+times|(?:one|two|three|four|five|six)\s+times)"
_HOURS = r"(?P<hours>\d+|one|two|three|four|five|six|eight|twelve|twenty-four)"

_Handler = Callable[[re.Match[str]], "_Fields | None"]


def _abbreviation(match: re.Match[str]) -> _Fields:
    return dict(_ABBREVIATIONS[match.group().lower()])


def _count(match: re.Match[str]) -> int:
    word = match.group("count").split()[0].lower()
    return int(word) if word.isdigit() else _COUNT_WORDS[word]


def _per_day(match: re.Match[str]) -> _Fields | None:
    count = _count(match)
    return {"times_per_period": count, "period": "daily"} if count > 0 else None


def _per_week(match: re.Match[str]) -> _Fields | None:
    count = _count(match)
    return {"times_per_period": count, "period": "weekly"} if count > 0 else None


def _interval(match: re.Match[str]) -> _Fields | None:
    raw = match.group("hours").lower()
    hours = int(raw) if raw.isdigit() else _COUNT_WORDS[raw]
    if hours <= 0:
        return None
    fields: _Fields = {"interval_hours": Decimal(hours)}
    if 24 % hours == 0:
        fields.update(times_per_period=24 // hours, period="daily")
    return fields


_SLOT_NAMES: tuple[Timing, ...] = ("morning", "afternoon", "night")


def _slots(match: re.Match[str]) -> _Fields | None:
    doses = [int(match.group(name)) for name in ("m", "a", "n")]
    timing = tuple(name for name, dose in zip(_SLOT_NAMES, doses, strict=True) if dose)
    if not timing:
        return None
    return {"times_per_period": len(timing), "period": "daily", "timing": timing}


def _as_needed(_: re.Match[str]) -> _Fields:
    return {"as_needed": True}


_PATTERNS: tuple[tuple[re.Pattern[str], _Handler], ...] = (
    (re.compile(_B + "(?:" + "|".join(_ABBREVIATIONS) + ")" + _E, re.IGNORECASE), _abbreviation),
    (re.compile(_B + _COUNT + r"\s+(?:a|per)\s+day" + _E, re.IGNORECASE), _per_day),
    (re.compile(_B + _COUNT + r"\s+daily" + _E, re.IGNORECASE), _per_day),
    (re.compile(_B + _COUNT + r"\s+(?:a|per)\s+week" + _E, re.IGNORECASE), _per_week),
    (re.compile(_B + _COUNT + r"\s+weekly" + _E, re.IGNORECASE), _per_week),
    (re.compile(_B + r"q(?P<hours>\d+)h" + _E, re.IGNORECASE), _interval),
    (re.compile(_B + r"every\s+" + _HOURS + r"\s+hours?" + _E, re.IGNORECASE), _interval),
    (re.compile(_B + r"as\s+(?:needed|required)" + _E, re.IGNORECASE), _as_needed),
    # Morning-afternoon-night: three single digits, with no letter, digit, dash, "/"
    # or decimal point attached on either side ("1-0-2025", "v1-0-1", "1-0-1/2").
    (
        re.compile(r"(?<![\w.,/\-–])(?P<m>\d)[-–](?P<a>\d)[-–](?P<n>\d)(?![\w/\-–]|[.,]\d)"),
        _slots,
    ),
)


# Timing phrases: matched only with dosing context in the same sentence.
_TIMING: tuple[tuple[re.Pattern[str], _Fields], ...] = (
    (
        re.compile(_B + r"(?:at\s+(?:night|bedtime)|every\s+night)" + _E, re.IGNORECASE),
        {"times_per_period": 1, "period": "daily", "timing": ("night",)},
    ),
    (
        re.compile(_B + r"(?:in\s+the\s+morning|every\s+morning)" + _E, re.IGNORECASE),
        {"times_per_period": 1, "period": "daily", "timing": ("morning",)},
    ),
)


def _timing_handler(fields: _Fields) -> _Handler:
    return lambda match: dict(fields) if has_dosing_context(match.string, *match.span()) else None


def find_frequencies(text: str) -> tuple[FrequencyMatch, ...]:
    """Find every dosing frequency in `text`, left to right, without overlaps."""
    patterns = _PATTERNS + tuple((p, _timing_handler(f)) for p, f in _TIMING)
    candidates = []
    for pattern, handler in patterns:
        for match in pattern.finditer(text):
            if match.group().lower() == "od" and eye_context(text, *match.span()) != "none":
                continue
            fields = handler(match)
            if fields is not None:
                candidates.append(
                    FrequencyMatch.model_validate(
                        {"start": match.start(), "end": match.end(), "text": match.group()} | fields
                    )
                )
    return non_overlapping(candidates)


def compare_frequencies(a: FrequencyMatch, b: FrequencyMatch) -> Comparison:
    """Compare two frequencies; without a times value on both sides, not comparable."""
    if a.times_per_period is None or b.times_per_period is None:
        return "not_comparable"
    if (a.times_per_period, a.period, a.as_needed) != (b.times_per_period, b.period, b.as_needed):
        return "different"
    if a.interval_hours is not None and b.interval_hours is not None:
        if a.interval_hours != b.interval_hours:
            return "different"
    if a.timing and b.timing and a.timing != b.timing:
        return "different"
    return "equal"
