from __future__ import annotations

from decimal import Decimal

import pytest
from _spans import assert_round_trip

from scribecheck.rules.frequency import compare_frequencies, find_frequencies

D = "daily"
W = "weekly"


@pytest.mark.parametrize(
    ("text", "matched", "times", "period", "interval", "as_needed", "timing"),
    [
        ("OD", "OD", 1, D, None, False, ()),
        ("500 mg od", "od", 1, D, None, False, ()),
        ("once daily", "once daily", 1, D, None, False, ()),
        ("once a day", "once a day", 1, D, None, False, ()),
        ("Daily", "Daily", 1, D, None, False, ()),
        ("BD", "BD", 2, D, None, False, ()),
        ("bd", "bd", 2, D, None, False, ()),
        ("BID", "BID", 2, D, None, False, ()),
        ("twice daily", "twice daily", 2, D, None, False, ()),
        ("twice a day", "twice a day", 2, D, None, False, ()),
        ("TDS", "TDS", 3, D, None, False, ()),
        ("TID", "TID", 3, D, None, False, ()),
        ("three times a day", "three times a day", 3, D, None, False, ()),
        ("three times daily", "three times daily", 3, D, None, False, ()),
        ("3 times a day", "3 times a day", 3, D, None, False, ()),
        ("2 times daily", "2 times daily", 2, D, None, False, ()),
        ("QDS", "QDS", 4, D, None, False, ()),
        ("QID", "QID", 4, D, None, False, ()),
        ("four times a day", "four times a day", 4, D, None, False, ()),
        ("mane", "mane", 1, D, None, False, ("morning",)),
        ("nocte", "nocte", 1, D, None, False, ("night",)),
        ("QHS", "QHS", 1, D, None, False, ("night",)),
        ("HS", "HS", 1, D, None, False, ("night",)),
        ("PRN", "PRN", None, None, None, True, ()),
        ("prn", "prn", None, None, None, True, ()),
        ("SOS", "SOS", None, None, None, True, ()),
        ("as needed", "as needed", None, None, None, True, ()),
        ("as required", "as required", None, None, None, True, ()),
        ("q8h", "q8h", 3, D, "8", False, ()),
        ("Q12H", "Q12H", 2, D, "12", False, ()),
        ("q4h", "q4h", 6, D, "4", False, ()),
        ("q36h", "q36h", None, None, "36", False, ()),
        ("every 8 hours", "every 8 hours", 3, D, "8", False, ()),
        ("every eight hours", "every eight hours", 3, D, "8", False, ()),
        ("every 6 hours", "every 6 hours", 4, D, "6", False, ()),
        ("weekly", "weekly", 1, W, None, False, ()),
        ("once a week", "once a week", 1, W, None, False, ()),
        ("once weekly", "once weekly", 1, W, None, False, ()),
        ("twice weekly", "twice weekly", 2, W, None, False, ()),
        ("1-0-1", "1-0-1", 2, D, None, False, ("morning", "night")),
        ("1–0–1", "1–0–1", 2, D, None, False, ("morning", "night")),
        ("1-1-1", "1-1-1", 3, D, None, False, ("morning", "afternoon", "night")),
        ("0-0-1", "0-0-1", 1, D, None, False, ("night",)),
        ("1-0-0", "1-0-0", 1, D, None, False, ("morning",)),
        ("2-0-2", "2-0-2", 2, D, None, False, ("morning", "night")),
        ("take 1-0-1.", "1-0-1", 2, D, None, False, ("morning", "night")),
        ("(1-0-1)", "1-0-1", 2, D, None, False, ("morning", "night")),
    ],
)
def test_frequencies(
    text: str,
    matched: str,
    times: int | None,
    period: str | None,
    interval: str | None,
    as_needed: bool,
    timing: tuple[str, ...],
) -> None:
    (match,) = find_frequencies(text)
    assert match.text == matched
    assert match.times_per_period == times
    assert match.period == period
    assert match.interval_hours == (None if interval is None else Decimal(interval))
    assert match.as_needed is as_needed
    assert match.timing == timing
    assert_round_trip(text, [match])


@pytest.mark.parametrize(
    "text",
    [
        "for five days",
        "for 8 hours",
        "1-0-2025",
        "12-10-25",
        "2025-1-0-1",
        "1-0-1-0",
        "1.1-0-1",
        "v1-0-1",
        "1-0-1/2",
        "1-0-1.5",
        "0-0-0",
        "every 4-6 hours",
        "every 4 to 6 hours",
        "pain at night",
        "worse in the morning",
        "two tablets a day",
        "odd",
        "BDS",
        "",
    ],
)
def test_not_frequencies(text: str) -> None:
    assert find_frequencies(text) == ()


def test_each_form_is_its_own_match() -> None:
    text = "Paracetamol 1 g q6h PRN; metformin 500 mg 1-0-1."
    matches = find_frequencies(text)
    assert [m.text for m in matches] == ["q6h", "PRN", "1-0-1"]
    assert_round_trip(text, matches)


def freq(text: str):  # type: ignore[no-untyped-def]
    (match,) = find_frequencies(text)
    return match


@pytest.mark.parametrize(
    ("a", "b", "result"),
    [
        ("TDS", "q8h", "equal"),
        ("BD", "1-0-1", "equal"),
        ("twice a day", "BID", "equal"),
        ("nocte", "QHS", "equal"),
        ("0-0-1", "nocte", "equal"),
        ("BD", "TDS", "different"),
        ("OD", "weekly", "different"),
        ("mane", "nocte", "different"),
        ("q8h", "q6h", "different"),
        ("PRN", "PRN", "not_comparable"),
        ("PRN", "BD", "not_comparable"),
        ("q36h", "q36h", "not_comparable"),
    ],
)
def test_compare_frequencies(a: str, b: str, result: str) -> None:
    assert compare_frequencies(freq(a), freq(b)) == result
    assert compare_frequencies(freq(b), freq(a)) == result
