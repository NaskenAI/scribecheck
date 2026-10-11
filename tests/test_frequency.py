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
        ("Metformin 500 mg OD", "OD", 1, D, None, False, ()),
        ("Take one tablet at night", "at night", 1, D, None, False, ("night",)),
        ("zopiclone 7.5 mg at bedtime", "at bedtime", 1, D, None, False, ("night",)),
        ("use every night", "every night", 1, D, None, False, ("night",)),
        ("take 2 tablets in the morning", "in the morning", 1, D, None, False, ("morning",)),
        ("every morning take 1 tab", "every morning", 1, D, None, False, ("morning",)),
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
        "drops OD",
        "Timolol eye drops 1 drop OD",
        "1 drop OD, 1 drop OS",
        "OS",
        "OU",
        "she sleeps badly every night",
        "stiff every morning",
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


def test_eye_context_does_not_cross_sentences() -> None:
    text = "Eyes are fine. Metformin 500 mg OD."
    (match,) = find_frequencies(text)
    assert (match.text, match.times_per_period) == ("OD", 1)
    assert_round_trip(text, [match])


def test_timing_phrase_equals_its_abbreviation() -> None:
    assert compare_frequencies(freq("Take one tablet at night"), freq("nocte")) == "equal"
    assert compare_frequencies(freq("take 1 tab in the morning"), freq("mane")) == "equal"


def test_two_ods_on_a_medication_line_stay_once_daily() -> None:
    text = "Metformin 500 mg OD, amlodipine 5 mg OD"
    matches = find_frequencies(text)
    assert [(m.text, m.times_per_period) for m in matches] == [("OD", 1), ("OD", 1)]
