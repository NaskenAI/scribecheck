from __future__ import annotations

from decimal import Decimal

import pytest
from _spans import assert_round_trip

from scribecheck.rules.numbers import find_numbers, parse_number


@pytest.mark.parametrize(
    ("text", "matched", "value"),
    [
        ("500", "500", "500"),
        ("take 0.5 today", "0.5", "0.5"),
        ("1,000", "1,000", "1000"),
        ("12,500 units", "12,500", "12500"),
        ("1,000,000", "1,000,000", "1000000"),
        ("500mg", "500", "500"),
        ("½ tablet", "½", "0.5"),
        ("¼", "¼", "0.25"),
        ("1½ tablets", "1½", "1.5"),
        ("zero", "zero", "0"),
        ("Twelve", "Twelve", "12"),
        ("ninety-nine", "ninety-nine", "99"),
        ("twenty-five", "twenty-five", "25"),
        ("twenty five", "twenty five", "25"),
        ("forty", "forty", "40"),
        ("five hundred", "five hundred", "500"),
        ("five hundred and fifty", "five hundred and fifty", "550"),
        ("a hundred", "a hundred", "100"),
        ("two thousand", "two thousand", "2000"),
        ("three thousand four hundred", "three thousand four hundred", "3400"),
        ("one thousand and twenty", "one thousand and twenty", "1020"),
        ("one and a half", "one and a half", "1.5"),
        ("2 and a half", "2 and a half", "2.5"),
        ("a half", "a half", "0.5"),
        ("one half", "one half", "0.5"),
        ("a quarter", "a quarter", "0.25"),
        ("half a tablet", "half", "0.5"),
        ("half an inhaler", "half", "0.5"),
        ("one", "one", "1"),
        ("zero point five", "zero point five", "0.5"),
        ("one point five", "one point five", "1.5"),
        ("two point two five", "two point two five", "2.25"),
        ("point five", "point five", "0.5"),
        ("1 point 5", "1 point 5", "1.5"),
    ],
)
def test_numbers(text: str, matched: str, value: str) -> None:
    (match,) = find_numbers(text)
    assert match.text == matched
    assert match.value == Decimal(value)
    assert_round_trip(text, [match])


@pytest.mark.parametrize(
    "text",
    [
        "one of the tablets",
        "half the time",
        "the one I take",
        "which one",
        "someone",
        "B12",
        "HbA1c",
        "nobody",
        "",
    ],
)
def test_not_numbers(text: str) -> None:
    assert find_numbers(text) == ()


def test_glued_digits_after_letters_are_skipped_but_others_found() -> None:
    text = "HbA1c 58 and B12 low"
    assert [m.value for m in find_numbers(text)] == [Decimal(58)]


@pytest.mark.parametrize("text", ["1,5", "1,0000", "1/2", "3/4 tablet", "1/2/2025"])
def test_ambiguous_comma_or_slash_numbers_are_skipped(text: str) -> None:
    assert find_numbers(text) == ()


def test_and_does_not_join_unrelated_numbers() -> None:
    assert [m.value for m in find_numbers("two and three")] == [Decimal(2), Decimal(3)]


def test_range_numbers_are_found_separately() -> None:
    assert [m.value for m in find_numbers("1-2 puffs")] == [Decimal(1), Decimal(2)]


def test_sentence_with_several_numbers() -> None:
    text = "Take one and a half tablets, twenty-five mg twice, then 1,000 units."
    matches = find_numbers(text)
    assert [m.value for m in matches] == [Decimal("1.5"), Decimal(25), Decimal(1000)]
    assert_round_trip(text, matches)


def test_sentence_final_full_stop_is_not_a_decimal_point() -> None:
    (match,) = find_numbers("The dose is 5.")
    assert match.text == "5"


@pytest.mark.parametrize(
    ("text", "value"),
    [("five hundred", "500"), (" 2.5 ", "2.5"), ("one and a half", "1.5"), ("1,000", "1000")],
)
def test_parse_number(text: str, value: str) -> None:
    assert parse_number(text) == Decimal(value)


@pytest.mark.parametrize("text", ["", "five hundred mg", "abc", "1 2"])
def test_parse_number_rejects_anything_else(text: str) -> None:
    assert parse_number(text) is None


@pytest.mark.parametrize("text", ["one point twenty-five", "point twenty"])
def test_unclear_spoken_decimals_are_skipped(text: str) -> None:
    assert find_numbers(text) == ()


@pytest.mark.parametrize(
    "text", ["at that point two nurses came", "the point five", "this point 3"]
)
def test_point_as_a_noun_is_not_a_decimal(text: str) -> None:
    assert all("point" not in m.text for m in find_numbers(text))
