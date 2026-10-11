from __future__ import annotations

from decimal import Decimal

import pytest
from _spans import assert_round_trip

from scribecheck.rules.units import Unit, compare_doses, convert, find_doses, parse_unit


@pytest.mark.parametrize(
    ("text", "matched", "value", "unit"),
    [
        ("500 mg", "500 mg", "500", Unit.MG),
        ("500mg", "500mg", "500", Unit.MG),
        ("5-mg tablets", "5-mg", "5", Unit.MG),
        ("10 milligram", "10 milligram", "10", Unit.MG),
        ("five hundred milligrams", "five hundred milligrams", "500", Unit.MG),
        ("1,000 mg", "1,000 mg", "1000", Unit.MG),
        ("0.5 mg", "0.5 mg", "0.5", Unit.MG),
        ("250 mcg", "250 mcg", "250", Unit.MCG),
        ("250 µg", "250 µg", "250", Unit.MCG),
        ("250 μg", "250 μg", "250", Unit.MCG),
        ("100 micrograms", "100 micrograms", "100", Unit.MCG),
        ("1 g", "1 g", "1", Unit.G),
        ("2 grams", "2 grams", "2", Unit.G),
        ("5 ml", "5 ml", "5", Unit.ML),
        ("5 mL", "5 mL", "5", Unit.ML),
        ("10 millilitres", "10 millilitres", "10", Unit.ML),
        ("1 L", "1 L", "1", Unit.L),
        ("1.5 litres", "1.5 litres", "1.5", Unit.L),
        ("2 liters", "2 liters", "2", Unit.L),
        ("10 units", "10 units", "10", Unit.UNIT),
        ("1000 IU", "1000 IU", "1000", Unit.IU),
        ("400 international units", "400 international units", "400", Unit.IU),
        ("two puffs", "two puffs", "2", Unit.PUFF),
        ("one puff", "one puff", "1", Unit.PUFF),
        ("2 drops", "2 drops", "2", Unit.DROP),
        ("one tablet", "one tablet", "1", Unit.TABLET),
        ("2 capsules", "2 capsules", "2", Unit.CAPSULE),
        ("one and a half tablets", "one and a half tablets", "1.5", Unit.TABLET),
        ("half a tablet", "half a tablet", "0.5", Unit.TABLET),
        ("½ tablet", "½ tablet", "0.5", Unit.TABLET),
        ("zero point five milligrams", "zero point five milligrams", "0.5", Unit.MG),
    ],
)
def test_doses(text: str, matched: str, value: str, unit: Unit) -> None:
    (dose,) = find_doses(text)
    assert dose.text == matched
    assert dose.value == Decimal(value)
    assert dose.unit is unit
    assert dose.start <= dose.number_start < dose.number_end <= dose.end
    assert_round_trip(text, [dose])


@pytest.mark.parametrize(
    "text",
    [
        "for five days",
        "one of the tablets",
        "1-2 puffs",
        "2 to 3 tablets",
        "2-3 mg",
        "take a tablet",
        "L knee",
        "the g in mg",
        "Vitamin B12 tablets",
        "take 1/2 tablet",
        "1,5 mg",
        "1,0000 mg",
        "15 mg/kg",
        "15 mg per kg",
        "500 mg/day",
        "Hb 120 g/L",
        "2 L/min oxygen",
        "one point twenty-five mg",
        "5 days of tablets",
        "",
    ],
)
def test_not_doses(text: str) -> None:
    assert find_doses(text) == ()


def test_sentence_with_several_doses() -> None:
    text = "Metformin 500mg BD, salbutamol two puffs PRN, and 1,000 IU vitamin D for five days."
    doses = find_doses(text)
    assert [(d.value, d.unit) for d in doses] == [
        (Decimal(500), Unit.MG),
        (Decimal(2), Unit.PUFF),
        (Decimal(1000), Unit.IU),
    ]
    assert_round_trip(text, doses)


@pytest.mark.parametrize(
    ("text", "unit"),
    [
        ("mg", Unit.MG),
        ("Milligrams", Unit.MG),
        ("µg", Unit.MCG),
        ("mcg", Unit.MCG),
        ("g", Unit.G),
        ("mL", Unit.ML),
        ("L", Unit.L),
        ("l", Unit.L),
        ("IU", Unit.IU),
        ("international unit", Unit.IU),
        ("units", Unit.UNIT),
        ("puffs", Unit.PUFF),
        ("drop", Unit.DROP),
        ("tablets", Unit.TABLET),
        ("capsule", Unit.CAPSULE),
    ],
)
def test_parse_unit(text: str, unit: Unit) -> None:
    assert parse_unit(text) is unit


@pytest.mark.parametrize("text", ["", "kg", "days", "mgs mg"])
def test_parse_unit_rejects_unknown(text: str) -> None:
    assert parse_unit(text) is None


def dose(text: str):  # type: ignore[no-untyped-def]
    (match,) = find_doses(text)
    return match


@pytest.mark.parametrize(
    ("text", "to", "value"),
    [
        ("0.5 mg", Unit.MCG, "500"),
        ("1 g", Unit.MG, "1000"),
        ("250 mcg", Unit.MG, "0.25"),
        ("1 L", Unit.ML, "1000"),
        ("500 ml", Unit.L, "0.5"),
        ("2 tablets", Unit.TABLET, "2"),
    ],
)
def test_convert(text: str, to: Unit, value: str) -> None:
    assert convert(dose(text), to) == Decimal(value)


@pytest.mark.parametrize(
    ("text", "to"),
    [("1 tablet", Unit.MG), ("5 mg", Unit.ML), ("10 units", Unit.IU), ("2 puffs", Unit.DROP)],
)
def test_convert_never_crosses_dimensions(text: str, to: Unit) -> None:
    assert convert(dose(text), to) is None


@pytest.mark.parametrize(
    ("a", "b", "result"),
    [
        ("0.5 mg", "500 mcg", "equal"),
        ("1 g", "1000 mg", "equal"),
        ("1 L", "1000 ml", "equal"),
        ("500 mg", "1 g", "different"),
        ("2 puffs", "1 puff", "different"),
        ("1 tablet", "500 mg", "not_comparable"),
        ("10 units", "10 IU", "not_comparable"),
        ("5 ml", "5 mg", "not_comparable"),
    ],
)
def test_compare_doses(a: str, b: str, result: str) -> None:
    assert compare_doses(dose(a), dose(b)) == result
    assert compare_doses(dose(b), dose(a)) == result


def test_point_as_a_noun_leaves_the_following_dose_whole() -> None:
    (found,) = find_doses("up to this point 3 tablets")
    assert (found.text, found.value, found.unit) == ("3 tablets", Decimal(3), Unit.TABLET)
