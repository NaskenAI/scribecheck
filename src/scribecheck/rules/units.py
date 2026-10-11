"""Doses: a number followed by a unit, with conversion inside mass and volume.

Mass (g, mg, mcg) and volume (L, ml) convert within themselves. Every other unit
(units, IU, puffs, drops, tablets, capsules) stands alone, so "1 tablet" and "500 mg"
are not comparable. "1/2" before a count unit ("1/2 tab") is 0.5 of it; other
fractions and mixed numbers ("1 1/2 tablets") give no match, as do ranges ("1-2
puffs", "one or two tablets") and rates or concentrations ("15 mg/kg", "120 g/L",
"500 mg per day").
"""

from __future__ import annotations

import re
from decimal import Decimal
from enum import Enum

from scribecheck.rules._match import Comparison, Match
from scribecheck.rules.numbers import NumberMatch, find_numbers


class Unit(str, Enum):
    MG = "mg"
    MCG = "mcg"
    G = "g"
    ML = "ml"
    L = "l"
    UNIT = "unit"
    IU = "iu"
    PUFF = "puff"
    DROP = "drop"
    TABLET = "tablet"
    CAPSULE = "capsule"


_FORMS: dict[str, Unit] = {
    "mg": Unit.MG,
    "milligram": Unit.MG,
    "milligrams": Unit.MG,
    "mcg": Unit.MCG,
    "µg": Unit.MCG,  # micro sign
    "μg": Unit.MCG,  # Greek mu
    "microgram": Unit.MCG,
    "micrograms": Unit.MCG,
    "g": Unit.G,
    "gram": Unit.G,
    "grams": Unit.G,
    "ml": Unit.ML,
    "millilitre": Unit.ML,
    "millilitres": Unit.ML,
    "milliliter": Unit.ML,
    "milliliters": Unit.ML,
    "l": Unit.L,
    "litre": Unit.L,
    "litres": Unit.L,
    "liter": Unit.L,
    "liters": Unit.L,
    "unit": Unit.UNIT,
    "units": Unit.UNIT,
    "iu": Unit.IU,
    "international unit": Unit.IU,
    "international units": Unit.IU,
    "puff": Unit.PUFF,
    "puffs": Unit.PUFF,
    "drop": Unit.DROP,
    "drops": Unit.DROP,
    "tab": Unit.TABLET,
    "tabs": Unit.TABLET,
    "tablet": Unit.TABLET,
    "tablets": Unit.TABLET,
    "capsule": Unit.CAPSULE,
    "capsules": Unit.CAPSULE,
}

# Factor to the dimension's base unit (mg for mass, ml for volume).
_MASS = {Unit.G: Decimal(1000), Unit.MG: Decimal(1), Unit.MCG: Decimal("0.001")}
_VOLUME = {Unit.L: Decimal(1000), Unit.ML: Decimal(1)}

_UNIT_ALTERNATIVES = "|".join(
    r"\s+".join(map(re.escape, form.split())) for form in sorted(_FORMS, key=len, reverse=True)
)
# A unit must not run into a letter, or into "/" or "per" (a rate such as mg/kg).
_UNIT_END = r"(?![A-Za-z])(?![ \t]*(?:/|per(?![A-Za-z])))"
# Right after a number: optional space or hyphen, then a unit.
_AFTER_NUMBER = re.compile(
    rf"(?:[ \t]*-[ \t]*|[ \t]*)(?P<unit>{_UNIT_ALTERNATIVES}){_UNIT_END}", re.IGNORECASE
)
# After "half": "a"/"an", then the unit ("half a tablet").
_AFTER_HALF = re.compile(
    rf"[ \t]+an?[ \t]+(?P<unit>{_UNIT_ALTERNATIVES}){_UNIT_END}", re.IGNORECASE
)
# "1/2 tab": half of a count unit, not inside a run of digits and slashes.
_HALF_SLASH = re.compile(
    r"(?<![\w.,/])1/2[ \t]*"
    r"(?P<unit>tabs?|tablets?|capsules?|puffs?|drops?)" + _UNIT_END,
    re.IGNORECASE,
)
# A whole number just before a fraction makes a mixed number ("1 1/2", "1-½", "one ½").
_WHOLE_BEFORE = re.compile(
    r"(?:\d|(?<![A-Za-z])(?:one|two|three|four|five|six|seven|eight|nine|ten))[ \t]*-?[ \t]*$",
    re.IGNORECASE,
)
_FRACTIONS = frozenset("½¼¾")
_RANGE_JOINER = re.compile(r"\s*(?:-|–|\bto\b|\bor\b)\s*", re.IGNORECASE)


class DoseMatch(Match):
    value: Decimal
    unit: Unit
    number_start: int
    number_end: int


def parse_unit(text: str) -> Unit | None:
    """The unit named by `text` (case-insensitive), or None."""
    return _FORMS.get(" ".join(text.split()).lower())


def find_doses(text: str) -> tuple[DoseMatch, ...]:
    """Find every number-plus-unit dose in `text`, left to right."""
    numbers = find_numbers(text)
    doses: list[DoseMatch] = []
    for index, number in enumerate(numbers):
        if _in_range(text, numbers, index):
            continue
        if number.text in _FRACTIONS and _WHOLE_BEFORE.search(text, 0, number.start):
            continue
        pattern = _AFTER_HALF if number.text.lower() == "half" else _AFTER_NUMBER
        found = pattern.match(text, number.end)
        if not found:
            continue
        unit = parse_unit(found.group("unit"))
        if unit is None:  # pragma: no cover - the pattern only matches known forms
            continue
        doses.append(
            DoseMatch(
                start=number.start,
                end=found.end(),
                text=text[number.start : found.end()],
                value=number.value,
                unit=unit,
                number_start=number.start,
                number_end=number.end,
            )
        )
    for found in _HALF_SLASH.finditer(text):
        if _WHOLE_BEFORE.search(text, 0, found.start()):
            continue
        unit = parse_unit(found.group("unit"))
        if unit is None:  # pragma: no cover - the pattern only matches known forms
            continue
        doses.append(
            DoseMatch(
                start=found.start(),
                end=found.end(),
                text=found.group(),
                value=Decimal("0.5"),
                unit=unit,
                number_start=found.start(),
                number_end=found.start() + 3,
            )
        )
    return tuple(sorted(doses, key=lambda d: d.start))


def convert(dose: DoseMatch, to: Unit) -> Decimal | None:
    """`dose` expressed in `to`, or None when the units are in different dimensions."""
    if dose.unit is to:
        return dose.value
    for table in (_MASS, _VOLUME):
        if dose.unit in table and to in table:
            return dose.value * table[dose.unit] / table[to]
    return None


def compare_doses(a: DoseMatch, b: DoseMatch) -> Comparison:
    """Whether two doses are equal after conversion, or not comparable at all."""
    converted = convert(b, a.unit)
    if converted is None:
        return "not_comparable"
    return "equal" if converted == a.value else "different"


def _in_range(text: str, numbers: tuple[NumberMatch, ...], index: int) -> bool:
    """True when the number is joined to a neighbouring number by -, to or or."""
    if index > 0:
        gap = text[numbers[index - 1].end : numbers[index].start]
        if _RANGE_JOINER.fullmatch(gap):
            return True
    if index + 1 < len(numbers):
        gap = text[numbers[index].end : numbers[index + 1].start]
        if _RANGE_JOINER.fullmatch(gap):
            return True
    return False
