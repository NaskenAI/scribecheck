"""Laterality: left, right or bilateral, with the body site.

A side counts only when it touches a body part from `BODY_PARTS` ("left knee",
"knee on the right", "right side of the chest", "both knees", "L knee"), or is an
unambiguous lateral word ("left-sided", "bilateral") or a quadrant (LLQ, LUQ, RLQ,
RUQ). "right" alone ("that's right") and "left" as a verb ("she left the clinic")
never count. The side word must be next to the part, so "left her knee brace" does
not match, at the cost of missing "left swollen knee". Matches never cross a line
break, and "side" followed by "effects" is not a body site.
"""

from __future__ import annotations

import re
from typing import Literal

from scribecheck.rules._match import Match, non_overlapping

Side = Literal["left", "right", "bilateral"]

# Canonical body part -> plural forms beyond the regular "+s".
_PARTS: dict[str, tuple[str, ...]] = {
    "head": (),
    "face": (),
    "forehead": (),
    "temple": (),
    "eye": (),
    "eyelid": (),
    "ear": (),
    "nostril": (),
    "cheek": (),
    "jaw": (),
    "tonsil": (),
    "neck": (),
    "shoulder": (),
    "arm": (),
    "upper arm": (),
    "armpit": (),
    "elbow": (),
    "forearm": (),
    "wrist": (),
    "hand": (),
    "palm": (),
    "finger": (),
    "thumb": (),
    "chest": (),
    "breast": (),
    "rib": (),
    "lung": (),
    "flank": (),
    "side": (),
    "abdomen": (),
    "iliac fossa": ("iliac fossae",),
    "groin": (),
    "hip": (),
    "buttock": (),
    "thigh": (),
    "leg": (),
    "lower leg": (),
    "knee": (),
    "calf": ("calves",),
    "shin": (),
    "ankle": (),
    "foot": ("feet",),
    "heel": (),
    "toe": (),
    "big toe": (),
    "kidney": (),
    "ovary": ("ovaries",),
    "testicle": (),
}
BODY_PARTS: tuple[str, ...] = tuple(_PARTS)

# Abdominal quadrants map to site "abdomen" with a qualifier.
_QUADRANTS = {"upper quadrant": "upper quadrant", "lower quadrant": "lower quadrant"}

_SINGULAR: dict[str, str] = {}
_PLURAL: dict[str, str] = {}
for _part, _irregular in _PARTS.items():
    _SINGULAR[_part] = _part
    for _plural in _irregular or (_part + "s",):
        _PLURAL[_plural] = _part
for _quadrant in _QUADRANTS:
    _SINGULAR[_quadrant] = _quadrant


def _alternatives(forms: dict[str, str]) -> str:
    ordered = sorted(forms, key=len, reverse=True)
    return "|".join(r"[ \t]+".join(map(re.escape, form.split())) for form in ordered)


_ANY_PART = _alternatives({**_SINGULAR, **_PLURAL})
_PLURAL_PART = _alternatives(_PLURAL)
_B = r"(?<![A-Za-z])"
_E = r"(?![A-Za-z])"
_OWNER = r"(?:the|my|his|her|their|your)[ \t]+"

_IGNORE = re.IGNORECASE
_SIDE_EFFECTS = re.compile(r"[ \t]*-?[ \t]*effects?(?![A-Za-z])", re.IGNORECASE)
_RULES: tuple[re.Pattern[str], ...] = (
    # left knee, left-sided chest, bilateral knee, left upper quadrant
    re.compile(
        rf"{_B}(?P<side>left|right|bilateral)(?:-sided)?[ \t]+(?P<part>{_ANY_PART}){_E}", _IGNORE
    ),
    # both knees
    re.compile(rf"{_B}(?P<both>both)[ \t]+(?P<part>{_PLURAL_PART}){_E}", _IGNORE),
    # knee on the right (side)
    re.compile(
        rf"{_B}(?P<part>{_ANY_PART})[ \t]+on[ \t]+the[ \t]+(?P<side>left|right)(?:[ \t]+side)?{_E}",
        _IGNORE,
    ),
    # right side of the chest
    re.compile(
        rf"{_B}(?P<side>left|right)[ \t]+side[ \t]+of[ \t]+(?:{_OWNER})?(?P<part>{_ANY_PART}){_E}",
        _IGNORE,
    ),
    # L knee, Rt hip: abbreviation case-sensitive, part any case
    re.compile(rf"{_B}(?P<abbr>Lt|Rt|L|R)[ \t]+(?P<part>(?i:{_ANY_PART})){_E}"),
    # LLQ, RUQ
    re.compile(rf"{_B}(?P<quadrant>[LR][LU]Q){_E}"),
    # left-sided, bilateral on their own
    re.compile(rf"{_B}(?:(?P<side>left|right)-sided|(?P<bilateral>bilateral)){_E}", _IGNORE),
)


class SideMatch(Match):
    side: Side
    site: str | None = None
    qualifier: str | None = None


def find_sides(text: str) -> tuple[SideMatch, ...]:
    """Find every side-plus-site mention in `text`, left to right, without overlaps."""
    candidates: list[SideMatch] = []
    for pattern in _RULES:
        for match in pattern.finditer(text):
            found = _build(match)
            if found.site == "side" and _SIDE_EFFECTS.match(text, found.end):
                continue
            candidates.append(found)
    return non_overlapping(candidates)


def _build(match: re.Match[str]) -> SideMatch:
    groups = match.groupdict()
    side: Side
    if groups.get("quadrant"):
        code = groups["quadrant"]
        side = "left" if code[0] == "L" else "right"
        qualifier = "upper quadrant" if code[1] == "U" else "lower quadrant"
        return _side(match, side, "abdomen", qualifier)
    if groups.get("both") or groups.get("bilateral"):
        side = "bilateral"
    elif groups.get("abbr"):
        side = "left" if groups["abbr"].startswith("L") else "right"
    else:
        side = "bilateral" if groups["side"].lower() == "bilateral" else _lr(groups["side"])
    part = groups.get("part")
    if part is None:
        return _side(match, side, None, None)
    key = " ".join(part.lower().split())
    canonical = _SINGULAR.get(key) or _PLURAL[key]
    if canonical in _QUADRANTS:
        return _side(match, side, "abdomen", _QUADRANTS[canonical])
    return _side(match, side, canonical, None)


def _lr(word: str) -> Side:
    return "left" if word.lower() == "left" else "right"


def _side(match: re.Match[str], side: Side, site: str | None, qualifier: str | None) -> SideMatch:
    return SideMatch(
        start=match.start(),
        end=match.end(),
        text=match.group(),
        side=side,
        site=site,
        qualifier=qualifier,
    )
