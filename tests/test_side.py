from __future__ import annotations

import pytest
from _spans import assert_round_trip

from scribecheck.rules.side import BODY_PARTS, find_sides


@pytest.mark.parametrize(
    ("text", "matched", "side", "site", "qualifier"),
    [
        ("left knee", "left knee", "left", "knee", None),
        ("Right hip", "Right hip", "right", "hip", None),
        ("my left knee has been swelling", "left knee", "left", "knee", None),
        ("both left knees", "left knees", "left", "knee", None),
        ("right eye", "right eye", "right", "eye", None),
        ("left breast", "left breast", "left", "breast", None),
        ("left lower leg", "left lower leg", "left", "lower leg", None),
        ("right big toe", "right big toe", "right", "big toe", None),
        ("left side", "left side", "left", "side", None),
        ("bilateral knee pain", "bilateral knee", "bilateral", "knee", None),
        ("both knees", "both knees", "bilateral", "knee", None),
        ("both feet", "both feet", "bilateral", "foot", None),
        ("left-sided chest pain", "left-sided chest", "left", "chest", None),
        ("right-sided weakness", "right-sided", "right", None, None),
        ("bilateral swelling", "bilateral", "bilateral", None, None),
        ("knee on the right", "knee on the right", "right", "knee", None),
        ("ankle on the left side", "ankle on the left side", "left", "ankle", None),
        ("right side of the chest", "right side of the chest", "right", "chest", None),
        ("L knee", "L knee", "left", "knee", None),
        ("R hip", "R hip", "right", "hip", None),
        ("Lt shoulder", "Lt shoulder", "left", "shoulder", None),
        ("Rt ankle", "Rt ankle", "right", "ankle", None),
        ("tender LLQ", "LLQ", "left", "abdomen", "lower quadrant"),
        ("LUQ", "LUQ", "left", "abdomen", "upper quadrant"),
        ("RLQ", "RLQ", "right", "abdomen", "lower quadrant"),
        ("RUQ pain", "RUQ", "right", "abdomen", "upper quadrant"),
        ("left upper quadrant", "left upper quadrant", "left", "abdomen", "upper quadrant"),
    ],
)
def test_sides(text: str, matched: str, side: str, site: str | None, qualifier: str | None) -> None:
    (match,) = find_sides(text)
    assert match.text == matched
    assert match.side == side
    assert match.site == site
    assert match.qualifier == qualifier
    assert_round_trip(text, [match])


@pytest.mark.parametrize(
    "text",
    [
        "that's right",
        "the right thing to do",
        "she left the clinic",
        "on the left",
        "she left her knee brace at home",
        "l knee",
        "right now",
        "turn right",
        "Left",
        "both of them",
        "LQ",
        "the right side effects",
        "left side-effects",
        "is that right\nknee",
        "left\nknee",
        "",
    ],
)
def test_not_sides(text: str) -> None:
    assert find_sides(text) == ()


def test_sentence_with_several_sides() -> None:
    text = "That's right, the left knee is swollen but the R hip is fine; RUQ tender."
    matches = find_sides(text)
    assert [(m.side, m.site) for m in matches] == [
        ("left", "knee"),
        ("right", "hip"),
        ("right", "abdomen"),
    ]
    assert_round_trip(text, matches)


def test_body_parts_list() -> None:
    assert {"knee", "hip", "chest", "abdomen", "eye", "foot"} <= set(BODY_PARTS)
    assert 30 <= len(BODY_PARTS) <= 60
