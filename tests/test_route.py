from __future__ import annotations

import pytest
from _spans import assert_round_trip

from scribecheck.rules.route import Route, find_routes


@pytest.mark.parametrize(
    ("text", "matched", "route"),
    [
        ("oral", "oral", Route.ORAL),
        ("taken orally", "orally", Route.ORAL),
        ("PO", "PO", Route.ORAL),
        ("500 mg po", "po", Route.ORAL),
        ("by mouth", "by mouth", Route.ORAL),
        ("IV", "IV", Route.INTRAVENOUS),
        ("given iv", "iv", Route.INTRAVENOUS),
        ("intravenous", "intravenous", Route.INTRAVENOUS),
        ("intravenously", "intravenously", Route.INTRAVENOUS),
        ("IM", "IM", Route.INTRAMUSCULAR),
        ("intramuscular", "intramuscular", Route.INTRAMUSCULAR),
        ("SC", "SC", Route.SUBCUTANEOUS),
        ("subcut", "subcut", Route.SUBCUTANEOUS),
        ("sub-cut", "sub-cut", Route.SUBCUTANEOUS),
        ("subcutaneously", "subcutaneously", Route.SUBCUTANEOUS),
        ("SL", "SL", Route.SUBLINGUAL),
        ("sublingual", "sublingual", Route.SUBLINGUAL),
        ("PR", "PR", Route.RECTAL),
        ("rectally", "rectally", Route.RECTAL),
        ("inhaled", "inhaled", Route.INHALED),
        ("blue inhaler", "inhaler", Route.INHALED),
        ("nebulised", "nebulised", Route.INHALED),
        ("nebulized", "nebulized", Route.INHALED),
        ("topical", "topical", Route.TOPICAL),
        ("topically", "topically", Route.TOPICAL),
        ("applied to the rash", "applied", Route.TOPICAL),
        ("apply to the skin", "apply", Route.TOPICAL),
        ("eye drops", "eye drops", Route.OPHTHALMIC),
        ("ear drops", "ear drops", Route.OTIC),
    ],
)
def test_routes(text: str, matched: str, route: Route) -> None:
    (match,) = find_routes(text)
    assert match.text == matched
    assert match.route is route
    assert_round_trip(text, [match])


@pytest.mark.parametrize(
    "text",
    [
        "im fine",
        "Im not sure",
        "pr exam",
        "PR interval",
        "sc",
        "sl",
        "she applied for a job",
        "IVF",
        "stage IV cancer",
        "grade IV",
        "type iv",
        "Class IV",
        "phase IV trial",
        "poem",
        "",
    ],
)
def test_not_routes(text: str) -> None:
    assert find_routes(text) == ()


def test_sentence_with_several_routes() -> None:
    text = "Ondansetron 4 mg IV, then by mouth; chloramphenicol eye drops."
    matches = find_routes(text)
    assert [m.route for m in matches] == [Route.INTRAVENOUS, Route.ORAL, Route.OPHTHALMIC]
    assert_round_trip(text, matches)
