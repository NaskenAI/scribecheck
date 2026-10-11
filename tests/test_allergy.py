from __future__ import annotations

import pytest
from _spans import assert_round_trip

from scribecheck.rules.allergy import AllergyMatch, find_allergies


def _find(text: str) -> tuple[AllergyMatch, ...]:
    results = find_allergies(text)
    assert_round_trip(text, results)
    for result in results:
        parts = [result.cue, *result.allergens]
        if result.reaction is not None:
            parts.append(result.reaction)
        for part in parts:
            assert text[part.start : part.end] == part.text
            assert result.start <= part.start < part.end <= result.end
    return results


@pytest.mark.parametrize(
    ("text", "allergens", "reaction"),
    [
        ("I'm allergic to penicillin.", ["penicillin"], None),
        ("She is allergic to amoxicillin and codeine", ["amoxicillin", "codeine"], None),
        ("allergic to penicillin, it gives me a rash", ["penicillin"], "rash"),
        ("Allergic to penicillin (hives)", ["penicillin"], "hives"),
        ("He has an allergy to latex.", ["latex"], None),
        ("Allergy to peanuts causes swelling", ["peanuts"], "swelling"),
        ("Allergies to aspirin, ibuprofen", ["aspirin", "ibuprofen"], None),
        ("Amoxicillin gives me a rash.", ["Amoxicillin"], "rash"),
        ("Penicillin, it gives me a rash.", ["Penicillin"], "rash"),
        ("Codeine gave me hives.", ["Codeine"], "hives"),
        ("Aspirin gives me a rash, hives or swelling", ["Aspirin"], "rash, hives or swelling"),
        ("Plasters bring me out in a rash", ["Plasters"], "rash"),
        ("Allergies: penicillin (rash)", ["penicillin"], "rash"),
        ("Allergy: co-trimoxazole", ["co-trimoxazole"], None),
        ("allergic to sulfa drugs - anaphylaxis", ["sulfa drugs"], "anaphylaxis"),
    ],
)
def test_patient_allergy(text: str, allergens: list[str], reaction: str | None) -> None:
    (result,) = _find(text)
    assert result.status == "allergic"
    assert result.experiencer == "patient"
    assert [a.text for a in result.allergens] == allergens
    assert (result.reaction.text if result.reaction else None) == reaction
    assert not result.possible and not result.hypothetical


@pytest.mark.parametrize(
    ("text", "cue"),
    [
        ("No allergies.", "allergies"),
        ("no known drug allergies", "no known drug allergies"),
        ("No known allergies", "No known allergies"),
        ("NKDA", "NKDA"),
        ("Allergies: NKDA", "NKDA"),
        ("nkda", "nkda"),
        ("NKA", "NKA"),
        ("No drug allergies", "No drug allergies"),
        ("I don't have any allergies", "allergies"),
        ("I'm not allergic to anything", "allergic to"),
        ("No allergies to medicines that I know of", "allergies to"),
    ],
)
def test_explicit_none(text: str, cue: str) -> None:
    (result,) = _find(text)
    assert result.status == "none"
    assert result.allergens == ()
    assert result.reaction is None
    assert result.cue.text == cue


def test_explicit_non_allergy() -> None:
    (result,) = _find("I'm not allergic to penicillin")
    assert result.status == "not_allergic"
    assert result.experiencer == "patient"
    assert [a.text for a in result.allergens] == ["penicillin"]


def test_family_statement_is_not_the_patients_allergy() -> None:
    (result,) = _find("My mother is allergic to penicillin")
    assert result.experiencer == "family"
    assert result.status == "allergic"
    assert [a.text for a in result.allergens] == ["penicillin"]


def test_possible_allergy() -> None:
    (result,) = _find("I might be allergic to penicillin")
    assert result.possible
    assert result.status == "allergic"


def test_several_statements_in_order() -> None:
    text = "My mother is allergic to penicillin. I'm allergic to latex, it gives me hives."
    family, patient = _find(text)
    assert family.experiencer == "family"
    assert patient.experiencer == "patient"
    assert [a.text for a in patient.allergens] == ["latex"]
    assert patient.reaction is not None and patient.reaction.text == "hives"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "The knee gives me trouble",
        "Walking gives me pain",
        "She has a rash on her arm",
        "allergy clinic next week",
        "Allergic rhinitis in spring",
    ],
)
def test_no_allergy_statement(text: str) -> None:
    assert _find(text) == ()


@pytest.mark.parametrize(
    "text",
    [
        # Questions are not statements.
        "Are you allergic to penicillin?",
        "Was the rash from an allergy to penicillin?",
        "Any allergies?",
        # Not an allergy at all.
        "Allergies: unknown",
        "Allergies: see above",
        "There was no allergic reaction to the vaccine",
        # Triggers that are not allergens, side effects, and general statements.
        "Stress gives me hives",
        "The heat gives me a rash",
        "Walking in the cold gives me wheezing",
        "Metformin gives me diarrhoea",
        "Amoxicillin causes rash in some people",
        "The rash, it gives me hives",
        # "None" with an exception, or a family statement the patient shares: the model decides.
        "No allergies except penicillin",
        "No allergies apart from penicillin which gives me hives",
        "NKDA other than codeine",
        "My mother is allergic to penicillin and so am I",
    ],
)
def test_no_result_when_unsure(text: str) -> None:
    assert _find(text) == ()


@pytest.mark.parametrize(
    ("text", "allergens"),
    [
        ("allergic to penicillin and has a rash on her arm", ["penicillin"]),
        ("Allergic to penicillin, codeine makes me sick", ["penicillin"]),
        ("Allergic to penicillin and codeine, which give me a rash", ["penicillin", "codeine"]),
    ],
)
def test_allergen_list_stops_at_a_new_clause(text: str, allergens: list[str]) -> None:
    (result,) = _find(text)
    assert [a.text for a in result.allergens] == allergens


def test_hypothetical_allergy_keeps_its_allergen() -> None:
    (result,) = _find("If you're allergic to penicillin, tell us")
    assert result.hypothetical
    assert [a.text for a in result.allergens] == ["penicillin"]


def test_historical_allergy() -> None:
    (result,) = _find("Previously allergic to penicillin")
    assert result.historical
    (current,) = _find("Allergic to penicillin")
    assert not current.historical


def test_relative_with_no_allergies_is_family() -> None:
    (result,) = _find("Mum: no allergies")
    assert result.experiencer == "family"
