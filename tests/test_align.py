"""rules/align.py: the patient's own current facts, with values from each drug's dosing phrase."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from scribecheck.note import parse_note
from scribecheck.rules._match import Match
from scribecheck.rules.align import DrugFact, Facts, SymptomFact, note_facts, transcript_facts
from scribecheck.transcript import parse_transcript


def _turns(text: str) -> Facts:
    return transcript_facts(parse_transcript(text))


def _drug(facts: Facts, generic: str) -> DrugFact:
    (found,) = [d for d in facts.drugs if generic in d.key]
    return found


def _texts(values: Sequence[Match]) -> list[str]:
    return [value.text for value in values]


# Dosing phrases


def test_values_stop_at_the_next_drug() -> None:
    facts = _turns("Patient: I take metformin 500 mg and gliclazide 80 mg twice daily.")
    metformin, gliclazide = _drug(facts, "metformin"), _drug(facts, "gliclazide")
    assert (_texts(metformin.doses), _texts(metformin.frequencies)) == (["500 mg"], [])
    assert _texts(gliclazide.doses) == ["80 mg"]
    assert _texts(gliclazide.frequencies) == ["twice daily"]


@pytest.mark.parametrize(
    "text",
    [
        "Patient: I take metformin. It is 500 mg twice daily.",  # next sentence
        "Patient: I take metformin\n500 mg twice daily",  # next line
        "Patient: I take metformin, 500 mg twice daily.",  # a comma separates
        "Patient: I take metformin and my water tablet 40 mg.",  # a conjunction separates
    ],
)
def test_values_outside_the_dosing_phrase_are_not_attributed(text: str) -> None:
    metformin = _drug(_turns(text), "metformin")
    assert (metformin.doses, metformin.frequencies, metformin.routes) == ((), (), ())


@pytest.mark.parametrize(
    ("text", "routes"),
    [
        ("Doctor: Take metformin 500 mg orally.", ["orally"]),
        ("Doctor: Take metformin 500 mg PO.", ["PO"]),
        ("Doctor: Take metformin 500 mg by mouth.", ["by mouth"]),
        ("Doctor: Use nystatin oral suspension 1 ml four times a day.", ["oral"]),
        ("Doctor: Take metformin 500 mg oral.", ["oral"]),
        ("Doctor: Take metformin 500 mg, oral.", []),  # the comma separates
        ("Doctor: Use nystatin for oral thrush.", []),  # a disease, not a route
        ("Doctor: Use nystatin 1 ml, oral hygiene matters.", []),
    ],
)
def test_route_words(text: str, routes: list[str]) -> None:
    drug = _drug(_turns(text), "nystatin" if "nystatin" in text else "metformin")
    assert _texts(drug.routes) == routes


def test_offsets_index_the_turn_text() -> None:
    transcript = parse_transcript("Doctor: Hello.\nPatient: I take metformin 500 mg.")
    (metformin,) = transcript_facts(transcript).drugs
    turn = transcript.turns[metformin.turn - 1]
    assert turn.text[metformin.match.start : metformin.match.end] == "metformin"
    (dose,) = metformin.doses
    assert turn.text[dose.start : dose.end] == "500 mg"


def test_note_offsets_index_the_note_text() -> None:
    note = parse_note("Medications:\nmetformin 1 g BD\n")
    (metformin,) = note_facts(note).drugs
    assert metformin.turn is None
    (dose,) = metformin.doses
    assert note.text[dose.start : dose.end] == "1 g"


def test_a_value_before_the_drug_belongs_to_it() -> None:
    metformin = _drug(_turns("Patient: 500 mg twice daily of metformin."), "metformin")
    assert (_texts(metformin.doses), _texts(metformin.frequencies)) == (["500 mg"], ["twice daily"])


def test_a_value_between_two_drugs_is_ambiguous() -> None:
    facts = _turns("Patient: I take metformin 500 mg gliclazide.")
    metformin, gliclazide = _drug(facts, "metformin"), _drug(facts, "gliclazide")
    assert (metformin.doses, gliclazide.doses) == ((), ())
    assert "doses" in metformin.ambiguous and "doses" in gliclazide.ambiguous


# Alignment keys


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("Patient: I take Glucophage.", {"metformin"}),
        ("Patient: I take Augmentin.", {"amoxicillin", "clavulanate"}),
        ("Patient: I take amoxicillin / clavulanate.", {"amoxicillin", "clavulanate"}),
        ("Patient: I take paracetamol.", {"acetaminophen"}),
    ],
)
def test_drugs_are_keyed_by_their_set_of_ingredients(text: str, key: set[str]) -> None:
    (drug,) = _turns(text).drugs
    assert drug.key == frozenset(key)


# Context


@pytest.mark.parametrize(
    "text",
    [
        "Patient: My mum takes metformin 500 mg.",  # family
        "Doctor: If the pain gets worse take metformin 500 mg.",  # hypothetical
    ],
)
def test_family_and_hypothetical_drug_mentions_are_dropped(text: str) -> None:
    assert _turns(text).drugs == ()


def test_family_side_and_symptom_mentions_are_dropped() -> None:
    facts = _turns("Patient: My mum's left knee is swollen and my dad has chest pain.")
    assert (facts.sides, facts.symptoms) == ((), ())


def test_historical_mentions_are_kept_and_marked() -> None:
    (symptom,) = _turns("Patient: I had chest pain last week.").symptoms
    assert "historical" in symptom.attributes


def test_possible_mentions_are_kept_and_marked() -> None:
    (symptom,) = _turns("Patient: Maybe some chest pain.").symptoms
    assert "possible" in symptom.attributes


# Symptom status


def _symptom(text: str) -> SymptomFact:
    (found,) = [s for s in _turns(text).symptoms if s.match.term == "chest pain"]
    return found


@pytest.mark.parametrize(
    ("text", "status", "presence"),
    [
        ("Patient: I've got chest pain.", "present", "patient"),
        ("Relative: She has had chest pain.", "present", "patient"),
        ("Carer: He gets chest pain.", "present", "patient"),
        ("Doctor: So you have chest pain.", "present", "clinician"),
        ("Nurse: She mentioned chest pain.", "present", "clinician"),
        ("I've got chest pain.", "present", "unattributed"),  # no speaker labels
        ("Patient: No chest pain.", "negated", None),
        ("Doctor: Any chest pain?\nPatient: Not that I've noticed.", "question", None),
        ("Doctor: You don't have chest pain?\nPatient: No.", "question", None),
    ],
)
def test_symptom_status(text: str, status: str, presence: str | None) -> None:
    found = _symptom(text)
    assert (found.status, found.presence) == (status, presence)


def test_a_bound_no_negates_the_question_mention() -> None:
    found = _symptom("Doctor: Any chest pain?\nPatient: No.")
    assert (found.status, found.presence, found.turn) == ("negated", None, 1)


def test_a_bare_yes_makes_the_question_mention_patient_side_presence() -> None:
    found = _symptom("Doctor: Any chest pain?\nPatient: Yes.")
    assert (found.status, found.presence, found.turn) == ("present", "patient", 1)


def test_note_symptom_status() -> None:
    facts = note_facts(parse_note("History:\nNo chest pain. Cough for a week.\n"))
    statuses = {s.match.term: s.status for s in facts.symptoms}
    assert statuses == {"chest pain": "negated", "cough": "present"}
