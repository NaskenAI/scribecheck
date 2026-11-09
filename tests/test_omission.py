"""Rules-mode Pass A: structured omission candidates at possible confidence.

Every case is synthetic. Every candidate's and unchecked item's evidence must slice back
exactly from its source.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from scribecheck.flags import Flag, verify_evidence
from scribecheck.note import parse_note
from scribecheck.rules.omission import OmissionResult, structured_omissions
from scribecheck.transcript import parse_transcript


def run(
    transcript: str, note: str, *, brands: Mapping[str, Sequence[str]] | None = None
) -> OmissionResult:
    parsed_transcript, parsed_note = parse_transcript(transcript), parse_note(note)
    result = structured_omissions(parsed_transcript, parsed_note, brands=brands)
    for item in (*result.candidates, *result.unchecked):
        verify_evidence(item, parsed_transcript, parsed_note)
    assert [c.id for c in result.candidates] == [
        f"F{i}" for i in range(1, len(result.candidates) + 1)
    ]
    for candidate in result.candidates:
        assert (candidate.type, candidate.verdict, candidate.confidence, candidate.origin) == (
            "omission",
            "absent",
            "possible",
            "rules",
        )
        assert candidate.note_evidence is None and candidate.transcript_evidence is not None
    return result


def facts(result: OmissionResult) -> list[tuple[str, str | None]]:
    return [(c.category, c.fact) for c in result.candidates]


def only(result: OmissionResult) -> Flag:
    (candidate,) = result.candidates
    return candidate


def reasons(result: OmissionResult) -> set[tuple[str, str, str]]:
    return {(u.category, u.item, u.reason) for u in result.unchecked}


NOTE = "History:\nFeeling well.\n"


# One positive per category


def test_medicine_missing() -> None:
    candidate = only(run("Patient: I take metformin every day.", NOTE))
    assert (candidate.rule_id, candidate.category, candidate.severity) == (
        "medicine.missing",
        "medicine",
        "supporting",
    )
    assert (candidate.fact, candidate.expected_section) == ("metformin", "Medications")
    assert candidate.transcript_evidence is not None
    assert candidate.transcript_evidence.text == "metformin"


def test_dose_missing() -> None:
    candidate = only(
        run(
            "Patient: I take metformin 500 mg twice daily.", "Medications:\nmetformin twice daily\n"
        )
    )
    assert (candidate.rule_id, candidate.severity, candidate.fact) == (
        "dose.missing",
        "supporting",
        "metformin 500 mg",
    )
    assert candidate.transcript_evidence is not None
    assert candidate.transcript_evidence.text == "metformin 500 mg"


def test_frequency_missing() -> None:
    candidate = only(
        run("Patient: I take metformin 500 mg twice daily.", "Medications:\nmetformin 500 mg\n")
    )
    assert (candidate.rule_id, candidate.severity, candidate.fact) == (
        "frequency.missing",
        "supporting",
        "metformin twice daily",
    )


def test_allergy_missing() -> None:
    candidate = only(run("Patient: I'm allergic to penicillin.", NOTE))
    assert (candidate.rule_id, candidate.category, candidate.severity) == (
        "allergy.missing",
        "allergy",
        "critical",
    )
    assert (candidate.fact, candidate.expected_section) == ("allergy: penicillin", "Allergies")


def test_side_missing() -> None:
    candidate = only(run("Patient: My left knee is swollen.", "History:\nKnee swelling.\n"))
    assert (candidate.rule_id, candidate.severity, candidate.fact) == (
        "side.missing",
        "supporting",
        "left knee",
    )
    assert candidate.expected_section == "History"
    assert candidate.transcript_evidence is not None
    assert candidate.transcript_evidence.text == "left knee"


def test_a_missing_drug_gives_only_the_medicine_candidate() -> None:
    result = run("Patient: I take metformin 500 mg twice daily.", NOTE)
    assert facts(result) == [("medicine", "metformin")]


def test_one_candidate_per_item() -> None:
    result = run(
        "Patient: I take metformin.\nDoctor: Metformin, right.\nPatient: Yes, metformin.", NOTE
    )
    assert facts(result) == [("medicine", "metformin")]


# Present in the note: no candidate


@pytest.mark.parametrize(
    ("transcript", "note"),
    [
        ("Patient: I take Glucophage.", "Medications:\nmetformin\n"),  # brand vs generic
        ("Patient: I take metformin.", "Medications:\nGlucophage\n"),
        ("Patient: I take Augmentin.", "Medications:\namoxicillin / clavulanate\n"),
        ("Patient: I take metformin.", "Plan:\nMetformin stopped.\n"),  # any attribute
        ("Patient: I take metformin 500 mg.", "Medications:\nmetformin 1 g\n"),  # Pass B's
        ("Patient: I take metformin twice daily.", "Medications:\nmetformin TDS\n"),
        ("Patient: I take metformin one tablet.", "Medications:\nmetformin 1 tablet\n"),
        ("Patient: I'm allergic to penicillin.", "Medications:\nPenicillin allergy noted.\n"),
        ("Patient: I'm allergic to penicillin.", "Allergies: NKDA\n"),  # a contradiction
        ("Patient: I'm allergic to penicillin.", "History:\nPenicillin causes a rash.\n"),
        ("Patient: My left knee is swollen.", "History:\nRight knee swelling.\n"),  # Pass B's
        ("Patient: My left knee is swollen.", "History:\nBoth knees swollen.\n"),
    ],
)
def test_present_in_the_note(transcript: str, note: str) -> None:
    assert run(transcript, note).candidates == ()


# Not established by the transcript: no candidate


@pytest.mark.parametrize(
    "transcript",
    [
        "Patient: My mum takes metformin.",  # family
        "Doctor: If the pain gets worse take ibuprofen.",  # hypothetical
        "Patient: I'm not taking metformin.",  # negated
        "Patient: Maybe metformin.",  # possible
        "Patient: I'm allergic to nothing.",  # no allergen
        "Patient: My husband is allergic to penicillin.",  # someone else's allergy
        "Patient: I might be allergic to penicillin.",  # possible allergy
        "Doctor: Are you allergic to penicillin?\nPatient: Not sure.",  # a question
        "Patient: My mum's left knee is swollen.",  # someone else's side
    ],
)
def test_not_the_patients_current_fact(transcript: str) -> None:
    result = run(transcript, NOTE)
    assert result.candidates == ()


@pytest.mark.parametrize(
    ("transcript", "reason"),
    [
        ("Patient: I used to take metformin.", "planned_or_past"),
        ("Doctor: Are you on metformin?\nPatient: Mm.", "planned_or_past"),  # a question
        ("Patient: I take metfromin.", "spelling_tolerance"),
    ],
)
def test_guarded_medicines_are_unchecked(transcript: str, reason: str) -> None:
    result = run(transcript, NOTE)
    assert result.candidates == ()
    assert reasons(result) == {("medicine", "metformin", reason)}


def test_a_historical_medicine_is_not_a_current_one() -> None:
    result = run("Patient: I had metformin years ago.", NOTE)
    assert (result.candidates, result.unchecked) == ((), ())


@pytest.mark.parametrize(
    ("transcript", "reason"),
    [
        (
            "Patient: I take metformin 500 mg.\nDoctor: The letter says metformin 1 g.",
            "transcript_inconsistent",
        ),
        ("Patient: I take metformin 500 mg. Actually it's 1 g.", "unattributed_value"),
        ("Patient: I take metformin 500 mg gliclazide.", "ambiguous_value"),
        ("Patient: I take metformin up to 500 mg.", "range_or_maximum"),
        ("Patient: I take two metformin 500 mg tablets.", "tablet_count"),
    ],
)
def test_guarded_doses_are_unchecked(transcript: str, reason: str) -> None:
    result = run(transcript, "Medications:\nmetformin\ngliclazide\n")
    assert [c.category for c in result.candidates] == []
    assert ("dose", "metformin", reason) in reasons(result)


@pytest.mark.parametrize(
    ("transcript", "reason"),
    [
        ("Patient: My left knee and my right knee hurt.", "transcript_inconsistent"),
        ("Patient: My knees hurt, the left knee most.", "plural_site"),
    ],
)
def test_guarded_sides_are_unchecked(transcript: str, reason: str) -> None:
    result = run(transcript, "History:\nKnee pain.\n")
    assert result.candidates == ()
    assert reasons(result) == {("side", "knee", reason)}


def test_unreviewed_names_may_establish_a_medicine() -> None:
    candidate = only(run("Patient: I take paracetamol every day.", NOTE))
    assert candidate.fact == "acetaminophen"


def test_omission_candidates_ignore_values_after_a_separator() -> None:
    result = run("Patient: I take metformin, 500 mg twice daily.", "Medications:\nmetformin\n")
    assert result.candidates == ()


# Change 1: a planned medicine may give a candidate, tagged for the Plan section


@pytest.mark.parametrize(
    "transcript",
    [
        "Doctor: Let's start metformin.",
        "Doctor: I'm going to put you on metformin.",
        "Doctor: We'll start metformin next week.",
        "Doctor: I'll prescribe metformin.",
    ],
)
def test_a_planned_medicine_missing_from_the_note(transcript: str) -> None:
    candidate = only(run(transcript, NOTE))
    assert (candidate.rule_id, candidate.fact, candidate.expected_section) == (
        "medicine.missing",
        "metformin (planned)",
        "Plan",
    )


def test_a_current_mention_outranks_a_planned_one() -> None:
    candidate = only(run("Patient: I take metformin.\nDoctor: Let's increase metformin.", NOTE))
    assert (candidate.fact, candidate.expected_section) == ("metformin", "Medications")


def test_planned_values_never_give_dose_or_frequency_candidates() -> None:
    result = run("Doctor: We'll start metformin 500 mg twice daily.", "Medications:\nmetformin\n")
    assert result.candidates == ()


@pytest.mark.parametrize(
    "transcript",
    [
        "Patient: I stopped metformin.",
        "Patient: I used to take metformin.",
        "Patient: I was taking metformin.",
        "Patient: I'm off metformin now.",
        "Patient: I quit metformin.",
        "Patient: I gave up aspirin.",
    ],
)
def test_past_and_stopped_use_never_gives_a_candidate(transcript: str) -> None:
    assert run(transcript, NOTE).candidates == ()


# Change 2: a side candidate only when the note names the site without a side


def test_no_side_candidate_when_the_note_does_not_name_the_site() -> None:
    assert run("Patient: My left knee is swollen.", NOTE).candidates == ()


def test_a_side_word_on_the_same_line_counts_as_present() -> None:
    assert run("Patient: My left knee is swollen.", "History:\nKnee pain, left.\n").candidates == ()


# Review findings


def test_a_drug_in_an_allergy_statement_is_not_a_medicine() -> None:
    result = run("Patient: I'm allergic to amoxicillin.", NOTE)
    assert [c.category for c in result.candidates] == ["allergy"]


@pytest.mark.parametrize(
    "transcript",
    [
        "Patient: He takes metformin.",
        "Patient: She is on metformin.",
        "Patient: He's allergic to penicillin.",
        "Patient: My neighbour takes metformin.",
        "Patient: My flatmate is allergic to penicillin.",
    ],
)
def test_someone_elses_facts_give_no_candidate(transcript: str) -> None:
    assert run(transcript, NOTE).candidates == ()


def test_a_parent_speaking_of_the_child_is_the_patient_side() -> None:
    assert [c.category for c in run("Parent: He takes metformin.", NOTE).candidates] == ["medicine"]


@pytest.mark.parametrize(
    "transcript",
    [
        "Patient: I don't take aspirin.",
        "Patient: I can't take aspirin.",
        "Patient: I didn't take aspirin.",
        "Patient: I haven't taken aspirin.",
    ],
)
def test_negated_use_gives_no_candidate(transcript: str) -> None:
    assert run(transcript, NOTE).candidates == ()


@pytest.mark.parametrize(
    "transcript",
    [
        "Patient: Aspirin. I stopped it last week.",
        "Patient: I take aspirin.\nDoctor: OK.\nPatient: Actually I stopped it.",
    ],
)
def test_a_stop_without_a_drug_name_leaves_medicines_unchecked(transcript: str) -> None:
    result = run(transcript, NOTE)
    assert result.candidates == ()
    assert reasons(result) == {("medicine", "aspirin", "planned_or_past")}


@pytest.mark.parametrize(
    "transcript",
    [
        "Doctor: Are you allergic to penicillin\nPatient: No.",
        "Doctor: Are you on metformin\nPatient: No.",
        "Patient: Should I take metformin.",
    ],
)
def test_questions_without_a_question_mark(transcript: str) -> None:
    assert run(transcript, NOTE).candidates == ()


def test_an_outgrown_allergy_is_not_current() -> None:
    result = run("Patient: I used to be allergic to penicillin but outgrew it.", NOTE)
    assert result.candidates == ()


@pytest.mark.parametrize(
    "transcript",
    [
        "Doctor: Is it your left knee?\nPatient: No, the right.",  # a question
        "Doctor: Raise your left knee for me.",  # an instruction
        "Patient: My left knee doesn't hurt.",  # negated
        "Patient: My left knee is fine.",
    ],
)
def test_sides_only_from_the_patients_own_statements(transcript: str) -> None:
    assert run(transcript, "History:\nKnee pain.\n").candidates == ()


def test_lying_on_one_side_is_not_a_body_site() -> None:
    assert (
        run("Patient: I sleep on my left side.", "History:\nPain in the side.\n").candidates == ()
    )


def test_a_person_is_never_an_allergen() -> None:
    result = run("Patient: I'm allergic to dust and my husband to cats.", NOTE)
    assert all("husband" not in (c.fact or "") for c in result.candidates)


@pytest.mark.parametrize(
    "note",
    [
        "Medications: metformin: 500 mg twice daily\n",
        "Medications: metformin (500 mg twice daily)\n",
        "Medications:\nmetformin, 500 mg twice daily\n",
        "Medications:\n- Metformin\n  - 500 mg\n  - twice daily\n",
        "Medications:\nMetformin\nDose: 500 mg\nFrequency: twice daily\n",
        "Medications: Metformin. Dose 500 mg. Twice daily.\n",
    ],
)
def test_note_layouts_do_not_hide_values(note: str) -> None:
    assert run("Patient: I take metformin 500 mg twice a day.", note).candidates == ()


def test_a_blank_line_ends_a_drugs_lines() -> None:
    result = run(
        "Patient: I take metformin 500 mg twice a day.",
        "Medications:\nmetformin\n\nPlan:\nreview in 500 mg twice daily clinic\n",
    )
    assert {c.category for c in result.candidates} == {"dose", "frequency"}


@pytest.mark.parametrize(
    ("allergen", "note_allergen"),
    [("peanuts", "peanut"), ("shellfish", "shell fish"), ("penicillin", "penicillins")],
)
def test_allergen_spelling_variants_count_as_present(allergen: str, note_allergen: str) -> None:
    result = run(f"Patient: I'm allergic to {allergen}.", f"Allergies: {note_allergen}\n")
    assert result.candidates == ()


def test_allergen_variants_give_one_candidate() -> None:
    result = run("Patient: I'm allergic to peanuts.\nPatient: Yes, allergic to peanut.", NOTE)
    assert [c.category for c in result.candidates] == ["allergy"]


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        ("Allergies: No allergies.\n", 0),
        ("Allergies: NKDA\n", 0),  # penicillin is a drug allergen
        ("Allergies: Not allergic to latex.\n", 1),  # covers latex only
        ("History: Mother is not allergic to anything.\n", 1),  # not the patient's
        ("Allergies: no known food allergies\n", 1),  # covers food only
    ],
)
def test_allergy_denials_cover_only_what_they_say(note: str, expected: int) -> None:
    result = run("Patient: I'm allergic to penicillin.", note)
    assert len(result.candidates) == expected


def test_nkda_does_not_cover_a_food_allergy() -> None:
    result = run("Patient: I'm allergic to peanuts.", "Allergies: NKDA\n")
    assert [c.category for c in result.candidates] == ["allergy"]


def test_count_units_display_in_the_plural() -> None:
    candidate = only(
        run(
            "Patient: I take insulin glargine 10 units at night.",
            "Medications:\ninsulin glargine at night\n",
        )
    )
    assert candidate.fact == "insulin glargine 10 units"


def test_a_year_is_not_a_tablet_count() -> None:
    candidate = only(
        run(
            "Patient: I take metformin 500 mg in the morning since 2019.",
            "Medications:\nmetformin in the morning\n",
        )
    )
    assert candidate.fact == "metformin 500 mg"


# A drug from a recounted past episode is not a current medicine (synthetic, after a
# development case where a past emergency visit's "don't give morphine" became a candidate).

EPISODE_NOTE = "Past medical history:\nSevere kidney infection, seen in A&E.\n"


def test_past_episode_morphine_regression() -> None:
    transcript = (
        "Doctor: Any other medical problems before?\n"
        "Patient: Once I had a really bad kidney infection, and they had me in A&E.\n"
        "Patient: And my GP called them to tell them they don't need to give me morphine.\n"
        "Patient: That was awful, but otherwise I'm well.\n"
    )
    result = run(transcript, EPISODE_NOTE)
    assert (result.candidates, result.unchecked) == ((), ())  # negated: not the patient's
    # Without the negation, the past episode alone keeps it from being a candidate.
    result = run(transcript.replace("don't need to give", "gave"), EPISODE_NOTE)
    assert result.candidates == ()
    assert reasons(result) == {("medicine", "morphine", "planned_or_past")}


@pytest.mark.parametrize(
    "transcript",
    [
        "Patient: They don't need to give me morphine.",
        "Patient: My GP said there's no need to give me morphine.",
        "Doctor: We shouldn’t give you morphine.",
        "Patient: The GP advised against morphine.",
    ],
)
def test_no_need_to_give_negates_the_drug(transcript: str) -> None:
    assert run(transcript, NOTE).candidates == ()


@pytest.mark.parametrize(
    "transcript",
    [
        "Patient: Once I had a kidney stone and they had me in A&E.\n"
        "Patient: They gave me morphine for the pain.",
        "Patient: I was admitted with pneumonia and they gave me amoxicillin.",
        "Patient: Back then I was on a drip. They gave me morphine.",
        "Patient: That time they put me on morphine.",
    ],
)
def test_a_drug_in_a_recounted_past_episode_is_past_use(transcript: str) -> None:
    result = run(transcript, NOTE)
    assert result.candidates == ()
    assert {u.reason for u in result.unchecked} == {"planned_or_past"}


@pytest.mark.parametrize(
    "transcript",
    [
        "Patient: I take morphine every day.",
        "Patient: I don't need to give up my metformin. I take it every day.",
        # The episode is in another speaker's turn, or two sentences back.
        "Doctor: Once you were in A&E?\nPatient: I take morphine every day.",
        "Patient: I was in A&E once. I'm fine now. I take morphine every day.",
        # Review findings: ordinary current use near an episode-like phrase.
        "Patient: I take metformin 500 mg twice a day at the time of breakfast and dinner.",
        "Patient: I take vitamins A and E and metformin 500 mg twice a day.",
        "Patient: Twice a day, er, the er metformin.",
        "Patient: I was in hospital last year for my heart. Now I take bisoprolol 5 mg daily.",
        "Patient: I was in hospital last year for my heart.\nPatient: I take bisoprolol 5 mg.",
        "Patient: I had a fall that time.\nPatient: I take bisoprolol 5 mg daily.",
        "Patient: I take bisoprolol 5 mg daily, as they told me in the hospital.",
        "Patient: I don't give myself my metformin 500 mg on Sundays.",
    ],
)
def test_current_use_still_raises_a_candidate(transcript: str) -> None:
    assert [c.category for c in run(transcript, NOTE).candidates] == ["medicine"]


# An explicit "no allergies" the note does not record


@pytest.mark.parametrize(
    ("said", "fact"),
    [
        ("Patient: NKDA.", "no known drug allergies"),
        ("Patient: No known drug allergies.", "no known drug allergies"),
        ("Patient: I don't have any allergies.", "no known allergies"),
        ("Doctor: Any allergies?\nPatient: Not allergic to anything.", "no known allergies"),
        ("Doctor: So no known allergies, OK.", "no known allergies"),
    ],
)
def test_no_allergy_status_missing(said: str, fact: str) -> None:
    candidate = only(run(said, NOTE))
    assert (candidate.category, candidate.rule_id, candidate.fact) == (
        "allergy",
        "allergy.status_missing",
        fact,
    )
    assert (candidate.severity, candidate.expected_section) == ("critical", "Allergies")


@pytest.mark.parametrize(
    "note",
    [
        "History:\nFeeling well.\nNKDA.\n",
        "History:\nFeeling well.\n\nAllergies:\nNo allergies.\n",
        "History:\nFeeling well.\n\nAllergies:\nNone known.\n",
        "History:\nFeeling well.\n\nAllergies:\nNil.\n",
        "History:\nAllergic to latex.\n",  # a named allergen: Pass B's business
        "History:\nNot allergic to penicillin.\n",
        "PMH:\nAllergy status unknown.\n",  # any allergy word records the status
        "History:\nFeeling well. NKA.\n",
    ],
)
def test_no_allergy_status_recorded_in_the_note(note: str) -> None:
    assert run("Patient: No known drug allergies.", note).candidates == ()


@pytest.mark.parametrize(
    "said",
    [
        "Doctor: Any allergies?",  # a question
        "Doctor: Do you have any allergies?\nPatient: Um.",
        "Patient: My son has no allergies.",  # someone else's
        "Patient: Maybe no allergies, I'm not sure.",  # possible
        # contradicted, even by a hedged, past or person-word allergen
        "Patient: No allergies.\nPatient: Oh, actually I'm allergic to penicillin.",
        "Patient: No known allergies. I might be allergic to penicillin though.",
        "Patient: No allergies. I was allergic to penicillin as a child.",
        "Patient: NKDA. But I'm allergic to my mum's dog.",
    ],
)
def test_no_allergy_status_candidate_guards(said: str) -> None:
    assert [
        c.fact for c in run(said, NOTE).candidates if c.rule_id == "allergy.status_missing"
    ] == []
