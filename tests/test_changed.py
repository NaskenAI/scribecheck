"""Pass B rules: certain changed-fact flags, and unchecked items with reasons.

Certain flags are disabled until they come only from allow-listed sentence shapes
(rules/changed.py, CERTAIN_FLAGS_ENABLED). Meanwhile an item that would have been a
certain flag is unchecked with reason "certain_disabled_pending_allowlist"; these tests
call it a candidate and check it as they would the flag, so the comparison logic stays
covered. Every case is synthetic, and every quote must slice back exactly from its source.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pytest

from scribecheck.flags import ChangedFactResult, NoteEvidence, TranscriptEvidence, verify_evidence
from scribecheck.note import parse_note
from scribecheck.rules.changed import (
    CERTAIN_DISABLED,
    CERTAIN_FLAGS_ENABLED,
    RULE_ID,
    SEVERITY,
    check_changed_facts,
)
from scribecheck.transcript import parse_transcript


def check(
    transcript: str, note: str, *, brands: Mapping[str, Sequence[str]] | None = None
) -> ChangedFactResult:
    parsed_transcript, parsed_note = parse_transcript(transcript), parse_note(note)
    result = check_changed_facts(parsed_transcript, parsed_note, brands=brands)
    for item in (*result.flags, *result.unchecked):
        verify_evidence(item, parsed_transcript, parsed_note)
    assert [flag.id for flag in result.flags] == [f"F{i}" for i in range(1, len(result.flags) + 1)]
    return result


@dataclass(frozen=True)
class Candidate:
    """An item that would have been a certain flag, as the flag would have reported it."""

    category: str
    rule_id: str
    severity: str
    transcript_value: str | None
    note_value: str | None
    transcript_evidence: TranscriptEvidence | None
    note_evidence: NoteEvidence | None


def candidates(result: ChangedFactResult) -> list[Candidate]:
    assert not CERTAIN_FLAGS_ENABLED and result.flags == ()
    return [
        Candidate(
            item.category,
            dict(RULE_ID)[item.category],
            dict(SEVERITY)[item.category],
            item.transcript_value,
            item.note_value,
            item.transcript_evidence[0] if item.transcript_evidence else None,
            item.note_evidence[0] if item.note_evidence else None,
        )
        for item in result.unchecked
        if item.reasons == (CERTAIN_DISABLED,)
    ]


def only_flag(result: ChangedFactResult) -> Candidate:
    (candidate,) = candidates(result)
    return candidate


def reasons(result: ChangedFactResult) -> set[tuple[str, str, str]]:
    """(category, item, primary reason) for every unchecked item that is not a candidate."""
    found = set()
    for item in result.unchecked:
        if item.reasons != (CERTAIN_DISABLED,):
            assert item.reasons[-1] == CERTAIN_DISABLED  # every item carries the marker
            found.add((item.category, item.item, item.reason))
    return found


def no_flags(result: ChangedFactResult) -> None:
    """No certain flag, and nothing that would have been one."""
    assert result.flags == () and candidates(result) == []


# Positive cases, one per flag type


def test_dose_changed() -> None:
    flag = only_flag(
        check(
            "Patient: I take metformin 500 mg twice daily.",
            "Medications:\nmetformin 1 g twice daily\n",
        )
    )
    assert (flag.rule_id, flag.category, flag.severity) == ("dose.mismatch", "dose", "critical")
    assert (flag.transcript_value, flag.note_value) == ("500 mg", "1 g")
    assert flag.transcript_evidence is not None and flag.note_evidence is not None
    assert (flag.transcript_evidence.text, flag.transcript_evidence.turn) == ("metformin 500 mg", 1)
    assert (flag.note_evidence.text, flag.note_evidence.section) == ("metformin 1 g", "Medications")


def test_equal_doses_in_different_units_are_not_a_change() -> None:
    result = check("Patient: I take metformin 1 g.", "Medications:\nmetformin 1000 mg\n")
    assert (result.flags, result.unchecked) == ((), ())


def test_frequency_changed() -> None:
    flag = only_flag(check("Doctor: Take ramipril 5 mg BD.", "Plan:\nramipril 5 mg TDS\n"))
    assert (flag.rule_id, flag.category, flag.severity) == (
        "frequency.mismatch",
        "frequency",
        "critical",
    )
    assert (flag.transcript_value, flag.note_value) == ("BD", "TDS")


def test_route_changed() -> None:
    flag = only_flag(check("Doctor: Take metformin 500 mg orally.", "Plan:\nmetformin 500 mg IV\n"))
    assert (flag.rule_id, flag.category, flag.severity) == ("route.mismatch", "route", "supporting")
    assert (flag.transcript_value, flag.note_value) == ("oral", "intravenous")


def test_side_changed() -> None:
    flag = only_flag(
        check("Patient: My left knee has been swelling.", "History:\nRight knee swelling.\n")
    )
    assert (flag.rule_id, flag.category, flag.severity) == ("side.mismatch", "side", "critical")
    assert (flag.transcript_value, flag.note_value) == ("left knee", "right knee")
    assert flag.transcript_evidence is not None and flag.note_evidence is not None
    assert (flag.transcript_evidence.text, flag.note_evidence.text) == ("left knee", "Right knee")


def test_quadrant_sides_align() -> None:
    flag = only_flag(check("Doctor: The pain is in the LLQ.", "Examination:\nTender RLQ.\n"))
    assert (flag.transcript_value, flag.note_value) == (
        "left lower quadrant",
        "right lower quadrant",
    )


def test_negation_reversed_denied_then_written_as_present() -> None:
    flag = only_flag(check("Patient: No chest pain at all.", "History:\nChest pain on exertion.\n"))
    assert (flag.rule_id, flag.category, flag.severity) == (
        "negation.reversed",
        "symptom",
        "critical",
    )
    assert (flag.transcript_value, flag.note_value) == ("denied", "present")
    assert flag.transcript_evidence is not None
    assert flag.transcript_evidence.text == "No chest pain"


def test_negation_reversed_present_then_written_as_denied() -> None:
    flag = only_flag(check("Patient: I've got chest pain.", "History:\nNo chest pain.\n"))
    assert (flag.transcript_value, flag.note_value) == ("present", "denied")


def test_negation_reversed_through_dialogue_negation() -> None:
    flag = only_flag(check("Doctor: Any chest pain?\nPatient: No.", "History:\nChest pain.\n"))
    assert (flag.rule_id, flag.transcript_value) == ("negation.reversed", "denied")
    assert flag.transcript_evidence is not None
    assert (flag.transcript_evidence.text, flag.transcript_evidence.turn) == ("No.", 2)


def test_a_bare_yes_is_patient_side_presence() -> None:
    flag = only_flag(check("Doctor: Any chest pain?\nPatient: Yes.", "History:\nNo chest pain.\n"))
    assert (flag.transcript_value, flag.note_value) == ("present", "denied")


def test_negation_without_speaker_labels_still_counts() -> None:
    flag = only_flag(check("No chest pain at all.", "History:\nChest pain.\n"))
    assert flag.transcript_value == "denied"


def test_combination_products_align_by_ingredient_set() -> None:
    flag = only_flag(
        check(
            "Patient: I take Augmentin 625 mg three times a day.",
            "Medications:\namoxicillin / clavulanate 625 mg twice daily\n",
        )
    )
    assert flag.rule_id == "frequency.mismatch"


def test_brand_aligns_with_its_generic() -> None:
    flag = only_flag(check("Patient: I take Glucophage 500 mg.", "Medications:\nmetformin 1 g\n"))
    assert flag.rule_id == "dose.mismatch"


def test_candidates_come_in_order() -> None:
    result = check(
        "Patient: I take metformin 500 mg twice daily. My left knee is swollen.",
        "History:\nRight knee swollen.\nMedications:\nmetformin 1 g three times a day\n",
    )
    assert [c.rule_id for c in candidates(result)] == [
        "dose.mismatch",
        "frequency.mismatch",
        "side.mismatch",
    ]


# No certain flag: the transcript must give one consistent value, matched on temporality


def test_dose_changed_during_the_consultation() -> None:
    result = check(
        "Doctor: You take metformin 500 mg twice daily.\n"
        "Doctor: Let's increase the metformin to 1 g twice daily.",
        "Plan:\nmetformin 1 g twice daily\n",
    )
    no_flags(result)
    assert ("dose", "metformin", "planned_or_past") in reasons(result)


def test_two_different_transcript_values() -> None:
    result = check(
        "Patient: I take metformin 500 mg.\nDoctor: The letter says metformin 1 g.",
        "Medications:\nmetformin 1 g\n",
    )
    no_flags(result)
    assert reasons(result) == {("dose", "metformin", "transcript_inconsistent")}


def test_both_knees_discussed() -> None:
    result = check(
        "Patient: My left knee hurts and my right knee is swollen.",
        "History:\nRight knee pain.\n",
    )
    no_flags(result)
    assert reasons(result) == {("side", "knee", "transcript_inconsistent")}


def test_symptom_present_last_week_and_absent_today() -> None:
    result = check(
        "Patient: Chest pain last week but no chest pain today.", "History:\nChest pain.\n"
    )
    no_flags(result)
    assert reasons(result) == {("symptom", "chest pain", "transcript_inconsistent")}


def test_a_historical_mention_blocks_certainty() -> None:
    result = check(
        "Patient: I had chest pain last week. No chest pain today.", "History:\nChest pain.\n"
    )
    no_flags(result)
    assert reasons(result) == {("symptom", "chest pain", "mixed_temporality")}


def test_note_with_two_values() -> None:
    result = check(
        "Patient: I take metformin 500 mg.", "Medications:\nmetformin 500 mg\nmetformin 1 g\n"
    )
    no_flags(result)
    assert reasons(result) == {("dose", "metformin", "note_inconsistent")}


def test_values_that_cannot_be_compared_are_unchecked_not_changed() -> None:
    result = check("Doctor: Take metformin 500 mg.", "Medications:\nmetformin 5 ml\n")
    no_flags(result)
    assert reasons(result) == {("dose", "metformin", "not_comparable")}


# No certain flag: OD, OS, OU


def test_od_as_once_daily() -> None:
    result = check("Doctor: Take ramipril 5 mg OD.", "Plan:\nramipril 5 mg BD\n")
    no_flags(result)
    assert reasons(result) == {("frequency", "ramipril", "eye_abbreviation")}


@pytest.mark.parametrize("abbreviation", ["OD", "OU"])
def test_eye_side_from_an_abbreviation(abbreviation: str) -> None:
    result = check(
        "Patient: The drops are for my left eye.", f"Plan:\nEye drops 1 drop {abbreviation}.\n"
    )
    no_flags(result)
    assert reasons(result) == {("side", "eye", "eye_abbreviation")}


# No certain flag: spelling tolerance


@pytest.mark.parametrize(
    ("transcript", "note"),
    [
        ("Patient: I take metfromin 500 mg.", "Medications:\nmetformin 1 g\n"),
        ("Patient: I take metformin 500 mg.", "Medications:\nmetfromin 1 g\n"),
    ],
)
def test_spelling_tolerant_match(transcript: str, note: str) -> None:
    result = check(transcript, note)
    no_flags(result)
    assert reasons(result) == {("dose", "metformin", "spelling_tolerance")}


# No certain flag: unreviewed data


@pytest.mark.parametrize(
    ("transcript", "note", "item"),
    [
        ("Patient: I take paracetamol 500 mg.", "Medications:\nacetaminophen 1 g\n", "dose"),
        ("Doctor: Domperidone 10 mg TDS.", "Plan:\ndomperidone 10 mg BD\n", "frequency"),
    ],
)
def test_unreviewed_drug_names(transcript: str, note: str, item: str) -> None:
    result = check(transcript, note)
    no_flags(result)
    assert {reason for _, _, reason in reasons(result)} == {"unreviewed_data"}
    assert {category for category, _, _ in reasons(result)} == {item}


def test_user_brand_resolved_through_an_international_name() -> None:
    result = check(
        "Patient: I take Examplol 500 mg.",
        "Medications:\nExamplol 1 g\n",
        brands={"Examplol": ["paracetamol"]},
    )
    no_flags(result)
    assert reasons(result) == {("dose", "acetaminophen", "unreviewed_data")}


@pytest.mark.parametrize(
    ("transcript", "note", "term"),
    [
        ("Patient: I'm not short of breath.", "History:\nShort of breath.\n", "dyspnea"),
        ("Patient: No tiredness.", "History:\nTiredness.\n", "fatigue"),
    ],
)
def test_unreviewed_symptom_forms(transcript: str, note: str, term: str) -> None:
    result = check(transcript, note)
    no_flags(result)
    assert reasons(result) == {("symptom", term, "unreviewed_data")}


# No certain flag: a route outside the drug's dosing phrase


@pytest.mark.parametrize(
    ("transcript", "note"),
    [
        ("Doctor: Use nystatin 1 ml for oral thrush.", "Plan:\nnystatin 1 ml IV\n"),
        ("Doctor: Take metformin 500 mg. Orally, with food.", "Plan:\nmetformin 500 mg IV\n"),
        (
            "Doctor: Take metformin 500 mg and aspirin 75 mg orally.",
            "Plan:\nmetformin 500 mg IV\naspirin 75 mg orally\n",
        ),
    ],
)
def test_route_outside_the_dosing_phrase(transcript: str, note: str) -> None:
    result = check(transcript, note)
    assert [c.rule_id for c in candidates(result) if c.category == "route"] == []


# No certain flag: who said the symptom was present


def test_clinician_only_presence() -> None:
    result = check("Doctor: So you have chest pain.\nPatient: Mm.", "History:\nNo chest pain.\n")
    no_flags(result)
    assert reasons(result) == {("symptom", "chest pain", "clinician_only")}


def test_presence_without_speaker_labels() -> None:
    result = check("I have chest pain.", "History:\nNo chest pain.\n")
    no_flags(result)
    assert reasons(result) == {("symptom", "chest pain", "no_speaker_labels")}


@pytest.mark.parametrize(
    "answer", ["Not that I've noticed.", "Yes, sometimes.", "Yeah but only at night."]
)
def test_unanswered_question(answer: str) -> None:
    result = check(f"Doctor: Any chest pain?\nPatient: {answer}", "History:\nNo chest pain.\n")
    no_flags(result)
    assert reasons(result) == {("symptom", "chest pain", "question_unanswered")}


def test_patient_presence_outweighs_a_clinician_summary() -> None:
    flag = only_flag(
        check(
            "Patient: I get chest pain.\nDoctor: So you have chest pain.",
            "History:\nNo chest pain.\n",
        )
    )
    assert flag.transcript_value == "present"


# Context: family, hypothetical and possible mentions


def test_family_mentions_are_not_the_patients() -> None:
    result = check("Patient: My mum has chest pain.", "History:\nNo chest pain.\n")
    assert (result.flags, result.unchecked) == ((), ())


def test_family_side_mention_does_not_make_the_transcript_inconsistent() -> None:
    flag = only_flag(
        check(
            "Patient: My mum's left knee is bad. My right knee is swollen.",
            "History:\nLeft knee swollen.\n",
        )
    )
    assert (flag.transcript_value, flag.note_value) == ("right knee", "left knee")


def test_hypothetical_mentions_are_not_facts() -> None:
    result = check(
        "Doctor: If you get chest pain, come straight back.", "History:\nNo chest pain.\n"
    )
    assert (result.flags, result.unchecked) == ((), ())


def test_possible_only_mention() -> None:
    result = check("Patient: Maybe some chest pain.", "History:\nNo chest pain.\n")
    no_flags(result)
    assert reasons(result) == {("symptom", "chest pain", "possible")}


# Alignment


def test_drug_on_one_side_only_is_unaligned() -> None:
    result = check("Patient: I take aspirin 75 mg.", "History:\nKnee pain.\n")
    no_flags(result)
    assert reasons(result) == {("drug", "aspirin", "unaligned")}


def test_partial_ingredient_overlap_does_not_align() -> None:
    result = check(
        "Patient: I take Augmentin 625 mg three times a day.",
        "Medications:\namoxicillin 500 mg twice daily\n",
    )
    no_flags(result)
    assert reasons(result) == {
        ("drug", "amoxicillin / clavulanate", "unaligned"),
        ("drug", "amoxicillin", "unaligned"),
    }


def test_a_dose_that_belongs_to_the_next_drug_is_not_attributed() -> None:
    result = check(
        "Patient: I take metformin and gliclazide 80 mg.",
        "Medications:\nmetformin 500 mg\ngliclazide 80 mg\n",
    )
    assert (result.flags, result.unchecked) == ((), ())


# Regression: every false certain flag the review reproduced. None may flag.

REVIEW_CASES = [
    # 1. questions, planned changes, past use
    (
        "Doctor: Are you on metformin 1000 mg?\nPatient: No, 500 mg.",
        "Meds: metformin 500 mg.",
        "planned_or_past",
    ),
    (
        "Patient: I take metformin 500 mg twice a day.\n"
        "Doctor: Let's increase it to 1000 mg twice a day.\nPatient: OK.",
        "Plan: metformin 1000 mg twice a day.",
        "planned_or_past",
    ),
    (
        "Patient: I take metformin 500 mg.",
        "Plan: increase metformin to 1000 mg.",
        "planned_or_past",
    ),
    ("Patient: I stopped the amlodipine 10 mg.", "Meds: amlodipine 5 mg.", "planned_or_past"),
    # 2. other people
    ("Patient: My husband takes amlodipine 10 mg.", "Meds: amlodipine 5 mg.", None),
    ("Patient: My wife has a cough.", "HPI: No cough.", None),
    # 3. worry and uncertainty
    ("Patient: I'm worried I'll get a headache.", "HPI: Denies headache.", "possible"),
    # 4. negation scope
    ("Patient: I have a cough and a fever.", "HPI: No treatment tried for cough.", None),
    # 5. a value that belongs to something else
    ("Patient: I take metformin, 20 mg atorvastatin.", "Meds: metformin 500 mg.", None),
    # 6. tablet count against strength
    ("Patient: I take two metformin 500 mg tablets.", "Meds: metformin 1000 mg.", "tablet_count"),
    # 7. ranges
    (
        "Patient: Ibuprofen 400 mg 2-3 times a day.",
        "Meds: ibuprofen 400 mg twice a day.",
        "range_or_maximum",
    ),
    # 8. a bare yes later taken back
    (
        "Doctor: Any chest pain?\nPatient: Yes.\nDoctor: Is it still there?\n"
        "Patient: No, it went away.",
        "HPI: No chest pain currently.",
        "status_change",
    ),
    # 9. plural site
    (
        "Patient: My knees hurt, especially the left knee.",
        "Exam: right knee tender.",
        "plural_site",
    ),
]  # fmt: skip


@pytest.mark.parametrize(("transcript", "note", "reason"), REVIEW_CASES)
def test_review_cases_give_no_certain_flag(transcript: str, note: str, reason: str | None) -> None:
    result = check(transcript, note)
    no_flags(result)
    if reason is not None:
        assert reason in {r for _, _, r in reasons(result)}


# Each guard, with a case of its own


@pytest.mark.parametrize(
    ("transcript", "note"),
    [
        ("Patient: Should I take amlodipine 10 mg?\nDoctor: No.", "Meds: amlodipine 5 mg."),
        ("Patient: I used to take amlodipine 10 mg.", "Meds: amlodipine 5 mg."),
        ("Patient: I was on amlodipine 10 mg.", "Meds: amlodipine 5 mg."),
        ("Doctor: Stop the amlodipine 10 mg.", "Meds: amlodipine 5 mg."),
        ("Patient: I take amlodipine 5 mg.", "Plan: Start amlodipine 10 mg from tomorrow."),
        ("Patient: I take amlodipine 5 mg.", "Plan: titrate amlodipine up to 10 mg."),
        ("Patient: I take amlodipine 5 mg.", "Meds: amlodipine 10 mg discontinued."),
        (
            "Patient: I take metformin 500 mg once a day.\n"
            "Doctor: Take it twice a day from now on.",
            "Plan: metformin 500 mg twice a day.",
        ),
    ],
)
def test_planned_or_past(transcript: str, note: str) -> None:
    result = check(transcript, note)
    no_flags(result)
    assert "planned_or_past" in {r for _, _, r in reasons(result)}


def test_an_orphan_value_blocks_every_drug() -> None:
    result = check(
        "Patient: I take metformin 1000 mg. Actually it's 500 mg.",
        "Medications:\nmetformin 500 mg\n",
    )
    no_flags(result)
    assert reasons(result) == {("dose", "metformin", "unattributed_value")}


def test_a_value_between_two_drugs_is_ambiguous() -> None:
    result = check(
        "Patient: I take metformin 500 mg gliclazide.",
        "Medications:\nmetformin 1 g\ngliclazide 80 mg\n",
    )
    no_flags(result)
    assert ("dose", "metformin", "ambiguous_value") in reasons(result)


@pytest.mark.parametrize(
    "transcript",
    [
        "Patient: Ibuprofen 400 mg two or three times a day.",
        "Patient: Ibuprofen 400 mg one to two times a day.",
        "Patient: Ibuprofen 400 mg at most twice a day.",
        "Patient: Ibuprofen up to 400 mg three times a day.",
    ],
)
def test_ranges_and_maximums(transcript: str) -> None:
    result = check(transcript, "Meds: ibuprofen 200 mg once a day.")
    no_flags(result)
    assert {r for _, _, r in reasons(result)} == {"range_or_maximum"}


@pytest.mark.parametrize(
    "transcript",
    [
        "Patient: Metformin, I take two 500 mg tablets.",
        "Patient: Metformin 500 mg x2 daily.",
        "Patient: Ramipril 5 mg cut in half.",
        "Patient: I take metformin one tablet.",
    ],
)
def test_tablet_counts_are_kept_apart_from_strength(transcript: str) -> None:
    drug = "ramipril" if "amipril" in transcript else "metformin"
    result = check(transcript, f"Meds: {drug} 1000 mg.")
    no_flags(result)


def test_identical_counts_compare_strengths() -> None:
    flag = only_flag(
        check("Patient: I take two metformin 500 mg tablets.", "Meds: two metformin 1 g tablets.")
    )
    assert (flag.transcript_value, flag.note_value) == ("500 mg", "1 g")


@pytest.mark.parametrize(
    ("transcript", "note"),
    [
        ("Patient: Both knees hurt, the left knee most.", "Exam: right knee tender."),
        ("Patient: My left knee hurts.", "Exam: Both knees tender, right knee worse."),
        ("Patient: The left knee swells.", "Exam: Bilateral knee swelling."),
    ],
)
def test_plural_and_bilateral_sites(transcript: str, note: str) -> None:
    result = check(transcript, note)
    no_flags(result)


@pytest.mark.parametrize(
    "answer", ["Yes, but it went away.", "It has resolved now.", "It settled last week."]
)
def test_resolution_words_block_certainty(answer: str) -> None:
    result = check(f"Patient: I've had chest pain.\nPatient: {answer}", "HPI: No chest pain.")
    no_flags(result)


@pytest.mark.parametrize(
    ("transcript", "note"),
    [
        ("Patient: I have a cough.", "HPI: No fever, cough or headache."),  # list negation
        ("Patient: I have a cough.", "HPI: Denies any trigger for cough."),
        ("Patient: I have a headache.", "HPI: Unable to sleep without headache waking her."),
        ("Patient: There's no obvious cause for my chest pain.", "HPI: Chest pain."),
        ("Patient: I have no clue why I get chest pain.", "HPI: Chest pain."),
    ],
)
def test_negation_scope(transcript: str, note: str) -> None:
    no_flags(check(transcript, note))


@pytest.mark.parametrize(
    "transcript",
    [
        "Patient: My mum takes her amlodipine 10 mg every day.",
        "Patient: My friend is on amlodipine 10 mg.",
        "Patient: My little boy has a fever.",
        "Patient: My husband's left knee is bad.",
    ],
)
def test_other_peoples_facts(transcript: str) -> None:
    note = "Meds: amlodipine 5 mg. HPI: No fever. Exam: right knee."
    no_flags(check(transcript, note))


@pytest.mark.parametrize(
    "transcript",
    [
        "Patient: The doctor said I might have a headache.",
        "Patient: I think I have a headache, not sure.",
        "Patient: I don't want a headache again.",
        "Doctor: We could try amlodipine 10 mg.",
        "Doctor: Amlodipine 10 mg is an option.",
    ],
)
def test_uncertainty(transcript: str) -> None:
    no_flags(check(transcript, "HPI: Denies headache. Meds: amlodipine 5 mg."))


@pytest.mark.parametrize(
    "question",
    ["Did you have a fever over the weekend?", "Have you had a fever?", "Was there a fever?"],
)
def test_a_bare_yes_to_a_past_tense_question_binds_nothing(question: str) -> None:
    result = check(f"Doctor: {question}\nPatient: Yes.", "HPI: No fever.")
    no_flags(result)


@pytest.mark.parametrize(
    "transcript",
    [
        "Patient: Metformin 500 mg, I eat three times a day.",
        "Doctor: Amoxicillin 500 mg, we gave IV fluids too.",
        "Patient: Nystatin for thrush, which is oral.",
        "Patient: Levothyroxine, my B12 injection is 1000 mcg.",
    ],
)
def test_values_after_a_separator_belong_to_nothing(transcript: str) -> None:
    note = (
        "Meds: metformin 500 mg once daily. amoxicillin 500 mg oral. nystatin topical. "
        "levothyroxine 100 mcg."
    )
    no_flags(check(transcript, note))


# A patient reading back advice is not reporting a symptom


def test_a_patient_echoing_safety_net_advice_establishes_no_presence() -> None:
    result = check(
        "Doctor: Come back if you get chest pain or fainting. Any of those, don't sit on it.\n"
        "Patient: Chest pain, fainting, won't stop.",
        "History:\nNo chest pain during episodes.\n",
    )
    no_flags(result)
    assert reasons(result) == {("symptom", "chest pain", "unconfirmed_presence")}


def test_a_fragment_answering_a_plain_question_is_presence() -> None:
    flag = only_flag(
        check(
            "Doctor: What brings you in today?\nPatient: Chest pain.", "History:\nNo chest pain.\n"
        )
    )
    assert flag.transcript_value == "present"


def test_a_fragment_after_a_conditional_question_is_not_presence() -> None:
    result = check(
        "Doctor: What would make you come back if it got worse?\nPatient: Chest pain.",
        "History:\nNo chest pain.\n",
    )
    no_flags(result)
    assert reasons(result) == {("symptom", "chest pain", "unconfirmed_presence")}


# Regression: the second review's 13 groups of false certain flags.
#
# Certain flags are disabled, so none of these can flag today, and several are still
# candidates under the deny-list rules. They are here so that when the allow-list
# re-enables certain flags, every one of them must stay flag-free.

SECOND_REVIEW_CASES = [
    # 1. a symptom in advice, plans, other people's history or an indication in the note
    ("Doctor: Any fever?\nPatient: No.", "Safety netting: fever, chest pain."),
    ("Doctor: Any fever?\nPatient: No.", "Medications: paracetamol PRN fever."),
    ("Doctor: Any fever?\nPatient: No.", "FHx: fever."),
    # 2. a planned change in a sentence with no drug name and no value
    (
        "Patient: I take metformin 500 mg twice daily.\nDoctor: I'd like to double it.",
        "Medications: metformin 1000 mg twice daily.",
    ),
    # 3. plan, past and stopped words the lists missed
    ("Patient: I take ramipril 5 mg daily.", "Plan: Ramipril upped to 10 mg daily."),
    ("Doctor: I'll put you on ramipril 10 mg daily.", "Medications: ramipril 5 mg daily."),
    # 4. a negated side
    ("Doctor: Where is the pain?\nPatient: Not my left knee.", "History: right knee pain."),
    # 5. a side in a question
    ("Doctor: Is it your left knee?\nPatient: No, the other one.", "History: Right knee pain."),
    # 6. "the other one", "either", "each"
    ("Patient: My left knee and the other knee.", "History: right knee pain."),
    # 7. a Yes or No to a question about something else than the symptom
    ("Doctor: Any blood in the cough?\nPatient: No.", "History: Reports cough."),
    ("Doctor: Is there any improvement in the cough?\nPatient: Yes.", "History: No cough now."),
    # 8. other people and speaker roles
    ("Patient: My flatmate has a cough.", "History: No cough."),
    ("Carer: My cough is bad.", "History: No cough."),
    # 9. negation scope
    ("Patient: No cough medicine helps me.", "History: Reports cough."),
    # 10. habit, hope and uncertainty
    ("Patient: When I get a fever I take paracetamol.", "History: No fever."),
    # 11. resolution and past words the lists missed
    ("Doctor: How is the cough?\nPatient: I'm over my cough.", "History: No cough now."),
    # 12. a tablet count outside the drug's sentence
    (
        "Patient: I take metformin 500 mg twice daily. I take two each time.",
        "Medications: metformin 1000 mg twice daily.",
    ),
    # 13. total daily dose
    ("Patient: I take metformin 500 mg twice daily.", "Medications: metformin total 1 g daily."),
]


@pytest.mark.parametrize(("transcript", "note"), SECOND_REVIEW_CASES)
def test_second_review_cases_give_no_certain_flag(transcript: str, note: str) -> None:
    assert check(transcript, note).flags == ()
