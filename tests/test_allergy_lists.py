from __future__ import annotations

import pytest
from _spans import assert_round_trip

from scribecheck.note import parse_note
from scribecheck.rules.allergy import AllergyMatch, find_allergies, find_note_allergies
from scribecheck.transcript import parse_transcript

# (allergen or None, reaction or None, kind, status, possible)
Item = tuple[str | None, str | None, str, str, bool]


def _check(text: str, results: tuple[AllergyMatch, ...]) -> list[Item]:
    assert_round_trip(text, results)
    items: list[Item] = []
    for result in results:
        cue = result.cue
        assert text[cue.start : cue.end] == cue.text
        assert result.start <= cue.start < cue.end <= result.end or cue.end <= result.start
        parts = list(result.allergens)
        if result.reaction is not None:
            parts.append(result.reaction)
        for part in parts:
            assert text[part.start : part.end] == part.text
            assert result.start <= part.start < part.end <= result.end
        assert len(result.allergens) <= 1 or result.status == "allergic"
        allergen = result.allergens[0].text if result.allergens else None
        reaction = result.reaction.text if result.reaction else None
        items.append((allergen, reaction, result.kind, result.status, result.possible))
    return items


def _note(text: str) -> list[Item]:
    note = parse_note(text)
    return _check(note.text, find_note_allergies(note))


def _text(text: str) -> list[Item]:
    return _check(text, find_allergies(text))


# Note side: Allergies sections


def test_nkda_section() -> None:
    assert _note("Allergies: NKDA") == [(None, None, "unknown", "none", False)]


def test_bulleted_section_with_allergy_and_intolerance() -> None:
    assert _note("Allergies:\n- Penicillin (rash)\n- Codeine – nausea") == [
        ("Penicillin", "rash", "allergy", "allergic", False),
        ("Codeine", "nausea", "intolerance", "allergic", False),
    ]


def test_drug_allergies_section_with_semicolons() -> None:
    assert _note("Drug allergies: penicillin - anaphylaxis; latex - rash") == [
        ("penicillin", "anaphylaxis", "allergy", "allergic", False),
        ("latex", "rash", "allergy", "allergic", False),
    ]


def test_comma_separated_items_with_reactions() -> None:
    assert _note("Allergies:\npenicillin - rash, codeine - nausea") == [
        ("penicillin", "rash", "allergy", "allergic", False),
        ("codeine", "nausea", "intolerance", "allergic", False),
    ]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("penicillin?", ("penicillin", None, "unknown", "allergic", True)),
        ("?penicillin", ("penicillin", None, "unknown", "allergic", True)),
        ("possible penicillin", ("penicillin", None, "unknown", "allergic", True)),
        ("penicillin (rash?)", ("penicillin", "rash", "allergy", "allergic", True)),
        ("Penicillin - possible rash", ("Penicillin", "rash", "allergy", "allergic", True)),
        ("penicillin (rash, hives)", ("penicillin", "rash, hives", "allergy", "allergic", False)),
        ("co-trimoxazole - rash", ("co-trimoxazole", "rash", "allergy", "allergic", False)),
        ("penicillin: rash", ("penicillin", "rash", "allergy", "allergic", False)),
        ("penicillin — urticaria", ("penicillin", "urticaria", "allergy", "allergic", False)),
        ("penicillin", ("penicillin", None, "unknown", "allergic", False)),
        ("sulfa drugs - wheeze", ("sulfa drugs", "wheeze", "allergy", "allergic", False)),
        ("codeine (headache)", ("codeine", "headache", "intolerance", "allergic", False)),
        ("codeine - felt odd", ("codeine", "felt odd", "unknown", "allergic", False)),
        (
            "amoxicillin - rash and nausea",
            ("amoxicillin", "rash and nausea", "allergy", "allergic", False),
        ),
        ("NKA", (None, None, "unknown", "none", False)),
        ("nil known", (None, None, "unknown", "none", False)),
        ("None known", (None, None, "unknown", "none", False)),
        ("No known allergies", (None, None, "unknown", "none", False)),
        ("none", (None, None, "unknown", "none", False)),
    ],
)
def test_single_item(body: str, expected: Item) -> None:
    assert _note(f"Allergies: {body}") == [expected]
    assert _note(f"## Allergies\n{body}\n") == [expected]


@pytest.mark.parametrize(
    "text",
    [
        "Allergies: see chart",
        "Allergies: as above",
        "Allergies: see above",
        "Allergies: unknown",
        "Allergies: not recorded",
        "Allergies: drugs",
        "## Allergies\n\n## Plan\nReview in 2 weeks.",
        "Allergies:\n\nPlan: review in 2 weeks.",
        "Allergies: rash with something she ate at a party last week",
    ],
)
def test_no_result(text: str) -> None:
    assert _note(text) == []


def test_bullet_styles() -> None:
    text = "Allergies:\n* penicillin (rash)\n• latex\n1. codeine - nausea\n2) aspirin"
    assert [item[0] for item in _note(text)] == ["penicillin", "latex", "codeine", "aspirin"]


def test_prose_line_in_allergies_section() -> None:
    text = "Allergies:\nPatient is allergic to penicillin, which gives her a rash."
    assert _note(text) == [("penicillin", "rash", "allergy", "allergic", False)]


def test_allergies_outside_the_section_are_found_once() -> None:
    text = (
        "History:\nKnee pain. She is allergic to latex.\n\n"
        "Allergies:\n- Penicillin (rash)\n\n"
        "Plan:\nAvoid penicillin."
    )
    assert _note(text) == [
        ("latex", None, "unknown", "allergic", False),
        ("Penicillin", "rash", "allergy", "allergic", False),
    ]


def test_section_items_use_the_heading_as_cue() -> None:
    note = parse_note("Allergies:\n- Penicillin (rash)")
    (result,) = find_note_allergies(note)
    assert result.cue.text == "Allergies"
    assert result.text == "Penicillin (rash)"


# Both sides: labelled lines and noun forms


def test_labelled_line_in_text() -> None:
    assert _text("Drug allergies: penicillin - anaphylaxis; latex - rash") == [
        ("penicillin", "anaphylaxis", "allergy", "allergic", False),
        ("latex", "rash", "allergy", "allergic", False),
    ]


def test_labelled_line_inside_another_section() -> None:
    text = "History:\nKnee pain for 2 weeks.\nDrug allergies: penicillin - anaphylaxis\n"
    assert _note(text) == [("penicillin", "anaphylaxis", "allergy", "allergic", False)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Allergies: penicillin - rash, codeine - nausea", 2),
        ("Allergic to: penicillin", 1),
        ("Known allergies: latex", 1),
        ("Allergies: penicillin?", 1),
    ],
)
def test_labelled_line_item_count(text: str, expected: int) -> None:
    assert len(_text(text)) == expected


def test_labelled_question_mark_is_possible_not_a_question() -> None:
    assert _text("Allergies: penicillin?") == [("penicillin", None, "unknown", "allergic", True)]


def test_penicillin_allergy_in_a_transcript_turn() -> None:
    transcript = parse_transcript("Doctor: Any allergies?\nPatient: Penicillin allergy.")
    turn = transcript.turns[1]
    (result,) = find_allergies(turn.text)
    (allergen,) = result.allergens
    assert transcript.quote(turn.number, allergen.start, allergen.end) == "Penicillin"
    assert result.status == "allergic"
    assert result.kind == "unknown"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Penicillin allergy", ("Penicillin", None, "unknown", "allergic", False)),
        ("She has a nut allergy", ("nut", None, "unknown", "allergic", False)),
        ("Penicillin allergy (rash)", ("Penicillin", "rash", "allergy", "allergic", False)),
        ("No penicillin allergy", ("penicillin", None, "unknown", "not_allergic", False)),
        ("allergy to penicillin", ("penicillin", None, "unknown", "allergic", False)),
        (
            "She had an allergic reaction to penicillin",
            ("penicillin", None, "unknown", "allergic", False),
        ),
        (
            "Allergic to codeine, it gives me nausea",
            ("codeine", "nausea", "intolerance", "allergic", False),
        ),
        ("Penicillin gives me wheezing", ("Penicillin", "wheezing", "allergy", "allergic", False)),
    ],
)
def test_free_text_forms(text: str, expected: Item) -> None:
    assert _text(text) == [expected]


@pytest.mark.parametrize(
    "text",
    [
        "Referred to the allergy clinic",
        "No drug allergy testing needed",
        "Food allergy",
        "She has an allergy",
        "Seasonal allergies",
        "There was no allergic reaction to the vaccine",
        "Metformin gives me diarrhoea",
        "Walking in the cold gives me wheezing",
        "Any penicillin allergy?",
    ],
)
def test_free_text_no_result(text: str) -> None:
    assert _text(text) == []


# Review findings: misread items


@pytest.mark.parametrize(
    "body",
    [
        "None reported",
        "none documented",
        "Nil reported",
        "Nothing known",
        "Nil known drug allergies",
        "NKDA (confirmed with patient)",
    ],
)
def test_more_explicit_none_items(body: str) -> None:
    expected = [(None, None, "unknown", "none", False)]
    assert _note(f"Allergies: {body}") == expected
    assert _text(f"Allergies: {body}") == expected


@pytest.mark.parametrize(
    "text",
    [
        "Allergies: N/A",
        "Allergies: ?NKDA",
        "Allergies: carries EpiPen",
        "Allergies: Reviewed, no change",
        "Allergies: history unclear",
    ],
)
def test_items_that_are_not_allergens(text: str) -> None:
    assert _note(text) == []
    assert _text(text) == []


def test_patient_denies_allergies_is_none() -> None:
    expected = [(None, None, "unknown", "none", False)]
    assert _note("Allergies: Patient denies allergies") == expected
    assert _text("Allergies: Patient denies allergies") == expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            "penicillin - rash; reviewed today",
            [("penicillin", "rash", "allergy", "allergic", False)],
        ),
        ("penicillin allergy", [("penicillin", None, "unknown", "allergic", False)]),
        (
            "Penicillin - rash. Codeine - nausea.",
            [
                ("Penicillin", "rash", "allergy", "allergic", False),
                ("Codeine", "nausea", "intolerance", "allergic", False),
            ],
        ),
        (
            "penicillin - rash, swelling",
            [("penicillin", "rash, swelling", "allergy", "allergic", False)],
        ),
        ("penicillin causes rash", [("penicillin", "rash", "allergy", "allergic", False)]),
        ("amoxicillin which gives rash", [("amoxicillin", "rash", "allergy", "allergic", False)]),
        (
            "penicillin. Codeine",
            [
                ("penicillin", None, "unknown", "allergic", False),
                ("Codeine", None, "unknown", "allergic", False),
            ],
        ),
        ("penicillin 500mg", [("penicillin", None, "unknown", "allergic", False)]),
        ("Amoxicillin 500 mg capsules", [("Amoxicillin", None, "unknown", "allergic", False)]),
        ("penicillin - patient unsure", [("penicillin", None, "unknown", "allergic", True)]),
        ("Penicillin [rash]", [("Penicillin", "rash", "allergy", "allergic", False)]),
        ("Codeine - itchy", [("Codeine", "itchy", "allergy", "allergic", False)]),
    ],
)
def test_item_reading(body: str, expected: list[Item]) -> None:
    assert _note(f"Allergies: {body}") == expected


@pytest.mark.parametrize(
    ("body", "allergens", "reaction"),
    [
        ("nuts and penicillin", ["nuts", "penicillin"], None),
        ("penicillin and codeine (rash)", ["penicillin", "codeine"], "rash"),
    ],
)
def test_and_joins_allergens_in_one_item(
    body: str, allergens: list[str], reaction: str | None
) -> None:
    for text in (f"Allergies: {body}",):
        note = parse_note(text)
        (result,) = find_note_allergies(note)
        assert [a.text for a in result.allergens] == allergens
        assert (result.reaction.text if result.reaction else None) == reaction
        (free,) = find_allergies(text)
        assert [a.text for a in free.allergens] == allergens


def test_wrapped_bullet_item() -> None:
    text = "Allergies:\n- Penicillin - rash and\n  swelling of lips\n- latex"
    note = parse_note(text)
    first, second = find_note_allergies(note)
    assert first.reaction is not None
    assert first.reaction.text == "rash and\n  swelling of lips"
    assert [a.text for a in second.allergens] == ["latex"]


def test_labelled_heading_with_bullets_on_following_lines() -> None:
    text = "History:\nKnee pain.\n\nKnown allergies:\n- latex\n- penicillin (rash)\n"
    assert [item[0] for item in _note(text)] == ["latex", "penicillin"]
    assert [item[0] for item in _text(text)] == ["latex", "penicillin"]


@pytest.mark.parametrize(
    "text",
    [
        "Childhood allergy",
        "Her egg allergy has resolved.",
        "She used to have a nut allergy",
    ],
)
def test_noun_form_no_result(text: str) -> None:
    assert _text(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Cow's milk allergy", ("Cow's milk", None, "unknown", "allergic", False)),
        (
            "Allergic reaction to wasp sting last summer",
            ("wasp sting", None, "unknown", "allergic", False),
        ),
    ],
)
def test_noun_form_allergen_span(text: str, expected: Item) -> None:
    assert _text(text) == [expected]


def test_statement_includes_closing_bracket() -> None:
    (result,) = find_allergies("Latex allergy (anaphylaxis)")
    assert result.text == "Latex allergy (anaphylaxis)"


def test_patient_denies_allergies_in_free_text() -> None:
    assert _text("Patient denies allergies.") == [(None, None, "unknown", "none", False)]


# Second review round


@pytest.mark.parametrize(
    ("text", "allergen"),
    [
        ("She is allergic to penicillin, which is used to treat infections.", "penicillin"),
        ("I used to get hay fever but I am allergic to latex.", "latex"),
        ("Her rash resolved but she has a penicillin allergy.", "penicillin"),
    ],
)
def test_no_longer_check_stays_in_its_clause(text: str, allergen: str) -> None:
    assert [item[0] for item in _text(text)] == [allergen]


@pytest.mark.parametrize(
    ("text", "allergens"),
    [
        ("## Allergies\n  Penicillin (rash)\n  Latex (hives)", ["Penicillin", "Latex"]),
        ("Allergies:\n  Penicillin\n  Latex", ["Penicillin", "Latex"]),
        ("Allergies: penicillin\nAllergic to: latex", ["penicillin", "latex"]),
        ("Allergies: St. John's wort", ["St. John's wort"]),
        ("Allergies: Linseed", ["Linseed"]),
        ("Allergies: Dried fruit - hives", ["Dried fruit"]),
        ("Allergies: Salt and vinegar crisps", ["Salt and vinegar crisps"]),
        ("Allergies: Co-codamol 30/500 - nausea", ["Co-codamol"]),
    ],
)
def test_section_items(text: str, allergens: list[str]) -> None:
    assert [item[0] for item in _note(text)] == allergens


@pytest.mark.parametrize(
    ("body", "reaction", "kind"),
    [
        ("Penicillin. Reaction: rash", "rash", "allergy"),
        ("Aspirin – asthma", "asthma", "allergy"),
        ("Penicillin - anaphylaxis — epipen", "anaphylaxis", "allergy"),
    ],
)
def test_section_reactions(body: str, reaction: str, kind: str) -> None:
    ((_, found, found_kind, _, _),) = _note(f"Allergies: {body}")
    assert (found, found_kind) == (reaction, kind)


# Third review round


@pytest.mark.parametrize(
    ("text", "allergens"),
    [
        ("Allergies: Vitamin B12", ["Vitamin B12"]),
        ("Allergies: Vitamin D3 - rash", ["Vitamin D3"]),
        ("Allergies: Fish, e.g. salmon", ["Fish"]),
    ],
)
def test_section_items_round_three(text: str, allergens: list[str]) -> None:
    assert [item[0] for item in _note(text)] == allergens


def test_no_longer_after_a_comma() -> None:
    assert _text("Penicillin allergy, no longer relevant.") == []
