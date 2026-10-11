from __future__ import annotations

from pathlib import Path

import pytest

from scribecheck.errors import InputError
from scribecheck.note import (
    DEFAULT_HEADING_MAP,
    CanonicalSection,
    Note,
    build_heading_map,
    parse_note,
    parse_note_file,
)

MARKDOWN_NOTE = """Telephone consultation, synthetic patient.

## History
Left knee swelling for 2 weeks.
Worse on stairs.

## Medications
Metformin 500 mg BD

## Allergies
Penicillin (rash)

### Plan:
Refer to physiotherapy.
"""

PLAIN_NOTE = """History:
Left knee swelling for 2 weeks.
BP: 120/80
Patient reports: pain on stairs.

Allergies: NKDA
Plan: refer to physiotherapy.

Safety netting
Return if fever develops.
"""


def assert_round_trip(note: Note) -> None:
    if note.preamble is None:
        assert note.preamble_start is None and note.preamble_end is None
    else:
        assert note.preamble_start is not None and note.preamble_end is not None
        assert note.text[note.preamble_start : note.preamble_end] == note.preamble
    for s in note.sections:
        assert note.text[s.start : s.end] == s.text
        assert note.text[s.heading_start : s.heading_end] == s.heading
        assert note.text[s.body_start : s.body_end] == s.body
        assert s.start <= s.heading_start < s.heading_end <= s.end
        assert s.body_start <= s.body_end <= s.end


def headings(note: Note) -> list[tuple[str, CanonicalSection | None]]:
    return [(s.heading, s.canonical) for s in note.sections]


def test_markdown_sections() -> None:
    note = parse_note(MARKDOWN_NOTE)
    assert headings(note) == [
        ("History", CanonicalSection.HISTORY),
        ("Medications", CanonicalSection.MEDICATIONS),
        ("Allergies", CanonicalSection.ALLERGIES),
        ("Plan", CanonicalSection.PLAN),
    ]
    history = note.sections[0]
    assert history.body == "Left knee swelling for 2 weeks.\nWorse on stairs."
    assert history.text == "## History\nLeft knee swelling for 2 weeks.\nWorse on stairs."
    assert note.sections[3].body == "Refer to physiotherapy."
    assert_round_trip(note)


def test_text_before_the_first_heading_belongs_to_no_section() -> None:
    note = parse_note(MARKDOWN_NOTE)
    assert note.preamble == "Telephone consultation, synthetic patient."
    assert note.preamble_start == 0
    assert all("Telephone consultation" not in s.text for s in note.sections)
    assert_round_trip(note)


def test_no_preamble_when_the_note_starts_with_a_heading() -> None:
    note = parse_note("\n\nPlan:\nReview in 2 weeks.\n")
    assert note.preamble is None
    assert_round_trip(note)


def test_note_without_headings_is_all_preamble() -> None:
    note = parse_note("Left knee swelling. Refer to physiotherapy.\n")
    assert note.sections == ()
    assert note.preamble == "Left knee swelling. Refer to physiotherapy."
    assert_round_trip(note)


@pytest.mark.parametrize("level", range(1, 7))
def test_markdown_heading_levels(level: int) -> None:
    note = parse_note(f"{'#' * level} Examination\nSwollen left knee.\n")
    (section,) = note.sections
    assert section.heading == "Examination"
    assert section.canonical is CanonicalSection.EXAMINATION
    assert section.body == "Swollen left knee."
    assert_round_trip(note)


def test_seven_hashes_is_not_a_heading() -> None:
    note = parse_note("####### Examination\nSwollen left knee.\n")
    assert note.sections == ()


def test_hash_without_space_is_not_a_heading() -> None:
    note = parse_note("#Examination\nSwollen left knee.\n")
    assert note.sections == ()


def test_markdown_heading_with_closing_hashes() -> None:
    note = parse_note("## Results ##\nHbA1c 58 mmol/mol\n")
    assert headings(note) == [("Results", CanonicalSection.RESULTS)]
    assert_round_trip(note)


@pytest.mark.parametrize(
    "line",
    [
        "Plan:",
        "Plan",
        "PLAN:",
        "plan :",
        "## Plan",
        "## Plan:",
        "**Plan:**",
        "**Plan**:",
        "**Plan**",
    ],
)
def test_headings_with_and_without_trailing_colons(line: str) -> None:
    note = parse_note(f"Assessment:\nLikely osteoarthritis.\n\n{line}\nRefer to physiotherapy.\n")
    assert [s.canonical for s in note.sections] == [
        CanonicalSection.ASSESSMENT,
        CanonicalSection.PLAN,
    ]
    assert note.sections[1].body == "Refer to physiotherapy."
    assert note.sections[0].body == "Likely osteoarthritis."
    assert_round_trip(note)


def test_bare_known_heading_without_colon_is_a_heading() -> None:
    note = parse_note(PLAIN_NOTE)
    assert ("Safety netting", CanonicalSection.PLAN) in headings(note)


def test_bare_unknown_line_without_colon_is_not_a_heading() -> None:
    note = parse_note("History:\nKnee pain\nWorse on stairs\n")
    assert len(note.sections) == 1
    assert note.sections[0].body == "Knee pain\nWorse on stairs"


def test_inline_content_after_heading() -> None:
    note = parse_note(PLAIN_NOTE)
    allergies = note.sections_for(CanonicalSection.ALLERGIES)
    assert len(allergies) == 1
    section = allergies[0]
    assert section.heading == "Allergies"
    assert section.body == "NKDA"
    assert section.text == "Allergies: NKDA"
    plan = note.sections_for(CanonicalSection.PLAN)[0]
    assert plan.body == "refer to physiotherapy."
    assert_round_trip(note)


def test_inline_content_after_markdown_heading() -> None:
    note = parse_note("## Allergies: NKDA\n## Plan\nReview.\n")
    assert note.sections[0].body == "NKDA"
    assert note.sections[0].canonical is CanonicalSection.ALLERGIES
    assert_round_trip(note)


def test_inline_heading_continues_onto_following_lines() -> None:
    note = parse_note("Medications: metformin 500 mg BD\nramipril 5 mg OD\nPlan: review\n")
    assert note.sections[0].body == "metformin 500 mg BD\nramipril 5 mg OD"
    assert_round_trip(note)


def test_colon_lines_with_content_and_unknown_names_are_not_headings() -> None:
    note = parse_note(PLAIN_NOTE)
    history = note.sections[0]
    assert history.canonical is CanonicalSection.HISTORY
    assert "BP: 120/80" in history.body
    assert "Patient reports: pain on stairs." in history.body
    assert [s.heading for s in note.sections] == [
        "History",
        "Allergies",
        "Plan",
        "Safety netting",
    ]


def test_unknown_standalone_heading_keeps_its_name() -> None:
    note = parse_note("Ideas, concerns and expectations:\nWorried about arthritis.\n")
    (section,) = note.sections
    assert section.heading == "Ideas, concerns and expectations"
    assert section.canonical is None
    assert section.body == "Worried about arthritis."
    assert_round_trip(note)


def test_unknown_markdown_heading_keeps_its_name() -> None:
    note = parse_note("# Social Context\nLives alone.\n")
    assert headings(note) == [("Social Context", None)]


def test_sentence_like_colon_lines_are_not_headings() -> None:
    note = parse_note("History:\nShe said the following when asked about her pain today:\nOK.\n")
    assert len(note.sections) == 1


def test_repeated_headings_are_separate_sections() -> None:
    text = "Plan:\nBloods today.\nExamination:\nNormal.\nPlan:\nReview in 2 weeks.\n"
    note = parse_note(text)
    plans = note.sections_for(CanonicalSection.PLAN)
    assert [s.body for s in plans] == ["Bloods today.", "Review in 2 weeks."]
    assert plans[0].start < plans[1].start
    assert_round_trip(note)


def test_heading_with_empty_body() -> None:
    note = parse_note("Results:\n\nPlan:\nReview.\n")
    results = note.sections[0]
    assert results.body == ""
    assert results.body_start == results.body_end
    assert results.text == "Results:"
    assert_round_trip(note)


def test_heading_lookup_collapses_whitespace_and_case() -> None:
    note = parse_note("PAST   MEDICAL  history:\nAppendicectomy 2015.\n")
    assert note.sections[0].canonical is CanonicalSection.HISTORY
    assert note.sections[0].heading == "PAST   MEDICAL  history"


def test_custom_heading_map_is_laid_over_the_defaults() -> None:
    heading_map = build_heading_map({"Meds Reviewed": CanonicalSection.MEDICATIONS})
    note = parse_note("Meds reviewed: metformin\nAllergies: NKDA\n", heading_map=heading_map)
    assert headings(note) == [
        ("Meds reviewed", CanonicalSection.MEDICATIONS),
        ("Allergies", CanonicalSection.ALLERGIES),
    ]
    assert note.sections[0].body == "metformin"


def test_custom_heading_map_overrides_a_default() -> None:
    heading_map = build_heading_map({"impression": CanonicalSection.EXAMINATION})
    note = parse_note("Impression:\nSwollen knee.\n", heading_map=heading_map)
    assert note.sections[0].canonical is CanonicalSection.EXAMINATION


def test_heading_map_without_defaults() -> None:
    heading_map = build_heading_map(
        {"Meds Reviewed": CanonicalSection.MEDICATIONS}, include_defaults=False
    )
    note = parse_note("Allergies:\nNKDA\nMeds reviewed:\nnone\n", heading_map=heading_map)
    assert headings(note) == [
        ("Allergies", None),
        ("Meds reviewed", CanonicalSection.MEDICATIONS),
    ]


def test_parse_note_accepts_a_plain_mapping() -> None:
    note = parse_note("Drugs: metformin\n", heading_map={"drugs": CanonicalSection.MEDICATIONS})
    assert headings(note) == [("Drugs", CanonicalSection.MEDICATIONS)]


def test_default_heading_map_covers_every_canonical_section() -> None:
    assert set(DEFAULT_HEADING_MAP.values()) == set(CanonicalSection)
    for name in ["history", "medications", "allergies", "examination", "results", "plan"]:
        assert name in DEFAULT_HEADING_MAP
    assert DEFAULT_HEADING_MAP["assessment"] is CanonicalSection.ASSESSMENT


def test_canonical_section_values_match_display_names() -> None:
    assert [s.value for s in CanonicalSection] == [
        "History",
        "Medications",
        "Allergies",
        "Examination",
        "Results",
        "Assessment",
        "Plan",
    ]


@pytest.mark.parametrize("text", ["", "  \n\t\n"])
def test_empty_note_is_an_input_error(text: str) -> None:
    with pytest.raises(InputError, match="empty"):
        parse_note(text)


def test_windows_line_endings_are_normalised() -> None:
    lf = parse_note(PLAIN_NOTE)
    crlf = parse_note(PLAIN_NOTE.replace("\n", "\r\n"))
    assert crlf == lf
    assert "\r" not in crlf.text


def test_models_are_immutable() -> None:
    note = parse_note(MARKDOWN_NOTE)
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError
        note.sections[0].body = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("text", [MARKDOWN_NOTE, PLAIN_NOTE])
def test_offsets_round_trip(text: str) -> None:
    assert_round_trip(parse_note(text))


def test_note_file(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_bytes(b"\xef\xbb\xbf" + MARKDOWN_NOTE.replace("\n", "\r\n").encode("utf-8"))
    assert parse_note_file(path) == parse_note(MARKDOWN_NOTE)


def test_note_file_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_bytes("Plan: café".encode("latin-1"))
    with pytest.raises(InputError, match="note.txt"):
        parse_note_file(path)


def test_empty_note_file_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.md"
    path.write_text("", encoding="utf-8")
    with pytest.raises(InputError, match="empty.md"):
        parse_note_file(path)


def test_unknown_colon_line_inside_a_section_is_not_a_heading() -> None:
    note = parse_note("Plan:\nThe patient was advised to:\n- rest\n- ice\n")
    (plan,) = note.sections
    assert plan.canonical is CanonicalSection.PLAN
    assert plan.body == "The patient was advised to:\n- rest\n- ice"


def test_unknown_colon_heading_after_a_blank_line_is_a_heading() -> None:
    note = parse_note("Plan:\nRest.\n\nIdeas and concerns:\nWorried.\n")
    assert headings(note) == [("Plan", CanonicalSection.PLAN), ("Ideas and concerns", None)]


def test_bare_known_heading_needs_a_blank_line_before_it() -> None:
    note = parse_note("Plan\nBloods\nreview in 2 weeks\n")
    (plan,) = note.sections
    assert plan.body == "Bloods\nreview in 2 weeks"


@pytest.mark.parametrize("line", ["Results:", "Results: HbA1c 58", "**Results**", "## Results"])
def test_explicit_headings_need_no_blank_line(line: str) -> None:
    note = parse_note(f"Plan:\nBloods today.\n{line}\n")
    assert [s.canonical for s in note.sections] == [
        CanonicalSection.PLAN,
        CanonicalSection.RESULTS,
    ]


def test_heading_map_values_ignore_case() -> None:
    heading_map = build_heading_map({"Rx": "plan", "Obs": "EXAMINATION"})
    assert heading_map["rx"] is CanonicalSection.PLAN
    assert heading_map["obs"] is CanonicalSection.EXAMINATION
    with pytest.raises(InputError, match="unknown section"):
        build_heading_map({"Rx": "treatment"})


def test_leading_bom_in_text_is_dropped() -> None:
    note = parse_note("\ufeffPlan:\nReview.\n")
    assert note.text == "Plan:\nReview.\n"
    assert note.sections[0].heading == "Plan"


# ACI-style notes: ALL-CAPS headings without colons

ACI_NOTE = """ENCOUNTER NOTE

CHIEF COMPLAINT

Left knee swelling.

HISTORY OF PRESENT ILLNESS

Two weeks of swelling, worse on stairs.

MEDICAL HISTORY

Asthma.

SURGICAL HISTORY

Appendicectomy.

REVIEW OF SYSTEMS

Denies fever.

VITALS

BP 120/80.

PHYSICAL EXAM

Swollen left knee.

ASSESSMENT AND PLAN

Likely osteoarthritis. Refer to physiotherapy.

DISCUSSION
Options discussed.

INSTRUCTIONS

Return if fever develops.
"""


def test_aci_headings() -> None:
    note = parse_note(ACI_NOTE)
    H, E, P = CanonicalSection.HISTORY, CanonicalSection.EXAMINATION, CanonicalSection.PLAN
    # A title followed by a blank line is not a heading; it stays before the first section.
    assert note.preamble == "ENCOUNTER NOTE"
    assert headings(note) == [
        ("CHIEF COMPLAINT", H),
        ("HISTORY OF PRESENT ILLNESS", H),
        ("MEDICAL HISTORY", H),
        ("SURGICAL HISTORY", H),
        ("REVIEW OF SYSTEMS", H),
        ("VITALS", E),
        ("PHYSICAL EXAM", E),
        ("ASSESSMENT AND PLAN", P),
        ("DISCUSSION", None),
        ("INSTRUCTIONS", P),
    ]
    assert note.sections[7].body == "Likely osteoarthritis. Refer to physiotherapy."
    assert_round_trip(note)


@pytest.mark.parametrize(
    ("heading", "section"),
    [
        ("CC", CanonicalSection.HISTORY),
        ("HPI", CanonicalSection.HISTORY),
        ("ROS", CanonicalSection.HISTORY),
        ("Past surgical history", CanonicalSection.HISTORY),
        ("PE", CanonicalSection.EXAMINATION),
        ("Vital signs", CanonicalSection.EXAMINATION),
        ("VITALS REVIEWED", CanonicalSection.EXAMINATION),
        ("A&P", CanonicalSection.PLAN),
        ("A/P", CanonicalSection.PLAN),
        ("Assessment & Plan", CanonicalSection.PLAN),
        ("Patient instructions", CanonicalSection.PLAN),
    ],
)
def test_new_default_headings(heading: str, section: CanonicalSection) -> None:
    note = parse_note(f"History:\nKnee pain.\n\n{heading}\nSome text.\n")
    assert headings(note)[-1] == (heading, section)


@pytest.mark.parametrize(
    "text",
    [
        "Allergies:\nPenicillin.\n\nNKDA\nNo other history.\n",
        "Allergies:\nPenicillin.\n\nNO KNOWN DRUG ALLERGIES\nNo other history.\n",
        "Plan:\nRest.\n\nNAD\nReview.\n",
        "Plan:\nRest.\n\nFOLLOW UP SOON\n\nReview.\n",  # followed by a blank line
        "Plan:\nRest.\nSTRICT REST\nReview.\n",  # no blank line before
    ],
)
def test_all_caps_body_lines_are_not_headings(text: str) -> None:
    assert len(parse_note(text).sections) == 1
