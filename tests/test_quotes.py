"""The fuzzy quote check: what it forgives, and what it never does."""

from __future__ import annotations

import pytest

from scribecheck.note import parse_note
from scribecheck.quotes import critical_tokens, find_in_note, find_in_transcript
from scribecheck.transcript import parse_transcript

T = parse_transcript(
    "Doctor: Any chest pain or breathlessness?\n"
    "Patient: I have chest pain on the left side, mostly at night.\n"
    "Doctor: And your medicines?\n"
    "Patient: I take metformin 5 mg daily, and my right knee hurts.\n"
)
N = parse_note("Allergies: NKDA\n\nMedications:\nMetformin 500 mg BD.\n")


def _slices(evidence: object) -> None:
    from scribecheck.flags import NoteEvidence, TranscriptEvidence

    if isinstance(evidence, TranscriptEvidence):
        assert T.quote(evidence.turn, evidence.start, evidence.end) == evidence.text
    elif isinstance(evidence, NoteEvidence):
        assert N.text[evidence.start : evidence.end] == evidence.text


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("I have chest pain on the left side", "I have chest pain on the left side"),
        ("i HAVE chest pain, on the left side!", "I have chest pain on the left side"),
        ("I have   chest pain\non the left side", "I have chest pain on the left side"),
        ("I have chest pian on the left side", "I have chest pain on the left side"),
        ("Patient: I take metformin 5 mg daily", "I take metformin 5 mg daily"),
        ("[T4] Patient: my right knee hurts...", "my right knee hurts"),
        (
            "I have chest pain … mostly at night",  # every part, in order
            "I have chest pain on the left side, mostly at night",
        ),
    ],
)
def test_forgives_case_spacing_punctuation_spelling_labels_ellipses(
    quote: str, expected: str
) -> None:
    found = find_in_transcript(T, quote)
    assert found is not None and found.text == expected
    _slices(found)


@pytest.mark.parametrize(
    "quote",
    [
        "no chest pain on the left side",  # negation added
        "I take metformin 50 mg daily",  # number changed
        "and my left knee hurts",  # side changed
        "I take metformin 5 g daily",  # unit changed
        "I take insulin 5 mg daily",  # drug changed
    ],
)
def test_never_forgives_a_critical_token(quote: str) -> None:
    assert find_in_transcript(T, quote) is None


@pytest.mark.parametrize(
    ("quote", "source"),
    [("no chest pain", "chest pain"), ("50 mg", "5 mg"), ("left knee", "right knee")],
)
def test_the_three_required_cases_fail(quote: str, source: str) -> None:
    transcript = parse_transcript(f"Patient: {source}\n")
    assert find_in_transcript(transcript, quote) is None
    assert find_in_note(parse_note(source), quote) is None


def test_short_exact_whole_tokens_verify_when_meaningful() -> None:
    for quote in ("NKDA", "nkda", "BD", "Metformin"):
        found = find_in_note(N, quote)
        assert found is not None and found.text.casefold() == quote.casefold()
        _slices(found)


def test_short_quotes_must_be_whole_tokens_and_not_trivial() -> None:
    assert find_in_note(N, "met") is None  # part of "Metformin"
    assert find_in_note(N, "B") is None
    assert find_in_note(N, "") is None
    note = parse_note("No fever. Took 2 doses.")
    assert find_in_note(note, "no") is None  # too short to be evidence
    assert find_in_note(note, "2") is None
    assert find_in_note(note, "500") is None


def test_short_fuzzy_quotes_are_rejected() -> None:
    assert find_in_note(N, "NKDB") is None  # under 8 characters, so no fuzzy match


def test_a_quote_spanning_two_turns_is_rejected() -> None:
    assert find_in_transcript(T, "Any chest pain or breathlessness? I have chest pain") is None


def test_critical_tokens() -> None:
    numbers, units, negations, sides, drugs = critical_tokens(
        "No left-sided pain; metformin 0.5 mg"
    )
    assert numbers == {"0.5": 1} and units == {"mg": 1}
    assert negations == {"no": 1} and sides == {"left": 1}
    assert drugs == {"metformin": 1}
    assert critical_tokens("doesn't hurt")[2]["not"] == 1


def _one(text: str) -> object:
    return parse_transcript(f"Patient: {text}\n")


@pytest.mark.parametrize(
    ("quote", "source"),
    [
        ("15 mg of warfarin", "I take 1.5 mg of warfarin daily."),  # decimal kept
        ("I take metoprolol 5", "I take metoprolol 50 mg every day."),  # no mid-number windows
        ("He doesnt have chest pain at night", "He does have chest pain at night."),
        ("I dont get chest pain at night", "I do get chest pain at night."),
        ("I take ten tablets of aspirin daily", "I take two tablets of aspirin daily."),
        ("Not on warfarin: since March", "I'm on warfarin since March."),  # not a label
        ("Known drug allergies noted here", "No known drug allergies noted here."),
        (
            "severe chest pain radiating … otherwise I feel fine today honestly",
            "Otherwise I feel fine today honestly.",
        ),  # every ellipsis part must match
    ],
)
def test_review_false_accepts_are_rejected(quote: str, source: str) -> None:
    transcript = _one(source)
    assert find_in_transcript(transcript, quote) is None  # type: ignore[arg-type]


def test_contraction_forms_agree() -> None:
    transcript = parse_transcript("Patient: It doesn't hurt when I walk.\n")
    assert find_in_transcript(transcript, "It does not hurt when I walk") is not None
    assert find_in_transcript(transcript, "It doesnt hurt when I walk") is not None


def test_speaker_labels_are_stripped_only_when_the_transcript_uses_them() -> None:
    transcript = parse_transcript("Patient: I take metformin 500 mg twice a day.\n")
    found = find_in_transcript(transcript, "Patient: I take metformin 500 mg twice a day")
    assert found is not None and found.text == "I take metformin 500 mg twice a day"
    assert find_in_transcript(transcript, "Mum: I take metformin 500 mg twice a day") is None
