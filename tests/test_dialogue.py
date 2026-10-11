from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from scribecheck.rules.dialogue import DialogueNegation, bind_negations
from scribecheck.transcript import Transcript, parse_transcript

# Words the tests treat as targets; in the pipeline the symptom lexicon supplies these.
TARGETS = ("chest pain", "fever", "cough", "headache", "pain", "rash")


def _targets(
    transcript: Transcript, words: Sequence[str] = TARGETS
) -> dict[int, list[tuple[int, int]]]:
    found: dict[int, list[tuple[int, int]]] = {}
    for turn in transcript.turns:
        lowered = turn.text.lower()
        spans: list[tuple[int, int]] = []
        for word in sorted(words, key=len, reverse=True):
            start = lowered.find(word)
            while start != -1:
                end = start + len(word)
                if not any(s < end and start < e for s, e in spans):
                    spans.append((start, end))
                start = lowered.find(word, end)
        if spans:
            found[turn.number] = sorted(spans)
    return found


def _bind(text: str) -> tuple[DialogueNegation, ...]:
    transcript = parse_transcript(text)
    results = bind_negations(transcript, _targets(transcript))
    for result in results:
        assert transcript.quote(result.question_turn, result.target.start, result.target.end) == (
            result.target.text
        )
        assert transcript.quote(result.answer_turn, result.answer.start, result.answer.end) == (
            result.answer.text
        )
    return results


@pytest.mark.parametrize(
    ("question", "answer", "target", "cue"),
    [
        ("Any chest pain?", "No.", "chest pain", "No"),
        ("Do you have a fever?", "Nope.", "fever", "Nope"),
        ("Have you had a rash?", "Never.", "rash", "Never"),
        ("Any headache at all?", "None.", "headache", "None"),
        ("Have you noticed a cough?", "No, I haven't.", "cough", "No"),
        ("Right. And any chest pain?", "no", "chest pain", "no"),
        ("How about a fever?", "Nah, no fever.", "fever", "Nah"),
    ],
)
def test_binds_single_target_question_to_negative_answer(
    question: str, answer: str, target: str, cue: str
) -> None:
    (result,) = _bind(f"Doctor: {question}\nPatient: {answer}")
    assert (result.question_turn, result.answer_turn) == (1, 2)
    assert result.target.text.lower() == target
    assert result.answer.text == cue


def test_binds_inside_a_longer_consultation() -> None:
    text = (
        "Doctor: What brings you in today?\n"
        "Patient: I've had a cough for a week.\n"
        "Doctor: Any chest pain?\n"
        "Patient: No.\n"
        "Doctor: Do you have a fever?\n"
        "Patient: Not really.\n"
    )
    (result,) = _bind(text)
    assert (result.question_turn, result.answer_turn) == (3, 4)
    assert result.target.text == "chest pain"


def test_json_transcript() -> None:
    data = {
        "consultation_id": "synthetic-1",
        "turns": [
            {"speaker": "clinician", "text": "Any chest pain?"},
            {"speaker": "patient", "text": "No."},
        ],
    }
    (result,) = _bind(json.dumps(data))
    assert result.target.text == "chest pain"


@pytest.mark.parametrize(
    "text",
    [
        # More than one target in the question.
        "Doctor: Any chest pain or fever?\nPatient: No.",
        "Doctor: Any chest pain? Any fever?\nPatient: No.",
        # Hedged answers.
        "Doctor: Any chest pain?\nPatient: Not really.",
        "Doctor: Any chest pain?\nPatient: No, not really.",
        "Doctor: Any chest pain?\nPatient: Sometimes.",
        "Doctor: Any chest pain?\nPatient: No, only when I climb stairs.",
        "Doctor: Any chest pain?\nPatient: A bit.",
        "Doctor: Any chest pain?\nPatient: No, just a bit tight.",
        "Doctor: Any chest pain?\nPatient: No, but it was there last week.",
        "Doctor: Any chest pain?\nPatient: Occasionally.",
        "Doctor: Any chest pain?\nPatient: Yes.",
        # The answer mentions another target.
        "Doctor: Any chest pain?\nPatient: No, but I've got a headache.",
        "Doctor: Any chest pain?\nPatient: No, a headache though.",
        # Speaker labels missing or unknown.
        "Any chest pain?\nNo.",
        "Nurse: Any chest pain?\nPatient: No.",
        "Doctor: Any chest pain?\nCarer: No.",
        # The negative answer is not in the very next turn.
        "Doctor: Any chest pain?\nDoctor: Take your time.\nPatient: No.",
        # Not a presence question.
        "Doctor: Is the pain getting better?\nPatient: No.",
        "Doctor: Any chest pain? Let me listen to your chest.\nPatient: No.",
        # The question is itself negated or hypothetical.
        "Doctor: No chest pain at all?\nPatient: No.",
        "Doctor: Any chest pain if you walk fast?\nPatient: No.",
        # The answer says more than a bare negative.
        "Doctor: Any chest pain?\nPatient: No idea.",
        "Doctor: Any chest pain?\nPatient: No, it's there when I walk.",
        "Doctor: Any chest pain?\nPatient: No, it's worse at night.",
        "Doctor: Any chest pain?\nPatient: Never at rest.",
        "Doctor: Any chest pain?\nPatient: No, I don't think so? Actually I do.",
        "Doctor: Any chest pain?\nPatient: No, it went away yesterday.",
        "Doctor: Any chest pain?\nPatient: No?",
        # The question offers an alternative.
        "Doctor: Any chest pain or anything like that?\nPatient: No.",
        "Doctor: Is there any chest pain or is it more of a pressure?\nPatient: No.",
        "Doctor: Any chest pain or tightness?\nPatient: No.",
        # The question is about family or another time.
        "Doctor: Any chest pain in your family?\nPatient: No.",
        "Doctor: Any history of chest pain?\nPatient: No.",
        "Doctor: Any chest pain before?\nPatient: No.",
        "Doctor: Did you have a fever last week?\nPatient: No.",
        "Doctor: Have you ever had a rash?\nPatient: Never.",
        "Doctor: Do you get chest pain on exertion?\nPatient: No.",
        "Doctor: Any chest pain with exercise?\nPatient: No.",
        "Doctor: Any chest pain at night?\nPatient: No.",
        # The question names no target.
        "Doctor: Anything else?\nPatient: No.",
    ],
)
def test_no_binding(text: str) -> None:
    assert _bind(text) == ()


def test_no_targets_given() -> None:
    transcript = parse_transcript("Doctor: Any chest pain?\nPatient: No.")
    assert bind_negations(transcript, {}) == ()


def test_unknown_turn_number_raises() -> None:
    transcript = parse_transcript("Doctor: Any chest pain?\nPatient: No.")
    with pytest.raises(ValueError):
        bind_negations(transcript, {9: [(0, 3)]})


def test_invalid_span_raises() -> None:
    transcript = parse_transcript("Doctor: Any chest pain?\nPatient: No.")
    with pytest.raises(ValueError):
        bind_negations(transcript, {1: [(4, 99)]})
