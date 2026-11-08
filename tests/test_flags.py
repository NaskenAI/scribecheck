from __future__ import annotations

import pytest

from scribecheck.flags import (
    Flag,
    NoteEvidence,
    TranscriptEvidence,
    UncheckedItem,
    note_evidence,
    transcript_evidence,
    verify_evidence,
)
from scribecheck.note import parse_note
from scribecheck.transcript import parse_transcript

TRANSCRIPT = parse_transcript("Doctor: How is the knee?\nPatient: My left knee has been swelling.")
NOTE = parse_note("Summary of visit.\n\nHistory:\nRight knee swelling for 2 weeks.\n")


def test_transcript_evidence_slices_its_turn() -> None:
    evidence = transcript_evidence(TRANSCRIPT, 2, 3, 12)
    assert evidence == TranscriptEvidence(text="left knee", turn=2, start=3, end=12)
    assert TRANSCRIPT.quote(evidence.turn, evidence.start, evidence.end) == evidence.text


def test_note_evidence_slices_the_note_and_names_its_section() -> None:
    start = NOTE.text.index("Right knee")
    evidence = note_evidence(NOTE, start, start + len("Right knee swelling"))
    assert (evidence.text, evidence.section) == ("Right knee swelling", "History")
    assert NOTE.text[evidence.start : evidence.end] == evidence.text


def test_note_evidence_in_the_preamble_has_no_section() -> None:
    evidence = note_evidence(NOTE, 0, len("Summary"))
    assert (evidence.text, evidence.section) == ("Summary", None)


@pytest.mark.parametrize(("turn", "start", "end"), [(0, 0, 1), (3, 0, 1), (2, 5, 4), (2, -1, 3)])
def test_transcript_evidence_rejects_offsets_outside_the_turn(
    turn: int, start: int, end: int
) -> None:
    with pytest.raises(ValueError):
        transcript_evidence(TRANSCRIPT, turn, start, end)


def test_transcript_evidence_rejects_an_end_past_the_turn() -> None:
    with pytest.raises(ValueError):
        transcript_evidence(TRANSCRIPT, 2, 0, len(TRANSCRIPT.turns[1].text) + 1)


@pytest.mark.parametrize(("start", "end"), [(5, 5), (-1, 3), (0, 10_000)])
def test_note_evidence_rejects_offsets_outside_the_note(start: int, end: int) -> None:
    with pytest.raises(ValueError):
        note_evidence(NOTE, start, end)


def _flag(**overrides: object) -> Flag:
    start = NOTE.text.index("Right knee")
    fields: dict[str, object] = {
        "id": "F1",
        "type": "changed_fact",
        "category": "side",
        "verdict": "changed",
        "severity": "critical",
        "confidence": "certain",
        "origin": "rules",
        "rule_id": "side.mismatch",
        "transcript_value": "left knee",
        "note_value": "right knee",
        "transcript_evidence": transcript_evidence(TRANSCRIPT, 2, 3, 12),
        "note_evidence": note_evidence(NOTE, start, start + len("Right knee")),
    }
    return Flag.model_validate(fields | overrides)


def test_flag_matches_the_spec_report_shape() -> None:
    flag = _flag()
    dumped = flag.model_dump(exclude_none=True)
    assert dumped["transcript_evidence"] == {
        "text": "left knee",
        "turn": 2,
        "start": 3,
        "end": 12,
    }
    assert dumped["note_evidence"]["section"] == "History"
    assert set(dumped) == {
        "id", "type", "category", "verdict", "severity", "confidence", "origin", "rule_id",
        "transcript_value", "note_value", "transcript_evidence", "note_evidence",
    }  # fmt: skip


def test_flag_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        _flag(confidence="sure")
    with pytest.raises(ValueError):
        _flag(severity="major")


def test_verify_evidence_accepts_exact_quotes() -> None:
    verify_evidence(_flag(), TRANSCRIPT, NOTE)


def test_verify_evidence_rejects_a_quote_that_does_not_slice_back() -> None:
    forged = TranscriptEvidence(text="right knee", turn=2, start=3, end=12)
    with pytest.raises(ValueError, match="transcript evidence"):
        verify_evidence(_flag(transcript_evidence=forged), TRANSCRIPT, NOTE)
    forged_note = NoteEvidence(text="Left knee", section="History", start=0, end=9)
    with pytest.raises(ValueError, match="note evidence"):
        verify_evidence(_flag(note_evidence=forged_note), TRANSCRIPT, NOTE)


def test_verify_evidence_checks_unchecked_items_too() -> None:
    item = UncheckedItem(
        category="side",
        item="knee",
        reasons=("transcript_inconsistent",),
        transcript_evidence=(TranscriptEvidence(text="nope", turn=2, start=3, end=12),),
    )
    with pytest.raises(ValueError):
        verify_evidence(item, TRANSCRIPT, NOTE)


def test_unchecked_items_need_a_reason_and_expose_the_primary_one() -> None:
    item = UncheckedItem(category="dose", item="metformin", reasons=("not_comparable", "possible"))
    assert item.reason == "not_comparable"
    with pytest.raises(ValueError):
        UncheckedItem(category="dose", item="metformin", reasons=())
