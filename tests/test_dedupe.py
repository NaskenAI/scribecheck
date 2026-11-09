"""One discrepancy, one flag, across passes, in both directions."""

from __future__ import annotations

from typing import Any

from scribecheck.dedupe import deduplicate
from scribecheck.flags import Flag, NoteEvidence, TranscriptEvidence


def _flag(kind: str, **changes: Any) -> Flag:
    base: dict[str, Any] = {
        "id": "F0",
        "type": kind,
        "category": "x",
        "verdict": {"omission": "absent", "changed_fact": "changed"}.get(kind, "unsupported"),
        "severity": "supporting",
        "confidence": "possible",
        "origin": "model",
    }
    base.update(changes)
    return Flag(**base)


def _t(start: int, end: int, turn: int = 2) -> TranscriptEvidence:
    return TranscriptEvidence(text="x" * (end - start), turn=turn, start=start, end=end)


def _n(start: int, end: int) -> NoteEvidence:
    return NoteEvidence(text="x" * (end - start), section=None, start=start, end=end)


CHANGE = _flag(
    "changed_fact",
    category="frequency",
    fact="ibuprofen",
    transcript_evidence=_t(10, 40),
    note_evidence=_n(50, 80),
    confidence="confirmed",
)


def test_pass_b_wins_over_pass_c_on_the_same_note_span() -> None:
    statement = _flag("unsupported", category="statement", note_evidence=_n(60, 70))
    assert deduplicate([CHANGE, statement]) == [CHANGE]


def test_pass_c_without_overlap_stays() -> None:
    statement = _flag("unsupported", category="statement", note_evidence=_n(90, 99))
    assert deduplicate([CHANGE, statement]) == [CHANGE, statement]


def test_pass_a_gives_way_to_pass_b_on_the_same_transcript_span() -> None:
    omission = _flag(
        "omission",
        category="frequency",
        fact="ibuprofen three times a day",
        transcript_evidence=_t(20, 30),
    )
    assert deduplicate([omission, CHANGE]) == [CHANGE]


def test_pass_a_gives_way_to_pass_b_on_the_same_drug() -> None:
    omission = _flag(
        "omission", category="dose", fact="Ibuprofen 400 mg", transcript_evidence=_t(0, 5, turn=4)
    )
    assert deduplicate([omission, CHANGE]) == [CHANGE]


def test_unrelated_omission_stays() -> None:
    omission = _flag(
        "omission",
        category="allergy",
        fact="allergy: penicillin",
        transcript_evidence=_t(0, 5, turn=6),
    )
    assert deduplicate([omission, CHANGE]) == [omission, CHANGE]


def test_drug_names_match_whole_words_only() -> None:
    iron = _flag("changed_fact", category="dose", fact="iron", confidence="confirmed")
    omission = _flag("omission", category="medicine", fact="Home environment is damp")
    assert deduplicate([omission, iron]) == [omission, iron]


def test_exam_rule_and_confirmed_model_verdict_on_one_sentence_keep_one() -> None:
    rule = _flag(
        "unsupported",
        category="examination",
        origin="rules",
        note_evidence=_n(5, 15),
        confidence="possible",
    )
    model = _flag(
        "unsupported", category="statement", note_evidence=_n(5, 20), confidence="confirmed"
    )
    assert deduplicate([rule, model]) == [model]


def test_omissions_are_never_merged_with_each_other() -> None:
    dose = _flag(
        "omission", category="dose", fact="metformin 500 mg", transcript_evidence=_t(7, 23)
    )
    frequency = _flag(
        "omission",
        category="frequency",
        fact="metformin twice a day",
        transcript_evidence=_t(7, 40),
    )
    assert deduplicate([dose, frequency]) == [dose, frequency]
