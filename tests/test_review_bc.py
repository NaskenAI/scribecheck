"""Regression tests for the Pass B, Pass C and dedupe review findings (mocked HTTP only)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from _mock_model import MockClaude
from test_pass_c import NOTE as C_NOTE
from test_pass_c import TRANSCRIPT as C_TRANSCRIPT
from test_pass_c import _run as run_c

from scribecheck import check
from scribecheck.config import Config
from scribecheck.dedupe import deduplicate
from scribecheck.flags import Flag, NoteEvidence, TranscriptEvidence
from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.budget import Budget
from scribecheck.model.endpoint import Secret
from scribecheck.model_passes import run_passes
from scribecheck.note import parse_note
from scribecheck.pass_a.base import Case
from scribecheck.pass_c import from_record
from scribecheck.quotes import find_in_note
from scribecheck.rules.examination import examination_flags
from scribecheck.transcript import parse_transcript


def _remote(modality: str) -> Any:
    return parse_transcript(
        json.dumps(
            {
                "consultation_id": "x",
                "modality": modality,
                "turns": [{"speaker": "Doctor", "text": "Hello"}],
            }
        )
    )


def _exam(note: str, modality: str = "telephone") -> list[str]:
    flags = examination_flags(_remote(modality), parse_note(note))
    return [f.note_evidence.text for f in flags if f.note_evidence is not None]


@pytest.mark.parametrize(
    "note",
    [
        "History: Knee pain, worse on examination by the physio last week.",
        "Plan: Book face-to-face exam: next week if no better.",
        "O/Ex reported",
        "Examination: Deferred to face-to-face appointment.",
        "Examination:\nPatient declined.",
        "Objective: Home BP 150/90 reported by patient.",
    ],
)
def test_examination_rule_false_positives(note: str) -> None:
    assert _exam(note) == []


def test_examination_rule_keeps_decimals_and_skips_advice() -> None:
    assert _exam("Examination:\nChest clear. Patient advised to rest.") == ["Chest clear"]
    assert _exam("Examination:\nNo lymphadenopathy, none palpable.") == [
        "No lymphadenopathy, none palpable"
    ]
    assert _exam("Examination:\nLooks well, temp 37.8 C") == ["Looks well, temp 37.8 C"]


def test_video_recognises_hands_on_findings_with_colons() -> None:
    note = (
        "Examination:\nLungs clear. Power 5/5 in all limbs. Pitting oedema to ankles. "
        "Neck: no masses. Air entry equal bilaterally. Chest: clear. Abdomen: soft. "
        "Throat red."
    )
    found = _exam(note, "video")
    assert "Throat red" not in found and "Chest: clear" in found and "Abdomen: soft" in found
    assert len(found) == 7


def _f(kind: str, fact: str | None = None, **extra: Any) -> Flag:
    verdict = {"omission": "absent", "changed_fact": "changed"}.get(kind, "unsupported")
    return Flag(
        id="F0",
        type=kind,
        category=extra.pop("category", "dose"),
        verdict=verdict,
        severity="supporting",
        confidence=extra.pop("confidence", "confirmed"),
        origin="model",
        fact=fact,
        **extra,
    )


def test_dedupe_compares_drugs_not_first_words() -> None:
    change = _f("changed_fact", "the dose of amlodipine")
    other = _f("omission", "Started metformin 500 mg for the diabetes", category="medicine")
    same = _f("omission", "Amlodipine 5 mg daily", category="dose")
    assert deduplicate([other, same, change]) == [other, change]


def test_dedupe_keeps_two_changes_on_different_drugs_in_one_line() -> None:
    line = NoteEvidence(text="Amlodipine 10 mg, ramipril 5 mg", section=None, start=0, end=31)
    a = _f("changed_fact", "amlodipine", note_evidence=line)
    b = _f("changed_fact", "ramipril", note_evidence=line)
    assert deduplicate([a, b]) == [a, b]


def test_dedupe_does_not_drop_a_small_overlap_or_the_examination_rule() -> None:
    change = _f(
        "changed_fact",
        "ramipril",
        note_evidence=NoteEvidence(text="x" * 20, section=None, start=0, end=20),
    )
    statement = _f(
        "unsupported",
        "started on aspirin",
        category="statement",
        note_evidence=NoteEvidence(text="x" * 40, section=None, start=15, end=55),
    )
    exam = _f(
        "unsupported",
        None,
        category="examination",
        confidence="possible",
        note_evidence=NoteEvidence(text="x" * 10, section=None, start=5, end=15),
    )
    assert deduplicate([change, statement, exam]) == [change, statement, exam]


def test_pass_c_skips_the_second_check_for_discarded_contradictions() -> None:
    from test_pass_c import _scripts

    scripts = _scripts({})
    check_ = scripts["statements"]

    def bad_quotes(listed: list[tuple[str, str]], body: Any) -> Any:
        out = check_(listed, body)
        for row in out["statements"]:
            if row["verdict"] == "contradicted":
                row["transcript_quote"] = "The patient reports a high fever daily"
        return out

    scripts["statements"] = bad_quotes
    scripts["statements"].__name__ = "bad"
    scripts["split"] = lambda _, __: {
        "statements": [{"text": "Fever for 3 days", "kind": "clinical"}]
    }
    mock = MockClaude(scripts)
    client = AnthropicClient("claude-sonnet-5-5", api_key=Secret("k"), transport=mock.transport)
    config = Config(mode="full", provider="anthropic", model="m", passes=("c",))
    case = Case("c0", parse_transcript(C_TRANSCRIPT), parse_note(C_NOTE))
    outcome = run_passes([case], config, client, Budget())["c0"]
    assert "statements_second" not in mock.stages()
    assert [u.reasons for u in outcome.unchecked] == [("quote_unverified",)]


def test_record_cue_must_be_in_the_statement_itself() -> None:
    note = parse_note(
        "History:\nPer chart: asthma. Started smoking again last month.\n\n"
        "Problem list:\nChest pain for 3 days, possible angina.\n"
    )
    smoking = find_in_note(note, "Started smoking again last month")
    angina = find_in_note(note, "Chest pain for 3 days, possible angina")
    asthma = find_in_note(note, "Per chart: asthma")
    assert smoking is not None and angina is not None and asthma is not None
    assert not from_record(note, smoking)
    assert not from_record(note, angina)  # a plain problem list is not imported
    assert from_record(note, asthma)


def test_video_examination_seen_not_touched_is_not_verifiable() -> None:
    outcome, _ = run_c(modality="video")
    exempt = {nv.text for nv, _ in outcome.not_verifiable}
    assert "Chest clear" not in exempt  # hands-on: the examination rule's
    outcome2, _ = run_c(modality="video")
    assert "Chest clear" not in {f.fact for f in outcome2.flags}


def test_rules_mode_reports_which_passes_ran() -> None:
    transcript = "Patient: I take metformin 500 mg twice a day.\n"
    report = check(transcript, "Plan:\nReview.\n", Config(passes=("a",)))
    assert report.passes == {"a": "rules", "b": "off", "c": "off"}
    assert json.loads(report.to_json())["passes"]["b"] == "off"
    full = check(transcript, "Plan:\nReview.\n")
    assert full.passes is None and "passes" not in json.loads(full.to_json())


def test_unchecked_items_overlapping_a_flag_are_dropped() -> None:
    _ = TranscriptEvidence  # imported for completeness of the evidence types
    report = check(
        "Doctor: Any medicines?\nPatient: I take ibuprofen 400 mg three times a day.\n",
        "Medications:\nIbuprofen 400 mg twice daily.\n",
    )
    spans = [(f.note_evidence.start, f.note_evidence.end) for f in report.flags if f.note_evidence]
    for item in report.unchecked:
        for q in item.note_evidence:
            assert not any(q.start < e and s < q.end for s, e in spans)
