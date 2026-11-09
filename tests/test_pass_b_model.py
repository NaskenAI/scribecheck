"""Pass B's model path: changed facts confirmed, rejected, consistent or unverifiable."""

from __future__ import annotations

from typing import Any

from _mock_model import MockClaude, text_of
from test_pass_a_pipeline import scripts

from scribecheck.checker import _rules
from scribecheck.config import Config
from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.budget import Budget
from scribecheck.model.endpoint import Secret
from scribecheck.model_passes import run_passes
from scribecheck.note import parse_note
from scribecheck.pass_a.base import Case, Outcome
from scribecheck.transcript import parse_transcript

KNEE = (
    "Doctor: What seems to be the problem?\n"
    "Patient: My left knee has been swollen for two weeks.\n"
    "Doctor: Are you taking anything for it?\n"
    "Patient: I take ibuprofen 400 mg three times a day.\n"
    "Doctor: It might be a meniscal tear.\n"
)
KNEE_NOTE = (
    "History:\nLeft knee swollen for 2 weeks.\n\n"
    "Medications:\nIbuprofen 400 mg twice daily.\n\n"
    "Assessment:\nMeniscal tear.\n"
)


def _run(
    changes: Any, second: Any, *, transcript: str = KNEE, note: str = KNEE_NOTE, **config: Any
) -> tuple[Outcome, Case, MockClaude]:
    mock = MockClaude(scripts(changes=changes, changes_second=second))
    client = AnthropicClient("claude-sonnet-5-5", api_key=Secret("k"), transport=mock.transport)
    settings = Config(mode="full", provider="anthropic", model="m", passes=("b",), **config)
    t, n = parse_transcript(transcript, consultation_id="b"), parse_note(note)
    rules = _rules(t, n, settings)
    case = Case("c0", t, n, warnings=tuple(rules.b_flags), pass_b_items=tuple(rules.b_unchecked))
    return run_passes([case], settings, client, Budget())["c0"], case, mock


def _frequency(listed: list[tuple[str, str]], _: Any) -> Any:
    rows = [{"id": i, "verdict": "consistent"} for i, _ in listed]
    rows[0] = {
        "id": listed[0][0],
        "verdict": "changed",
        "kind": "frequency",
        "item": "ibuprofen",
        "transcript_value": "three times a day",
        "note_value": "twice daily",
        "transcript_quote": "I take ibuprofen 400 mg three times a day",
        "note_quote": "Ibuprofen 400 mg twice daily",
        "severity": "supporting",
    }
    return {"findings": rows}


def _uphold(value: bool) -> Any:
    return lambda listed, _: {"claims": [{"id": i, "upheld": value} for i, _ in listed]}


def test_warning_confirmed_with_the_model_severity() -> None:
    outcome, case, mock = _run(_frequency, _uphold(True))
    assert [f.rule_id for f in case.warnings] == ["frequency.mismatch"]
    (flag,) = outcome.flags
    assert (flag.type, flag.category, flag.confidence, flag.severity, flag.origin) == (
        "changed_fact",
        "frequency",
        "confirmed",
        "supporting",
        "model",
    )
    assert flag.note_evidence is not None and flag.transcript_evidence is not None
    assert outcome.resolved_warnings == {0}
    assert mock.stages() == ["changes", "changes_second"]


def test_rejected_by_the_second_check_is_unchecked() -> None:
    outcome, _, _ = _run(_frequency, _uphold(False))
    assert outcome.flags == []
    assert [u.reasons for u in outcome.unchecked] == [("second_check_rejected",)]


def test_rejected_can_be_shown_as_possible() -> None:
    outcome, _, _ = _run(_frequency, _uphold(False), show_rejected_as_possible=True)
    assert [(f.confidence, f.category) for f in outcome.flags] == [("possible", "frequency")]


def test_consistent_verdict_drops_the_warning() -> None:
    outcome, _, mock = _run(
        lambda listed, _: {"findings": [{"id": i, "verdict": "consistent"} for i, _ in listed]},
        _uphold(True),
    )
    assert outcome.flags == [] and outcome.resolved_warnings == {0}
    assert "changes_second" not in mock.stages()


def test_certainty_change_with_hedged_sentences_as_hints() -> None:
    def certainty(listed: list[tuple[str, str]], body: Any) -> Any:
        assert "[T5] It might be a meniscal tear." in text_of(body)
        rows: list[dict[str, Any]] = [{"id": i, "verdict": "consistent"} for i, _ in listed]
        rows.append(
            {
                "id": "new1",
                "verdict": "changed",
                "kind": "certainty",
                "item": "diagnosis",
                "transcript_value": "might be a meniscal tear",
                "note_value": "meniscal tear",
                "transcript_quote": "It might be a meniscal tear",
                "note_quote": "Meniscal tear.",
                "severity": "critical",
            }
        )
        return {"findings": rows}

    outcome, _, _ = _run(certainty, _uphold(True))
    (flag,) = outcome.flags
    assert (flag.category, flag.rule_id, flag.confidence, flag.severity) == (
        "certainty",
        "certainty.changed",
        "confirmed",
        "critical",
    )


def test_drug_found_in_the_note_only_goes_to_the_model_as_a_substitution() -> None:
    transcript = "Doctor: Any medicines?\nPatient: I take amoxicillin 500 mg three times a day.\n"
    note = "Medications:\nClarithromycin 500 mg twice daily.\n"

    def substitution(listed: list[tuple[str, str]], body: Any) -> Any:
        assert any("drug in the note only" in text for _, text in listed)
        return {
            "findings": [
                {
                    "id": listed[0][0],
                    "verdict": "changed",
                    "kind": "drug_substituted",
                    "item": "antibiotic",
                    "transcript_value": "amoxicillin",
                    "note_value": "clarithromycin",
                    "transcript_quote": "I take amoxicillin 500 mg three times a day",
                    "note_quote": "Clarithromycin 500 mg twice daily",
                    "severity": "critical",
                }
            ]
        }

    outcome, case, _ = _run(substitution, _uphold(True), transcript=transcript, note=note)
    assert [f.rule_id for f in outcome.flags] == ["drug.substituted"]
    assert outcome.resolved_items  # the rules' unaligned item is settled


def test_unverifiable_quote_is_unchecked() -> None:
    def bad_quote(listed: list[tuple[str, str]], _: Any) -> Any:
        found = _frequency(listed, _)
        found["findings"][0]["note_quote"] = "Ibuprofen 800 mg four times daily"
        return found

    outcome, _, mock = _run(bad_quote, _uphold(True))
    assert outcome.flags == []
    assert "quote_recheck" in mock.stages()
    # The rules warning stays as it was: nothing settled it.
    assert outcome.resolved_warnings == set() and outcome.unchecked == []


def test_failed_second_check_keeps_the_rules_warning() -> None:
    outcome, _, _ = _run(_frequency, lambda listed, _: {"claims": []})
    assert outcome.flags == [] and outcome.resolved_warnings == set()


def test_new_finding_with_unverifiable_quote_is_unchecked() -> None:
    def new(listed: list[tuple[str, str]], _: Any) -> Any:
        rows: list[dict[str, Any]] = [{"id": i, "verdict": "consistent"} for i, _ in listed]
        rows.append(
            {
                "id": "new1",
                "verdict": "changed",
                "kind": "side",
                "item": "knee",
                "transcript_quote": "My right knee has been swollen",
                "note_quote": "Left knee swollen for 2 weeks",
            }
        )
        return {"findings": rows}

    outcome, _, _ = _run(new, _uphold(True))
    assert [u.reasons for u in outcome.unchecked] == [("quote_unverified",)]
