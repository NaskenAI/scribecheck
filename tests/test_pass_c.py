"""Pass C: statements split, exempted, checked and second-checked."""

from __future__ import annotations

import json
from typing import Any

from _mock_model import MockClaude
from test_pass_a_pipeline import scripts

from scribecheck.config import Config
from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.budget import Budget
from scribecheck.model.endpoint import Secret
from scribecheck.model_passes import run_passes
from scribecheck.note import parse_note
from scribecheck.pass_a.base import Case, Outcome
from scribecheck.transcript import parse_transcript

TRANSCRIPT = (
    "Doctor: What brings you in?\n"
    "Patient: I've had a cough for two weeks.\n"
    "Doctor: Any fever?\n"
    "Patient: No fever at all.\n"
)
NOTE = (
    "History:\nCough for 2 weeks. Fever for 3 days. History of type 2 diabetes. "
    "Hypertension per records.\n\n"
    "Imported history:\nAsthma.\n\n"
    "Examination:\nChest clear.\n\n"
    "Vitals:\nBP 128/82.\n\n"
    "Results:\nCRP 4.\n\n"
    "Demographics:\n45-year-old man.\n\n"
    "Plan:\nStarted on amoxicillin.\n"
)
STATEMENTS = [
    ("Cough for 2 weeks", "clinical"),
    ("Fever for 3 days", "clinical"),
    ("History of type 2 diabetes", "record_history"),
    ("Hypertension per records", "record_history"),
    ("Asthma", "record_history"),
    ("Chest clear", "examination"),
    ("BP 128/82", "vital"),
    ("CRP 4", "result"),
    ("45-year-old man", "demographic"),
    ("Started on amoxicillin", "clinical"),
]
VERDICTS = {
    "Cough for 2 weeks": ("supported", "I've had a cough for two weeks"),
    "Fever for 3 days": ("contradicted", "No fever at all"),
    "History of type 2 diabetes": ("unsupported", ""),
    "Started on amoxicillin": ("unsupported", ""),
}


def _scripts(upheld: dict[str, bool], statements: list[tuple[str, str]] = STATEMENTS) -> Any:
    def split(_: Any, __: Any) -> Any:
        return {"statements": [{"text": t, "kind": k} for t, k in statements]}

    def check(listed: list[tuple[str, str]], _: Any) -> Any:
        return {
            "statements": [
                {
                    "id": i,
                    "verdict": VERDICTS[t][0],
                    "transcript_quote": VERDICTS[t][1],
                    "severity": "critical",
                }
                for i, t in listed
            ]
        }

    def second(listed: list[tuple[str, str]], _: Any) -> Any:
        rows = []
        for i, text in listed:
            statement = text.split("] ", 1)[1]
            rows.append({"id": i, "upheld": upheld.get(statement, True)})
        return {"statements": rows}

    return scripts(split=split, statements=check, statements_second=second)


def _run(
    upheld: dict[str, bool] | None = None, *, modality: str | None = "in-person", **config: Any
) -> tuple[Outcome, MockClaude]:
    mock = MockClaude(_scripts(upheld or {}))
    client = AnthropicClient("claude-sonnet-5-5", api_key=Secret("k"), transport=mock.transport)
    settings = Config(mode="full", provider="anthropic", model="m", passes=("c",), **config)
    data: dict[str, Any] = {
        "consultation_id": "c",
        "turns": [
            {"speaker": s, "text": t}
            for s, t in (line.split(": ", 1) for line in TRANSCRIPT.strip().splitlines())
        ],
    }
    if modality:
        data["modality"] = modality
    case = Case("c0", parse_transcript(json.dumps(data)), parse_note(NOTE))
    return run_passes([case], settings, client, Budget())["c0"], mock


def test_verdicts_confirmed() -> None:
    outcome, mock = _run()
    flags = {f.fact: f for f in outcome.flags}
    assert set(flags) == {
        "Fever for 3 days",
        "History of type 2 diabetes",
        "Started on amoxicillin",
    }
    fever = flags["Fever for 3 days"]
    assert (fever.verdict, fever.confidence, fever.rule_id) == (
        "contradicted",
        "confirmed",
        "statement.contradicted",
    )
    assert (
        fever.transcript_evidence is not None
        and fever.transcript_evidence.text == "No fever at all"
    )
    assert flags["Started on amoxicillin"].verdict == "unsupported"
    assert flags["Started on amoxicillin"].transcript_evidence is None
    assert mock.stages()[0] == "split"


def test_invented_history_without_a_record_cue_is_checked_not_exempted() -> None:
    outcome, _ = _run()
    exempt = {nv.text for nv, _ in outcome.not_verifiable}
    assert "History of type 2 diabetes" not in exempt
    assert "History of type 2 diabetes" in {f.fact for f in outcome.flags}


def test_exemptions_are_not_verifiable_and_never_flagged() -> None:
    outcome, _ = _run()
    exempt = {(nv.text, kind) for nv, kind in outcome.not_verifiable}
    assert exempt == {
        ("Hypertension per records", "record_history"),
        ("Asthma", "record_history"),  # in an imported section
        ("Chest clear", "examination"),  # face-to-face
        ("BP 128/82", "vital"),
        ("CRP 4", "result"),
        ("45-year-old man", "demographic"),
    }
    flagged = {f.fact for f in outcome.flags}
    assert not flagged & {text for text, _ in exempt}


def test_exemptions_are_configurable() -> None:
    def checked(listed: list[tuple[str, str]], _: Any) -> Any:
        return {
            "statements": [
                {
                    "id": i,
                    "verdict": "supported",
                    "transcript_quote": "I've had a cough for two weeks",
                }
                for i, _ in listed
            ]
        }

    mock = MockClaude(
        scripts(
            split=lambda _, __: {"statements": [{"text": "BP 128/82", "kind": "vital"}]},
            statements=checked,
        )
    )
    client = AnthropicClient("claude-sonnet-5-5", api_key=Secret("k"), transport=mock.transport)
    config = Config(mode="full", provider="anthropic", model="m", passes=("c",), exempt=("result",))
    case = Case("c0", parse_transcript(TRANSCRIPT), parse_note(NOTE))
    outcome = run_passes([case], config, client, Budget())["c0"]
    assert outcome.not_verifiable == [] and "statements" in mock.stages()


def test_not_upheld_is_unchecked_or_possible() -> None:
    outcome, _ = _run({"Started on amoxicillin": False})
    assert "Started on amoxicillin" not in {f.fact for f in outcome.flags}
    assert ("second_check_rejected",) in [u.reasons for u in outcome.unchecked]
    shown, _ = _run({"Started on amoxicillin": False}, show_rejected_as_possible=True)
    amox = next(f for f in shown.flags if f.fact == "Started on amoxicillin")
    assert amox.confidence == "possible"


def test_remote_examination_findings_are_left_to_the_examination_rule() -> None:
    outcome, _ = _run(modality="telephone")
    assert "Chest clear" not in {nv.text for nv, _ in outcome.not_verifiable}
    assert "Chest clear" not in {f.fact for f in outcome.flags}


def test_unanchored_statement_is_unchecked() -> None:
    mock = MockClaude(_scripts({}, statements=[("Patient smokes forty a day", "clinical")]))
    client = AnthropicClient("claude-sonnet-5-5", api_key=Secret("k"), transport=mock.transport)
    config = Config(mode="full", provider="anthropic", model="m", passes=("c",))
    case = Case("c0", parse_transcript(TRANSCRIPT), parse_note(NOTE))
    outcome = run_passes([case], config, client, Budget())["c0"]
    assert [u.reasons for u in outcome.unchecked] == [("quote_unverified",)]
