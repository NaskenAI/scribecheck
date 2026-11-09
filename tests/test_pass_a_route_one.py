"""Route one reproduced from the OmissionBench harness, end to end on mocked HTTP."""

from __future__ import annotations

from typing import Any

from _mock_model import MockClaude, text_of

from scribecheck import prompts
from scribecheck.config import Config
from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.budget import Budget
from scribecheck.model.endpoint import Secret
from scribecheck.note import parse_note
from scribecheck.pass_a import Case, run_pass_a
from scribecheck.pass_a.base import plain_transcript
from scribecheck.pass_a.route_one import parse_keyed_verdicts, quote_in_note
from scribecheck.transcript import parse_transcript

TRANSCRIPT = (
    "Doctor: What brings you in today?\n"
    "Patient: I've had a cough for two weeks and no fever.\n"
    "Doctor: Any medicines or allergies?\n"
    "Patient: I take metformin 500 mg twice a day. I'm allergic to penicillin.\n"
    "Doctor: It sounds like a viral cough. Come back if you get short of breath.\n"
)
NOTE = (
    "History:\nCough for 2 weeks. Apyrexial.\n\n"
    "Medications:\nMetformin 500 mg BD.\n\n"
    "Plan:\nLikely viral illness. Safety-netting advice given.\n"
)

FACTS = [
    ("f1", "Cough for two weeks", "I've had a cough for two weeks"),
    ("f2", "Takes metformin 500 mg twice a day", "I take metformin 500 mg twice a day"),
    ("f3", "Allergic to penicillin", "I'm allergic to penicillin"),
    ("f4", "Return if short of breath", "Come back if you get short of breath"),
    ("f5", "Working impression: viral cough", "It sounds like a viral cough"),
    ("f6", "Denies fever", "the patient denies any fever whatsoever"),
    ("f7", "Doctor greeted the patient", "What brings you in today?"),
]
CHECK = {
    "f1": "full",
    "f2": "full",
    "f3": "absent",
    "f4": "partial",
    "f5": "absent",
    "f6": "absent",
}


def _scripts(checked_once: list[bool]) -> dict[str, Any]:
    def extract(_: Any, body: dict[str, Any]) -> str:
        facts = [{"id": i, "fact": f, "evidence": e} for i, f, e in FACTS]
        return "Here you go:\n" + __import__("json").dumps({"facts": facts})

    def audit(listed: list[tuple[str, str]], body: dict[str, Any]) -> Any:
        severity = {"f3": "critical", "f4": "critical", "f6": "supporting"}
        kept = [
            {"id": i, "fact": t, "severity": severity.get(i, "supporting"), "why": "x"}
            for i, t in listed
            if i != "f7"
        ]
        return {"facts": kept, "dropped": [{"id": "f7", "reason": "not-note-bearing"}]}

    def check(listed: list[tuple[str, str]], body: dict[str, Any]) -> Any:
        ids = [i for i, _ in listed]
        if not checked_once:
            checked_once.append(True)
            ids = [i for i in ids if i != "f2"]  # one id missing: a targeted retry follows
        return {"verdicts": {i: CHECK[i] for i in ids}}

    def refute(listed: list[tuple[str, str]], body: dict[str, Any]) -> Any:
        verdicts = {i: {"found": "no", "quote": ""} for i, _ in listed}
        verdicts["f5"] = {"found": "full", "quote": "Likely viral illness."}  # in the note
        verdicts["f6"] = {"found": "full", "quote": "Patient is afebrile today"}  # not in it
        return {"verdicts": verdicts}

    def requote(listed: list[tuple[str, str]], body: dict[str, Any]) -> Any:
        return {"quotes": [{"id": i, "quote": ""} for i, _ in listed]}

    return {
        "r1_extract": extract,
        "r1_audit": audit,
        "r1_check": check,
        "r1_refute": refute,
        "quote_recheck": requote,
    }


def _run() -> tuple[MockClaude, Any, Any]:
    mock = MockClaude(_scripts([]))
    client = AnthropicClient(
        "claude-sonnet-5-5",
        api_key=Secret("sk-test"),
        transport=mock.transport,
        sleep=lambda _: None,
    )
    config = Config(
        mode="full", provider="anthropic", model="claude-sonnet-5-5", pipeline="route_one"
    )
    transcript = parse_transcript(TRANSCRIPT, consultation_id="r1")
    note = parse_note(NOTE)
    case = Case("c0", transcript, note)
    outcome = run_pass_a([case], config, client, Budget(), live=True)["c0"]
    return mock, outcome, case


def test_end_to_end_stages_and_flags() -> None:
    mock, outcome, _ = _run()
    assert mock.stages() == [
        "r1_extract",
        "r1_audit",
        "r1_check",
        "r1_check",
        "r1_refute",
        "quote_recheck",
    ]
    by_fact = {f.fact: f for f in outcome.flags}
    assert set(by_fact) == {"Allergic to penicillin", "Return if short of breath", "Denies fever"}
    allergy = by_fact["Allergic to penicillin"]
    assert (allergy.verdict, allergy.confidence, allergy.severity) == (
        "absent",
        "confirmed",
        "critical",
    )
    assert allergy.rule_id == "route_one.missing" and allergy.second_check == "refute"
    assert allergy.transcript_evidence is not None
    assert allergy.transcript_evidence.text == "I'm allergic to penicillin"
    partial = by_fact["Return if short of breath"]
    assert (partial.verdict, partial.confidence) == ("partial", "possible")


def test_verified_rescue_clears_and_unverified_rescue_does_not() -> None:
    _, outcome, _ = _run()
    facts = {f.fact for f in outcome.flags}
    assert "Working impression: viral cough" not in facts  # rescued with a quote in the note
    assert "Denies fever" in facts  # the refuter's quote is not in the note: absence stands


def test_unverifiable_transcript_quote_keeps_the_flag() -> None:
    _, outcome, _ = _run()
    fever = next(f for f in outcome.flags if f.fact == "Denies fever")
    assert fever.verdict == "absent"  # the verdict is unchanged
    assert fever.confidence == "possible"
    assert fever.evidence_verified is False and fever.transcript_evidence is None
    assert not [u for u in outcome.unchecked if "quote_unverified" in u.reasons]


def test_missing_id_retry_asks_only_about_the_missing_id() -> None:
    mock, outcome, _ = _run()
    retry = [body for stage, body in mock.calls if stage == "r1_check"][1]
    assert "f2: Takes metformin" in text_of(retry) and "f1:" not in text_of(retry)
    assert not [u for u in outcome.unchecked if u.item == "presence check"]


def test_prompts_are_the_verbatim_harness_files_without_constrained_output() -> None:
    mock, _, case = _run()
    extract = mock.calls[0][1]
    assert "output_config" not in extract or "format" not in extract["output_config"]
    assert "system" not in extract
    expected = prompts.render(
        prompts.load("route_one", "pipeline_extract"), transcript=plain_transcript(case.transcript)
    )
    assert text_of(extract) == expected
    assert extract["output_config"]["effort"] == "medium"  # the harness's pinned effort
    audit = mock.calls[1][1]
    assert audit["output_config"]["effort"] == "high" and audit["max_tokens"] == 24000


def test_extraction_and_audit_never_see_the_note() -> None:
    mock, _, _ = _run()
    for stage, body in mock.calls:
        if stage in ("r1_extract", "r1_audit"):
            text = text_of(body)
            for line in NOTE.splitlines():
                if len(line) > 12:
                    assert line not in text


def test_ported_parsers() -> None:
    assert parse_keyed_verdicts(
        {"f1": True, "F2": "Missing"}, ["f1", "f2"], ("full", "partial", "absent")
    ) == {
        "f1": "full",
        "f2": "absent",
    }
    assert quote_in_note("  Metformin   500 mg BD ", NOTE)
    assert not quote_in_note("short", NOTE)
