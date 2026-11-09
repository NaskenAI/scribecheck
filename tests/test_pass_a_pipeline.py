"""scribecheck's own Pass A pipeline, end to end on mocked HTTP."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from _mock_model import MockClaude, text_of

from scribecheck.checker import _rules
from scribecheck.config import Config
from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.budget import Budget
from scribecheck.model.endpoint import Secret
from scribecheck.note import parse_note
from scribecheck.pass_a import Case, run_pass_a
from scribecheck.pass_a.blind import BlindnessError, assert_blind
from scribecheck.transcript import parse_transcript

TRANSCRIPT = (
    "Doctor: What brings you in today?\n"
    "Patient: I've had a cough for two weeks and no fever.\n"
    "Doctor: Any medicines or allergies?\n"
    "Patient: I take metformin 500 mg twice a day. I'm allergic to penicillin.\n"
    "Doctor: It sounds like a viral cough. Come back if you get short of breath.\n"
)
NOTE = (
    "History:\nCough for 2 weeks. No fever.\n\n"
    "Medications:\nMetformin 500 mg BD.\n\n"
    "Plan:\nLikely viral illness. Safety-netting advice given.\n"
)
EXTRACTED = [
    ("Cough for two weeks", "symptom", "I've had a cough for two weeks"),
    ("No fever", "symptom", "no fever"),
    ("Takes metformin 500 mg twice a day", "medicine", "I take metformin 500 mg twice a day"),
    ("Allergic to penicillin", "allergy", "I'm allergic to penicillin"),
    ("Return if short of breath", "follow_up", "Come back if you get short of breath"),
    ("Working impression: viral cough", "assessment", "It sounds like a viral cough"),
    ("Doctor asked what brings the patient in", "history", "What brings you in today?"),
]
VERDICTS: dict[str, tuple[str, str]] = {
    "Cough for two weeks": ("present", "Cough for 2 weeks."),
    "No fever": ("present", "No fever."),
    "Takes metformin 500 mg twice a day": ("present", "Metformin 500 mg BD."),
    "Allergic to penicillin": ("absent", ""),
    "Return if short of breath": ("partial", "Safety-netting advice given."),
    "Working impression: viral cough": ("present", "Likely viral illness."),
    "Doctor asked what brings the patient in": ("absent", ""),
    "Takes metformin": ("present", "Metformin 500 mg BD."),
}

Scripts = dict[str, Callable[[list[tuple[str, str]], dict[str, Any]], Any]]


def scripts(**overrides: Any) -> Scripts:
    def extract(_: Any, body: dict[str, Any]) -> Any:
        return {
            "facts": [
                {"fact": f, "category": c, "attributes": [], "evidence": e} for f, c, e in EXTRACTED
            ]
        }

    def audit(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        out = []
        for fid, text in listed:
            peripheral = text.startswith("Doctor asked")
            severity = (
                "critical"
                if "penicillin" in text or "short of breath" in text
                else ("peripheral" if peripheral else "supporting")
            )
            out.append({"id": fid, "keep": True, "core": True, "severity": severity})
        return {"facts": out}

    def check(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "verdicts": [
                {"id": fid, "verdict": VERDICTS[t][0], "note_quote": VERDICTS[t][1]}
                for fid, t in listed
            ]
        }

    def second_look(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {"facts": [{"id": fid, "found": False} for fid, _ in listed]}

    def candidate(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "findings": [
                {
                    "id": fid,
                    "confirmed": True,
                    "transcript_quote": "I'm allergic to penicillin",
                    "severity": "critical",
                }
                for fid, _ in listed
            ]
        }

    def requote(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {"quotes": [{"id": i, "quote": ""} for i, _ in listed]}

    def no_changes(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {"findings": [{"id": i, "verdict": "consistent"} for i, _ in listed]}

    def no_statements(_: Any, __: dict[str, Any]) -> Any:
        return {"statements": []}

    base: Scripts = {
        "changes": no_changes,
        "changes_second": lambda listed, _: {"claims": []},
        "split": no_statements,
        "statements": no_statements,
        "statements_second": no_statements,
        "extract": extract,
        "audit": audit,
        "check": check,
        "second_look": second_look,
        "candidate": candidate,
        "quote_recheck": requote,
    }
    base.update(overrides)
    return base


def run(
    mock: MockClaude,
    *,
    transcript: str = TRANSCRIPT,
    note: str = NOTE,
    candidates: bool = True,
    **config: Any,
) -> tuple[Any, Case]:
    client = AnthropicClient(
        "claude-sonnet-5-5",
        api_key=Secret("sk-test"),
        transport=mock.transport,
        sleep=lambda _: None,
    )
    settings = Config(mode="full", provider="anthropic", model="claude-sonnet-5-5", **config)
    t, n = parse_transcript(transcript, consultation_id="p1"), parse_note(note)
    rules = _rules(t, n, settings)
    found = [*rules.a_flags, *rules.b_flags]
    case = Case("c0", t, n, tuple(found) if candidates else ())
    return run_pass_a([case], settings, client, Budget(), live=True)["c0"], case


def test_end_to_end() -> None:
    mock = MockClaude(scripts())
    outcome, case = run(mock)
    assert mock.stages() == ["extract", "audit", "check", "second_look", "candidate"]
    flags = {(f.fact, f.verdict, f.confidence, f.severity, f.origin) for f in outcome.flags}
    assert flags == {
        ("Return if short of breath", "partial", "possible", "critical", "model"),
        ("allergy: penicillin", "absent", "confirmed", "critical", "rules"),
    }
    # The model's own "Allergic to penicillin" omission gives way to the confirmed rules
    # candidate: one discrepancy, one flag.
    partial = next(f for f in outcome.flags if f.verdict == "partial")
    assert partial.note_evidence is not None and partial.note_evidence.text.startswith("Safety")
    assert partial.transcript_evidence is not None
    assert partial.expected_section == "Plan"
    assert outcome.resolved_candidates == {0}
    assert outcome.tally.calls == 5


def test_extraction_and_audit_never_contain_note_text() -> None:
    mock = MockClaude(scripts())
    run(mock)
    for stage, body in mock.calls:
        if stage in ("extract", "audit"):
            text = text_of(body)
            assert "NOTE:" not in text
            for line in NOTE.splitlines():
                if len(line) > 10:
                    assert line not in text
            assert body["messages"][0]["content"][0]["text"].startswith("TRANSCRIPT")
            assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_blindness_rail_raises_on_a_leak() -> None:
    note = "History: a long synthetic note about a cough lasting two weeks with no fever at all."
    with pytest.raises(BlindnessError):
        assert_blind("TRANSCRIPT ... " + note, [note], "test")
    assert_blind("TRANSCRIPT only", [note], "test")


def test_present_with_an_unverifiable_quote_is_unchecked() -> None:
    def check(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        rows = []
        for fid, t in listed:
            verdict, quote = VERDICTS[t]
            if t == "No fever":
                quote = "Patient is entirely afebrile"
            rows.append({"id": fid, "verdict": verdict, "note_quote": quote})
        return {"verdicts": rows}

    mock = MockClaude(scripts(check=check))
    outcome, _ = run(mock)
    assert "quote_recheck" in mock.stages()
    unverified = [u for u in outcome.unchecked if u.reasons == ("quote_unverified",)]
    assert [u.item for u in unverified] == ["No fever"]
    assert "No fever" not in {f.fact for f in outcome.flags}


def test_facts_without_a_transcript_quote_are_dropped() -> None:
    def extract(_: Any, body: dict[str, Any]) -> Any:
        return {
            "facts": [
                {
                    "fact": "Smokes 20 a day",
                    "category": "family_social",
                    "attributes": [],
                    "evidence": "I smoke twenty cigarettes a day",
                },
                {
                    "fact": "Cough for two weeks",
                    "category": "symptom",
                    "attributes": [],
                    "evidence": "I've had a cough for two weeks",
                },
            ]
        }

    mock = MockClaude(scripts(extract=extract))
    outcome, _ = run(mock, candidates=False)
    texts = {f.text: f for f in outcome.facts}
    assert "Smokes 20 a day" not in texts
    assert mock.stages()[:2] == ["extract", "quote_recheck"]
    # The rules layer adds the medicine the model did not name; the allergy too.
    assert texts["Takes metformin"].origin == "rules"
    assert texts["Allergic to penicillin"].origin == "rules"


def test_rejected_candidate_gives_no_flag() -> None:
    def candidate(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "findings": [{"id": i, "confirmed": False, "severity": "peripheral"} for i, _ in listed]
        }

    def check(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        rows = []
        for fid, t in listed:
            verdict, quote = VERDICTS[t]
            if "penicillin" in t:  # the model's own fact is found present this time
                verdict, quote = "present", "Metformin 500 mg BD."
            rows.append({"id": fid, "verdict": verdict, "note_quote": quote})
        return {"verdicts": rows}

    mock = MockClaude(scripts(candidate=candidate, check=check))
    outcome, _ = run(mock)
    assert outcome.resolved_candidates == {0}
    assert not [f for f in outcome.flags if f.origin == "rules"]


KNEE = (
    "Doctor: What seems to be the problem?\n"
    "Patient: My left knee has been swollen for two weeks.\n"
    "Doctor: Are you taking anything for it?\n"
    "Patient: I take ibuprofen 400 mg three times a day.\n"
)
KNEE_NOTE = (
    "History:\nLeft knee swollen for 2 weeks.\n\nMedications:\nIbuprofen 400 mg twice daily.\n"
)


def test_changed_value_warning_confirmed_takes_the_model_severity() -> None:
    def extract(_: Any, body: dict[str, Any]) -> Any:
        return {
            "facts": [
                {
                    "fact": "Left knee swollen for two weeks",
                    "category": "symptom",
                    "attributes": [],
                    "evidence": "My left knee has been swollen",
                }
            ]
        }

    def check(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        rows = []
        for i, t in listed:
            quote = "Ibuprofen 400 mg twice daily." if "ibuprofen" in t else "Left knee swollen"
            rows.append({"id": i, "verdict": "present", "note_quote": quote})
        return {"verdicts": rows}

    def candidate(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "findings": [
                {
                    "id": i,
                    "confirmed": True,
                    "severity": "supporting",
                    "transcript_quote": "ibuprofen 400 mg three times a day",
                    "note_quote": "Ibuprofen 400 mg twice daily",
                }
                for i, _ in listed
            ]
        }

    mock = MockClaude(scripts(extract=extract, check=check, candidate=candidate))
    outcome, case = run(mock, transcript=KNEE, note=KNEE_NOTE)
    assert [f.rule_id for f in case.candidates] == ["frequency.mismatch"]
    (flag,) = outcome.flags
    assert (flag.type, flag.confidence, flag.severity, flag.second_check) == (
        "changed_fact",
        "confirmed",
        "supporting",
        "model",
    )  # the rules default was critical
    assert flag.note_evidence is not None and flag.note_evidence.text.startswith("Ibuprofen")


def test_candidate_with_unverifiable_quote_is_unchecked() -> None:
    def candidate(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "findings": [
                {
                    "id": i,
                    "confirmed": True,
                    "severity": "critical",
                    "transcript_quote": "she is allergic to latex gloves",
                }
                for i, _ in listed
            ]
        }

    mock = MockClaude(scripts(candidate=candidate))
    outcome, _ = run(mock)
    assert [u.reasons for u in outcome.unchecked if u.category == "allergy"] == [
        ("quote_unverified",)
    ]


def test_clinical_standard_skips_peripheral_facts() -> None:
    mock = MockClaude(scripts())
    run(mock, candidates=False)
    clinical = next(text_of(b) for s, b in mock.calls if s == "check")
    mock_strict = MockClaude(scripts())
    run(mock_strict, candidates=False, standard="strict")
    strict = next(text_of(b) for s, b in mock_strict.calls if s == "check")
    assert "Doctor asked" not in clinical and "Doctor asked" in strict


def test_checks_are_batched_about_ten_facts_per_call() -> None:
    many = [
        (f"Fact number {i} about the cough", "symptom", "I've had a cough for two weeks")
        for i in range(23)
    ]

    def extract(_: Any, body: dict[str, Any]) -> Any:
        return {
            "facts": [
                {"fact": f, "category": c, "attributes": [], "evidence": e} for f, c, e in many
            ]
        }

    def check(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "verdicts": [
                {"id": i, "verdict": "present", "note_quote": "Cough for 2 weeks."}
                for i, _ in listed
            ]
        }

    mock = MockClaude(scripts(extract=extract, check=check))
    run(mock, candidates=False)
    assert mock.stages().count("check") == 3  # 23 facts, 10 per call
    body = next(b for s, b in mock.calls if s == "check")
    assert [p.get("cache_control") for p in body["messages"][0]["content"]][:2] == [
        {"type": "ephemeral"},
        {"type": "ephemeral"},
    ]


def test_long_transcript_is_extracted_in_overlapping_chunks() -> None:
    turns = []
    for i in range(1, 41):
        speaker = "Doctor" if i % 2 else "Patient"
        turns.append(f"{speaker}: Synthetic line number {i} about a persistent dry cough.")
    long_transcript = "\n".join(turns) + "\n"
    seen: list[str] = []

    def extract(_: Any, body: dict[str, Any]) -> Any:
        seen.append(body["messages"][0]["content"][0]["text"])
        return {
            "facts": [
                {
                    "fact": "Persistent dry cough",
                    "category": "symptom",
                    "attributes": [],
                    "evidence": "about a persistent dry cough",
                }
            ]
        }

    def check(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "verdicts": [
                {"id": i, "verdict": "present", "note_quote": "Cough for 2 weeks."}
                for i, _ in listed
            ]
        }

    mock = MockClaude(scripts(extract=extract, check=check))
    outcome, _ = run(
        mock, transcript=long_transcript, candidates=False, chunk_chars=1200, chunk_overlap_turns=2
    )
    assert len(seen) >= 3
    assert all(s.startswith("TRANSCRIPT, PART") for s in seen)
    assert not any("NOTE" in s or "Metformin" in s for s in seen)
    last_of_first = seen[0].strip().splitlines()[-1].split("]")[0] + "]"  # e.g. "[T18]"
    assert last_of_first in seen[1]  # neighbouring chunks overlap
    assert [f.text for f in outcome.facts] == ["Persistent dry cough"]  # merged across chunks
