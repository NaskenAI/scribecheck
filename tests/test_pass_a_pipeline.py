"""scribecheck's own Pass A pipeline, end to end on mocked HTTP."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from _mock_model import MockClaude, text_of
from pydantic import BaseModel

from scribecheck.checker import _rules
from scribecheck.config import Config
from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.budget import Budget
from scribecheck.model.endpoint import Secret
from scribecheck.model.structured import strict_schema
from scribecheck.note import parse_note
from scribecheck.pass_a import Case, pipeline, run_pass_a
from scribecheck.pass_a.blind import BlindnessError, assert_blind
from scribecheck.pass_a.pipeline import is_core_kind, is_logistics
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


@pytest.fixture
def v5(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run Pass A under the scribecheck_v5 set (kept for the record; the default is v4)."""
    monkeypatch.setattr(pipeline, "PROMPTS", "scribecheck_v5")


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


def test_the_default_is_the_frozen_v4_candidate() -> None:
    assert pipeline.PROMPTS == "scribecheck_v4"
    mock = MockClaude(scripts())
    outcome, _ = run(mock)
    # No partial look: the frozen candidate runs as the v4 full development run did.
    assert mock.stages() == ["extract", "audit", "check", "second_look", "candidate"]
    assert outcome.tally.calls == 5
    assert all("partial_look" not in step for step in outcome.trace.values())


def test_the_v4_audit_has_no_v5_backstops() -> None:
    rows = (
        ("If the cough gets worse, ring the practice", "follow_up", "If the cough gets worse"),
        ("Does not need to attend in person", "plan", "You don't need to come in"),
        ("No fever", "symptom", "If the cough gets worse"),
    )

    def audit(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        by_text = dict((text, fid) for fid, text in listed)
        return {
            "facts": [
                # Dropped: v5 would keep it (never_drop).
                {"id": by_text[rows[0][0]], "keep": False, "core": False, "severity": "critical"},
                # Core logistics: v5 would make it contextual.
                {"id": by_text[rows[1][0]], "keep": True, "core": True, "severity": "supporting"},
                # A contextual denial: v5 would make it core (core_kind).
                {"id": by_text[rows[2][0]], "keep": True, "core": False, "severity": "supporting"},
            ]
        }

    mock = MockClaude(scripts(extract=_facts(*rows), audit=audit, check=_verdicts))
    outcome, _ = run(mock, transcript=LOGISTICS_TRANSCRIPT, note=NKDA_NOTE, candidates=False)
    steps = {s["fact"]: s for s in outcome.trace.values()}
    assert steps[rows[0][0]]["audit"] == "dropped"
    assert (steps[rows[1][0]]["core"], steps[rows[1][0]]["checked"]) == (True, True)
    assert (steps[rows[2][0]]["core"], steps[rows[2][0]]["checked"]) == (False, False)
    assert all("forced" not in step for step in steps.values())


def test_the_v4_second_look_asks_and_keeps_no_absence() -> None:
    """The frozen candidate's second-look request is the v4 run's: its output schema has no
    `absence`, and an answer that sends one anyway never puts it on a flag."""

    class _Looked(BaseModel):  # as at ee8e2ac, the v4 full run's commit
        id: str
        found: bool
        note_quote: str = ""
        transcript_quote: str = ""

    class Looked(BaseModel):
        facts: list[_Looked]

    bodies: list[dict[str, Any]] = []

    def second_look(listed: list[tuple[str, str]], body: dict[str, Any]) -> Any:
        bodies.append(body)
        return {"facts": [{"id": fid, "found": False, "absence": "clear"} for fid, _ in listed]}

    outcome, _ = run(MockClaude(scripts(second_look=second_look)), candidates=False)
    assert bodies and all(
        b["output_config"]["format"]["schema"] == strict_schema(Looked) for b in bodies
    )
    absent = [f for f in outcome.flags if f.verdict == "absent"]
    assert absent and all(f.absence is None for f in absent)
    assert all("absence" not in step for step in outcome.trace.values())


@pytest.mark.usefixtures("v5")
def test_end_to_end_v5() -> None:
    mock = MockClaude(scripts())
    outcome, case = run(mock)
    assert mock.stages() == [
        "extract",
        "audit",
        "check",
        "second_look",
        "candidate",
        "partial_look",
    ]
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
    assert outcome.tally.calls == 6  # the partial look runs for the one partial verdict


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


@pytest.mark.usefixtures("v5")
def test_trace_records_every_stage() -> None:
    def extract(_: Any, __: dict[str, Any]) -> Any:
        facts = [
            {"fact": f, "category": c, "attributes": [], "evidence": e} for f, c, e in EXTRACTED
        ]
        facts.append(
            {"fact": "Has a cat", "category": "family_social", "evidence": "I have a cat at home"}
        )
        return {"facts": facts}

    def audit(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "facts": [
                {
                    "id": fid,
                    "keep": not text.startswith("Working impression"),
                    "core": text not in ("Cough for two weeks", "No fever"),
                    "severity": "critical" if "penicillin" in text else "supporting",
                }
                for fid, text in listed
            ]
        }

    def second_look(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "facts": [
                {"id": fid, "found": "penicillin" not in t, "note_quote": "Likely viral illness."}
                for fid, t in listed
            ]
        }

    mock = MockClaude(scripts(extract=extract, audit=audit, second_look=second_look))
    outcome, _ = run(mock, candidates=False)
    steps = {s["fact"]: s for s in outcome.trace.values()}
    assert steps["Has a cat"]["evidence_verified"] is False and "audit" not in steps["Has a cat"]
    dropped = steps["Working impression: viral cough"]
    assert (dropped["audit"], dropped["core"], dropped["severity"]) == ("dropped", None, None)
    cough = steps["Cough for two weeks"]
    assert (cough["audit"], cough["core"], cough["checked"]) == ("kept", False, False)
    # A pertinent negative the audit called contextual is core all the same, and tagged.
    no_fever = steps["No fever"]
    assert (no_fever["core"], no_fever["checked"], no_fever["forced"]) == (
        True,
        True,
        ["core_kind"],
    )
    penicillin = steps["Allergic to penicillin"]
    assert (penicillin["verdict"], penicillin["second_look"]) == ("absent", "confirmed")
    assert penicillin["severity"] == "critical"
    assert steps["Return if short of breath"]["verdict"] == "partial"
    asked = steps["Doctor asked what brings the patient in"]
    assert (asked["verdict"], asked["second_look"]) == ("absent", "rescued")
    assert asked["note_quote"] == "Likely viral illness"


def test_trace_marks_an_unverified_present_quote() -> None:
    def check(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        rows = []
        for fid, t in listed:
            verdict, quote = VERDICTS[t]
            rows.append({"id": fid, "verdict": verdict, "note_quote": quote})
            if t == "No fever":
                rows[-1]["note_quote"] = "Patient is entirely afebrile"
        return {"verdicts": rows}

    outcome, _ = run(MockClaude(scripts(check=check)), candidates=False)
    step = next(s for s in outcome.trace.values() if s["fact"] == "No fever")
    assert (step["verdict"], step["note_quote_verified"]) == ("present", False)
    assert "note_quote" not in step


NKDA_TRANSCRIPT = (
    "Doctor: What brings you in today?\n"
    "Patient: I've had a cough for two weeks.\n"
    "Doctor: Any allergies?\n"
    "Patient: No known drug allergies.\n"
)
NKDA_NOTE = "History:\nCough for 2 weeks.\n"


def _facts(*rows: tuple[str, str, str]) -> Callable[[Any, dict[str, Any]], Any]:
    def extract(_: Any, __: dict[str, Any]) -> Any:
        return {
            "facts": [
                {"fact": f, "category": c, "attributes": [], "evidence": e} for f, c, e in rows
            ]
        }

    return extract


def _verdicts(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
    return {
        "verdicts": [
            {"id": fid, "verdict": "present", "note_quote": "Cough for 2 weeks"}
            if "cough" in t.lower()
            else {"id": fid, "verdict": "absent"}
            for fid, t in listed
        ]
    }


def _confirm_status(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
    return {
        "findings": [
            {
                "id": fid,
                "confirmed": True,
                "transcript_quote": "No known drug allergies",
                "severity": "critical",
            }
            for fid, _ in listed
        ]
    }


def test_the_no_allergy_status_becomes_a_rules_fact() -> None:
    cough = ("Cough for two weeks", "symptom", "I've had a cough for two weeks")
    mock = MockClaude(scripts(extract=_facts(cough), check=_verdicts))
    outcome, _ = run(mock, transcript=NKDA_TRANSCRIPT, note=NKDA_NOTE, candidates=False)
    rules = [f for f in outcome.facts if f.origin == "rules"]
    assert [(f.text, f.category) for f in rules] == [("No known drug allergies", "allergy")]


def test_a_model_allergy_fact_stands_in_for_the_rules_fact() -> None:
    rows = (
        ("Cough for two weeks", "symptom", "I've had a cough for two weeks"),
        ("No known drug allergies", "allergy", "No known drug allergies"),
    )
    mock = MockClaude(scripts(extract=_facts(*rows), check=_verdicts, candidate=_confirm_status))
    outcome, case = run(mock, transcript=NKDA_TRANSCRIPT, note=NKDA_NOTE)
    assert not [f for f in outcome.facts if f.origin == "rules"]
    assert [c.rule_id for c in case.candidates] == ["allergy.status_missing"]
    # One discrepancy, one flag: the confirmed rules candidate replaces the model's absence.
    assert [(f.fact, f.origin, f.confidence) for f in outcome.flags] == [
        ("no known drug allergies", "rules", "confirmed")
    ]


def test_a_confirmed_status_never_removes_an_allergen_flag() -> None:
    rows = (
        ("Cough for two weeks", "symptom", "I've had a cough for two weeks"),
        ("Takes cetirizine for allergic rhinitis", "medicine", "No known drug allergies"),
        ("Possible shellfish allergy", "allergy", "No known drug allergies"),
    )
    mock = MockClaude(scripts(extract=_facts(*rows), check=_verdicts, candidate=_confirm_status))
    outcome, _ = run(mock, transcript=NKDA_TRANSCRIPT, note=NKDA_NOTE)
    # Neither model fact is an allergy status: the rules fact is added, and confirming the
    # status candidate leaves both model flags standing.
    assert [f.text for f in outcome.facts if f.origin == "rules"] == ["No known drug allergies"]
    assert sorted(f.fact or "" for f in outcome.flags) == [
        "Possible shellfish allergy",
        "Takes cetirizine for allergic rhinitis",
        "no known drug allergies",
    ]


def test_medicines_and_allergies_are_always_core() -> None:
    def audit(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "facts": [
                {"id": fid, "keep": True, "core": False, "severity": "supporting"}
                for fid, _ in listed
            ]
        }

    outcome, _ = run(MockClaude(scripts(audit=audit)), candidates=False)
    steps = {s["fact"]: s for s in outcome.trace.values()}
    for fact in ("Takes metformin 500 mg twice a day", "Allergic to penicillin"):
        assert (steps[fact]["core"], steps[fact]["checked"]) == (True, True), fact
    for fact in ("Cough for two weeks", "Working impression: viral cough"):
        assert (steps[fact]["core"], steps[fact]["checked"]) == (False, False), fact


LOGISTICS_TRANSCRIPT = (
    "Doctor: You don't need to come in, we can do this on the phone.\n"
    "Doctor: If the cough gets worse, ring us straight away.\n"
)


@pytest.mark.usefixtures("v5")
def test_attendance_logistics_are_never_core() -> None:
    rows = (
        ("Does not need to attend in person", "plan", "You don't need to come in"),
        ("If the cough gets worse, ring the practice", "follow_up", "If the cough gets worse"),
    )

    def audit(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "facts": [
                {"id": fid, "keep": True, "core": True, "severity": "supporting"}
                for fid, _ in listed
            ]
        }

    mock = MockClaude(scripts(extract=_facts(*rows), audit=audit, check=_verdicts))
    outcome, _ = run(mock, transcript=LOGISTICS_TRANSCRIPT, note=NKDA_NOTE, candidates=False)
    steps = {s["fact"]: s for s in outcome.trace.values()}
    # The audit called both core; the logistics fact is not, the safety-net still is.
    assert (steps[rows[0][0]]["core"], steps[rows[0][0]]["checked"]) == (False, False)
    assert (steps[rows[1][0]]["core"], steps[rows[1][0]]["checked"]) == (True, True)


@pytest.mark.parametrize(
    ("category", "text", "expected"),
    [
        ("plan", "Does not need to attend in person", True),
        ("plan", "Consultation by phone; no need to come in person", True),
        ("plan", "Partner will take them by taxi", True),
        ("plan", "Book fasting blood tests", False),
        ("plan", "Patient should attend face-to-face for review of the mole", False),
        ("plan", "Must not drive; arrange transport to the clinic", False),
        ("plan", "Practice nurse will phone to arrange the INR check", False),
        ("plan", "Referred to cardiology; hospital will call with an appointment", False),
        ("plan", "Come in person at 3pm on Tuesday", False),
        ("follow_up", "If symptoms recur, contact the practice for phone assessment", False),
        ("follow_up", "Review booked in 2 weeks", False),
        ("plan", "Not to drive for one month", False),
        ("plan", "Avoid heavy lifting", False),
        ("history", "Does not need to attend in person", False),
    ],
)
def test_is_logistics(category: str, text: str, expected: bool) -> None:
    assert is_logistics(category, text) is expected


SPECIFIC_TRANSCRIPT = (
    "Patient: I can't run very far any more without pain in my knee.\n"
    "Doctor: Come back if your temperature goes above 38.\n"
)
SPECIFIC_NOTE = (
    "History:\nPain comes on 20 min into a run.\n\nPlan:\nContact us if the temperature persists.\n"
)


def test_second_look_rescues_specific_and_keeps_vaguer_as_absent() -> None:
    """Both directions through the pipeline, with the second look answering as the v4 prompt
    asks: the more specific running statement carries the fact; the vaguer temperature
    advice does not carry the threshold."""
    rows = (
        ("Cannot run far without knee pain", "symptom", "I can't run very far any more"),
        ("Return if temperature above 38", "follow_up", "Come back if your temperature"),
    )

    def look(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "facts": [
                {"id": fid, "found": True, "note_quote": "Pain comes on 20 min into a run"}
                if "run" in text
                else {"id": fid, "found": False}
                for fid, text in listed
            ]
        }

    def absent(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {"verdicts": [{"id": fid, "verdict": "absent"} for fid, _ in listed]}

    mock = MockClaude(scripts(extract=_facts(*rows), check=absent, second_look=look))
    outcome, _ = run(mock, transcript=SPECIFIC_TRANSCRIPT, note=SPECIFIC_NOTE, candidates=False)
    steps = {s["fact"]: s for s in outcome.trace.values()}
    assert steps["Cannot run far without knee pain"]["second_look"] == "rescued"
    assert steps["Return if temperature above 38"]["second_look"] == "confirmed"
    assert [f.fact for f in outcome.flags] == ["Return if temperature above 38"]


def _partial_look(missing: bool, quote: str) -> Callable[[Any, dict[str, Any]], Any]:
    def look(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "facts": [
                {
                    "id": fid,
                    "element": "short of breath" if "short of breath" in text else "",
                    "missing": missing,
                    "transcript_quote": quote,
                    "absence": "clear",
                }
                for fid, text in listed
            ]
        }

    return look


@pytest.mark.usefixtures("v5")
def test_a_partial_whose_element_is_missing_becomes_an_omission_flag() -> None:
    look = _partial_look(True, "Come back if you get short of breath")
    outcome, _ = run(MockClaude(scripts(partial_look=look)), candidates=False)
    element = next(f for f in outcome.flags if f.rule_id == "pass_a.partial_element")
    assert (element.verdict, element.confidence, element.severity, element.absence) == (
        "absent",
        "confirmed",
        "critical",
        "clear",
    )
    assert element.fact == "Return if short of breath: missing short of breath"
    assert element.transcript_evidence is not None
    assert element.transcript_evidence.text == "Come back if you get short of breath"
    assert element.note_evidence is not None  # the vaguer note text it matched
    assert not [f for f in outcome.flags if f.verdict == "partial"]  # replaced
    step = next(s for s in outcome.trace.values() if s["fact"] == "Return if short of breath")
    assert (step["partial_look"], step["partial_element"]) == ("flagged", "short of breath")


@pytest.mark.usefixtures("v5")
@pytest.mark.parametrize(
    ("missing", "quote", "label"),
    [
        (False, "Come back if you get short of breath", "carried"),
        (True, "words the transcript never says", "unverified"),
    ],
)
def test_a_carried_or_unverified_element_leaves_the_partial(
    missing: bool, quote: str, label: str
) -> None:
    look = _partial_look(missing, quote)
    outcome, _ = run(MockClaude(scripts(partial_look=look)), candidates=False)
    assert not [f for f in outcome.flags if f.rule_id == "pass_a.partial_element"]
    assert [f.fact for f in outcome.flags if f.verdict == "partial"] == [
        "Return if short of breath"
    ]
    step = next(s for s in outcome.trace.values() if s["fact"] == "Return if short of breath")
    assert step["partial_look"] == label


@pytest.mark.usefixtures("v5")
def test_the_default_partial_look_finds_no_element() -> None:
    outcome, _ = run(MockClaude(scripts()), candidates=False)
    step = next(s for s in outcome.trace.values() if s["fact"] == "Return if short of breath")
    assert step["partial_look"] == "no_element" and "partial_element" not in step


@pytest.mark.usefixtures("v5")
def test_the_audit_never_drops_a_plan_fact_unless_logistics() -> None:
    rows = (
        ("If the cough gets worse, ring the practice", "follow_up", "If the cough gets worse"),
        ("Does not need to attend in person", "plan", "You don't need to come in"),
        ("Cough for two weeks", "symptom", "If the cough gets worse"),
    )

    def audit(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "facts": [
                {"id": fid, "keep": False, "core": False, "severity": "supporting"}
                for fid, _ in listed
            ]
        }

    mock = MockClaude(scripts(extract=_facts(*rows), audit=audit, check=_verdicts))
    outcome, _ = run(mock, transcript=LOGISTICS_TRANSCRIPT, note=NKDA_NOTE, candidates=False)
    steps = {s["fact"]: s for s in outcome.trace.values()}
    safety = steps[rows[0][0]]
    assert (safety["audit"], safety["core"], safety["checked"], safety["forced"]) == (
        "kept",
        True,
        True,
        ["never_drop"],
    )
    for dropped in (rows[1][0], rows[2][0]):
        assert steps[dropped]["audit"] == "dropped" and "forced" not in steps[dropped]


@pytest.mark.usefixtures("v5")
def test_the_second_look_grades_each_absence() -> None:
    def second_look(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {"facts": [{"id": fid, "found": False, "absence": "probable"} for fid, _ in listed]}

    outcome, _ = run(MockClaude(scripts(second_look=second_look)), candidates=False)
    absent = [f for f in outcome.flags if f.verdict == "absent"]
    assert absent and all(f.absence == "probable" for f in absent)
    step = next(s for s in outcome.trace.values() if s["fact"] == "Allergic to penicillin")
    assert step["absence"] == "probable"


@pytest.mark.parametrize(
    ("category", "text", "expected"),
    [
        ("examination", "Chest clear on auscultation", True),
        ("symptom", "Denies chest pain", True),
        ("symptom", "No fever", True),
        ("symptom", "Patient denied fever", True),
        ("symptom", "Cough for two weeks", False),
        ("symptom", "Not sleeping well", False),
        ("symptom", "No longer able to climb stairs", False),
        ("symptom", "No change in the pain", False),
        ("symptom", "Nil by mouth", False),
        ("history", "No previous surgery", False),
    ],
)
def test_is_core_kind(category: str, text: str, expected: bool) -> None:
    assert is_core_kind(category, text) is expected


@pytest.mark.usefixtures("v5")
def test_a_partial_the_rules_candidate_already_flags_is_not_flagged_twice() -> None:
    # The confirmed metformin candidate removes the partial metformin flag (one
    # discrepancy, one flag), so the partial look must not bring it back.
    verdicts = dict(VERDICTS)
    verdicts["Takes metformin 500 mg twice a day"] = ("partial", "Metformin 500 mg BD.")

    def check(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "verdicts": [
                {"id": fid, "verdict": verdicts[t][0], "note_quote": verdicts[t][1]}
                for fid, t in listed
            ]
        }

    def look(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "facts": [
                {
                    "id": fid,
                    "element": "twice a day",
                    "missing": True,
                    "transcript_quote": "I take metformin 500 mg twice a day",
                }
                for fid, text in listed
                if "metformin" in text
            ]
        }

    def candidate(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "findings": [
                {
                    "id": fid,
                    "confirmed": True,
                    "transcript_quote": "I take metformin 500 mg twice a day",
                    "severity": "critical",
                }
                for fid, text in listed
                if "metformin" in text
            ]
        }

    note = NOTE.replace("Metformin 500 mg BD.", "Metformin 500 mg.")
    mock = MockClaude(scripts(check=check, partial_look=look, candidate=candidate))
    outcome, _ = run(mock, note=note)
    metformin = [f for f in outcome.flags if "metformin" in (f.fact or "").lower()]
    assert [(f.rule_id, f.origin) for f in metformin] == [("frequency.missing", "rules")]
    step = next(
        s for s in outcome.trace.values() if s["fact"] == "Takes metformin 500 mg twice a day"
    )
    assert step["partial_look"] == "superseded"
