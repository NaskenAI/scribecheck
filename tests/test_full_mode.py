"""Full mode end to end: every pass, deduplication, the decision rule, --passes, route one."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import pytest
from _mock_model import MockClaude

import scribecheck.model
from scribecheck.checker import check
from scribecheck.cli import main
from scribecheck.config import Config
from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.endpoint import Secret
from scribecheck.report import render_markdown

DEMO = resources.files("scribecheck") / "demo"
TRANSCRIPT = (DEMO / "demo-003.transcript.json").read_text(encoding="utf-8")
NOTE = (DEMO / "demo-003.note.md").read_text(encoding="utf-8")
FACTS = [
    (
        "Left knee swollen and painful for two weeks",
        "symptom",
        "My left knee has been swollen and painful for two weeks",
    ),
    ("Twisted the knee playing football", "history", "since I twisted it playing football"),
    (
        "Takes ibuprofen 400 mg three times a day",
        "medicine",
        "I take ibuprofen 400 mg three times a day",
    ),
    (
        "Plan: X-ray and physiotherapy referral",
        "plan",
        "I'd like an X-ray and a physiotherapy referral",
    ),
]
PRESENT = {
    "Twisted the knee playing football": "after a twisting injury playing football",
    "Plan: X-ray and physiotherapy referral": "Refer to physiotherapy",
    "Takes ibuprofen 400 mg three times a day": "Ibuprofen 400 mg twice daily",
    "Left knee swollen and painful for two weeks": "Knee swollen and painful for 2 weeks",
}


def demo_scripts() -> dict[str, Any]:
    def extract(_: Any, __: Any) -> Any:
        return {
            "facts": [
                {"fact": f, "category": c, "attributes": [], "evidence": e} for f, c, e in FACTS
            ]
        }

    def audit(listed: list[tuple[str, str]], _: Any) -> Any:
        return {
            "facts": [
                {"id": i, "keep": True, "core": True, "severity": "supporting"} for i, _ in listed
            ]
        }

    def check_(listed: list[tuple[str, str]], _: Any) -> Any:
        rows = []
        for i, text in listed:
            quote = next(
                (q for k, q in PRESENT.items() if k in text or text in k),
                "Knee swollen and painful for 2 weeks",
            )
            rows.append({"id": i, "verdict": "present", "note_quote": quote})
        return {"verdicts": rows}

    def candidate(listed: list[tuple[str, str]], _: Any) -> Any:
        return {
            "findings": [
                {
                    "id": i,
                    "confirmed": True,
                    "severity": "supporting",
                    "transcript_quote": "My left knee",
                }
                for i, _ in listed
            ]
        }

    def changes(listed: list[tuple[str, str]], _: Any) -> Any:
        return {
            "findings": [
                {
                    "id": i,
                    "verdict": "changed",
                    "kind": "frequency",
                    "item": "ibuprofen",
                    "transcript_value": "three times a day",
                    "note_value": "twice daily",
                    "transcript_quote": "I take ibuprofen 400 mg three times a day",
                    "note_quote": "Ibuprofen 400 mg twice daily",
                    "severity": "critical",
                }
                for i, _ in listed
            ]
        }

    def split(_: Any, __: Any) -> Any:
        return {
            "statements": [
                {"text": "Ibuprofen 400 mg twice daily", "kind": "clinical"},
                {"text": "X-ray", "kind": "clinical"},
            ]
        }

    def statements(listed: list[tuple[str, str]], _: Any) -> Any:
        return {
            "statements": [
                {"id": i, "verdict": "unsupported", "severity": "critical"}
                if "Ibuprofen" in t
                else {"id": i, "verdict": "supported", "transcript_quote": "I'd like an X-ray"}
                for i, t in listed
            ]
        }

    return {
        "extract": extract,
        "audit": audit,
        "check": check_,
        "second_look": lambda listed, _: {"facts": []},
        "candidate": candidate,
        "quote_recheck": lambda listed, _: {"quotes": []},
        "changes": changes,
        "changes_second": lambda listed, _: {
            "claims": [{"id": i, "upheld": True} for i, _ in listed]
        },
        "split": split,
        "statements": statements,
        "statements_second": lambda listed, _: {
            "statements": [{"id": i, "upheld": True} for i, _ in listed]
        },
    }


@pytest.fixture
def mock(monkeypatch: pytest.MonkeyPatch) -> MockClaude:
    claude = MockClaude(demo_scripts())

    def create(config: Config) -> AnthropicClient:
        return AnthropicClient(
            "claude-sonnet-5-5",
            api_key=Secret("k"),
            transport=claude.transport,
            sleep=lambda _: None,
        )

    monkeypatch.setattr(scribecheck.model, "create_client", create)
    return claude


FULL = Config(mode="full", provider="anthropic", model="claude-sonnet-5-5")


def test_demo_case_end_to_end(mock: MockClaude) -> None:
    report = check(TRANSCRIPT, NOTE, FULL)
    stages = set(mock.stages())
    assert {"extract", "audit", "check", "candidate", "changes", "split", "statements"} <= stages
    flags = {(f.type, f.category, f.confidence) for f in report.flags}
    assert ("omission", "side", "confirmed") in flags  # the rules candidate, confirmed
    assert ("changed_fact", "frequency", "confirmed") in flags  # Pass B
    # Pass C's "Ibuprofen 400 mg twice daily" is the same discrepancy as Pass B's: dropped.
    assert not [f for f in report.flags if f.type == "unsupported"]
    # Pass A's ibuprofen fact gives way to the changed fact too.
    assert len([f for f in report.flags if "ibuprofen" in (f.fact or "").lower()]) == 1
    assert report.passes == {"a": "scribecheck", "b": "model", "c": "model"}
    assert report.decision == "warn"


def test_fail_on_critical_applies_to_the_combined_flags(
    mock: MockClaude, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "t.json").write_text(TRANSCRIPT, encoding="utf-8")
    (tmp_path / "n.md").write_text(NOTE, encoding="utf-8")
    (tmp_path / "scribecheck.toml").write_text(
        'mode = "full"\n[model]\nprovider = "anthropic"\nname = "claude-sonnet-5-5"\n',
        encoding="utf-8",
    )
    assert main(["check", "-t", "t.json", "-n", "n.md"]) == 0
    assert main(["check", "-t", "t.json", "-n", "n.md", "--fail-on", "critical"]) == 1


def test_passes_a_only_runs_and_reports_pass_a(mock: MockClaude) -> None:
    report = check(TRANSCRIPT, NOTE, FULL.model_copy(update={"passes": ("a",)}))
    assert not {"changes", "split", "statements"} & set(mock.stages())
    assert {f.type for f in report.flags} == {"omission"}
    assert report.passes == {"a": "scribecheck", "b": "off", "c": "off"}


def test_route_one_is_unaffected(mock: MockClaude) -> None:
    from test_pass_a_route_one import _scripts

    mock.scripts = _scripts([True])
    report = check(TRANSCRIPT, NOTE, FULL.model_copy(update={"pipeline": "route_one"}))
    assert set(mock.stages()) <= {
        "r1_extract",
        "r1_audit",
        "r1_check",
        "r1_refute",
        "quote_recheck",
    }
    assert report.passes == {"a": "route_one", "b": "rules", "c": "rules"}


def test_not_verifiable_statements_are_listed_and_counted(mock: MockClaude) -> None:
    mock.scripts["split"] = lambda _, __: {"statements": [{"text": "X-ray", "kind": "result"}]}
    report = check(TRANSCRIPT, NOTE, FULL)
    assert report.counts.not_verifiable == 1
    data = json.loads(report.to_json())
    assert data["not_verifiable"][0]["kind"] == "result"


def test_cli_passes_option(
    mock: MockClaude,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "t.json").write_text(TRANSCRIPT, encoding="utf-8")
    (tmp_path / "n.md").write_text(NOTE, encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["check", "-t", "t.json", "-n", "n.md", "--passes", "a,x"])
    assert "choose one or more of a, b and c" in capsys.readouterr().err
    assert main(["check", "-t", "t.json", "-n", "n.md", "--passes", "a", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["mode"] == "rules" and {f["type"] for f in data["flags"]} <= {"omission"}


def _partial_plan(mock: MockClaude) -> None:
    """The plan is judged partial and the rules candidate rejected: only a partial remains."""
    check_ = mock.scripts["check"]

    def partial_check(listed: list[tuple[str, str]], body: Any) -> Any:
        rows = check_(listed, body)["verdicts"]
        for row, (_, text) in zip(rows, listed, strict=True):
            if text.startswith("Plan:"):
                row["verdict"] = "partial"
        return {"verdicts": rows}

    mock.scripts["check"] = partial_check
    mock.scripts["candidate"] = lambda listed, _: {
        "findings": [{"id": i, "confirmed": False} for i, _ in listed]
    }


def test_partial_verdicts_are_information_by_default(mock: MockClaude) -> None:
    _partial_plan(mock)
    report = check(TRANSCRIPT, NOTE, FULL.model_copy(update={"passes": ("a",)}))
    assert report.flags == () and report.counts.omission == 0
    assert report.decision == "pass"
    assert [(i.id, i.verdict, i.fact) for i in report.info] == [
        ("I1", "partial", "Plan: X-ray and physiotherapy referral")
    ]
    assert report.info[0].note_evidence is not None
    assert report.info[0].note_evidence.text == "Refer to physiotherapy"
    data = json.loads(report.to_json())
    assert data["info"][0]["id"] == "I1" and data["flags"] == []
    markdown = render_markdown(report)
    assert "## Information" in markdown and '"Refer to physiotherapy"' in markdown


def test_partial_as_flags_restores_the_warning(mock: MockClaude) -> None:
    _partial_plan(mock)
    config = FULL.model_copy(update={"passes": ("a",), "partial_as_flags": True})
    report = check(TRANSCRIPT, NOTE, config)
    assert [(f.id, f.verdict) for f in report.flags] == [("F1", "partial")]
    assert report.info == () and report.counts.omission == 1
    assert report.decision == "warn"
    assert "info" not in json.loads(report.to_json())
    assert "## Information" not in render_markdown(report)


def test_pass_a_trace_follows_each_fact(mock: MockClaude) -> None:
    _partial_plan(mock)
    assert check(TRANSCRIPT, NOTE, FULL.model_copy(update={"passes": ("a",)})).pass_a_trace is None
    config = FULL.model_copy(update={"passes": ("a",), "pass_a_trace": True})
    report = check(TRANSCRIPT, NOTE, config)
    assert report.pass_a_trace is not None
    steps = {t.fact: t for t in report.pass_a_trace}
    plan = steps["Plan: X-ray and physiotherapy referral"]
    assert (plan.audit, plan.core, plan.severity, plan.checked) == (
        "kept",
        True,
        "supporting",
        True,
    )
    assert (plan.verdict, plan.note_quote, plan.second_look) == (
        "partial",
        "Refer to physiotherapy",
        None,
    )
    knee = steps["Left knee swollen and painful for two weeks"]
    assert knee.verdict == "present" and knee.note_quote == "Knee swollen and painful for 2 weeks"
    assert all(t.evidence_verified and t.origin == "model" for t in report.pass_a_trace)
    assert json.loads(report.to_json())["pass_a_trace"][0]["id"] == "f1"
