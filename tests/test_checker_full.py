"""Full mode through the CLI and the reports (mocked HTTP)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _mock_model import MockClaude
from test_pass_a_pipeline import NOTE, TRANSCRIPT, scripts

import scribecheck.model
from scribecheck.cli import main
from scribecheck.config import Config
from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.endpoint import Secret
from scribecheck.report import render_markdown

CONFIG = """
mode = "full"
[model]
provider = "anthropic"
name = "claude-sonnet-5-5"
"""


@pytest.fixture
def folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MockClaude:
    claude = MockClaude(scripts())

    def create(config: Config) -> AnthropicClient:
        return AnthropicClient(
            "claude-sonnet-5-5",
            api_key=Secret("sk-test"),
            transport=claude.transport,
            sleep=lambda _: None,
        )

    monkeypatch.setattr(scribecheck.model, "create_client", create)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scribecheck.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "a.transcript.txt").write_text(TRANSCRIPT, encoding="utf-8")
    (tmp_path / "a.note.md").write_text(NOTE, encoding="utf-8")
    return claude


def test_check_full_mode_runs_live_and_warns(
    folder: MockClaude, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["check", "-t", "a.transcript.txt", "-n", "a.note.md", "--format", "json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["mode"] == "full" and data["decision"] == "warn"
    assert data["model"]["name"] == "claude-sonnet-5-5"
    assert data["pass_a"]["pipeline"] == "scribecheck"
    assert {f["confidence"] for f in data["flags"]} <= {"confirmed", "possible"}
    assert "POST /v1/messages/batches" not in folder.paths


def test_fail_on_critical_can_fail_in_full_mode(folder: MockClaude) -> None:
    code = main(["check", "-t", "a.transcript.txt", "-n", "a.note.md", "--fail-on", "critical"])
    assert code == 1  # the penicillin allergy omission is confirmed and critical


def test_batch_full_mode_uses_batches_unless_live(
    folder: MockClaude, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["batch", ".", "--format", "json"]) == 0
    assert "POST /v1/messages/batches" in folder.paths
    err = capsys.readouterr().err
    assert "model calls" in err and "cost unknown" in err
    folder.paths.clear()
    assert main(["batch", ".", "--live", "--format", "json"]) == 0
    assert "POST /v1/messages/batches" not in folder.paths


def test_full_mode_markdown_banner(folder: MockClaude) -> None:
    from scribecheck.checker import check_files

    report = check_files(
        "a.transcript.txt",
        "a.note.md",
        Config(mode="full", provider="anthropic", model="claude-sonnet-5-5"),
    )
    text = render_markdown(report)
    assert "Full mode: rules plus Pass A with the model (pipeline scribecheck" in text
    assert "not built yet" in text


def test_route_one_full_mode_reports_unverified_evidence_count(
    folder: MockClaude, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_pass_a_route_one import _scripts

    folder.scripts = _scripts([])
    from scribecheck.checker import check_files

    config = Config(
        mode="full", provider="anthropic", model="claude-sonnet-5-5", pipeline="route_one"
    )
    report = check_files("a.transcript.txt", "a.note.md", config)
    assert report.pass_a is not None and report.pass_a.prompts == "route_one"
    unverified = [f for f in report.flags if f.evidence_verified is False]
    assert report.pass_a.unverified_evidence == len(unverified) == 1
    assert json.loads(report.to_json())["flags"][-1].get("evidence_verified") in (False, None)
