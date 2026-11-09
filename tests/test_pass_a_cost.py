"""Cost and calls per note, shared-call accounting, batches and the cost cap (mocked HTTP)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _mock_model import MockClaude
from test_pass_a_pipeline import NOTE, TRANSCRIPT, scripts

import scribecheck.model
from scribecheck.checker import check_files, check_folder
from scribecheck.config import Config, PriceTable
from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.endpoint import Secret

PRICES = PriceTable(input=1.0, output=1.0, batch_multiplier=0.5)
PER_CALL = 120 / 1_000_000  # the mock's 100 input + 20 output tokens at $1 per million


@pytest.fixture
def mock(monkeypatch: pytest.MonkeyPatch) -> MockClaude:
    claude = MockClaude(scripts())

    def create(config: Config) -> AnthropicClient:
        return AnthropicClient(
            config.model or "claude-sonnet-5-5",
            api_key=Secret("sk-test"),
            transport=claude.transport,
            sleep=lambda _: None,
        )

    monkeypatch.setattr(scribecheck.model, "create_client", create)
    return claude


def _config(**changes: Any) -> Config:
    base: dict[str, Any] = {
        "mode": "full",
        "provider": "anthropic",
        "model": "claude-sonnet-5-5",
        "prices": PRICES,
    }
    base.update(changes)
    return Config(**base)


def _write(folder: Path, case: str, transcript: str, note: str) -> None:
    (folder / f"{case}.transcript.txt").write_text(transcript, encoding="utf-8")
    (folder / f"{case}.note.md").write_text(note, encoding="utf-8")


def test_usage_and_cost_per_note(tmp_path: Path, mock: MockClaude) -> None:
    _write(tmp_path, "a", TRANSCRIPT, NOTE)
    report = check_files(tmp_path / "a.transcript.txt", tmp_path / "a.note.md", _config())
    assert report.usage is not None and report.model is not None and report.pass_a is not None
    calls = len(mock.calls)
    assert report.usage.calls == calls and report.usage.shared_calls == 0
    assert report.usage.input_tokens == 100 * calls
    assert report.usage.cost_usd == pytest.approx(PER_CALL * calls)
    assert report.model.version == "claude-sonnet-5-5-mock"
    assert report.model.endpoint_host == "api.anthropic.com"
    assert (report.pass_a.pipeline, report.pass_a.prompts, report.pass_a.batch) == (
        "scribecheck",
        "scribecheck_v3",
        False,
    )
    data = json.loads(report.to_json())
    assert data["usage"]["calls"] == calls and data["mode"] == "full"


def test_without_prices_cost_is_unknown(tmp_path: Path, mock: MockClaude) -> None:
    _write(tmp_path, "a", TRANSCRIPT, NOTE)
    report = check_files(
        tmp_path / "a.transcript.txt", tmp_path / "a.note.md", _config(prices=None)
    )
    assert report.usage is not None and report.usage.cost_usd is None
    assert report.usage.calls == len(mock.calls)


def test_shared_extraction_is_split_between_notes(tmp_path: Path, mock: MockClaude) -> None:
    _write(tmp_path, "a", TRANSCRIPT, NOTE)
    _write(tmp_path, "b", TRANSCRIPT, NOTE.replace("No fever.", "Afebrile."))
    result = check_folder(tmp_path, _config(), live=True)
    assert not result.problems
    assert mock.stages().count("extract") == 1 and mock.stages().count("audit") == 1
    usages = [c.report.usage for c in result.cases]
    assert all(u is not None and u.shared_calls == 2 for u in usages)
    total = sum(u.cost_usd or 0 for u in usages if u is not None)
    assert total == pytest.approx(PER_CALL * len(mock.calls))


def test_folder_runs_use_batches_by_default(tmp_path: Path, mock: MockClaude) -> None:
    _write(tmp_path, "a", TRANSCRIPT, NOTE)
    result = check_folder(tmp_path, _config())
    assert "POST /v1/messages/batches" in mock.paths
    assert "POST /v1/messages" not in mock.paths
    report = result.cases[0].report
    assert report.pass_a is not None and report.pass_a.batch is True
    calls = len(mock.calls)
    assert report.usage is not None
    assert report.usage.cost_usd == pytest.approx(PER_CALL * calls * 0.5)  # batch price


def test_live_option_makes_single_calls(tmp_path: Path, mock: MockClaude) -> None:
    _write(tmp_path, "a", TRANSCRIPT, NOTE)
    check_folder(tmp_path, _config(), live=True)
    assert "POST /v1/messages/batches" not in mock.paths


def test_cost_cap_stops_the_pipeline_and_marks_the_rest(tmp_path: Path, mock: MockClaude) -> None:
    _write(tmp_path, "a", TRANSCRIPT, NOTE)
    config = _config(cost_cap_usd=PER_CALL * 2)  # extraction and audit, then stop
    report = check_files(tmp_path / "a.transcript.txt", tmp_path / "a.note.md", config)
    assert mock.stages() == ["extract", "audit"]
    capped = [u for u in report.unchecked if u.reasons == ("cost_cap_reached",)]
    assert capped and all(u.category != "pass_a" for u in capped)
    assert report.usage is not None and report.usage.calls == 2
