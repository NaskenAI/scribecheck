"""Golden-file tests for the JSON and Markdown reports.

After an intended change to report output, regenerate the files and review the diff:

    SCRIBECHECK_UPDATE_GOLDEN=1 python -m pytest tests/test_report_golden.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scribecheck import check_files
from scribecheck.checker import DEMO_PREAMBLE, demo_cases, demo_reports
from scribecheck.report import Report, batch_json, render_batch_markdown, render_markdown

GOLDEN = Path(__file__).parent / "golden"
FIXTURES = Path(__file__).parent / "fixtures" / "report"
UPDATE = os.environ.get("SCRIBECHECK_UPDATE_GOLDEN") == "1"


def _compare(name: str, actual: str) -> None:
    path = GOLDEN / name
    if UPDATE:
        path.write_text(actual, encoding="utf-8", newline="\n")
    expected = path.read_text(encoding="utf-8")
    assert actual == expected, f"{name} differs from its golden file; see the module docstring"


def _reports() -> dict[str, Report]:
    reports = {r.consultation_id or "": r for r in demo_reports()}
    reports["guarded"] = check_files(
        FIXTURES / "guarded.transcript.txt", FIXTURES / "guarded.note.md"
    )
    return reports


@pytest.mark.parametrize("name", ["demo-001", "demo-002", "demo-003", "guarded"])
def test_json_report(name: str) -> None:
    _compare(f"{name}.json", _reports()[name].to_json())


@pytest.mark.parametrize("name", ["demo-001", "demo-002", "demo-003", "guarded"])
def test_markdown_report(name: str) -> None:
    _compare(f"{name}.md", render_markdown(_reports()[name]))


def test_demo_batch_markdown() -> None:
    _compare("demo-batch.md", render_batch_markdown(demo_cases(), [], preamble=DEMO_PREAMBLE))


def test_batch_json_shape() -> None:
    data = json.loads(batch_json(demo_cases(), ["x.note.md: no matching transcript"]))
    assert data["schema_version"] == "1.0"
    assert [c["id"] for c in data["cases"]] == ["demo-001", "demo-002", "demo-003"]
    assert data["cases"][0]["transcript"] == "demo-001.transcript.json"
    assert data["cases"][0]["report"]["consultation_id"] == "demo-001"
    assert data["problems"] == ["x.note.md: no matching transcript"]


def test_json_follows_the_spec_schema() -> None:
    data = json.loads(_reports()["demo-003"].to_json())
    assert list(data) == [
        "schema_version",
        "consultation_id",
        "mode",
        "standard",
        "model",
        "decision",
        "counts",
        "flags",
        "unchecked",
    ]
    assert data["model"] is None
    assert list(data["counts"]) == [
        "omission",
        "changed_fact",
        "unsupported",
        "not_verifiable",
        "unchecked",
    ]
    omission, change = data["flags"]
    assert omission["note_evidence"] is None  # written explicitly, as in the spec
    assert "transcript_value" not in omission  # other empty fields are left out
    assert list(change)[:8] == [
        "id",
        "type",
        "category",
        "verdict",
        "severity",
        "confidence",
        "origin",
        "rule_id",
    ]


def test_markdown_cells_escape_pipes_and_newlines() -> None:
    from scribecheck.report import _cell

    assert _cell("a | b\nc") == "a \\| b c"
    assert _cell("back\\slash") == "back\\\\slash"
