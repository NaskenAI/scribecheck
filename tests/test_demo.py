"""`scribecheck demo`: offline, quick, and showing real output for every bundled case."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from importlib import resources

import pytest

from scribecheck.checker import demo_reports
from scribecheck.cli import main

# Blocks every socket connection and name lookup, then runs the demo and reports whether
# the HTTP client was ever imported.
_OFFLINE_DEMO = """
import socket, sys

def blocked(*args, **kwargs):
    raise RuntimeError("network blocked")

socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.getaddrinfo = blocked
socket.create_connection = blocked

from scribecheck.cli import main
code = main(["demo", "--format", "json"])
print("httpx loaded:", "httpx" in sys.modules, file=sys.stderr)
raise SystemExit(code)
"""


def _manifest() -> dict[str, object]:
    folder = resources.files("scribecheck") / "demo"
    data: dict[str, object] = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return data


def test_demo_runs_under_the_network_guard(capsys: pytest.CaptureFixture[str]) -> None:
    started = time.monotonic()
    assert main(["demo"]) == 0
    assert time.monotonic() - started < 60
    out = capsys.readouterr().out
    assert "synthetic consultations" in out
    assert "not yet been reviewed by a clinician" in out


def test_demo_in_a_fresh_process_with_sockets_blocked() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _OFFLINE_DEMO],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "httpx loaded: False" in result.stderr
    data = json.loads(result.stdout)
    assert len(data["cases"]) == 3


def test_demo_data_is_marked_synthetic_and_unreviewed() -> None:
    manifest = _manifest()
    assert manifest["synthetic"] is True
    assert manifest["reviewed"] is False
    folder = resources.files("scribecheck") / "demo"
    for case in manifest["cases"]:  # type: ignore[attr-defined]
        transcript = json.loads(
            (folder / f"{case['id']}.transcript.json").read_text(encoding="utf-8")
        )
        assert transcript["synthetic"] is True and transcript["reviewed"] is False
        note = (folder / f"{case['id']}.note.md").read_text(encoding="utf-8")
        assert note.startswith("SYNTHETIC DEMO: not a real patient\n")


def test_every_demo_case_shows_its_expected_findings() -> None:
    expected = {c["id"]: c["expected"] for c in _manifest()["cases"]}  # type: ignore[attr-defined]
    reports = demo_reports()
    assert [r.consultation_id for r in reports] == list(expected)
    for report in reports:
        assert report.flags, report.consultation_id
        assert {f.rule_id for f in report.flags} == set(expected[report.consultation_id])
        assert report.decision == "warn"
