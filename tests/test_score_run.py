"""score_run.py on a synthetic subset: prepare, then score (no network, synthetic text)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "score_run.py"
SECRET_T = "The parrot recited Shakespeare in the waiting room"
SECRET_N = "Purple elephants were noted in the corridor"
CLEAN = f"History:\nCough. {SECRET_N}.\n"
CLEAN_KEY = f"syn|C1|clean|{hashlib.sha256(CLEAN.encode()).hexdigest()[:12]}"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("score_run", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_run"] = module  # dataclasses need their module registered
    spec.loader.exec_module(module)
    return module


def _pair(n: int, kind: str = "omit", **extra: Any) -> dict[str, Any]:
    pair = {
        "key": "syn|C1",
        "id": "C1",
        "stratum": "syn",
        "pair_id": f"syn|C1|{kind}|p{n}",
        "type": kind,
        "clean": CLEAN,
        "errored": f"History:\nNote {n}. {SECRET_N}.\n",
        "fact": "Takes metformin 500 mg twice daily",
        "target": {"fact": "x"},
        "residual_level": "complete" if n % 2 else "partial-weak",
        "severity": "critical",
    }
    pair.update(extra)
    return pair


@pytest.fixture
def subset(tmp_path: Path) -> dict[str, Path]:
    pairs = [_pair(1), _pair(2), _pair(3), _pair(4, "change")]
    (tmp_path / "pairs.json").write_text(json.dumps({"pairs": pairs}), encoding="utf-8")
    keys = [f"{p['pair_id']}|err" for p in pairs] + [CLEAN_KEY]
    (tmp_path / "spec.json").write_text(json.dumps({"note_keys": keys}), encoding="utf-8")
    transcripts = tmp_path / "transcripts" / "syn"
    transcripts.mkdir(parents=True)
    (transcripts / "C1.txt").write_text(f"Doctor: Hello.\nPatient: {SECRET_T}.\n", encoding="utf-8")
    return {
        "pairs": tmp_path / "pairs.json",
        "spec": tmp_path / "spec.json",
        "transcripts": tmp_path / "transcripts",
        "root": tmp_path,
    }


def _flag(
    fid: str,
    kind: str = "omission",
    confidence: str = "confirmed",
    fact: str = "metformin",
    severity: str = "critical",
    **extra: Any,
) -> dict[str, Any]:
    verdict = {"omission": "absent", "changed_fact": "changed"}.get(kind, "unsupported")
    return {
        "id": fid,
        "type": kind,
        "category": "medicine",
        "verdict": verdict,
        "severity": severity,
        "confidence": confidence,
        "origin": "model",
        "fact": fact,
        "transcript_evidence": None,
        "note_evidence": None,
        **extra,
    }


def _report(
    flags: list[dict[str, Any]], unchecked: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "consultation_id": "x",
        "mode": "full",
        "standard": "strict",
        "model": {"name": "m", "version": "m-1"},
        "decision": "warn",
        "counts": {},
        "flags": flags,
        "unchecked": unchecked or [],
        "usage": {
            "calls": 10,
            "shared_calls": 2,
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_write_tokens": 0,
            "cache_read_tokens": 50,
            "cost_usd": 0.01,
        },
    }


def _run(subset: dict[str, Path], reports: dict[str, dict[str, Any]]) -> Path:
    run = subset["root"] / "run"
    run.mkdir()
    cases = [
        {"id": key.replace("|", "__"), "transcript": "t", "note": "n", "report": r}
        for key, r in reports.items()
    ]
    (run / "batch.json").write_text(
        json.dumps(
            {"schema_version": "1.0", "cases": cases, "problems": [], "elapsed_seconds": 50.0}
        ),
        encoding="utf-8",
    )
    return run


def _score(
    subset: dict[str, Path], run: Path, capsys: pytest.CaptureFixture[str], *extra: str
) -> str:
    module = _load()
    assert (
        module.main(
            [
                "score",
                "--run",
                str(run),
                "--pairs",
                str(subset["pairs"]),
                "--spec",
                str(subset["spec"]),
                *extra,
            ]
        )
        == 0
    )
    return capsys.readouterr().out


def test_prepare_resolves_err_and_clean_keys(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out = subset["root"] / "cases"
    module = _load()
    assert (
        module.main(
            [
                "prepare",
                "--pairs",
                str(subset["pairs"]),
                "--transcripts",
                str(subset["transcripts"]),
                "--spec",
                str(subset["spec"]),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "prepared 5 notes" in printed and "omission 3" in printed and "clean 1" in printed
    clean_case = CLEAN_KEY.replace("|", "__")
    assert (out / f"{clean_case}.note.md").read_text(encoding="utf-8") == CLEAN
    assert (out / f"{clean_case}.transcript.txt").is_file()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cases"]["syn__C1__omit__p1__err"]["pair_id"] == "syn|C1|omit|p1"
    assert SECRET_T not in printed and SECRET_N not in printed


def test_prepare_rejects_an_unknown_key(subset: dict[str, Path]) -> None:
    subset["spec"].write_text(json.dumps({"note_keys": ["syn|C9|omit|zz|err"]}), encoding="utf-8")
    with pytest.raises(KeyError, match="no pair"):
        _load().main(
            [
                "prepare",
                "--pairs",
                str(subset["pairs"]),
                "--transcripts",
                str(subset["transcripts"]),
                "--spec",
                str(subset["spec"]),
                "--out",
                str(subset["root"] / "cases"),
            ]
        )


def _standard_reports() -> dict[str, dict[str, Any]]:
    return {
        "syn|C1|omit|p1|err": _report(
            [_flag("F1", fact="metformin 500 mg twice a day"), _flag("F2", confidence="possible")]
        ),
        "syn|C1|omit|p2|err": _report([_flag("F1", confidence="possible", fact="asthma")]),
        # p3 has only a peripheral omission: not flagged under the decision rule
        "syn|C1|omit|p3|err": _report([_flag("F1", severity="peripheral", fact="unrelated")]),
        "syn|C1|change|p4|err": _report([_flag("F1", kind="changed_fact")]),
        # the clean note has only non-Pass-A warnings: not a false alarm
        CLEAN_KEY: _report(
            [_flag("F1", kind="changed_fact"), _flag("F2", kind="unsupported", origin="rules")],
            unchecked=[{"category": "x", "item": "y", "reasons": ["quote_unverified"]}],
        ),
    }


def test_detection_and_false_alarms_use_pass_a_omissions_only(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out = _score(subset, _run(subset, _standard_reports()), capsys)
    assert "detection on omission notes: 2/3 (66.7%)" in out
    assert "false alarms on clean notes: 0/1 (0.0%)" in out  # its warnings are not Pass A
    assert "other warnings: changed_fact/medicine 2, unsupported/medicine 1" in out
    assert "clean notes with any other warning: 1/1" in out


def test_k_and_c_sweep(subset: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    out = _score(subset, _run(subset, _standard_reports()), capsys)
    rows = {
        line.split()[0] + line.split()[1]: line
        for line in out.splitlines()
        if line.strip().startswith(("possible ", "confirmed "))
    }
    assert "3/3 (100.0%)" in rows["possible1"]  # p1, p2, p3 each have one flag
    assert "1/3 (33.3%)" in rows["possible2"]
    assert "0/3 (0.0%)" in rows["possible3"]
    assert "2/3 (66.7%)" in rows["confirmed1"]  # p1 and p3 have a confirmed flag
    assert all("0/1 (0.0%)" in line for line in rows.values())


def test_correct_fact_matching_and_show_facts(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    run = _run(subset, _standard_reports())
    out = _score(subset, run, capsys)
    assert "decision rule: 1/2 (50.0%) of detected notes name the removed fact" in out
    assert "syn|C1|omit|p1  F1" in out and "  yes" in out
    assert "metformin" not in out  # fact text only with --show-facts
    assert "syn|C1|omit|p2  -  0.00  no" in out  # "asthma" is not comparable
    shown = _score(subset, run, capsys, "--show-facts")
    assert "removed fact: Takes metformin 500 mg twice daily" in shown
    assert SECRET_N not in shown and SECRET_T not in shown


def test_usage_time_reasons_and_missing(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    reports = _standard_reports()
    del reports["syn|C1|omit|p2|err"]
    out = _score(subset, _run(subset, reports), capsys)
    assert "Coverage: 4/5 notes have a report" in out
    assert "missing (counted as not flagged): syn|C1|omit|p2|err" in out
    assert "calls 10.0 own + 2.0 shared" in out
    assert "cost: $0.0400 in all, $0.0100 per note" in out
    assert "wall clock: 50 s in all, 12.5 s per note" in out
    assert "unchecked items by primary reason: quote_unverified 1" in out
    assert "model ids reported: m-1" in out


def test_agreement_with_gate1b_stores(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    gate = subset["root"] / "gate1b" / "gate1b-power" / "_state"
    gate.mkdir(parents=True)
    rows = [
        {"note_key": "syn|C1|omit|p1|err", "arm": "power-gepa-high", "aggregate": 3},
        {"note_key": "syn|C1|omit|p3|err", "arm": "power-gepa-high", "aggregate": 2},
        {"note_key": CLEAN_KEY, "arm": "power-gepa-high", "aggregate": 80},
    ]
    (gate / "power-gepa-high-r1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    out = _score(
        subset,
        _run(subset, _standard_reports()),
        capsys,
        "--gate1b",
        str(subset["root"] / "gate1b"),
    )
    assert "with Gate 1(b) power-gepa-high (decision rule): 2/3 (66.7%)" in out
    assert "differs: syn|C1|omit|p3|err (ours not, Gate 1(b) power-gepa-high flagged)" in out


@pytest.mark.parametrize(
    ("removed", "flagged"),
    [
        ("Metformin 500 mg twice daily", "Lisinopril 10 mg twice daily"),
        ("Atorvastatin 20 mg at night", "Amlodipine 5 mg once daily"),
        ("Family history of breast cancer in mother", "Family history of heart disease in father"),
        ("Blood pressure 150/90", "Blood pressure 120/80"),
        ("Denies chest pain", "Chest pain on exertion"),
    ],
)
def test_fact_matching_rejects_different_facts(removed: str, flagged: str) -> None:
    assert not _load().comparable(removed, flagged)


def test_fact_matching_accepts_the_same_fact_in_other_words() -> None:
    module = _load()
    assert module.comparable("Takes metformin 500 mg twice daily", "metformin 500 mg BD")
    assert module.comparable("No chest pain", "Denies chest pain")


def test_failed_notes_extra_reports_and_missing_agreement(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    reports = _standard_reports()
    reports["syn|C1|omit|p2|err"]["unchecked"] = [
        {"category": "pass_a", "item": "extraction", "reasons": ["cost_cap_reached"]}
    ]
    reports["syn|C1|omit|p9|err"] = _report([])
    del reports["syn|C1|omit|p3|err"]
    gate = subset["root"] / "gate1b" / "gate1b-power" / "_state"
    gate.mkdir(parents=True)
    rows = [
        {"note_key": f"syn|C1|omit|p{n}|err", "arm": "power-gepa-high", "aggregate": 3}
        for n in (1, 2, 3)
    ]
    (gate / "power-gepa-high-r1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    out = _score(subset, _run(subset, reports), capsys, "--gate1b", str(subset["root"] / "gate1b"))
    assert "Pass A failed (excluded from the rates): syn|C1|omit|p2|err (cost_cap_reached)" in out
    assert "report not in the spec (ignored): syn|C1|omit|p9|err" in out
    assert "detection on omission notes: 1/2 (50.0%)" in out  # p2 excluded; p3 missing = 0
    assert "(decision rule): 1/3 (33.3%); 2 without a usable report" in out


def test_manifest_mapping_and_round_trip_check(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load()
    out = subset["root"] / "cases"
    module.main(
        [
            "prepare",
            "--pairs",
            str(subset["pairs"]),
            "--transcripts",
            str(subset["transcripts"]),
            "--spec",
            str(subset["spec"]),
            "--out",
            str(out),
        ]
    )
    capsys.readouterr()
    reports = _standard_reports()
    run = _run(subset, reports)
    printed = _score(subset, run, capsys, "--manifest", str(out / "manifest.json"))
    assert "Coverage: 5/5 notes have a report" in printed
    subset["spec"].write_text(json.dumps({"note_keys": ["syn|C1__x|omit|err"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not map"):
        module.main(
            [
                "prepare",
                "--pairs",
                str(subset["pairs"]),
                "--transcripts",
                str(subset["transcripts"]),
                "--spec",
                str(subset["spec"]),
                "--out",
                str(subset["root"] / "cases2"),
            ]
        )
