"""score_run.py on a synthetic subset: prepare, then score (no network, synthetic text)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
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
    verdict: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    verdict = verdict or {"omission": "absent", "changed_fact": "changed"}.get(kind, "unsupported")
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


def _sweep(out: str) -> dict[str, str]:
    rows = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) > 3 and parts[0] in ("possible", "confirmed") and parts[2].isdigit():
            rows[f"{parts[0]}/{parts[1]}/{parts[2]}"] = line
    return rows


def test_k_and_c_sweep(subset: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    rows = _sweep(_score(subset, _run(subset, _standard_reports()), capsys))
    assert len(rows) == 2 * 3 * 5
    assert "3/3 (100.0%)" in rows["possible/all/1"]  # p1, p2, p3 each have one flag
    assert "1/3 (33.3%)" in rows["possible/all/2"]
    assert "0/3 (0.0%)" in rows["possible/all/3"]
    assert "2/3 (66.7%)" in rows["confirmed/all/1"]  # p1 and p3 have a confirmed flag
    assert all("0/1 (0.0%)" in line for line in rows.values())


def test_sweep_severity_floors(subset: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    reports = _standard_reports()
    reports["syn|C1|omit|p2|err"] = _report(
        [_flag("F1", severity="supporting"), _flag("F2", severity="supporting")]
    )
    rows = _sweep(_score(subset, _run(subset, reports), capsys))
    # p1 critical (x2), p2 supporting (x2), p3 peripheral (x1)
    assert rows["possible/all/1"].split()[3] == "3/3"
    assert rows["possible/supporting+/1"].split()[3] == "2/3"
    assert rows["possible/critical/1"].split()[3] == "1/3"
    assert rows["possible/all/2"].split()[3] == "2/3"
    assert rows["possible/supporting+/2"].split()[3] == "2/3"
    assert rows["possible/critical/2"].split()[3] == "1/3"


def test_correct_detection_counts_only_notes_naming_the_removed_fact(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out = _score(subset, _run(subset, _standard_reports()), capsys)
    # p1 and p2 are detected under the decision rule; only p1's flag names metformin 500 mg
    assert "correct detection (a counted flag names the removed fact): 1/3 (33.3%)" in out
    assert _sweep(out)["possible/all/1"].split()[5] == "1/3"


def test_route_one_point(subset: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    reports = {
        # counts: critical, absent, confirmed
        "syn|C1|omit|p1|err": _report([_flag("F1", fact="metformin 500 mg twice a day")]),
        # each misses one condition: partial; supporting; possible
        "syn|C1|omit|p2|err": _report([_flag("F1", verdict="partial")]),
        "syn|C1|omit|p3|err": _report(
            [_flag("F1", severity="supporting"), _flag("F2", confidence="possible")]
        ),
        "syn|C1|change|p4|err": _report([]),
        CLEAN_KEY: _report([_flag("F1", verdict="partial"), _flag("F2", severity="supporting")]),
    }
    out = _score(subset, _run(subset, reports), capsys)
    section = out.split("== Route one's rule")[1].split("==")[0]
    assert "detection on omission notes: 1/3 (33.3%)" in section
    assert "correct detection: 1/3 (33.3%)" in section
    assert "false alarms on clean notes: 0/1 (0.0%)" in section
    assert "route one's rule: 1/1 (100.0%) of detected notes name the removed fact" in out


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


def test_partial_verdicts_are_information(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    reports = _standard_reports()
    # p2: a partial among the flags (a run from before the change); p3: in the info list
    reports["syn|C1|omit|p2|err"] = _report([_flag("F1", verdict="partial", confidence="possible")])
    reports["syn|C1|omit|p3|err"] = _report([])
    reports["syn|C1|omit|p3|err"]["info"] = [_flag("I1", verdict="partial", confidence="possible")]
    reports[CLEAN_KEY] = _report([_flag("F1", verdict="partial", confidence="possible")])
    run = _run(subset, reports)
    out = _score(subset, run, capsys)
    assert "partial verdicts: information; 3 listed as information" in out
    assert "detection on omission notes: 1/3 (33.3%)" in out  # p1 only
    assert "false alarms on clean notes: 0/1 (0.0%)" in out
    out = _score(subset, run, capsys, "--partial-as-flags")
    assert "counted as flags (--partial-as-flags); 0 listed as information" in out
    assert "detection on omission notes: 3/3 (100.0%)" in out
    assert "false alarms on clean notes: 1/1 (100.0%)" in out


def _misses(
    subset: dict[str, Path], run: Path, capsys: pytest.CaptureFixture[str], *extra: str
) -> str:
    args = ["misses", "--run", str(run), "--pairs", str(subset["pairs"])]
    assert _load().main([*args, "--spec", str(subset["spec"]), *extra]) == 0
    return capsys.readouterr().out


def _step(fid: str, fact: str, **extra: Any) -> dict[str, Any]:
    step = {"id": fid, "fact": fact, "category": "medicine", "origin": "model"}
    step.update({"evidence_verified": True, "audit": "kept", "core": True})
    step.update({"severity": "critical", "checked": True})
    return {**step, **extra}


def test_misses_classes_from_the_trace(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    reports = _standard_reports()
    traces = {
        # p1: judged present, with the note quote the report holds
        "syn|C1|omit|p1|err": [
            _step("f1", "Takes metformin 500 mg twice a day", verdict="present", note_quote="Q1"),
            _step("f2", "Cough", verdict="absent", second_look="confirmed"),
        ],
        # p2: only an unrelated fact: not extracted
        "syn|C1|omit|p2|err": [_step("f1", "Has asthma", verdict="present")],
        # p3: extracted but contextual, so never checked
        "syn|C1|omit|p3|err": [
            _step("f1", "Metformin 500 mg twice daily", core=False, checked=False)
        ],
    }
    for key, trace in traces.items():
        reports[key]["pass_a_trace"] = trace
    reports["syn|C1|omit|p1|err"]["flags"] = []
    out = _misses(subset, _run(subset, reports), capsys)
    assert "3 omission notes, 3 with a trace" in out
    lines = out.splitlines()
    p1 = next(i for i, line in enumerate(lines) if "syn|C1|omit|p1" in line)
    assert lines[p1].endswith("judged present")
    assert "extracted f1 (model) / audit kept (core, critical) / check present" in lines[p1 + 1]
    assert any("syn|C1|omit|p2" in line and line.endswith("not extracted") for line in lines)
    assert any(
        "syn|C1|omit|p3" in line and line.endswith("contextual, never checked") for line in lines
    )
    assert "   judged present: 1" in out and "   not extracted: 1" in out
    assert "complete (2): contextual, never checked 1, judged present 1" in out
    assert "partial-weak (1): not extracted 1" in out
    assert "metformin" not in out.lower() and "Q1" not in out  # facts only with --show-facts
    shown = _misses(subset, _run_again(subset, reports), capsys, "--show-facts")
    assert "removed fact: Takes metformin 500 mg twice daily" in shown
    assert 'note quote:   "Q1"' in shown


def _run_again(subset: dict[str, Path], reports: dict[str, dict[str, Any]]) -> Path:
    shutil.rmtree(subset["root"] / "run")
    return _run(subset, reports)


def test_misses_without_a_trace_use_the_flags(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    reports = _standard_reports()
    reports["syn|C1|omit|p2|err"] = _report(
        [_flag("F1", verdict="partial", confidence="possible", fact="metformin 500 mg twice")]
    )
    out = _misses(subset, _run(subset, reports), capsys)
    assert "3 omission notes, 0 with a trace" in out
    assert "detected: 1" in out and "judged partial: 1" in out and "no matching flag: 1" in out
    assert "flag F1: partial, possible, critical" in out


@pytest.mark.parametrize(
    ("step", "label"),
    [
        (None, "not extracted"),
        ({"evidence_verified": False}, "not extracted"),
        ({"audit": "dropped"}, "dropped by the audit"),
        ({"audit": "kept", "checked": False}, "contextual, never checked"),
        ({"audit": "kept", "checked": True, "verdict": "partial"}, "judged partial"),
        (
            {"audit": "kept", "checked": True, "verdict": "absent", "second_look": "rescued"},
            "rescued by the second look",
        ),
        (
            {"audit": "kept", "checked": True, "verdict": "absent", "second_look": "confirmed"},
            "detected",
        ),
        ({"audit": "kept", "checked": True}, "unchecked"),  # the check failed
        ({"audit": "failed"}, "unchecked"),
        (
            {"audit": "kept", "checked": True, "verdict": "present", "note_quote_verified": False},
            "unchecked",
        ),
    ],
)
def test_classify_miss(step: dict[str, Any] | None, label: str) -> None:
    assert _load().classify_miss(step) == label


def _b3_row(key: str, *facts: tuple[str, str, str]) -> dict[str, Any]:
    return {
        "note_key": key,
        "tier": "B3",
        "parse_failure": False,
        "detail": {
            "missing_facts": [
                {"id": f"f{i}", "fact": fact, "severity": sev, "verdict": verdict}
                for i, (fact, sev, verdict) in enumerate(facts, start=1)
            ]
        },
    }


def test_route_one_correct_detection_from_gate1b(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    gate = subset["root"] / "gate1b" / "gate1b-pipeline" / "_state"
    gate.mkdir(parents=True)
    rows = [
        # p1: a critical absent fact naming the removed fact: detected and correct
        _b3_row("syn|C1|omit|p1|err", ("Metformin 500 mg twice a day", "critical", "absent")),
        # p2: a critical absent fact about something else: detected, not correct
        _b3_row("syn|C1|omit|p2|err", ("Asthma since childhood", "critical", "absent")),
        # p3: the removed fact only as partial, or as supporting: not detected
        _b3_row(
            "syn|C1|omit|p3|err",
            ("Metformin 500 mg twice daily", "critical", "partial"),
            ("Metformin 500 mg twice daily", "supporting", "absent"),
        ),
        _b3_row(CLEAN_KEY, ("Cough for a week", "critical", "absent")),  # a false alarm
    ]
    (gate / "gate1b-B3.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    run = _run(subset, _standard_reports())
    out = _score(subset, run, capsys, "--gate1b", str(subset["root"] / "gate1b"))
    section = out.split("== Route one (Gate 1(b) B3)")[1].split("==")[0]
    assert "detection on omission notes: 2/3 (66.7%)" in section
    assert "correct detection: 1/3 (33.3%)" in section
    assert "false alarms on clean notes: 1/1 (100.0%)" in section
    assert "correct-fact rate: 1/2 (50.0%)" in section
    assert "syn|C1|omit|p1  0.87  yes" in section and "syn|C1|omit|p2  0.00  no" in section
    assert "metformin" not in section.lower()
    shown = _score(subset, run, capsys, "--gate1b", str(subset["root"] / "gate1b"), "--show-facts")
    assert "route one fact: Metformin 500 mg twice a day" in shown


def test_misses_weak_match(subset: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    reports = _standard_reports()
    # p1's removed fact is buried in a compound fact: comparable, but below the threshold
    compound = (
        "Medicines: metformin 500, ramipril, atorvastatin, aspirin, omeprazole, sertraline, "
        "bisoprolol"
    )
    reports["syn|C1|omit|p1|err"]["pass_a_trace"] = [_step("f1", compound, verdict="partial")]
    reports["syn|C1|omit|p1|err"]["flags"] = []
    for key in ("syn|C1|omit|p2|err", "syn|C1|omit|p3|err"):
        reports[key]["pass_a_trace"] = [_step("f1", "Has asthma", verdict="present")]
    out = _misses(subset, _run(subset, reports), capsys, "--show-facts")
    line = next(x for x in out.splitlines() if "syn|C1|omit|p1" in x)
    assert line.endswith("weak match (compound?)")
    assert f"closest fact: {compound}" in out
    assert "   weak match (compound?): 1" in out and "   not extracted: 2" in out


@pytest.mark.parametrize(
    ("removed", "candidate", "same"),
    [
        # a removed fact naming a drug matches only a candidate naming the same drug
        ("Occasionally takes Xanax for sleep", "Takes melatonin for sleep", False),
        ("Occasionally takes Xanax for sleep", "Takes alprazolam occasionally for sleep", True),
        ("Continue Lipitor 80 mg daily", "Continue atorvastatin 80 mg daily", True),
        ("Continue Lipitor 80 mg daily", "Continue rosuvastatin 80 mg daily", False),
        ("Vitamin D, melatonin and Aleve for cramps", "Takes Aleve sometimes for cramps", True),
        # a removed fact giving numbers needs one of them and no other
        ("Seek review if temperature above 38", "Seek review if temperature is high", False),
        ("Fever above 37.5-38 or spiking", "Seek review if temperature above 38", True),
        ("Seek review if temperature above 38", "Seek review if temperature above 39", False),
        # neither: unchanged
        ("Antibiotics not needed", "Antibiotics judged not needed", True),
    ],
)
def test_comparable_needs_the_same_drug_and_numbers(
    removed: str, candidate: str, same: bool
) -> None:
    assert _load().comparable(removed, candidate) is same
