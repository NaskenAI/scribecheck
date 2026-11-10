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


def test_spec_lists_every_errored_and_clean_note(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out = subset["root"] / "full_spec.json"
    assert _load().main(["spec", "--pairs", str(subset["pairs"]), "--out", str(out)]) == 0
    spec = json.loads(out.read_text(encoding="utf-8"))
    assert spec["note_keys"] == sorted(
        [f"syn|C1|omit|p{n}|err" for n in (1, 2, 3)] + ["syn|C1|change|p4|err", CLEAN_KEY]
    )
    assert spec["consultations"] == ["syn|C1"]
    printed = capsys.readouterr().out
    assert "5 notes over 1 consultations: 4 errored, 1 clean" in printed
    assert SECRET_N not in printed


def test_reference_methods_detection_and_false_alarms(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    gate = subset["root"] / "gate1b" / "gate1b-power" / "_state"
    gate.mkdir(parents=True)
    rows = [
        {"note_key": "syn|C1|omit|p1|err", "arm": "power-gepa-high", "aggregate": 3},
        {"note_key": "syn|C1|omit|p2|err", "arm": "power-gepa-high", "aggregate": 50},
        {"note_key": CLEAN_KEY, "arm": "power-gepa-high", "aggregate": 4},
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
    section = out.split("== Reference methods on the same notes")[1].split("==")[0]
    # p3 has no row: left out of its denominator and counted
    assert "Gate 1(b) power-gepa-high: detection 1/2 (50.0%), false alarms 1/1 (100.0%)" in section
    assert "(1 omission and 0 clean notes without a usable record)" in section
    graded = out.split("== Graded note score")[1].split("== Per-note")[0]
    # The clean note has no Pass A omission flag, so any threshold is within 1/1
    assert "at Gate 1(b) power-gepa-high's false alarms 1/1: grade >= (0, 0, 0, 1, 0, 0)" in graded
    # p2's only flag is possible, so its grade is zero and it is never flagged
    assert "detection 2/3 (66.7%), correct (words) 1/3 (33.3%), false alarms 0/1" in graded


# graded note score


def _note(module: ModuleType, kind: str, *flags: dict[str, Any], info: int = 0) -> Any:
    note = module.Note(f"k{id(flags)}", kind, _pair(1), {"mode": "full"})
    note.pass_a = [module.Flag.model_validate(f) for f in flags]
    note.info = [module.Flag.model_validate(_flag(f"I{i}", verdict="partial")) for i in range(info)]
    return note


def test_grade_orders_confirmed_critical_first() -> None:
    module = _load()
    crit = _note(module, "omission", _flag("F1"))
    supp_two = _note(module, "omission", _flag("F1", severity="supporting"), _flag("F2"))
    possible = _note(module, "omission", _flag("F1", confidence="possible"))
    partials = _note(module, "omission", info=3)
    assert module.grade(crit) == (0, 1, 0, 1, 0, 0)
    assert module.grade(supp_two) == (0, 1, 1, 2, 0, 0)
    assert module.grade(possible) == module.NO_GRADE  # only confirmed absent flags count
    assert module.grade(partials) == (0, 0, 0, 0, 3, 3)
    # A clearly absent flag ranks first, ahead of any number of others
    clear = _note(module, "omission", _flag("F1", severity="supporting", absence="clear"))
    assert module.grade(clear) == (1, 0, 1, 1, 0, 0) > module.grade(supp_two)
    assert module.grade(supp_two) > module.grade(crit) > module.grade(partials)


def test_matched_grade_takes_the_most_lenient_threshold_within_the_rate() -> None:
    module = _load()
    omissions = [
        _note(module, "omission", _flag("F1"), _flag("F2")),
        _note(module, "omission", _flag("F1")),
        _note(module, "omission", _flag("F1", severity="supporting")),
        _note(module, "omission", info=1),
    ]
    cleans = [
        _note(module, "clean", _flag("F1")),
        _note(module, "clean", info=2),
        _note(module, "clean"),
        _note(module, "clean"),
    ]
    # 1 false alarm in 4: down to the supporting omission's (0, 1, 1, 0, 0), which flags only
    # the critical clean note
    assert module.matched_grade(omissions, cleans, 1, 4) == (0, 0, 1, 1, 0, 0)
    # 0 in 65: only thresholds above the clean note's critical flag
    assert module.matched_grade(omissions, cleans, 0, 65) == (0, 2, 0, 2, 0, 0)
    # 2 in 4 reaches the one-partial omission's (0, 0, 0, 1, 1), past the partials-only clean
    assert module.matched_grade(omissions, cleans, 2, 4) == (0, 0, 0, 0, 1, 1)
    assert (
        module.matched_grade(
            omissions[:1], [_note(module, "clean", _flag("F1"), _flag("F2"))], 0, 1
        )
        is None
    )


# model-assisted matching


class FakeClient:
    """A model client that answers each judge request from `answer` and counts calls."""

    provider = "anthropic"
    model = "m"

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.calls = 0

    @property
    def endpoint(self) -> Any:
        from scribecheck.model.endpoint import Endpoint

        return Endpoint.from_url("https://api.anthropic.com")

    def send(self, request: Any, *, correction: str | None = None) -> Any:
        from scribecheck.model.base import RawReply, Usage

        self.calls += 1
        task = request.messages[0].blocks[0].text
        return RawReply(json.dumps(self.answer(task)), "end", Usage(100, 20), "m-1")


def _yes(fact_ids: list[str], flag_ids: list[str] | None = None) -> dict[str, Any]:
    flags = flag_ids or []
    return {
        "facts": {"same": bool(fact_ids), "ids": fact_ids},
        "flags": {"same": bool(flags), "ids": flags},
    }


def _items(module: ModuleType) -> list[Any]:
    judge = module.fact_judge
    report = {
        "pass_a_trace": [_step("f1", "Takes metformin 500 mg twice a day")],
        "flags": [_flag("F1", fact="metformin")],
    }
    return [
        judge.item("p1", ["Takes metformin 500 mg twice daily"], report),
        judge.item("p2", ["Has asthma"], report),
    ]


def test_judge_caches_and_judges_a_changed_fact_list_afresh(tmp_path: Path) -> None:
    from scribecheck.model.budget import Budget

    module = _load()
    judge = module.fact_judge
    client = FakeClient(
        lambda task: _yes(["f1"], ["F1"]) if "metformin" in task.split("EXTRACTED")[0] else _yes([])
    )
    path = tmp_path / "cache.json"
    items = _items(module)
    counts = judge.judge(items, client, Budget(), judge.load_cache(path), "m", path)
    assert (counts["judged"], client.calls) == (2, 2)
    cache = judge.load_cache(path)
    assert judge.lookup(cache, items[0]).flags.ids == ["F1"]
    assert judge.lookup(cache, items[1]).facts.same is False
    # Again: nothing to call.
    counts = judge.judge(items, client, Budget(), cache, "m", path)
    assert (counts["cached"], client.calls) == (2, 2)
    # A changed fact list is judged afresh; the sent texts never include a transcript or note.
    changed = judge.Item("p1", items[0].removed, (("f1", "Takes metformin 1 g"),), items[0].flags)
    assert judge.lookup(cache, changed) is None
    judge.judge([changed], client, Budget(), cache, "m", path)
    assert client.calls == 3


def test_judge_rejects_ids_it_was_not_sent(tmp_path: Path) -> None:
    from scribecheck.model.budget import Budget

    module = _load()
    judge = module.fact_judge
    path = tmp_path / "cache.json"
    items = _items(module)[:1]
    counts = judge.judge(items, FakeClient(lambda _: _yes(["f9"])), Budget(), {}, "m", path)
    assert counts["invalid ids"] == 1
    assert judge.lookup(judge.load_cache(path), items[0]) is None
    bad = judge.Judgement.model_validate(
        {"facts": {"same": True, "ids": []}, "flags": {"same": False, "ids": []}}
    )
    assert judge.problem(bad, items[0]) == "facts: 'same' disagrees with the ids"


def test_judge_stops_without_the_key(
    subset: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = subset["root"] / "judge.toml"
    config.write_text(
        'mode = "full"\n[model]\nprovider = "anthropic"\nname = "claude-sonnet-5-5"\n',
        encoding="utf-8",
    )
    run = _run(subset, _standard_reports())
    args = ["judge", "--run", str(run), "--pairs", str(subset["pairs"])]
    args += ["--spec", str(subset["spec"]), "--config", str(config)]
    assert _load().main(args) == 2
    assert not (run / "fact_judgements.json").exists()
    assert "key" in capsys.readouterr().err


def _cache(
    subset: dict[str, Path], reports: dict[str, dict[str, Any]], answers: dict[str, Any]
) -> Path:
    """A judgement cache for model "m": `answers` by pair id."""
    module = _load()
    pairs = {p["pair_id"]: p for p in json.loads(subset["pairs"].read_text())["pairs"]}
    entries = {}
    for pair_id, answer in answers.items():
        it = module.fact_judge.item(
            pair_id, module.removed_facts(pairs[pair_id]), reports[f"{pair_id}|err"]
        )
        entries[pair_id] = {"key": it.key("m"), "judgement": answer}
    path = subset["root"] / "fact_judgements.json"
    path.write_text(json.dumps({"model": "m", "entries": entries}), encoding="utf-8")
    return path


def test_score_with_judgements_lists_where_words_and_model_differ(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    reports = _standard_reports()
    # p1: words yes (metformin 500 mg), model no; p2: words no (asthma), model yes
    cache = _cache(
        subset,
        reports,
        {"syn|C1|omit|p1": _yes([]), "syn|C1|omit|p2": _yes([], ["F1"])},
    )
    out = _score(subset, _run(subset, reports), capsys, "--judgements", str(cache))
    section = out.split("== Model-assisted correct detection")[1].split("== Graded")[0]
    assert "omission notes without a valid judgement: 1" in section
    assert "no judgement: syn|C1|omit|p3" in section
    assert "decision rule: correct detection by words 1/3 (33.3%), by the model 1/3" in section
    assert "words yes, model no: syn|C1|omit|p1  F1" in section
    assert "words no, model yes: syn|C1|omit|p2  F1" in section
    assert "detected notes where they differ: 2/2" in section
    assert "metformin" not in section  # fact text only with --show-facts
    # A stale entry (the flag list changed) counts as missing, never as a match.
    reports["syn|C1|omit|p2|err"]["flags"] = [_flag("F1", fact="asthma, wheeze")]
    out = _score(subset, _run_again(subset, reports), capsys, "--judgements", str(cache))
    assert "omission notes without a valid judgement: 2" in out


def test_misses_with_judgements_classify_the_named_fact(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    reports = _standard_reports()
    reports["syn|C1|omit|p2|err"]["pass_a_trace"] = [
        _step("f1", "Has asthma", verdict="present"),
        _step("f2", "Diabetes tablet twice a day", verdict="absent", second_look="confirmed"),
    ]
    for key in ("syn|C1|omit|p1|err", "syn|C1|omit|p3|err"):
        reports[key]["pass_a_trace"] = [_step("f1", "Has asthma", verdict="present")]
    cache = _cache(subset, reports, {"syn|C1|omit|p2": _yes(["f2"])})
    out = _misses(subset, _run(subset, reports), capsys, "--judgements", str(cache))
    line = next(x for x in out.splitlines() if "syn|C1|omit|p2" in x)
    assert line.endswith("detected  (words: not extracted)")
    assert "notes classed by words for want of a valid judgement: 2" in out
    assert "words 'not extracted' -> model 'detected': 1" in out


def test_sample_is_fixed_and_shaped_for_review(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    reports = _standard_reports()
    answers = {f"syn|C1|omit|p{n}": _yes([]) for n in (1, 2, 3)}
    cache = _cache(subset, reports, answers)
    run = _run(subset, reports)
    rows = []
    for name in ("a.csv", "b.csv"):
        out = subset["root"] / name
        args = ["sample", "--run", str(run), "--pairs", str(subset["pairs"])]
        args += ["--spec", str(subset["spec"]), "--judgements", str(cache), "--out", str(out)]
        assert _load().main([*args, "--n", "2"]) == 0
        rows.append(out.read_text(encoding="utf-8"))
    assert rows[0] == rows[1]  # the seed fixes the sample
    lines = rows[0].splitlines()
    assert lines[0].split(",") == list(_load().SAMPLE_COLUMNS)
    assert len(lines) == 3
    assert "wrote" in capsys.readouterr().out


def test_severity_agreement_kappa_and_table(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load()
    assert module.kappa({("critical", "critical"): 5, ("supporting", "supporting"): 5}) == 1.0
    assert module.kappa({}) is None
    reports = _standard_reports()
    # p1 graded critical (agrees), p2 supporting (lower), p3 matched only by the model
    reports["syn|C1|omit|p1|err"]["pass_a_trace"] = [
        _step("f1", "Takes metformin 500 mg twice daily")
    ]
    reports["syn|C1|omit|p2|err"]["pass_a_trace"] = [
        _step("f1", "Takes metformin 500 mg twice daily", severity="supporting")
    ]
    reports["syn|C1|omit|p3|err"]["pass_a_trace"] = [
        _step("f1", "Diabetes tablet", severity="peripheral")
    ]
    cache = _cache(
        subset,
        reports,
        {f"syn|C1|omit|p{n}": _yes(["f1"]) for n in (1, 2, 3)},
    )
    md = subset["root"] / "severity.md"
    args = ["severity", "--run", str(_run(subset, reports)), "--pairs", str(subset["pairs"])]
    args += ["--spec", str(subset["spec"]), "--judgements", str(cache), "--out", str(md)]
    assert module.main(args) == 0
    text = md.read_text(encoding="utf-8")
    words, model = text.split("### Word matcher")[1].split("### Model-assisted")
    assert "graded by the audit: 2" in words and "we grade lower: 1" in words
    assert "| supporting | 1 | 0 | 0 | 1 |" in words
    assert "graded by the audit: 3" in model and "we grade lower: 2" in model
    assert "metformin" not in text.lower()  # counts only
    assert text == capsys.readouterr().out


# v5 mechanisms


def test_v5_mechanisms_with_and_only(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    reports = _standard_reports()
    # p1: one partial-element flag, clearly absent: its mechanism alone flags the note
    reports["syn|C1|omit|p1|err"]["flags"] = [
        _flag(
            "F1",
            fact="Takes metformin 500 mg twice daily: missing 500 mg",
            rule_id="pass_a.partial_element",
            absence="clear",
        )
    ]
    # p2: a never-dropped fact's flag beside an untagged one: 'with' only
    reports["syn|C1|omit|p2|err"]["flags"] = [_flag("F1", fact="asthma"), _flag("F2", fact="cat")]
    reports["syn|C1|omit|p2|err"]["pass_a_trace"] = [
        _step("f1", "asthma", forced=["never_drop"]),
        _step("f2", "cat"),
    ]
    # the clean note: a core-kind flag alone, a false alarm that mechanism costs
    reports[CLEAN_KEY]["flags"] = [_flag("F1", fact="No fever", absence="probable")]
    reports[CLEAN_KEY]["pass_a_trace"] = [_step("f1", "No fever", forced=["core_kind"])]
    out = _score(subset, _run(subset, reports), capsys)
    section = out.split("== v5 mechanisms")[1].split("\n==")[0]
    rows = {" ".join(line.split()[:2]): line for line in section.splitlines()[4:]}
    assert "detection 1/3 (33.3%), correct (words) 1/3 (33.3%)" in rows["partial_element with"]
    assert "false alarms 0/1" in rows["partial_element only"]
    assert "detection 1/3" in rows["never_drop with"] and "detection 0/3" in rows["never_drop only"]
    assert (
        "detection 0/3" in rows["core_kind with"] and "false alarms 1/1" in rows["core_kind only"]
    )
    assert "detection 1/3" in rows["clear only"] and "false alarms 1/1" in rows["probable only"]


def test_v5_mechanisms_absent_before_v5(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out = _score(subset, _run(subset, _standard_reports()), capsys)
    assert "no flag carries a v5 tag (a run before v5)" in out


# baselines' flagged facts


def test_baseline_flagged_facts_follow_route_ones_rule() -> None:
    module = _load()
    b3 = {
        "baseline_facts": [
            "critical | absent | Takes metformin | twice daily",
            "supporting | absent | Has asthma",
            "critical | present | No fever",
        ]
    }
    assert module.baseline_flagged_facts(b3) == ["Takes metformin | twice daily"]
    assert module.baseline_flagged_facts({"n_absent_critical": 1}) is None
    assert module.baseline_flagged_facts(None) is None
    # The word matcher's route-one facts read the same text when a row has no detail.
    assert module.route_one_facts(b3) == ["Takes metformin | twice daily"]
    assert module.route_one_facts({"n_absent_critical": 1}) is None


def test_route_two_is_never_judged_for_correct_detection() -> None:
    module = _load()
    assert set(module.BASELINES) == {"released B3 r1"}
    assert not set(module.SCORE_ONLY) & set(module.BASELINES)
    assert "not applicable: score-only method" in module.NOT_APPLICABLE


def test_baseline_correct_detection_from_the_judge(
    subset: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    p1, p2 = "syn|C1|omit|p1|err", "syn|C1|omit|p2|err"
    b3 = {
        p1: {"n_absent_critical": 1, "baseline_facts": ["critical | absent | Metformin dose"]},
        p2: {"n_absent_critical": 1},  # flagged, but no fact text kept
        CLEAN_KEY: {"n_absent_critical": 0},
    }
    power = {p1: {"aggregate": 3}}
    monkeypatch.setattr(
        module, "released_rows", lambda _: {"b3 r1": b3, "power r1": power, "power r3": {}}
    )
    notes = module._omission_notes(
        module.argparse.Namespace(
            run=_run(subset, _standard_reports()),
            pairs=subset["pairs"],
            spec=subset["spec"],
            manifest=None,
        )
    )
    items = module.baseline_items(subset["root"], notes)
    assert items["released B3 r1"][p2] is None
    assert [i for i, _ in items["released B3 r1"][p1].flags] == ["B1"]
    assert set(items) == {"released B3 r1"}  # route two is score-only: never judged
    path = subset["root"] / "fact_judgements.json"
    client = FakeClient(lambda task: _yes([], ["B1"]))
    from scribecheck.model.budget import Budget

    found = [it for by in items.values() for it in by.values() if it is not None]
    module.fact_judge.judge(found, client, Budget(), {}, "m", path)
    assert client.calls == 1
    run = subset["root"] / "run"
    args = ["score", "--run", str(run), "--pairs", str(subset["pairs"]), "--spec"]
    args += [str(subset["spec"]), "--released", str(subset["root"]), "--judgements", str(path)]
    assert module.main(args) == 0
    out = capsys.readouterr().out
    refs = out.split("== Reference methods")[1].split("\n==")[0]
    b3_line = refs.split("released B3 r1:")[1].split("released power")[0]
    assert "correct (model) 1/3 (33.3%) (1 flagged notes without fact text" in b3_line
    for name in ("power-gepa-high r1:", "power-gepa-high r3:"):
        line = refs.split(name)[1].split("released")[0]
        assert "correct detection: not applicable: score-only method" in line
    assert "released B3 r1 itself: correct (model) 1/3" in out
    assert "Metformin dose" not in out  # baseline fact text is never printed


def _no_split(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Released rows as the split wrote them before it kept baseline_facts."""
    b3 = {"syn|C1|omit|p1|err": {"n_absent_critical": 1}, CLEAN_KEY: {"n_absent_critical": 0}}
    monkeypatch.setattr(
        module, "released_rows", lambda _: {"b3 r1": b3, "power r1": {}, "power r3": {}}
    )


def test_judge_refuses_before_any_call_when_the_split_was_not_rerun(
    subset: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    _no_split(module, monkeypatch)

    def no_client(*_: Any) -> Any:
        raise AssertionError("no client may be made")

    monkeypatch.setattr(module, "create_client", no_client)
    monkeypatch.setattr(module, "load_config", no_client)
    run = _run(subset, _standard_reports())
    args = ["judge", "--run", str(run), "--pairs", str(subset["pairs"]), "--spec"]
    args += [str(subset["spec"]), "--config", "x.toml", "--released", str(subset["root"])]
    assert module.main(args) == 2
    assert "re-run the split" in capsys.readouterr().err


def test_score_says_when_the_split_was_not_rerun(
    subset: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    _no_split(module, monkeypatch)
    run = _run(subset, _standard_reports())
    args = ["score", "--run", str(run), "--pairs", str(subset["pairs"]), "--spec"]
    assert module.main([*args, str(subset["spec"]), "--released", str(subset["root"])]) == 0
    refs = capsys.readouterr().out.split("== Reference methods")[1].split("\n==")[0]
    b3_line = refs.split("released B3 r1:")[1].split("released power")[0]
    assert "no development row carries baseline_facts: re-run the split" in b3_line


def test_a_partial_element_flag_takes_its_facts_forced_tags() -> None:
    module = _load()
    note = _note(module, "omission")
    note.report = {
        "pass_a_trace": [
            _step("f1", "Chest clear", audit="dropped", checked=False),
            _step("f2", "Chest clear", forced=["core_kind"]),
        ]
    }
    flag = module.Flag.model_validate(
        _flag(
            "F1", fact="Chest clear: missing left base crackles", rule_id="pass_a.partial_element"
        )
    )
    assert module.mechanisms_of(note, flag) == {"partial_element", "core_kind"}


# The stability runs: a stratified fixed-seed spec, and agreement between runs


def _cells_pairs() -> list[dict[str, Any]]:
    """40 synthetic pairs over two strata, three kinds and two severities, with a clean
    note per consultation (ten consultations)."""
    pairs = []
    for i in range(40):
        stratum = "aa" if i % 10 < 7 else "bb"  # a consultation's notes share its stratum
        consultation = f"{stratum}|C{i % 10}"
        pairs.append(
            {
                "key": consultation,
                "id": f"C{i % 10}",
                "stratum": stratum,
                "pair_id": f"{consultation}|{('omit', 'add', 'change')[i % 3]}|p{i}",
                "type": ("omit", "add", "change")[i % 3],
                "severity": "critical" if i % 5 else "supporting",
                "clean": f"Clean note {i % 10}",
            }
        )
    return pairs


def test_allocate_is_proportional_and_exact() -> None:
    module = _load()
    assert module.allocate({"a": 265, "b": 65}, 100) == {"a": 80, "b": 20}
    shares = module.allocate({"x": 5, "y": 3, "z": 1}, 4)
    assert sum(shares.values()) == 4 and shares["x"] >= shares["y"] >= shares["z"]
    assert module.allocate({"x": 0}, 0) == {"x": 0}


def test_the_sample_is_fixed_by_its_seed_and_stratified() -> None:
    module = _load()
    cells = module.spec_cells(_cells_pairs())
    assert sum(len(k) for k in cells["errored"].values()) == 40
    assert sum(len(k) for k in cells["clean"].values()) == 10
    first = module.sample_keys(cells, 20, seed=7)
    assert first == module.sample_keys(cells, 20, seed=7)
    assert first != module.sample_keys(cells, 20, seed=8)
    assert len(first) == len(set(first)) == 20
    clean = [k for k in first if "|clean|" in k]
    assert len(clean) == 4  # 10 of 50 notes are clean: 4 of 20
    for cell, keys in cells["errored"].items():
        picked = sum(1 for k in first if k in keys)
        assert abs(picked - 16 * len(keys) / 40) < 1, cell  # within one of its share
    with pytest.raises(ValueError):
        module.sample_keys(cells, 51, seed=7)


def test_spec_writes_a_sample_with_its_seed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load()
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps({"pairs": _cells_pairs()}), encoding="utf-8")
    out = tmp_path / "spec.json"
    args = ["spec", "--pairs", str(pairs), "--out", str(out)]
    assert module.main([*args, "--sample", "20", "--seed", "7"]) == 0
    spec = json.loads(out.read_text(encoding="utf-8"))
    assert (spec["sample"], spec["seed"], len(spec["note_keys"])) == (20, 7, 20)
    touched = {k.split("|")[0] + "|" + k.split("|")[1] for k in spec["note_keys"]}
    assert set(spec["consultations"]) == touched
    assert "20 notes" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        module.main([*args, "--sample", "20"])  # a sample needs a seed
    with pytest.raises(SystemExit):
        module.main([*args, "--sample", "0", "--seed", "7"])
    # Without --sample, every note, as before.
    assert module.main(args) == 0
    assert len(json.loads(out.read_text(encoding="utf-8"))["note_keys"]) == 50


def test_fleiss_kappa() -> None:
    module = _load()
    assert module.fleiss_kappa([[True, True, True], [False, False, False]]) == 1.0
    assert module.fleiss_kappa([[True, True, True], [True, True, True]]) is None
    split = module.fleiss_kappa([[True, False, True], [False, True, False]])
    assert split is not None and split < 0


def _stability_run(subset: dict[str, Path], name: str, flagged_p2: bool, clean: bool) -> Path:
    reports = _standard_reports()
    if not flagged_p2:
        reports["syn|C1|omit|p2|err"] = _report([])
    if clean:
        reports[CLEAN_KEY] = _report([_flag("F1", fact="asthma")])
    run = subset["root"] / name
    run.mkdir()
    cases = [
        {"id": key.replace("|", "__"), "transcript": "t", "note": "n", "report": r}
        for key, r in reports.items()
    ]
    (run / "batch.json").write_text(json.dumps({"cases": cases, "problems": []}), "utf-8")
    return run


def test_stability_reports_spread_and_per_note_agreement(
    subset: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load()
    runs = [
        _stability_run(subset, "r1", flagged_p2=True, clean=False),
        _stability_run(subset, "r2", flagged_p2=False, clean=False),
        _stability_run(subset, "r3", flagged_p2=True, clean=True),
    ]
    args = ["stability", "--runs", *map(str, runs), "--pairs", str(subset["pairs"])]
    assert module.main([*args, "--spec", str(subset["spec"])]) == 0
    out = capsys.readouterr().out
    assert "r1: detection 2/3 (66.7%), correct 1/3 (33.3%), false alarms 0/1 (0.0%)" in out
    assert "r2: detection 1/3 (33.3%)" in out
    assert "detection: min 1/3 (33.3%), max 2/3 (66.7%), mean 1.7/3" in out
    assert "false alarms: min 0/1 (0.0%), max 1/1 (100.0%), mean 0.3/1" in out
    assert "omission notes: same in every run 2/3 (66.7%)" in out
    assert "clean notes: same in every run 0/1 (0.0%)" in out
    assert "omission notes by the number of runs whose flag names the removed fact: " in out
    assert "3 runs 1, 2 runs 0, 1 runs 0, 0 runs 2" in out
    assert "Notes whose result differs between runs: 2" in out
    assert "syn|C1|omit|p2|err: flagged in 2/3" in out
    assert f"{CLEAN_KEY}: flagged in 1/3" in out
    assert SECRET_N not in out and "metformin" not in out  # counts and ids only
    with pytest.raises(SystemExit):
        module.main(["stability", "--runs", str(runs[0]), "--pairs", "p", "--spec", "s"])
