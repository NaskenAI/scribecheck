"""gate2.py on a synthetic held-out-shaped set (no network, synthetic text)."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "gate2.py"
CLEAN = "History:\nCough. Purple elephants were noted in the corridor.\n"
CLEAN_KEY = f"syn|C1|clean|{hashlib.sha256(CLEAN.encode()).hexdigest()[:12]}"
FACTS = {
    "p1": "Takes metformin 500 mg twice daily",
    "p2": "Has asthma since childhood",
    "p3": "Blood test for ferritin",
    "p4": "Added: allergic to penicillin",
    "p5": "Referred to physiotherapy",
    "p6": "Changed: dose of amlodipine",
}
KINDS = {"p1": "omit", "p2": "omit", "p3": "omit", "p4": "add", "p5": "omit", "p6": "change"}
OFFICIAL = ["p1", "p2", "p3", "p4"]
SMALL = {
    "held-out pairs": 6,
    "official pairs": 4,
    "official omission pairs": 3,
    "official add or change pairs": 1,
    "supplementary pairs": 2,
    "clean twins": 1,
}


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gate2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["gate2"] = module
    spec.loader.exec_module(module)
    return module


def pid(name: str) -> str:
    return f"syn|C1|{KINDS[name]}|{name}"


def _flag(
    fid: str, fact: str, severity: str = "critical", confidence: str = "confirmed"
) -> dict[str, Any]:
    return {
        "id": fid,
        "type": "omission",
        "category": "medicine",
        "verdict": "absent",
        "severity": severity,
        "confidence": confidence,
        "origin": "model",
        "fact": fact,
        "transcript_evidence": None,
        "note_evidence": None,
    }


def _report(
    flags: list[dict[str, Any]], unchecked: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "consultation_id": "x",
        "mode": "full",
        "standard": "strict",
        "flags": flags,
        "unchecked": unchecked or [],
        "pass_a_trace": [{"id": "f1", "fact": "Some extracted fact"}],
        "usage": {"cost_usd": 0.05},
    }


METFORMIN = [_flag("F1", "metformin 500 mg twice daily"), _flag("F2", "metformin dose")]
RUNS = {
    # p1: flagged in every run (two critical absences: also at both fixed points), naming the
    # fact; p2: flagged in runs 1 and 3, naming another fact; p3: run 2 only; the clean note:
    # run 2 only. Majority: p1 and p2 of 3; the clean note not flagged.
    "r1": {"p1": METFORMIN, "p2": [_flag("F1", "inhaler", "supporting", "possible")]},
    "r2": {"p1": METFORMIN, "p3": [_flag("F1", "ferritin test")], "clean": [_flag("F1", "x")]},
    "r3": {"p1": METFORMIN, "p2": [_flag("F1", "inhaler")]},
}


@pytest.fixture
def held(tmp_path: Path) -> dict[str, Path]:
    pairs = [
        {
            "key": "syn|C1",
            "id": "C1",
            "stratum": "syn",
            "pair_id": pid(name),
            "type": KINDS[name],
            "clean": CLEAN,
            "errored": f"History:\nNote {name}.\n",
            "fact": fact,
        }
        for name, fact in FACTS.items()
    ]
    paths = {"root": tmp_path, "pairs": tmp_path / "pairs.json", "ids": tmp_path / "ids.json"}
    paths["pairs"].write_text(json.dumps({"pairs": pairs}), encoding="utf-8")
    paths["ids"].write_text(
        json.dumps({"records": [{"pair_id": pid(n)} for n in OFFICIAL]}), encoding="utf-8"
    )
    return paths


def _specs(module: ModuleType, held: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "EXPECTED", SMALL)
    assert (
        module.main(
            [
                "specs",
                "--pairs",
                str(held["pairs"]),
                "--ids",
                str(held["ids"]),
                "--out",
                str(held["root"]),
            ]
        )
        == 0
    )


def _key(name: str) -> str:
    return CLEAN_KEY if name == "clean" else f"{pid(name)}|err"


def _write_run(
    root: Path, name: str, flags: dict[str, list[dict[str, Any]]], spec: str, **extra: Any
) -> Path:
    keys = json.loads((root / spec).read_text())["note_keys"]
    by_key = {_key(n): f for n, f in flags.items()}
    reports = {k: _report(by_key.get(k, [])) for k in keys}
    reports.update(extra)
    run = root / name
    run.mkdir()
    cases = [{"id": k.replace("|", "__"), "report": r} for k, r in reports.items() if r is not None]
    (run / "batch.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
    return run


def _manifest(root: Path, spec: str) -> Path:
    keys = json.loads((root / spec).read_text())["note_keys"]
    path = root / f"{spec}.manifest.json"
    path.write_text(
        json.dumps({"cases": {k.replace("|", "__"): {"note_key": k} for k in keys}}),
        encoding="utf-8",
    )
    return path


def _row(name: str, replicate: str, **extra: Any) -> dict[str, Any]:
    row = {
        "pair_id": None if name == "clean" else pid(name),
        "note_key": _key(name),
        "note_role": "clean" if name == "clean" else "errored",
        "model": "openai/gpt-5.4",
        "dataset_version": "factorial-v1-armconfirm",
        "replicate": replicate,
    }
    row.update(extra)
    return row


def _b3(name: str, replicate: str, *facts: str) -> dict[str, Any]:
    missing = [{"severity": "critical", "verdict": "absent", "fact": f} for f in facts]
    missing.append({"severity": "supporting", "verdict": "absent", "fact": "not counted"})
    return _row(name, replicate, tier="B3", arm="B3", detail={"missing_facts": missing})


def _write_raw(root: Path) -> Path:
    """Route one: p1 in replicates 1 and 2 (naming metformin), p2 in 2 and 3 (naming another
    fact): majority p1 and p2. Replicate 1 has no clean row (as the released B3 r1). A
    stale earlier shard of replicate 3 flags p3, and a later one replaces it. Route two
    flags p1 in every replicate: majority p1."""
    raw = root / "raw"
    judges = raw / "judgements" / "judges"

    def shard(family: str, folder: str, rows: list[dict[str, Any]]) -> None:
        path = judges / family / folder / "records.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"records": rows}), encoding="utf-8")

    shard(
        "w2-pipeline",
        "20260814-000000-a1-r1",
        [_b3("p1", "1", "metformin twice a day"), _b3("p2", "1"), _b3("p3", "1")],
    )
    shard(
        "w2-pipeline",
        "20260825-000000-b2-r2",
        [
            _b3("p1", "2", "metformin 500 mg"),
            _b3("p2", "2", "wheeze"),
            _b3("p3", "2"),
            _b3("clean", "2"),
        ],
    )
    shard(
        "w2-pipeline",
        "20260825-000000-b2-r3",
        [_b3("p1", "3"), _b3("p2", "3", "wheeze"), _b3("p3", "3", "ferritin"), _b3("clean", "3")],
    )
    shard("w2-pipeline", "20260826-000000-c3-r3", [_b3("p3", "3")])
    # A row of another model or dataset version is never used.
    shard("w2-pipeline", "20260814-000000-a1-r9", [_b3("p3", "1", "x") | {"model": "other"}])
    power = [
        _row(n, r, arm="power-gepa-high", cell="power-gepa-high", aggregate=5 if n == "p1" else 15)
        for n in ("p1", "p2", "p3", "clean")
        for r in ("1", "2", "3")
    ]
    shard("w2-power", "20260818-000000-d7-r1", power)
    return raw


class FakeClient:
    """A judge that says a flag states the removed fact when they share a drug word."""

    provider = "anthropic"
    model = "m"

    def __init__(self) -> None:
        self.calls = 0
        self.tasks: list[str] = []

    @property
    def endpoint(self) -> Any:
        from scribecheck.model.endpoint import Endpoint

        return Endpoint.from_url("https://api.anthropic.com")

    def send(self, request: Any, *, correction: str | None = None) -> Any:
        from scribecheck.model.base import RawReply, Usage

        self.calls += 1
        task = request.messages[0].blocks[0].text
        self.tasks.append(task)
        removed = task.split("REMOVED FACT:\n")[1].split("\n")[0].lower()
        flags = task.split("FLAGGED FACTS:\n")[1].split("\n\n")[0]
        ids = [
            m.group(1)
            for m in re.finditer(r"^(\S+): (.*)$", flags, re.MULTILINE)
            if any(w in m.group(2).lower() for w in ("metformin", "ferritin") if w in removed)
        ]
        answer = {"facts": {"same": False, "ids": []}, "flags": {"same": bool(ids), "ids": ids}}
        return RawReply(json.dumps(answer), "end", Usage(100, 20), "m-1")


@pytest.fixture
def scored(held: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from scribecheck.model.budget import Budget

    module = _load()
    _specs(module, held, monkeypatch)
    root = held["root"]
    runs = [_write_run(root, name, flags, "primary_spec.json") for name, flags in RUNS.items()]
    s1 = _write_run(
        root,
        "s1",
        {"p5": [_flag("F1", "physio")], "p6": [_flag("F1", "x")], "clean": [_flag("F1", "y")]},
        "supplementary_spec.json",
    )
    manifest = _manifest(root, "primary_spec.json")
    raw = _write_raw(root)
    # Judge through the module's own item builders, as `judge` does.
    pairs = module._pairs(held["pairs"])
    notes = [module._notes(r, pairs, root / "primary_spec.json", manifest) for r in runs]
    items = [
        module.our_item(i, n)
        for i, ns in enumerate(notes, start=1)
        for n in ns
        if n.kind == "omission" and module.has_flags(n)
    ]
    by_rep = module.route_one_items(module.load_released(raw), notes[0])
    items += [it for by in by_rep.values() for it in by.values() if it is not None]
    client = FakeClient()
    cache = root / "fact_judgements.json"
    module.fact_judge.judge(
        items, client, Budget(), module.fact_judge.load_cache(cache), "m", cache
    )
    classify = module.action_classify
    labels = {"p1": "medication_change", "p2": "not_action", "p3": "test"}
    entries = {
        pid(n): {"key": classify.key(FACTS[n], "m"), "label": lab} for n, lab in labels.items()
    }
    actions = root / "action_classes.json"
    actions.write_text(json.dumps({"model": "m", "entries": entries}), encoding="utf-8")
    return {
        "module": module,
        "runs": runs,
        "s1": s1,
        "manifest": manifest,
        "s_manifest": _manifest(root, "supplementary_spec.json"),
        "raw": raw,
        "cache": cache,
        "actions": actions,
        "client": client,
        "items": items,
        "held": held,
    }


# statistics


def test_wilson_matches_known_values() -> None:
    module = _load()
    low, high = module.wilson(5, 10)
    assert (round(low, 4), round(high, 4)) == (0.2366, 0.7634)
    low, high = module.wilson(0, 10)
    assert (low, round(high, 4)) == (0.0, 0.2775)
    assert module.wilson(0, 0) == (0.0, 1.0)


def test_mcnemar_exact() -> None:
    module = _load()
    assert module.mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert module.mcnemar_exact(6, 0) == module.mcnemar_exact(0, 6)
    assert module.mcnemar_exact(3, 3) == 1.0
    assert module.mcnemar_exact(0, 0) == 1.0
    assert module.mcnemar_exact(10, 2) == pytest.approx(2 * (1 + 12 + 66) / 4096)


def test_newcombe_paired_interval() -> None:
    module = _load()
    d, low, high = module.newcombe_paired(20, 10, 2, 99)
    assert d == pytest.approx(8 / 131)
    assert low < d < high and low > 0  # a clear gain: the interval excludes zero
    d, low, high = module.newcombe_paired(20, 3, 3, 105)
    assert d == 0 and low < 0 < high
    # Swapping the methods mirrors the interval.
    d, low, high = module.newcombe_paired(20, 10, 2, 99)
    assert module.newcombe_paired(20, 2, 10, 99) == pytest.approx((-d, -high, -low))
    _, low, high = module.newcombe_paired(0, 5, 0, 0)
    assert -1 <= low <= high <= 1


def test_strict_majority_of_three() -> None:
    module = _load()
    assert module.strict_majority([True, True, False])
    assert not module.strict_majority([True, False, False])
    assert not module.strict_majority([True])  # missing runs count as not flagged


# specs and run checks


def test_specs_split_official_and_supplementary(
    held: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load()
    _specs(module, held, monkeypatch)
    primary = json.loads((held["root"] / "primary_spec.json").read_text())["note_keys"]
    supplementary = json.loads((held["root"] / "supplementary_spec.json").read_text())["note_keys"]
    assert primary == sorted(f"{pid(n)}|err" for n in OFFICIAL) + [CLEAN_KEY]
    assert supplementary == sorted(f"{pid(n)}|err" for n in ("p5", "p6")) + [CLEAN_KEY]
    assert "syn|C1" not in capsys.readouterr().out  # counts only


def test_specs_stop_and_write_nothing_when_counts_differ(
    held: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load()  # the real EXPECTED counts: 230 held-out pairs
    out = held["root"] / "specs"
    assert (
        module.main(
            ["specs", "--pairs", str(held["pairs"]), "--ids", str(held["ids"]), "--out", str(out)]
        )
        == 2
    )
    assert not out.exists()
    assert "held-out pairs 6, expected 230" in capsys.readouterr().err


def test_build_specs_rejects_unknown_official_ids() -> None:
    module = _load()
    with pytest.raises(ValueError, match="1 official ids not in the pairs file"):
        module.build_specs([], {"nowhere"}, {})


def _check(
    scored: dict[str, Any], run: Path, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str]:
    held = scored["held"]
    code = scored["module"].main(
        [
            "check-run",
            "--run",
            str(run),
            "--pairs",
            str(held["pairs"]),
            "--spec",
            str(held["root"] / "primary_spec.json"),
            "--manifest",
            str(scored["manifest"]),
        ]
    )
    return code, capsys.readouterr().out


def test_check_run_keeps_a_complete_run(
    scored: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _check(scored, scored["runs"][0], capsys)
    assert code == 0
    assert (
        "5 notes, 0 without a report, 0 with Pass A failed, cost cap reached: no, $0.25: usable"
        in out
    )


def test_check_run_discards_a_capped_run(
    scored: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    capped = _report([], [{"category": "pass_b", "item": "x", "reasons": ["cost_cap_reached"]}])
    run = _write_run(
        scored["held"]["root"], "capped", {}, "primary_spec.json", **{_key("p3"): capped}
    )
    code, out = _check(scored, run, capsys)
    assert code == 3 and "cost cap reached: yes" in out and "DISCARD" in out


def test_check_run_discards_a_run_with_too_many_failures(
    scored: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    failed = _report([], [{"category": "pass_a", "item": "x", "reasons": ["model_error"]}])
    run = _write_run(
        scored["held"]["root"], "failing", {}, "primary_spec.json", **{_key("p3"): failed}
    )
    code, out = _check(scored, run, capsys)
    assert code == 3 and "1 with Pass A failed" in out  # 1 of 5 notes is over 2%
    missing = _write_run(
        scored["held"]["root"], "missing", {}, "primary_spec.json", **{_key("p3"): None}
    )
    code, out = _check(scored, missing, capsys)
    assert code == 3 and "1 without a report" in out


# released baselines and the judge


def test_released_rows_by_replicate_with_the_latest_shard(scored: dict[str, Any]) -> None:
    module = scored["module"]
    released = module.load_released(scored["raw"])
    b3 = released["route one (B3)"]
    assert CLEAN_KEY not in b3["1"]  # missing from replicate 1, as released
    assert module.route_one_flag(b3["3"][_key("p3")]) is False  # the later shard replaced it
    keys = [_key(n) for n in ("p1", "p2", "p3", "clean")]
    flags = module.baseline_flags(released, "route one (B3)", keys)
    assert flags[_key("p1")] == [True, True, False]
    assert flags[CLEAN_KEY] == [False, False, False]
    power = module.baseline_flags(released, "route two (power-gepa-high)", keys)
    assert power[_key("p1")] == [True, True, True] and not any(power[_key("p2")])


def test_judge_items_cover_flagged_notes_and_route_ones_facts(scored: dict[str, Any]) -> None:
    ids = sorted(it.pair_id for it in scored["items"])
    assert ids == sorted(
        [f"r{i}|{pid('p1')}" for i in (1, 2, 3)]
        + [f"r1|{pid('p2')}", f"r3|{pid('p2')}", f"r2|{pid('p3')}"]
        + [f"B3 r1|{pid('p1')}", f"B3 r2|{pid('p1')}", f"B3 r2|{pid('p2')}", f"B3 r3|{pid('p2')}"]
    )
    route_one = next(it for it in scored["items"] if it.pair_id == f"B3 r2|{pid('p1')}")
    assert route_one.flags == (("B1", "metformin 500 mg"),)  # critical absent facts only
    # Fact texts only: no transcript or note text is ever sent.
    assert not any("History:" in t or "Patient:" in t for t in scored["client"].tasks)


# score


def _score(scored: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> str:
    held, root = scored["held"], scored["held"]["root"]
    out = root / "evidence" / "gate2.txt"
    args = ["score", "--runs", *map(str, scored["runs"]), "--supplementary-run", str(scored["s1"])]
    args += ["--pairs", str(held["pairs"]), "--primary-spec", str(root / "primary_spec.json")]
    args += ["--supplementary-spec", str(root / "supplementary_spec.json")]
    args += [
        "--primary-manifest",
        str(scored["manifest"]),
        "--supplementary-manifest",
        str(scored["s_manifest"]),
    ]
    args += [
        "--raw",
        str(scored["raw"]),
        "--judgements",
        str(scored["cache"]),
        "--actions",
        str(scored["actions"]),
    ]
    args += ["--commit", "abc", "--prereg", "def", "--out", str(out)]
    assert scored["module"].main(args) == 0
    printed = capsys.readouterr().out
    assert out.read_text(encoding="utf-8") == printed
    return printed


def _section(out: str, title: str) -> str:
    return out.split(f"== {title}")[1].split("\n== ")[0]


def test_primary_endpoints_by_majority(
    scored: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    out = _score(scored, capsys)
    assert "role-played benchmark data (OmissionBench held-out subset)" in out
    primary = _section(out, "Primary endpoints")
    assert "detection: 2/3 (66.7%), Wilson 95%" in primary
    assert "false alarms: 0/1 (0.0%)" in primary
    assert "run 1: detection 2/3 (66.7%), false alarms 0/1" in primary
    assert "run 2: detection 2/3 (66.7%), false alarms 1/1" in primary
    correct = _section(out, "Secondary 1")
    assert "correct detection: 1/3 (33.3%)" in correct  # p1 only
    assert "flagged notes without a valid judgement (counted as not named): 0" in correct


def test_fixed_points_and_paired_comparison(
    scored: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    out = _score(scored, capsys)
    fixed = _section(out, "Secondary 2")
    assert "at 1/65, grade >= (0, 1, 0, 1, 2, 2):" in fixed
    assert "detection 1/3 (33.3%)" in fixed and "correct detection 1/3" in fixed
    published = _section(out, "Secondary 3")
    assert (
        "route one (B3): published 28/131 at 1/47; recomputed from the released rows 2/3"
        in published
    )
    assert "DOES NOT MATCH" in published  # a synthetic set is not the published one
    assert out.splitlines()[4].startswith("# WARNING: the released rows do not reproduce")
    assert "rows per replicate r1 3, r2 4, r3 4" in published
    paired = _section(out, "Secondary 4")
    assert (
        "decision rule vs route one (B3): both 2, ours only 0, theirs only 0, neither 1; "
        "exact McNemar p = 1.0000" in paired
    )
    assert (
        "decision rule vs route two (power-gepa-high): both 1, ours only 1, theirs only 0" in paired
    )
    assert "false alarms: ours 0/1, theirs 0/1; no higher than the baseline's: yes" in paired


def test_route_one_correct_detection_plan_action_and_supplementary(
    scored: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    out = _score(scored, capsys)
    route_one = _section(out, "Secondary 5")
    # p1 named in replicates 1 and 2; p2's "wheeze" never names asthma.
    assert "correct detection: 1/3 (33.3%)" in route_one
    assert "replicate 3: correct detection 0/3" in route_one
    actions = _section(out, "Secondary 6")
    assert "plan-action subset: 2/3 omission notes" in actions  # p1 and p3
    assert "detection 1/2 (50.0%)" in actions and "correct detection 1/2" in actions
    assert "medication_change: 1/1 (100.0%); 1/1" in actions
    assert "not_action: 1/1 (100.0%); 0/1" in actions
    supplementary = _section(out, "Secondary 7")
    assert "detection: 1/1 (100.0%)" in supplementary
    assert "false alarms: 1/1 (100.0%)" in supplementary
    assert "change notes with a counted Pass A flag: 1/1" in supplementary


def test_score_writes_counts_only(
    scored: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    out = _score(scored, capsys)
    assert "syn|C1" not in out and "C1" not in out  # no pair ids
    for text in [*FACTS.values(), "metformin", "inhaler", "wheeze", "Purple elephants", "History"]:
        assert text not in out


def test_a_failed_note_counts_as_not_flagged(scored: dict[str, Any]) -> None:
    module = scored["module"]
    note = module.Note(
        "k",
        "omission",
        {},
        _report([_flag("F1", "x")], [{"category": "pass_a", "reasons": ["model_error"]}]),
    )
    note.pass_a = [module.Flag.model_validate(_flag("F1", "x"))]
    assert module.decision_point(note) == []
    assert module.fixed_point((0, 0, 0, 0, 0, 0))(note) == []


def test_sample_draws_majority_flagged_notes_for_review(
    scored: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    held, root = scored["held"], scored["held"]["root"]
    out = root / "spot.csv"
    args = ["sample", "--runs", *map(str, scored["runs"]), "--pairs", str(held["pairs"])]
    args += ["--spec", str(root / "primary_spec.json"), "--manifest", str(scored["manifest"])]
    args += ["--judgements", str(scored["cache"]), "--out", str(out)]
    assert scored["module"].main(args) == 0
    rows = {r["pair_id"]: r for r in csv.DictReader(out.open(encoding="utf-8"))}
    assert set(rows) == {pid("p1"), pid("p2")}  # the majority-flagged omission notes
    assert rows[pid("p1")]["judge_runs_correct"] == "3"
    assert rows[pid("p1")]["judge_majority_correct"] == "yes"
    assert rows[pid("p2")]["runs_flagged"] == "2" and rows[pid("p2")]["r2_counted_flags"] == ""
    assert rows[pid("p1")]["r1_judge_flag_ids"] == "F1 F2"
    assert "2 of 2 majority-flagged" in capsys.readouterr().out


def test_the_supplementary_spec_never_includes_official_pairs(
    held: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    _specs(module, held, monkeypatch)
    primary = set(json.loads((held["root"] / "primary_spec.json").read_text())["note_keys"])
    supplementary = set(
        json.loads((held["root"] / "supplementary_spec.json").read_text())["note_keys"]
    )
    assert primary & supplementary == {CLEAN_KEY}


def test_the_discard_boundary_is_two_percent() -> None:
    module = _load()
    assert not module.RunCheck(100, 0, 2, False, None).discard  # 2% is kept
    assert module.RunCheck(100, 1, 2, False, None).discard  # over 2%
    assert not module.RunCheck(198, 0, 3, False, None).discard  # 3/198 = 1.5%
    assert module.RunCheck(198, 0, 4, False, None).discard  # 4/198 = 2.02%
    assert module.RunCheck(198, 0, 0, True, None).discard  # the cost cap alone
