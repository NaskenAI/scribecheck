"""scripts/compare_gate1b.py on synthetic released and new records.

All ids and scores are invented. Text fields carry SENTINEL so the tests can prove no text
reaches the output.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SENTINEL = "SENTINEL-TEXT-DO-NOT-LEAK"


def _load() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))  # the script imports recompute_published beside it
    spec = importlib.util.spec_from_file_location("compare_gate1b", SCRIPTS / "compare_gate1b.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up while it loads
    spec.loader.exec_module(module)
    return module


COMPARE = _load()
CONSULTATIONS = ("s|c1", "s|c2")
NOTES = ("c1-omit", "c1-add", "c1-clean", "c2-omit", "c2-add", "c2-clean")


def _row(note: str, **fields: Any) -> dict[str, Any]:
    consultation, kind = note.split("-")
    clean_key = f"s|{consultation}|clean|abc"
    errored = {
        "pair_id": f"s|{consultation}|{kind}|f1",
        "note_key": f"s|{consultation}|{kind}|f1|err",
        "note_role": "errored",
        "pair_type": kind,
    }
    clean = {"pair_id": None, "note_key": clean_key, "note_role": "clean"}
    return {
        **(clean if kind == "clean" else errored),
        "clean_key": clean_key,
        "stratum": "s",
        "consultation": consultation,
        "note_text": SENTINEL,
        **fields,
    }


def key(note: str) -> str:
    return str(_row(note)["note_key"])


POWER = {"model": "openai/gpt-5.4", "arm": "power-gepa-high", "dataset_version": "factorial-v1"}
B3 = {
    "model": "openai/gpt-5.4",
    "arm": "B3",
    "cell": "B3",
    "tier": "B3",
    "dataset_version": "factorial-v1",
}

# Route two: aggregates per note (None = parse failure).
R1 = (3, 9, 10, 10, 5, 10)  # flags (below 10): T T F F T F
R3 = (4, 10, 10, 9, 5, 10)  # T F F T T F  -> r1 vs r3 agree on 4 of 6
NEW = (2, 9, 10, 10, 6, None)  # T T F F T -  -> vs r1 5/6, vs r3 3/6

# Route one: released n_absent_critical (None = no count, nothing graded absent) ...
B3_RELEASED = (1, 0, None, 2, None, 1)  # flags T F F T F T
CRITICAL_ABSENT = {"severity": "critical", "verdict": "absent", "fact": SENTINEL}
CRITICAL_PARTIAL = {"severity": "critical", "verdict": "partial", "fact": SENTINEL}
SUPPORTING_ABSENT = {"severity": "supporting", "verdict": "absent", "fact": SENTINEL}
# ... and the new run's detail.missing_facts.
B3_NEW = (
    [CRITICAL_ABSENT],  # T, 1
    [SUPPORTING_ABSENT],  # F, 0
    [],  # F, 0
    [CRITICAL_PARTIAL],  # F, 0  (released T, 2)
    [CRITICAL_ABSENT, CRITICAL_ABSENT],  # T, 2  (released F, 0)
    [CRITICAL_ABSENT],  # T, 1
)


def _power_rows(scores: tuple[float | None, ...], replicate: int, **overrides: Any) -> list[Any]:
    run = {**POWER, **overrides}
    return [
        _row(note, **run, replicate=replicate, aggregate=score)
        if score is not None
        else _row(note, **run, replicate=replicate, aggregate=None, parse_failure=True)
        for note, score in zip(NOTES, scores, strict=True)
    ]


def _b3_released() -> list[Any]:
    rows = []
    for note, count in zip(NOTES, B3_RELEASED, strict=True):
        extra = {} if count is None else {"n_absent_critical": count}
        rows.append(_row(note, **B3, replicate=1, aggregate=0.5, **extra))
    return rows


def _write(path: Path, rows: list[Any], *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    else:
        path.write_text(json.dumps({"records": rows}), encoding="utf-8")


def build(tmp_path: Path, *, drop_b3: str | None = None) -> dict[str, Path]:
    released = tmp_path / "judge_scores"
    stem = "judgements__judges__"
    _write(
        released / f"{stem}w2-power__20260818-100000-aaaaaaa-r1__w2_power.json", _power_rows(R1, 1)
    )
    _write(
        released / f"{stem}w2-power__20260818-110000-aaaaaaa-r3__w2_power.json", _power_rows(R3, 3)
    )
    _write(released / f"{stem}w2-pipeline__20260819-100000-bbbbbbb-r1__pipe.json", _b3_released())
    # Decoys that must be ignored: another model, the held-out version, another consultation.
    decoys = _power_rows((1, 1, 1, 1, 1, 1), 1, model="google/gemini-3.1-pro-preview")
    decoys += _power_rows((1, 1, 1, 1, 1, 1), 1, dataset_version="factorial-v1-armconfirm")
    decoys.append(_row("c9-omit", **POWER, replicate=1, aggregate=1))
    _write(released / f"{stem}w2-power__20260818-120000-ccccccc-r1__w2_power.json", decoys)
    new_power = tmp_path / "new" / "power-gepa-high-r1.jsonl"
    _write(new_power, _power_rows(NEW, 1), jsonl=True)
    new_b3 = tmp_path / "new" / "gate1b-B3.jsonl"
    b3_rows = [
        _row(note, **B3, replicate=1, aggregate=0.5, detail={"missing_facts": facts})
        for note, facts in zip(NOTES, B3_NEW, strict=True)
        if note != drop_b3
    ]
    _write(new_b3, b3_rows, jsonl=True)
    spec = tmp_path / "subset_spec.json"
    return {"released": released, "power": new_power, "b3": new_b3, "spec": spec}


def _spec(paths: dict[str, Path]) -> None:
    argv = ["spec", "--released", str(paths["released"]), "--out", str(paths["spec"])]
    assert COMPARE.main([*argv, "--consultations", *CONSULTATIONS]) == 0


def _compare(paths: dict[str, Path], *extra: str) -> int:
    argv = ["compare", "--released", str(paths["released"]), "--spec", str(paths["spec"])]
    return int(COMPARE.main([*argv, *extra]))


def test_spec_lists_the_notes_every_reference_set_has(tmp_path: Path) -> None:
    paths = build(tmp_path)
    _spec(paths)
    spec = json.loads(paths["spec"].read_text())
    assert spec["note_keys"] == sorted(key(n) for n in NOTES)
    assert spec["consultations"] == list(CONSULTATIONS)
    assert set(spec) == {"note_keys", "consultations"}  # the harness reads note_keys


def test_route_two_agreement_and_pass_mark(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = build(tmp_path)
    _spec(paths)
    capsys.readouterr()
    assert _compare(paths, "--new-power", str(paths["power"])) == 1
    out = capsys.readouterr().out
    assert SENTINEL not in out
    # The new run's parse failure (c2-clean) counts as a disagreement.
    assert f"new vs released r1: agree 5/6 (83.3%) | disagree: {key('c2-clean')}" in out
    assert "new vs released r3: agree 3/6 (50.0%)" in out
    assert "released r1 vs r3: agree 4/6 (66.7%)" in out
    assert "aggregate, new vs released r1: equal 3/5, mean |difference| 0.400, max 1" in out
    assert "pass mark: 66.7% (90%, or the released r1-vs-r3 agreement if lower)" in out
    assert "route two: FAIL" in out


def test_route_one_counts_margins_and_agreement_for_information(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = build(tmp_path)
    _spec(paths)
    capsys.readouterr()
    assert _compare(paths, "--new-b3", str(paths["b3"])) == 0
    out = capsys.readouterr().out
    assert SENTINEL not in out
    assert "valid new records: 6/6" in out
    assert "detection on omission notes: new 1/2, released r1 2/2, difference 1 (margin 4)" in out
    assert "false alarms on clean notes: new 1/2, released r1 1/2, difference 0 (margin 1)" in out
    disagree = ", ".join(sorted([key("c2-omit"), key("c2-add")]))
    assert f"per-note agreement (information only): 4/6 (66.7%) | disagree: {disagree}" in out
    assert "below 80%: investigate before M4" in out
    assert "n_absent_critical, new vs released r1: equal 4/6, mean |difference| 0.667, max 2" in out
    assert "route one: PASS" in out


def test_route_one_fails_without_a_valid_record_for_every_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = build(tmp_path, drop_b3="c1-add")
    _spec(paths)
    capsys.readouterr()
    assert _compare(paths, "--new-b3", str(paths["b3"])) == 1
    out = capsys.readouterr().out
    assert f"valid new records: 5/6 | missing or failed: {key('c1-add')}" in out
    assert "route one: FAIL" in out


def test_route_two_pass_mark_is_ninety_unless_released_agreement_is_lower() -> None:
    verdict = COMPARE.route_two_passes
    assert verdict(new_r1=0.92, new_r3=0.90, released=0.95) == (0.90, True)
    assert verdict(new_r1=0.92, new_r3=0.89, released=0.95) == (0.90, False)
    assert verdict(new_r1=0.80, new_r3=0.85, released=0.80) == (0.80, True)


def test_route_one_flag_and_count() -> None:
    flag, count = COMPARE.route_one_flag, COMPARE.route_one_count
    assert flag({"detail": {"missing_facts": [CRITICAL_ABSENT]}}) is True
    assert flag({"detail": {"missing_facts": []}}) is False  # parsed, nothing missing
    assert flag({"n_absent_critical": 0}) is False and flag({}) is False
    assert flag({"parse_failure": True}) is None
    assert count({"detail": {"missing_facts": [CRITICAL_ABSENT, CRITICAL_PARTIAL]}}) == 1
    assert count({"n_absent_critical": 3}) == 3 and count({}) == 0
    assert count({"parse_failure": True}) is None
