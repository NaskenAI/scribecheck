"""scripts/recompute_published.py on a synthetic fixture with hand-computed results.

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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recompute_published.py"
SENTINEL = "SENTINEL-TEXT-DO-NOT-LEAK"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("recompute_published", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up while it loads
    spec.loader.exec_module(module)
    return module


RECOMPUTE = _load()

# pair_id -> type, as in pairs/dataset_v2.json
TYPES = {
    "s|c1|e1|f1": "omit",
    "s|c2|e1|f1": "omit",
    "s|c3|e1|f1": "omit",
    "s|c4|e1|f1": "omit",
    "s|c5|e1|f1": "add",
    "s|c6|e1|f1": "change",
    "s|c7|e1|f1": "omit",
}


def _clean(consultation: str, aggregate: float | None, flagged: bool | None) -> dict[str, Any]:
    return {
        "pair_id": None,
        "clean_key": f"s|{consultation}|clean|a",
        "note_key": f"s|{consultation}|clean",
        "note_role": "clean",
        "stratum": "s",
        "consultation": consultation,
        "aggregate": aggregate,
        "flagged": flagged,
        "rationale": SENTINEL,
    }


def _errored(consultation: str, aggregate: float | None, flagged: bool | None) -> dict[str, Any]:
    return {
        "pair_id": f"s|{consultation}|e1|f1",
        "clean_key": f"s|{consultation}|clean|a",
        "note_key": f"s|{consultation}|e1|err",
        "note_role": "errored",
        "stratum": "s",
        "consultation": consultation,
        "aggregate": aggregate,
        "flagged": flagged,
        "note_text": SENTINEL,
    }


FILE_A = [
    _clean("c1", 0.8, False),
    _clean("c2", 0.5, False),
    _clean("c3", 0.3, True),
    _clean("c5", 0.9, False),
    _clean("c5", 0.9, False),  # identical duplicate: one note
    _clean("c6", 0.4, None),  # unknown flag: not in the clean flag rate
    _clean("c7", 0.6, False),
    _clean("c8", 0.7, False),
    _clean("c8", 0.2, False),  # conflicting duplicate: excluded
    _clean("c9", 0.5, False),
    _errored("c1", 0.2, True),  # omission win
    _errored("c2", 0.5, False),  # omission tie
    _errored("c3", 0.7, True),  # omission loss
    _errored("c4", 0.1, True),  # omission, no clean twin
    _errored("c5", 0.1, True),  # add, win
    _errored("c6", 0.4, None),  # change, tie
    _errored("c7", None, False),  # omission, missing score
    _errored("c9", 0.1, True),  # pair_id not in --pairs
]
FILE_B = [_clean("c1", 0.9, False), _errored("c1", 0.3, True)]


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def build_fixture(root: Path) -> tuple[Path, Path]:
    records = root / "records"
    _write(records / "w1" / "scored.json", {"records": REPLICATED})
    _write(records / "w2" / "scored.json", {"records": SMOKE})
    _write(records / "w3" / "summary.json", {"text": SENTINEL})
    _write(records / "w4" / "empty.json", {"records": []})
    pairs = root / "dataset_v2.json"
    _write(pairs, {"pairs": [{"pair_id": k, "type": v, "key": k[:4]} for k, v in TYPES.items()]})
    return records, pairs


def test_compute_file_a() -> None:
    result = RECOMPUTE.compute(FILE_A, TYPES)
    assert (result.omit_n, result.omit_wins, result.omit_ties) == (3, 1, 1)
    assert result.omit_discrimination == pytest.approx(0.5)
    assert (result.change_n, result.change_wins, result.change_ties) == (2, 1, 1)
    assert result.change_discrimination == pytest.approx(0.75)
    assert (result.detection_n, result.detection_flagged) == (5, 3)
    assert result.detection_rate == pytest.approx(0.6)
    assert (result.clean_n, result.clean_flagged) == (6, 1)
    assert result.clean_flag_rate == pytest.approx(1 / 6)
    assert (result.no_twin, result.missing_score, result.unknown_type, result.conflicts) == (
        1,
        1,
        1,
        1,
    )


def test_rates_with_no_pairs_are_none() -> None:
    result = RECOMPUTE.compute([_errored("c4", 0.1, True)], TYPES)
    assert result.omit_n == 0 and result.omit_discrimination is None
    assert result.clean_n == 0 and result.clean_flag_rate is None
    assert result.no_twin == 1


def test_conflicting_errored_duplicates_are_excluded() -> None:
    rows = [_clean("c1", 0.8, False), _errored("c1", 0.2, True), _errored("c1", 0.9, True)]
    result = RECOMPUTE.compute(rows, TYPES)
    assert result.omit_n == 0 and result.detection_n == 0 and result.conflicts == 1


def test_labels() -> None:
    paths = [
        Path("records/w1/scored.json"),
        Path("judge_scores/judgements__judges__m1__run1__scored.json"),
        Path("judge_scores/judgements__judges__m2__run1__scored.json"),
        Path("judge_scores/validation__sitting__scored.json"),
    ]
    assert RECOMPUTE.labels(paths) == [
        "w1",
        "run1:judgements__judges__m1__run1__scored",
        "run1:judgements__judges__m2__run1__scored",
        "sitting",
    ]


def test_bad_pairs_file_exits_1(tmp_path: Path) -> None:
    records, _ = build_fixture(tmp_path)
    assert RECOMPUTE.main([str(records), "--pairs", str(tmp_path / "missing.json")]) == 1


# Replicates, arms, models, parse failures and smoke runs

RUN_A = {
    "model": "openai/gpt-5.4",
    "arm": "B2",
    "cell": "B2-F-bin-k1",
    "criterion": "F",
    "format": "bin",
    "k": 1,
    "tier": "B2",
    "reasoning_effort": "high",
    "dataset_version": "factorial-v1",
}
RUN_B = {**RUN_A, "arm": "B3", "cell": "B3-F-bin-k1", "tier": "B3"}
RUN_C = {**RUN_A, "model": "google/gemini-3.1-pro-preview"}
RUN_D = {k: v for k, v in RUN_C.items() if k != "tier"}  # same as C, but no tier field
RUN_E = {**RUN_A, "dataset_version": "factorial-v1-armsmoke"}


def _run(run: dict[str, Any], replicate: int, row: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {**row, **run, "replicate": replicate, **extra}


# Group A: (clean score, clean flag, errored score, errored flag) per replicate and consultation.
A_SCORES = {
    1: {"c1": (0.8, False, 0.2, True), "c2": (0.5, False, 0.5, True),
        "c3": (0.3, True, 0.7, False), "c5": (0.9, False, 0.1, True)},
    2: {"c1": (0.8, False, 0.2, False), "c2": (0.5, True, 0.4, False),
        "c3": (0.3, False, 0.2, True), "c5": (0.9, False, 0.9, False)},
    3: {"c1": (0.8, False, 0.9, True), "c2": (0.5, False, 0.6, False),
        "c3": (0.3, True, 0.3, False), "c5": (0.9, None, 0.95, True)},
}  # fmt: skip
GROUP_A = [
    row
    for replicate, scores in A_SCORES.items()
    for consultation, (clean, clean_flag, err, err_flag) in scores.items()
    for row in (
        _run(RUN_A, replicate, _clean(consultation, clean, clean_flag)),
        _run(RUN_A, replicate, _errored(consultation, err, err_flag)),
    )
] + [_run(RUN_A, 1, _errored("c4", 0.1, True), parse_failure=True)]
GROUP_B = [
    _run(RUN_B, 1, _clean("c1", 0.6, False)),
    _run(RUN_B, 1, _errored("c1", 0.2, True)),
    _run(RUN_B, 1, _clean("c8", 0.5, False)),
    _run(RUN_B, 1, _errored("c8", 0.1, True), pair_type="add"),  # not in --pairs
]
GROUP_C = [_run(RUN_C, 1, _clean("c1", 0.5, False)), _run(RUN_C, 1, _errored("c1", 0.5, False))]
GROUP_D = [_run(RUN_D, 1, _clean("c1", 0.4, False)), _run(RUN_D, 1, _errored("c1", 0.1, True))]
REPLICATED = GROUP_A + GROUP_B + GROUP_C + GROUP_D
SMOKE = [_run(RUN_E, 1, _clean("c1", 0.5, False)), _run(RUN_E, 1, _errored("c1", 0.1, True))]


def _groups() -> dict[str, Any]:
    results = RECOMPUTE.analyse("w1", REPLICATED, TYPES)
    return {
        f"{g.key_values['arm']}/{g.key_values['model']}/{g.key_values['tier']}": g for g in results
    }


def test_groups_split_by_arm_model_and_missing_fields() -> None:
    groups = _groups()
    assert set(groups) == {
        "B2/openai/gpt-5.4/B2",
        "B3/openai/gpt-5.4/B3",
        "B2/google/gemini-3.1-pro-preview/B2",
        "B2/google/gemini-3.1-pro-preview/(missing)",
    }
    assert all(g.comparable for g in groups.values())


def test_group_a_per_replicate_and_combined() -> None:
    a = _groups()["B2/openai/gpt-5.4/B2"]
    assert a.parse_failures == 1 and a.rows == len(GROUP_A) - 1
    assert list(a.replicates) == ["1", "2", "3"]
    r1, r2, r3 = a.replicates.values()
    assert (r1.omit_n, r1.omit_wins, r1.omit_ties) == (3, 1, 1)
    assert (r2.omit_n, r2.omit_wins, r2.omit_ties) == (3, 3, 0)
    assert (r3.omit_n, r3.omit_wins, r3.omit_ties) == (3, 0, 1)
    assert a.omit_mean == pytest.approx((0.5 + 1.0 + 1 / 6) / 3)
    assert [r.change_discrimination for r in (r1, r2, r3)] == [1.0, 0.5, 0.0]
    assert a.change_mean == pytest.approx(0.5)
    assert [(r.detection_flagged, r.detection_n) for r in (r1, r2, r3)] == [(2, 3), (1, 3), (1, 3)]
    assert [(r.clean_flagged, r.clean_n) for r in (r1, r2, r3)] == [(1, 4), (1, 4), (1, 3)]
    # c1 is flagged in replicates 1 and 3 (majority); c3 only in replicate 2 (not).
    assert (a.majority_detection_flagged, a.majority_detection_n) == (1, 3)
    # Clean c3 is flagged in replicates 1 and 3; clean c5 has two known flags, both False.
    assert (a.majority_clean_flagged, a.majority_clean_n) == (1, 4)


def test_types_come_from_pairs_then_rows() -> None:
    b = _groups()["B3/openai/gpt-5.4/B3"]
    (r1,) = b.replicates.values()
    assert (r1.types_from_pairs, r1.types_from_rows, r1.unknown_type) == (1, 1, 0)
    assert (r1.omit_n, r1.change_n, r1.change_wins) == (1, 1, 1)


def test_smoke_runs_are_not_comparable() -> None:
    (smoke,) = RECOMPUTE.analyse("w2", SMOKE, TYPES)
    assert not smoke.comparable


def test_majority_needs_more_than_half() -> None:
    assert RECOMPUTE.strict_majority([True, False]) is False
    assert RECOMPUTE.strict_majority([True, True, False]) is True
    assert RECOMPUTE.strict_majority([True]) is True


def test_main_prints_groups_sections_and_no_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records, pairs = build_fixture(tmp_path)
    assert RECOMPUTE.main([str(records), "--pairs", str(pairs)]) == 0
    out = capsys.readouterr().out
    assert SENTINEL not in out
    comparable, _, smoke = out.partition("Not comparable to published figures")
    assert comparable.count("[w1]") == 4 and "[w2]" not in comparable
    assert smoke.count("[w2]") == 1 and "dataset=factorial-v1-armsmoke" in smoke
    block = comparable[comparable.index("arm=B2") :]
    assert "parse failures excluded 1" in block
    assert "omission discrimination: mean 0.556" in block
    assert "r1 0.500 (n 3, wins 1, ties 1)" in block
    assert "majority 1/3 0.333" in block and "majority 1/4 0.250" in block
    assert "skipped 2 file(s)" in out and "parse failures excluded in total: 1" in out
