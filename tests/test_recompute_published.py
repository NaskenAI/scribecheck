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


def test_source_reads_family_and_folder_timestamp() -> None:
    source = RECOMPUTE.source
    raw = Path("raw/judgements/judges/w2-power/20260818-162144-d710faf-r1/w2_power.json")
    flat = Path("judge_scores/judgements__judges__w2-power__20260819-090000-bbbbbbb-r2__x.json")
    assert source(raw) == RECOMPUTE.Source("w2-power", "20260818-162144")
    assert source(flat) == RECOMPUTE.Source("w2-power", "20260819-090000")
    assert source(Path("records/w1/scored.json")) == RECOMPUTE.Source("w1", None)
    assert source(Path("judge_scores/validation__sitting__scored.json")) == RECOMPUTE.Source(
        "sitting", None
    )
    undated = Path("raw/judgements/judges/w2-strong/latest/scored.json")
    assert source(undated) == RECOMPUTE.Source("w2-strong", None)


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
# Group F: three replicates where parse failures hit notes that are scored in other
# replicates. Each failed row would be a win and a flag if it were counted.
RUN_F = {**RUN_A, "arm": "checklist", "cell": "checklist", "tier": "B3"}
FAILED = {"parse_failure": True}
GROUP_F = [
    _run(RUN_F, replicate, _clean(consultation, 0.8, False))
    for replicate in (1, 2, 3)
    for consultation in ("c1", "c2", "c3")
] + [
    _run(RUN_F, 1, _errored("c1", 0.2, True)),
    _run(RUN_F, 1, _errored("c2", 0.3, True)),
    _run(RUN_F, 1, _errored("c3", 0.1, True)),
    _run(RUN_F, 2, _errored("c1", 0.2, True)),
    _run(RUN_F, 2, _errored("c2", 0.9, False)),
    _run(RUN_F, 2, _errored("c3", 0.0, True), **FAILED),
    _run(RUN_F, 3, _errored("c1", 0.0, True), **FAILED),
    _run(RUN_F, 3, _errored("c2", 0.0, True), **FAILED),
    _run(RUN_F, 3, _errored("c3", 0.0, True), **FAILED),
]
REPLICATED = GROUP_A + GROUP_B + GROUP_C + GROUP_D + GROUP_F
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
        "checklist/openai/gpt-5.4/B3",
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
    # c1 is flagged in replicates 1 and 3 (majority); c3 only in replicate 2 (not). c4 failed
    # to parse in its only replicate: not flagged in all three, and absent from known-only.
    assert (a.majority_detection_flagged, a.majority_detection_n) == (1, 4)
    assert (a.known_majority_detection_flagged, a.known_majority_detection_n) == (1, 3)
    # Clean c3 is flagged in replicates 1 and 3; clean c5 has two known flags, both False.
    assert (a.majority_clean_flagged, a.majority_clean_n) == (1, 4)
    assert (a.known_majority_clean_flagged, a.known_majority_clean_n) == (1, 4)


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
    # Over all three replicates, a lost replicate counts as not flagged.
    assert RECOMPUTE.strict_majority([True, False], replicates=3) is False  # 1 of 2, 1 lost
    assert RECOMPUTE.strict_majority([True, True], replicates=3) is True  # 2 of 2, 1 lost
    assert RECOMPUTE.strict_majority([True], replicates=3) is False  # 1 of 1, 2 lost


def test_parse_failures_drop_out_of_every_replicate_metric() -> None:
    f = _groups()["checklist/openai/gpt-5.4/B3"]
    assert f.parse_failures == 4 and f.rows == len(GROUP_F) - 4
    assert list(f.replicates) == ["1", "2", "3"]
    r1, r2, r3 = f.replicates.values()
    assert (r1.omit_n, r1.omit_wins) == (3, 3)
    assert (r2.omit_n, r2.omit_wins) == (2, 1)  # c3's failed row is not a win
    assert (r3.omit_n, r3.omit_discrimination) == (0, None)
    assert f.omit_mean == pytest.approx((1.0 + 0.5) / 2)
    assert [(r.detection_flagged, r.detection_n) for r in (r1, r2, r3)] == [(3, 3), (1, 2), (0, 0)]


def test_majority_readings_on_lost_replicates() -> None:
    f = _groups()["checklist/openai/gpt-5.4/B3"]
    # c1: 2 of 2 known, 1 lost -> flagged in both readings.
    # c2: 1 of 2 known, 1 lost -> flagged in neither.
    # c3: 1 of 1 known, 2 lost -> flagged only in the known-only reading.
    assert (f.majority_detection_flagged, f.majority_detection_n) == (1, 3)
    assert (f.known_majority_detection_flagged, f.known_majority_detection_n) == (2, 3)
    assert (f.majority_clean_flagged, f.majority_clean_n) == (0, 3)
    assert (f.known_majority_clean_flagged, f.known_majority_clean_n) == (0, 3)


def test_main_prints_groups_sections_and_no_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records, pairs = build_fixture(tmp_path)
    assert RECOMPUTE.main([str(records), "--pairs", str(pairs)]) == 0
    out = capsys.readouterr().out
    assert SENTINEL not in out
    comparable, _, smoke = out.partition("Not comparable to published figures")
    assert comparable.count("[w1]") == 5 and "[w2]" not in comparable
    assert smoke.count("[w2]") == 1 and "dataset=factorial-v1-armsmoke" in smoke
    block = comparable[comparable.index("arm=B2") :]
    assert "parse failures excluded 1" in block
    assert "omission discrimination: mean 0.556" in block
    assert "r1 0.500 (n 3, wins 1, ties 1)" in block
    detection = next(line for line in block.splitlines() if "detection:" in line)
    assert "| majority 1/4 0.250 | known-only majority 1/3 0.333" in detection
    clean = next(line for line in block.splitlines() if "clean flag rate:" in line)
    assert "| majority 1/4 0.250 | known-only majority 1/4 0.250" in clean
    assert "skipped 2 file(s)" in out and "parse failures excluded in total: 5" in out


# Run shards: one run spread over several timestamped folders of one family.

SHARD_RUN = {**RUN_A, "arm": "power-gepa-high", "cell": "power"}
SHARD_FOLDERS = (
    "20260818-162144-d710faf-r1",
    "20260818-170000-a1b2c3d-r1",
    "20260819-090000-bbbbbbb-r1",
)
CLEAN_SCORES = (("c1", 0.8), ("c2", 0.5), ("c3", 0.3))
SHARDS = {
    SHARD_FOLDERS[0]: [
        _run(SHARD_RUN, 1, _clean("c1", 0.8, False)),
        _run(SHARD_RUN, 1, _errored("c1", 0.9, False)),  # replaced by shard 2
        _run(SHARD_RUN, 1, _clean("c2", 0.5, False)),
        _run(SHARD_RUN, 1, _errored("c2", 0.2, True)),  # replaced by shard 3
    ],
    SHARD_FOLDERS[1]: [
        _run(SHARD_RUN, 1, _clean("c3", 0.3, True)),
        _run(SHARD_RUN, 1, _errored("c3", 0.1, True)),
        _run(SHARD_RUN, 1, _errored("c1", 0.2, True)),
    ],
    SHARD_FOLDERS[2]: [
        _run(SHARD_RUN, 1, _errored("c2", 0.0, True), parse_failure=True),  # latest, then fails
        *[_run(SHARD_RUN, 2, _clean(c, a, False)) for c, a in CLEAN_SCORES],
        _run(SHARD_RUN, 2, _errored("c1", 0.2, True)),
        _run(SHARD_RUN, 2, _errored("c2", 0.6, False)),
        _run(SHARD_RUN, 2, _errored("c3", 0.1, True)),
    ],
}  # fmt: skip


def _shard_files() -> list[tuple[Any, list[dict[str, Any]]]]:
    return [(RECOMPUTE.Source("w2-power", folder[:15]), rows) for folder, rows in SHARDS.items()]


def test_shards_of_one_run_form_one_group() -> None:
    (group,) = RECOMPUTE.analyse_files(_shard_files(), TYPES)
    assert group.label == "w2-power" and group.files == 3
    assert list(group.replicates) == ["1", "2"] and group.replicate_count == 2
    assert group.coverage == {"1": (3, 3), "2": (3, 3)}
    assert group.coverage_label == "partial coverage"


def test_latest_shard_replaces_older_rows_before_parse_failures_drop() -> None:
    (group,) = RECOMPUTE.analyse_files(_shard_files(), TYPES)
    assert group.replaced == 2 and group.parse_failures == 1
    r1, r2 = group.replicates.values()
    assert (r1.omit_n, r1.omit_wins) == (2, 2)  # c1 from shard 2 (a win), c2 failed in shard 3
    assert (r1.detection_flagged, r1.detection_n) == (2, 2)
    assert (r1.clean_flagged, r1.clean_n) == (1, 3)
    assert (r2.omit_n, r2.omit_wins) == (3, 2)
    assert group.omit_mean == pytest.approx((1.0 + 2 / 3) / 2)
    assert (group.majority_detection_flagged, group.majority_detection_n) == (2, 3)
    assert (group.known_majority_detection_flagged, group.known_majority_detection_n) == (2, 3)
    assert (group.majority_clean_flagged, group.majority_clean_n) == (0, 3)


def _covering(run: dict[str, Any], errored: int, clean: int, replicate: int = 1) -> list[Any]:
    """A replicate covering `errored` errored notes over `clean` consultations."""
    rows = [_run(run, replicate, _clean(f"k{i}", 0.9, False)) for i in range(clean)]
    for i in range(errored):
        row = _errored(f"k{i % clean}", 0.1, True)
        row.update(
            pair_id=f"s|k{i % clean}|e{i}|f1", note_key=f"s|k{i % clean}|e{i}", pair_type="omit"
        )
        rows.append(_run(run, replicate, row))
    return rows


RUN_CONFIRM = {**RUN_A, "arm": "pipeline", "dataset_version": "factorial-v1-armconfirm"}
RUN_DEV = {**RUN_A, "arm": "ragas"}
RUN_PARTIAL = {**RUN_A, "arm": "geval"}
COVERAGE_FILES = {
    "complete": _covering(RUN_A, 495, 112) + _covering(RUN_A, 495, 112, replicate=2),
    "confirm": _covering(RUN_CONFIRM, 151, 47),
    "development": _covering(RUN_DEV, 265, 65) + _covering(RUN_DEV, 265, 65, replicate=2),
    # replicate 2 is one errored note short, so the group is partial
    "partial": _covering(RUN_PARTIAL, 265, 65) + _covering(RUN_PARTIAL, 264, 65, replicate=2),
}


def _coverage_groups() -> dict[str, Any]:
    files = [(RECOMPUTE.Source(name, None), rows) for name, rows in COVERAGE_FILES.items()]
    return {g.label: g for g in RECOMPUTE.analyse_files(files, TYPES)}


def test_coverage_labels() -> None:
    groups = _coverage_groups()
    assert groups["complete"].coverage == {"1": (495, 112), "2": (495, 112)}
    assert groups["complete"].coverage_label == "complete"
    assert groups["confirm"].coverage_label == "complete"
    assert groups["development"].coverage_label == "development complete"
    assert groups["partial"].coverage == {"1": (265, 65), "2": (264, 65)}
    assert groups["partial"].coverage_label == "partial coverage"


def test_coverage_counts_parse_failures() -> None:
    rows = _covering(RUN_DEV, 265, 65)
    rows[-1]["parse_failure"] = True
    (group,) = RECOMPUTE.analyse_files([(RECOMPUTE.Source("dev", None), rows)], TYPES)
    assert group.coverage == {"1": (265, 65)} and group.parse_failures == 1
    assert group.coverage_label == "development complete"


def test_missing_or_null_dataset_version_is_not_comparable() -> None:
    missing = {k: v for k, v in RUN_A.items() if k != "dataset_version"}
    rows = [
        _run(missing, 1, _clean("c1", 0.5, False)),
        _run({**RUN_A, "dataset_version": None}, 1, _clean("c1", 0.5, False)),
    ]
    groups = RECOMPUTE.analyse("w5", rows, TYPES)
    assert sorted(g.key_values["dataset_version"] for g in groups) == ["(missing)", "(null)"]
    assert not any(g.comparable for g in groups)


def test_summary_lists_complete_then_development_groups(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records, pairs = build_fixture(tmp_path)
    judges = records / "judgements" / "judges"
    for folder, rows in SHARDS.items():
        _write(judges / "w2-power" / folder / "w2_power.json", {"records": rows})
    for name, rows in COVERAGE_FILES.items():
        _write(judges / f"w2-{name}" / "20260820-120000-c0ffee1-r1" / "w2.json", {"records": rows})
    assert RECOMPUTE.main([str(records), "--pairs", str(pairs), "--summary"]) == 0
    out = capsys.readouterr().out
    assert SENTINEL not in out
    complete, heading, development = out.partition(
        "Development complete — not comparable to published figures"
    )
    assert heading
    development = development.partition("== Partial coverage ==")[0]
    lines = [line for line in complete.splitlines() if "family=" in line]
    assert [line.split("family=")[1].split()[0] for line in lines] == ["w2-complete", "w2-confirm"]
    first = lines[0]
    assert first.startswith("openai/gpt-5.4 family=w2-complete arm=B2 cell=B2-F-bin-k1")
    assert "dataset=factorial-v1 " in first
    assert "| omit r1 1.000 r2 1.000 |" in first and "| add/change r1 – r2 – |" in first
    assert "| detection r1 1.000 r2 1.000 | clean r1 0.000 r2 0.000 |" in first
    assert "| majority detection 1.000 clean 0.000" in first
    assert "| known-only majority detection 1.000 clean 0.000" in first
    dev_lines = [line for line in development.splitlines() if "family=" in line]
    assert [line.split("family=")[1].split()[0] for line in dev_lines] == ["w2-development"]
    for absent in ("w2-partial", "w2-power", "[w1]", "w2-confirm"):
        assert absent not in development
    assert "w2-partial" not in complete and "w2-power" not in complete


# Paired discrimination drops a pair when either side failed to parse
# (as the harness does: record/figures/extract_data.py:80-87).


def test_pair_is_dropped_when_its_clean_twin_failed_to_parse() -> None:
    rows = [
        _run(RUN_A, 1, _clean("c1", 0.9, False), parse_failure=True),
        _run(RUN_A, 1, _errored("c1", 0.1, True)),
        _run(RUN_A, 1, _clean("c2", 0.9, False)),
        _run(RUN_A, 1, _errored("c2", 0.1, True), parse_failure=True),
        _run(RUN_A, 1, _clean("c3", 0.9, False)),
        _run(RUN_A, 1, _errored("c3", 0.1, True)),
    ]
    (group,) = RECOMPUTE.analyse("w6", rows, TYPES)
    (r1,) = group.replicates.values()
    assert (r1.omit_n, r1.omit_wins) == (1, 1)  # only c3 is paired


# Detection at matched false alarms, by the harness's convention
# (record/figures/extract_data.py:106-135): thresholds are every observed score v and v+1e-6,
# a note is flagged when its score is strictly below the threshold, and the chosen point has
# the highest detection with at most k clean notes flagged, then fewer clean notes flagged,
# then the larger threshold.


def _point(err: list[float], clean: list[float], k: int) -> tuple[float, int, int, int, int]:
    p = RECOMPUTE.matched_threshold(err, clean, k)
    return (p.threshold, p.det_k, p.n_err, p.fa_k, p.n_clean)


def test_matched_threshold_is_strict_and_at_most_k() -> None:
    err, clean = [1.0, 5.0, 9.0, 10.0, 10.0], [9.0, 10.0, 10.0, 10.0]
    # Errored notes AT the threshold (10) are not flagged.
    assert _point(err, clean, 1) == (10.0, 3, 5, 1, 4)
    # Three clean notes tie at 10, so no threshold flags exactly two or three: at most k.
    assert _point(err, clean, 2) == (10.0, 3, 5, 1, 4)
    assert _point(err, clean, 3) == (10.0, 3, 5, 1, 4)


def test_matched_threshold_prefers_fewer_false_alarms_over_a_larger_threshold() -> None:
    # Thresholds 2 and 6 both detect 2 of 3; 2 flags no clean note and 6 flags one.
    assert _point([1.0, 1.0, 8.0], [2.0, 6.0], 1) == (2.0, 2, 3, 0, 2)
    assert _point([1.0, 1.0, 8.0], [2.0, 6.0], 2) == (8.000001, 3, 3, 2, 2)


def test_matched_threshold_with_fewer_clean_notes_than_k() -> None:
    assert _point([1.0, 2.0, 5.0, 9.0], [3.0, 8.0], 3) == (9.000001, 4, 4, 2, 2)


def test_matched_threshold_needs_more_than_two_distinct_scores() -> None:
    assert RECOMPUTE.matched_threshold([0.0, 1.0], [1.0, 1.0], 1) is None
    assert RECOMPUTE.matched_threshold([], [1.0, 2.0, 3.0], 1) is None


# Group M: a powered 0-10 judge over two replicates; c4's replicate-2 row failed to parse.
RUN_M = {**RUN_A, "arm": "power-gepa-high", "cell": "power-gepa-high", "format": "score"}
M_SCORES = {
    1: {"c1": (10, 3), "c2": (10, 9.5), "c3": (10, 10), "c4": (9, 6)},
    2: {"c1": (10, 2), "c2": (10, 10), "c3": (10, 8), "c4": (10, None)},
}
GROUP_M = [
    row
    for replicate, scores in M_SCORES.items()
    for consultation, (clean, err) in scores.items()
    for row in (
        _run(RUN_M, replicate, _clean(consultation, clean, False)),
        _run(RUN_M, replicate, _errored(consultation, err, False), parse_failure=err is None),
    )
]


def _group_m() -> Any:
    (group,) = RECOMPUTE.analyse("w2-power", GROUP_M, TYPES)
    return group


def test_matched_false_alarms_per_replicate_and_majority() -> None:
    m = _group_m().matched[1]
    r1, r2 = m.per_replicate["1"], m.per_replicate["2"]
    assert (r1.threshold, r1.det_k, r1.n_err, r1.fa_k, r1.n_clean) == (10.0, 3, 4, 1, 4)
    # c4 failed in replicate 2, so it is not among the scores swept there.
    assert (r2.threshold, r2.det_k, r2.n_err, r2.fa_k, r2.n_clean) == (10.0, 2, 3, 0, 4)
    # Flagged in both replicates (strict majority of 2): c1 only, over all 4 omission notes.
    assert m.majority == (1, 4, 0, 4)


def test_route_two_below_10_rule() -> None:
    rule = _group_m().rules["route-two-below-10"]
    # parse failure = not flagged, on the full denominator
    assert rule.per_replicate == {"1": (3, 4, 1, 4), "2": (2, 4, 0, 4)}
    assert rule.majority == (1, 4, 0, 4)
    assert rule.known_majority == (2, 4, 0, 4)  # c4's only known flag is True


RUN_B3 = {**RUN_A, "arm": "B3", "cell": "B3", "tier": "B3", "format": "score"}
CRITICAL_ABSENT = [
    {"severity": "critical", "verdict": "absent", "fact": SENTINEL},
    {"severity": "supporting", "verdict": "present", "fact": SENTINEL},
]
SUPPORTING_ABSENT = [
    {"severity": "supporting", "verdict": "absent", "fact": SENTINEL},
    {"severity": "critical", "verdict": "present", "fact": SENTINEL},
]
GROUP_B3 = [
    _run(RUN_B3, 1, _clean("c1", 0.9, None), n_absent_critical=0),
    _run(RUN_B3, 1, _clean("c2", 0.9, None), detail={"missing_facts": CRITICAL_ABSENT}),
    _run(RUN_B3, 1, _errored("c1", 0.2, None), detail={"missing_facts": CRITICAL_ABSENT}),
    _run(RUN_B3, 1, _errored("c2", 0.2, None), detail={"missing_facts": SUPPORTING_ABSENT}),
    _run(RUN_B3, 1, _errored("c3", 0.2, None), n_absent_critical=2),
    _run(RUN_B3, 1, _errored("c4", 0.2, None), parse_failure=True),
]


def test_critical_absent_reads_detail_then_the_split_count() -> None:
    rule = RECOMPUTE.critical_absent
    assert rule({"detail": {"missing_facts": CRITICAL_ABSENT}}) is True
    assert rule({"detail": {"missing_facts": SUPPORTING_ABSENT}}) is False
    assert rule({"detail": {"missing_facts": []}, "n_absent_critical": 3}) is False
    assert rule({"n_absent_critical": 2}) is True and rule({"n_absent_critical": 0}) is False
    assert rule({"detail": "text"}) is None and rule({}) is None


def test_pipeline_critical_absent_rule() -> None:
    (group,) = RECOMPUTE.analyse("w2-pipeline", GROUP_B3, TYPES)
    rule = group.rules["pipeline-critical-absent"]
    assert rule.per_replicate == {"1": (2, 4, 1, 2)}
    assert rule.majority == (2, 4, 1, 2)
    # Groups with neither detail.missing_facts nor the split's count get no such rule.
    assert "pipeline-critical-absent" not in _group_m().rules


def test_report_prints_rules_and_matched_points_without_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for group in RECOMPUTE.analyse("w2-pipeline", GROUP_B3, TYPES) + [_group_m()]:
        print(RECOMPUTE.format_group(group))
    out = capsys.readouterr().out
    assert SENTINEL not in out
    assert "rule pipeline-critical-absent: r1 det 2/4 fa 1/2 | majority det 2/4 fa 1/2" in out
    # B3's scores take two values only, so no replicate has a point and neither has a majority.
    assert "matched false alarms k1: r1 – | majority –" in out
    assert (
        "rule route-two-below-10: r1 det 3/4 fa 1/4 | r2 det 2/4 fa 0/4 "
        "| majority det 1/4 fa 0/4 | known-only majority det 2/4 fa 0/4"
    ) in out
    assert (
        "matched false alarms k1: r1 det 3/4 fa 1/4 (t 10) | r2 det 2/3 fa 0/4 (t 10) "
        "| majority det 1/4 fa 0/4"
    ) in out


def test_summary_lists_partial_coverage_groups(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records, pairs = build_fixture(tmp_path)
    judges = records / "judgements" / "judges"
    for name, rows in COVERAGE_FILES.items():
        _write(judges / f"w2-{name}" / "20260820-120000-c0ffee1-r1" / "w2.json", {"records": rows})
    assert RECOMPUTE.main([str(records), "--pairs", str(pairs), "--summary"]) == 0
    out = capsys.readouterr().out
    _, heading, partial = out.partition("== Partial coverage ==")
    assert heading
    line = next(line for line in partial.splitlines() if "family=w2-partial" in line)
    assert "coverage r1 errored 265 clean 65 | r2 errored 264 clean 65" in line
    assert "family=w2-complete" not in partial and "dataset=factorial-v1-armsmoke" not in partial
