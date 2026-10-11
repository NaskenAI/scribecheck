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
    _write(records / "w1" / "scored.json", {"records": FILE_A})
    _write(records / "w2" / "scored.json", {"records": FILE_B})
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


def test_main_prints_one_row_per_file_and_no_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records, pairs = build_fixture(tmp_path)
    assert RECOMPUTE.main([str(records), "--pairs", str(pairs)]) == 0
    out = capsys.readouterr().out
    assert SENTINEL not in out
    rows = [line.split() for line in out.splitlines() if line.startswith("w")]
    assert [row[0] for row in rows] == ["w1", "w2"]
    w1, w2 = rows
    # omission n, wins, ties, discrimination
    assert w1[1:5] == ["3", "1", "1", "0.500"]
    assert w1[5:9] == ["2", "1", "1", "0.750"]
    assert w1[9:12] == ["5", "3", "0.600"]
    assert w1[12:15] == ["6", "1", "0.167"]
    assert w1[15:] == ["1", "1", "1", "1"]
    assert w2[1:5] == ["1", "1", "0", "1.000"]
    assert w2[5:9] == ["0", "0", "0", "–"]
    assert "skipped 2 file(s)" in out


def test_bad_pairs_file_exits_1(tmp_path: Path) -> None:
    records, _ = build_fixture(tmp_path)
    assert RECOMPUTE.main([str(records), "--pairs", str(tmp_path / "missing.json")]) == 1
