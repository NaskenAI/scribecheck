"""scripts/split_omissionbench.py on a synthetic fixture with the published structure.

Everything here is invented: consultation ids, notes and transcripts are placeholders, and
every text field carries SENTINEL so the tests can prove no text reaches output or logs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "split_omissionbench.py"
SENTINEL = "SENTINEL-TEXT-DO-NOT-LEAK"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("split_omissionbench", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up while it loads
    spec.loader.exec_module(module)
    return module


SPLIT = _load()


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        path.write_text(json.dumps(data), encoding="utf-8")
    else:
        path.write_text(data, encoding="utf-8")


def _keys(prefix: str, count: int) -> list[tuple[str, str]]:
    """(stratum, consultation) pairs spread over two strata."""
    return [("gp" if i % 2 else "ed", f"{prefix}{i:03d}") for i in range(count)]


DEV_KEYS = _keys("d", 65)
HELDOUT_KEYS = _keys("h", 47)
OTHER_KEYS = _keys("x", 5)  # outside the eval set


TYPES = ("omit", "add", "change")
LEVELS = ("complete", "fragment", "restatement")


def _pairs_for(keys: list[tuple[str, str]], total: int, eval_set: bool) -> list[dict[str, Any]]:
    per_key, extra = divmod(total, len(keys))
    pairs = []
    for index, (stratum, consultation) in enumerate(keys):
        for edit in range(per_key + (1 if index < extra else 0)):
            pairs.append(
                {
                    "pair_id": f"{stratum}|{consultation}|edit{edit}|fact{edit}",
                    "key": f"{stratum}|{consultation}",
                    "stratum": stratum,
                    "id": consultation,
                    "eval_set": eval_set,
                    "split": "eval" if eval_set else "gepa_dev",
                    "clean": f"clean note {SENTINEL}",
                    "errored": f"errored note {SENTINEL}",
                    "trace": "complete",
                    "type": TYPES[len(pairs) % len(TYPES)],
                    "residual_level": LEVELS[len(pairs) % len(LEVELS)],
                }
            )
    return pairs


def _score_row(stratum: str, consultation: str, pair_id: str | None) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "clean_key": None if pair_id else f"{stratum}|{consultation}",
        "note_key": f"note-{stratum}-{consultation}-{pair_id or 'clean'}",
        "note_role": "errored" if pair_id else "clean",
        "stratum": stratum,
        "consultation": consultation,
        "aggregate": 0.5,
        "flagged": True,
        "rationale": f"judge text {SENTINEL}",
        "note_text": f"note {SENTINEL}",
    }


def build_fixture(root: Path, *, dev_pairs: int = 265) -> tuple[Path, Path]:
    """A raw folder with the published layout; returns (raw dir, held-out records file)."""
    raw = root / "raw"
    dev = _pairs_for(DEV_KEYS, dev_pairs, True)
    heldout = _pairs_for(HELDOUT_KEYS, 230, True)
    other = _pairs_for(OTHER_KEYS, 10, False)
    _write(
        raw / "pairs" / "dataset_v2.json",
        {
            "_attribution": "OmissionBench (synthetic fixture), CC BY 4.0",
            "version": 2,
            "summary": {"heldout_example": f"{SENTINEL}"},
            "pairs": dev + heldout + other,
        },
    )
    for stratum, consultation in DEV_KEYS + HELDOUT_KEYS + OTHER_KEYS:
        _write(raw / "transcripts" / stratum / f"{consultation}.txt", f"Doctor: hi {SENTINEL}\n")
    _write(raw / "LICENSE.txt", "CC BY 4.0 (synthetic)")
    _write(raw / "transcripts" / "ATTRIBUTION.md", "PriMock57 and ACI-Bench (synthetic)")
    _write(raw / "transcripts" / "gp" / "attribution.txt", "per-stratum attribution (synthetic)")
    _write(raw / "README.md", f"dataset card {SENTINEL}")

    # Fact data: certain dev entries, certain held-out entries and uncertain ones.
    sheets = raw / "fact_sheets"
    for stratum, consultation in DEV_KEYS[:3] + HELDOUT_KEYS[:2]:
        _write(sheets / stratum / f"{consultation}.json", {"facts": [f"fact {SENTINEL}"]})
    _write(sheets / "overview.json", {"facts": [SENTINEL]})
    _write(
        raw / "pairs" / "fact_sites.json",
        {
            dev[0]["pair_id"]: {"site": "History"},
            heldout[0]["pair_id"]: {"site": "Plan"},
            "global-notes": {"site": SENTINEL},
        },
    )
    d_stratum, d_consultation = DEV_KEYS[1]
    _write(
        raw / "pairs" / "factorial_severity.json",
        {
            "rubric_version": 1,
            "rubric": {"text": SENTINEL},
            "facts": {
                f"{DEV_KEYS[0][0]}|{DEV_KEYS[0][1]}|fact0": {"severity": "critical"},
                "uid-by-fields": {
                    "stratum": d_stratum,
                    "consultation": d_consultation,
                    "severity": "supporting",
                },
                f"{HELDOUT_KEYS[0][0]}|{HELDOUT_KEYS[0][1]}|fact0": {"severity": "critical"},
                "orphan-uid": {"severity": "peripheral", "text": SENTINEL},
            },
        },
    )

    # Judge logs, review packs and score files.
    rows = [_score_row(p["stratum"], p["id"], p["pair_id"]) for p in dev[:20] + heldout[:20]]
    rows += [_score_row(s, c, None) for s, c in DEV_KEYS[:5] + HELDOUT_KEYS[:5] + OTHER_KEYS[:1]]
    _write(raw / "judgements" / "w1" / "scored_records.json", {"model": "x", "records": rows})
    # The published held-out subset: 151 of the 230 held-out pairs, covering all 47
    # consultations (the first pair of each, then the next ones in order).
    firsts = {p["key"]: p for p in reversed(heldout)}
    subset = list(firsts.values()) + [p for p in heldout if p not in firsts.values()]
    heldout_rows = [_score_row(p["stratum"], p["id"], p["pair_id"]) for p in subset[:151]]
    heldout_rows += [_score_row(s, c, None) for s, c in HELDOUT_KEYS]
    records = raw / "judgements" / "w2" / "pipeline_scored_records.json"
    _write(records, {"records": heldout_rows})
    _write(raw / "judgements" / "w1" / "judge.log", f"log {SENTINEL}")
    _write(raw / "validation" / "pack.json", {"items": [SENTINEL]})
    _write(raw / "notes" / "n1.md", f"note {SENTINEL}")
    return raw, records


def _run(raw: Path, records: Path, dev_out: Path, heldout_out: Path) -> int:
    argv = [
        "--raw", str(raw),
        "--heldout-records", str(records),
        "--dev-out", str(dev_out),
        "--heldout-out", str(heldout_out),
    ]  # fmt: skip
    return int(SPLIT.main(argv))


def _all_text(folder: Path) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in folder.rglob("*") if p.is_file())


@pytest.fixture
def split(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[Path, Path, str]:
    raw, records = build_fixture(tmp_path)
    dev_out, heldout_out = tmp_path / "out" / "dev", tmp_path / "out" / "heldout"
    assert _run(raw, records, dev_out, heldout_out) == 0
    captured = capsys.readouterr()
    return dev_out, heldout_out, captured.out + captured.err


def _pairs(folder: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((folder / "pairs" / "dataset_v2.json").read_text())
    return data


def test_pairs_are_split_by_consultation(split: tuple[Path, Path, str]) -> None:
    dev_out, heldout_out, _ = split
    dev, heldout = _pairs(dev_out), _pairs(heldout_out)
    assert len(dev["pairs"]) == 265 and len(heldout["pairs"]) == 230
    assert {p["key"] for p in dev["pairs"]} == {f"{s}|{c}" for s, c in DEV_KEYS}
    assert {p["key"] for p in heldout["pairs"]} == {f"{s}|{c}" for s, c in HELDOUT_KEYS}
    for data in (dev, heldout):
        assert set(data) == {"_attribution", "version", "pairs"}  # "summary" is not scalar
        assert all(p["eval_set"] for p in data["pairs"])


def test_transcripts_and_attribution(split: tuple[Path, Path, str]) -> None:
    dev_out, heldout_out, _ = split
    dev_transcripts = {p.stem for p in (dev_out / "transcripts").rglob("*.txt")}
    heldout_transcripts = {p.stem for p in (heldout_out / "transcripts").rglob("*.txt")}
    assert dev_transcripts - {"attribution"} == {c for _, c in DEV_KEYS}
    assert heldout_transcripts - {"attribution"} == {c for _, c in HELDOUT_KEYS}
    for out in (dev_out, heldout_out):
        assert (out / "LICENSE.txt").is_file()
        assert (out / "transcripts" / "ATTRIBUTION.md").is_file()
        assert (out / "transcripts" / "gp" / "attribution.txt").is_file()
        assert not (out / "README.md").exists()


def test_fact_data_is_filtered_with_certainty(split: tuple[Path, Path, str]) -> None:
    dev_out, heldout_out, log = split
    sheets = {
        p.relative_to(dev_out / "fact_sheets").as_posix()
        for p in (dev_out / "fact_sheets").rglob("*.json")
    }
    assert sheets == {f"{s}/{c}.json" for s, c in DEV_KEYS[:3]}
    sites = json.loads((dev_out / "pairs" / "fact_sites.json").read_text())
    assert list(sites) == [f"{DEV_KEYS[0][0]}|{DEV_KEYS[0][1]}|edit0|fact0"]
    severity = json.loads((dev_out / "pairs" / "factorial_severity.json").read_text())
    assert set(severity) == {"rubric_version", "facts"}
    assert set(severity["facts"]) == {f"{DEV_KEYS[0][0]}|{DEV_KEYS[0][1]}|fact0", "uid-by-fields"}
    assert not (heldout_out / "fact_sheets").exists()
    for listed in ("fact_sheets/overview.json", "global-notes", "orphan-uid"):
        assert listed in log


def test_judge_scores_are_reduced_and_development_only(split: tuple[Path, Path, str]) -> None:
    dev_out, _, log = split
    files = sorted((dev_out / "judge_scores").iterdir())
    assert [f.name for f in files] == [
        "judgements__w1__scored_records.json",
        "judgements__w2__pipeline_scored_records.json",
    ]
    w1 = json.loads(files[0].read_text())["records"]
    assert len(w1) == 20 + 5
    for row in w1:
        assert set(row) == set(SPLIT.SCORE_FIELDS)
        key = row["clean_key"] or "|".join(row["pair_id"].split("|")[:2])
        assert key in {f"{s}|{c}" for s, c in DEV_KEYS}
    assert json.loads(files[1].read_text())["records"] == []
    assert "dropped" in log


def test_nothing_leaks(split: tuple[Path, Path, str]) -> None:
    dev_out, heldout_out, log = split
    assert SENTINEL not in log
    for sub in ("judgements", "validation", "notes"):
        assert not (dev_out / sub).exists() and not (heldout_out / sub).exists()
    dev_text = _all_text(dev_out)
    # Development pairs and transcripts carry the sentinel by design; nothing else may.
    for path in (dev_out / "judge_scores").rglob("*"):
        assert SENTINEL not in path.read_text()
    assert SENTINEL not in (dev_out / "pairs" / "factorial_severity.json").read_text()
    assert SENTINEL not in (dev_out / "pairs" / "fact_sites.json").read_text()
    assert dev_text  # sanity


def _assert_aborted(code: int, *folders: Path) -> None:
    assert code == 1
    for folder in folders:
        assert not folder.exists() or not any(folder.iterdir())


def test_wrong_counts_abort_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw, records = build_fixture(tmp_path, dev_pairs=266)
    dev_out, heldout_out = tmp_path / "dev", tmp_path / "heldout"
    _assert_aborted(_run(raw, records, dev_out, heldout_out), dev_out, heldout_out)
    err = capsys.readouterr().err
    assert "266" in err and "265" in err and SENTINEL not in err


def test_missing_transcript_aborts(tmp_path: Path) -> None:
    raw, records = build_fixture(tmp_path)
    stratum, consultation = DEV_KEYS[0]
    (raw / "transcripts" / stratum / f"{consultation}.txt").unlink()
    dev_out, heldout_out = tmp_path / "dev", tmp_path / "heldout"
    _assert_aborted(_run(raw, records, dev_out, heldout_out), dev_out, heldout_out)


def test_non_empty_output_aborts(tmp_path: Path) -> None:
    raw, records = build_fixture(tmp_path)
    dev_out, heldout_out = tmp_path / "dev", tmp_path / "heldout"
    _write(dev_out / "old.txt", "earlier run")
    assert _run(raw, records, dev_out, heldout_out) == 1
    assert [p.name for p in dev_out.iterdir()] == ["old.txt"]
    assert not heldout_out.exists()


def test_output_inside_raw_aborts(tmp_path: Path) -> None:
    raw, records = build_fixture(tmp_path)
    dev_out, heldout_out = raw / "dev", tmp_path / "heldout"
    _assert_aborted(_run(raw, records, dev_out, heldout_out), dev_out, heldout_out)


def test_text_in_a_kept_score_field_aborts(tmp_path: Path) -> None:
    raw, records = build_fixture(tmp_path)
    path = raw / "judgements" / "w1" / "scored_records.json"
    data = json.loads(path.read_text())
    data["records"][0]["aggregate"] = "x" * 300
    _write(path, data)
    dev_out, heldout_out = tmp_path / "dev", tmp_path / "heldout"
    _assert_aborted(_run(raw, records, dev_out, heldout_out), dev_out, heldout_out)


def test_success_prints_development_breakdown(split: tuple[Path, Path, str]) -> None:
    dev_out, _, log = split
    dev = _pairs(dev_out)["pairs"]
    for name in ("type", "residual_level"):
        counts: dict[str, int] = {}
        for pair in dev:
            counts[pair[name]] = counts.get(pair[name], 0) + 1
        expected = ", ".join(f"{label} {count}" for label, count in sorted(counts.items()))
        assert f"development pairs by {name}: {expected}" in log
    assert sum(int(part.split()[-1]) for part in expected.split(", ")) == 265
