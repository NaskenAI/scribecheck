"""The clinician review sheets are generated from the data files and kept up to date."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "scribecheck" / "data"
REVIEW = ROOT / "docs" / "review"
SHEETS = {
    "inn_usan_review.csv": [
        "international_name", "us_name", "notes", "decision", "clinician_comment"
    ],
    "symptom_phrases_review.csv": [
        "phrase", "maps_to_term", "symp_id", "kind", "notes", "decision", "clinician_comment"
    ],
    "extra_ingredients_review.csv": ["ingredient", "notes", "decision", "clinician_comment"],
}  # fmt: skip


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("review")
    script = ROOT / "scripts" / "build_review_sheets.py"
    subprocess.run([sys.executable, str(script), "--output-dir", str(out)], check=True)
    return out


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_committed_sheets_match_the_data(generated: Path, name: str) -> None:
    assert (REVIEW / name).read_bytes() == (generated / name).read_bytes(), (
        f"docs/review/{name} is stale: run python scripts/build_review_sheets.py"
    )


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_headers_and_empty_decision_columns(generated: Path, name: str) -> None:
    with (generated / name).open(encoding="utf-8", newline="") as handle:
        assert next(csv.reader(handle)) == SHEETS[name]
    for row in _rows(generated / name):
        assert row["decision"] == "" and row["clinician_comment"] == ""


def test_inn_sheet_lists_every_pair(generated: Path) -> None:
    pairs = json.loads((DATA / "inn_usan.json").read_text(encoding="utf-8"))["pairs"]
    rows = _rows(generated / "inn_usan_review.csv")
    assert [(r["international_name"], r["us_name"]) for r in rows] == [
        (p["international"], " + ".join(p["us"])) for p in pairs
    ]


def test_extra_ingredient_sheet_lists_every_name(generated: Path) -> None:
    extra = json.loads((DATA / "ingredients_extra.json").read_text(encoding="utf-8"))
    rows = _rows(generated / "extra_ingredients_review.csv")
    assert [r["ingredient"] for r in rows] == [e["name"] for e in extra["ingredients"]]
    assert all(r["notes"] for r in rows)


def test_symptom_sheet_covers_lay_merges_exclusions_and_withheld(generated: Path) -> None:
    symptoms = json.loads((DATA / "symptoms.json").read_text(encoding="utf-8"))
    rows = _rows(generated / "symptom_phrases_review.csv")
    keys = [(r["phrase"], r["kind"]) for r in rows]
    assert len(keys) == len(set(keys))
    expected = set()
    for term in symptoms["terms"]:
        expected |= {(lay["text"], "lay") for lay in term["lay"]}
        for merged in term["merged"]:
            expected |= {(merged["label"], "merge")}
            expected |= {(s, "merge") for s in merged["synonyms"]}
    for forms in symptoms["excluded_forms"].values():
        expected |= {(form, "excluded_synonym") for form in forms}
    expected |= {(w["text"], "lay") for w in symptoms["withheld"]}
    assert set(keys) == expected
    assert {r["kind"] for r in rows} == {"lay", "merge", "excluded_synonym"}
    (white,) = [r for r in rows if r["phrase"] == "white discharge"]
    assert white["maps_to_term"] == "vaginal discharge"
    assert white["notes"] == "maps to vaginal discharge; unsafe without the patient's sex"
    assert all(r["symp_id"].startswith("SYMP:") for r in rows)
