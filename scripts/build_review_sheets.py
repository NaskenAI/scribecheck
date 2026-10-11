"""Write the clinician review sheets in docs/review/ from the packaged data files.

Usage:
    python scripts/build_review_sheets.py [--output-dir docs/review]

Reads only src/scribecheck/data/inn_usan.json, symptoms.json and ingredients_extra.json,
so the sheets always match the data. The decision and clinician_comment columns are
left empty for the reviewer. Re-run after any change to those files; a test fails when
the committed sheets are stale.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "scribecheck" / "data"
DEFAULT_OUTPUT = ROOT / "docs" / "review"
REVIEW_COLUMNS = ["decision", "clinician_comment"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sheets = {
        "inn_usan_review.csv": (
            ["international_name", "us_name", "notes"],
            inn_usan_rows(_load("inn_usan.json")),
        ),
        "symptom_phrases_review.csv": (
            ["phrase", "maps_to_term", "symp_id", "kind", "notes"],
            symptom_rows(_load("symptoms.json")),
        ),
        "extra_ingredients_review.csv": (
            ["ingredient", "notes"],
            extra_ingredient_rows(_load("ingredients_extra.json")),
        ),
    }
    for name, (columns, rows) in sheets.items():
        (args.output_dir / name).write_bytes(_csv(columns + REVIEW_COLUMNS, rows))
    return 0


def _load(name: str) -> Any:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def inn_usan_rows(data: dict[str, Any]) -> list[list[str]]:
    return [
        [pair["international"], " + ".join(pair["us"]), pair.get("note", "")]
        for pair in data["pairs"]
    ]


def extra_ingredient_rows(data: dict[str, Any]) -> list[list[str]]:
    return [[entry["name"], entry["note"]] for entry in data["ingredients"]]


def symptom_rows(data: dict[str, Any]) -> list[list[str]]:
    """Lay phrases, withheld phrases, merged SYMP forms and excluded SYMP forms."""
    by_label = {term["label"]: term for term in data["terms"]}
    for term in data["terms"]:
        for merged in term["merged"]:
            by_label[merged["label"]] = merged
    rows = []
    for term in data["terms"]:
        for lay in term["lay"]:
            rows.append([lay["text"], term["label"], term["id"], "lay", ""])
    for item in data["withheld"]:
        rows.append([item["text"], item["term"], item["id"], "lay", item["note"]])
    for term in data["terms"]:
        for merged in term["merged"]:
            note = f"SYMP term {merged['label']} ({merged['id']}) folded into {term['label']}"
            for form in [merged["label"], *merged["synonyms"]]:
                rows.append([form, term["label"], merged["id"], "merge", note])
    for label, forms in data["excluded_forms"].items():
        for form in forms:
            kind = "label" if form.casefold() == label.casefold() else "exact synonym"
            note = f"SYMP {kind} not matched: {data['excluded_reasons'][form]}"
            rows.append([form, label, by_label[label]["id"], "excluded_synonym", note])
    return rows


def _csv(columns: list[str], rows: list[list[str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(row + [""] * len(REVIEW_COLUMNS))
    return buffer.getvalue().encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
