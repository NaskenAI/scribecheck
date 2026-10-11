"""Recompute the published OmissionBench numbers from released scored-records files.

Usage:
    python scripts/recompute_published.py <records-dir> --pairs <pairs/dataset_v2.json>

Gate 1(a): validates the scoring code against the benchmark's released judgements, with no
API calls. Every *.json under <records-dir> with a "records" array of rows carrying note_key
is one table row, labelled with its parent folder (for the split's flattened copies,
"a__b__c.json", the original parent "b"). Per file:

- Paired discrimination, as the dataset card defines it: each errored note is compared with
  its clean twin (the clean row with the same clean_key in the same file). A win is an
  errored aggregate strictly lower than the clean one; a tie counts half. Omission pairs
  and add-or-change pairs are reported separately, with each pair's type from --pairs.
- Detection rate: the share of errored omission notes flagged.
- Clean-twin flag rate: the share of distinct clean notes flagged.

Identical duplicate rows count once; a note whose duplicates disagree on aggregate or
flagged is left out and counted as a conflict. An errored note with no clean twin, a null
aggregate on either side, or a pair_id missing from --pairs is counted, not scored. A null
flagged leaves the note out of that rate's denominator. Only numbers and folder names are
printed, never any other field, and no network call is made.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ADD_OR_CHANGE = frozenset({"add", "change"})


@dataclass
class FileResult:
    """Counts for one scored-records file; rates are None when there is nothing to rate."""

    omit_n: int = 0
    omit_wins: int = 0
    omit_ties: int = 0
    change_n: int = 0
    change_wins: int = 0
    change_ties: int = 0
    detection_n: int = 0
    detection_flagged: int = 0
    clean_n: int = 0
    clean_flagged: int = 0
    no_twin: int = 0
    missing_score: int = 0
    unknown_type: int = 0
    conflicts: int = 0

    @property
    def omit_discrimination(self) -> float | None:
        return _discrimination(self.omit_wins, self.omit_ties, self.omit_n)

    @property
    def change_discrimination(self) -> float | None:
        return _discrimination(self.change_wins, self.change_ties, self.change_n)

    @property
    def detection_rate(self) -> float | None:
        return self.detection_flagged / self.detection_n if self.detection_n else None

    @property
    def clean_flag_rate(self) -> float | None:
        return self.clean_flagged / self.clean_n if self.clean_n else None


def _discrimination(wins: int, ties: int, n: int) -> float | None:
    return (wins + 0.5 * ties) / n if n else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("records_dir", type=Path)
    parser.add_argument("--pairs", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        types = load_types(args.pairs)
    except (OSError, ValueError) as error:
        print(f"recompute_published: cannot read --pairs: {type(error).__name__}", file=sys.stderr)
        return 1
    if not args.records_dir.is_dir():
        print("recompute_published: <records-dir> is not a folder", file=sys.stderr)
        return 1

    files, skipped = [], 0
    for path in sorted(args.records_dir.rglob("*.json")):
        rows = _records(path)
        if rows:
            files.append((path, rows))
        else:
            skipped += 1
    names = labels([path for path, _ in files])
    print_table(
        [(name, compute(rows, types)) for name, (_, rows) in zip(names, files, strict=True)]
    )
    print(f"\n{len(files)} file(s) scored, skipped {skipped} file(s) with no scored records")
    return 0


def load_types(path: Path) -> dict[str, str]:
    """pair_id -> type ("omit", "add" or "change") from pairs/dataset_v2.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("pairs"), list):
        raise ValueError("no top-level 'pairs' list")
    return {
        p["pair_id"]: p["type"]
        for p in data["pairs"]
        if isinstance(p, dict)
        and isinstance(p.get("pair_id"), str)
        and isinstance(p.get("type"), str)
    }


def _records(path: Path) -> list[dict[str, Any]]:
    """The file's scored rows, or [] when it is not a non-empty scored-records file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    rows = data.get("records") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        return []
    if not all(isinstance(row, dict) and "note_key" in row for row in rows):
        return []
    return rows


def labels(paths: Sequence[Path]) -> list[str]:
    """Parent folder names; the original parent for flattened "a__b__c.json" copies."""
    names = []
    for path in paths:
        parts = path.name.split("__")
        names.append(parts[-2] if len(parts) >= 3 else path.parent.name)
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return [
        f"{name}:{path.stem}" if counts[name] > 1 else name
        for name, path in zip(names, paths, strict=True)
    ]


def _distinct(rows: Iterable[dict[str, Any]], key: str) -> tuple[dict[str, dict[str, Any]], int]:
    """One row per key; keys whose rows disagree on aggregate or flagged are dropped."""
    seen: dict[str, dict[str, Any]] = {}
    conflicted: set[str] = set()
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str):
            continue
        if value in seen:
            first = seen[value]
            if (first.get("aggregate"), first.get("flagged")) != (
                row.get("aggregate"),
                row.get("flagged"),
            ):
                conflicted.add(value)
        else:
            seen[value] = row
    for value in conflicted:
        del seen[value]
    return seen, len(conflicted)


def _score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def compute(rows: Sequence[dict[str, Any]], types: Mapping[str, str]) -> FileResult:
    result = FileResult()
    clean_rows = [r for r in rows if r.get("note_role") == "clean" or r.get("pair_id") is None]
    errored_rows = [
        r for r in rows if r.get("note_role") != "clean" and r.get("pair_id") is not None
    ]
    clean, clean_conflicts = _distinct(clean_rows, "clean_key")
    errored, errored_conflicts = _distinct(errored_rows, "note_key")
    result.conflicts = clean_conflicts + errored_conflicts

    for note in clean.values():
        if isinstance(note.get("flagged"), bool):
            result.clean_n += 1
            result.clean_flagged += note["flagged"]

    for note in errored.values():
        kind = types.get(note["pair_id"])
        if kind is None:
            result.unknown_type += 1
            continue
        if kind == "omit" and isinstance(note.get("flagged"), bool):
            result.detection_n += 1
            result.detection_flagged += note["flagged"]
        twin = clean.get(note.get("clean_key"))  # type: ignore[arg-type]
        if twin is None:
            result.no_twin += 1
            continue
        errored_score, clean_score = _score(note.get("aggregate")), _score(twin.get("aggregate"))
        if errored_score is None or clean_score is None:
            result.missing_score += 1
            continue
        win, tie = errored_score < clean_score, errored_score == clean_score
        if kind == "omit":
            result.omit_n += 1
            result.omit_wins += win
            result.omit_ties += tie
        elif kind in ADD_OR_CHANGE:
            result.change_n += 1
            result.change_wins += win
            result.change_ties += tie
        else:
            result.unknown_type += 1
    return result


COLUMNS = (
    "file",
    "omit_n", "omit_win", "omit_tie", "omit_disc",
    "addchg_n", "addchg_win", "addchg_tie", "addchg_disc",
    "det_n", "det_flag", "det_rate",
    "clean_n", "clean_flag", "clean_rate",
    "no_twin", "no_score", "no_type", "conflicts",
)  # fmt: skip


def _rate(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def print_table(rows: Sequence[tuple[str, FileResult]]) -> None:
    cells = [list(COLUMNS)]
    for name, r in rows:
        cells.append(
            [
                name,
                str(r.omit_n), str(r.omit_wins), str(r.omit_ties), _rate(r.omit_discrimination),
                str(r.change_n), str(r.change_wins), str(r.change_ties),
                _rate(r.change_discrimination),
                str(r.detection_n), str(r.detection_flagged), _rate(r.detection_rate),
                str(r.clean_n), str(r.clean_flagged), _rate(r.clean_flag_rate),
                str(r.no_twin), str(r.missing_score), str(r.unknown_type), str(r.conflicts),
            ]
        )  # fmt: skip
    widths = [max(len(row[i]) for row in cells) for i in range(len(COLUMNS))]
    for row in cells:
        print(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        )


if __name__ == "__main__":
    raise SystemExit(main())
