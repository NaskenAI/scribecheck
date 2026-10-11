"""Recompute the published OmissionBench numbers from released scored-records files.

Usage:
    python scripts/recompute_published.py <records-dir> --pairs <pairs/dataset_v2.json>

Gate 1(a): validates the scoring code against the benchmark's released judgements, with no
API calls. Every *.json under <records-dir> with a "records" array of rows carrying note_key
is read; it is labelled with its parent folder (for the split's flattened copies,
"a__b__c.json", the original parent "b").

Rows with parse_failure true are left out and counted. The rest are grouped by file and
GROUP_FIELDS (model, arm, cell, criterion, format, k, tier, reasoning effort, dataset
version); a missing field is its own value, so groups never merge by accident. Within a
group every metric is computed per replicate:

- Paired discrimination, as the dataset card defines it: each errored note is compared with
  its clean twin (the clean row with the same clean_key in the same group and replicate). A
  win is an errored aggregate strictly lower than the clean one; a tie counts half. Omission
  and add-or-change pairs are reported separately. A pair's type comes from --pairs, or from
  the row's pair_type when its pair_id is not there.
- Detection rate: the share of errored omission notes flagged.
- Clean-twin flag rate: the share of distinct clean notes flagged.

Replicates are then combined as the dataset card does: discrimination is the mean over
replicates (each replicate's value is listed too), and detection and clean flag rates get a
strict-majority result, where a note counts as flagged if it is flagged in more than half of
the replicates whose flag for it is known.

Within one replicate, identical duplicate rows count once; a note whose duplicates disagree
on aggregate or flagged is left out and counted as a conflict. An errored note with no clean
twin, a null aggregate on either side, or an unknown type is counted, not scored. Groups
from smoke runs or the frozen master list (NOT_COMPARABLE) are printed in a separate section.
Only numbers, folder names and short group values are printed, never any other field, and
no network call is made.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ADD_OR_CHANGE = frozenset({"add", "change"})
PAIR_TYPES = frozenset({"omit", "add", "change"})
GROUP_FIELDS = (
    "model",
    "arm",
    "cell",
    "criterion",
    "format",
    "k",
    "tier",
    "reasoning_effort",
    "dataset_version",
)
NOT_COMPARABLE = ("armsmoke", "pairs_master_frozen")
MISSING = "(missing)"
MAX_VALUE = 60


@dataclass
class ReplicateResult:
    """Counts for one replicate of one group; rates are None when there is nothing to rate."""

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
    types_from_pairs: int = 0
    types_from_rows: int = 0
    omission_flags: dict[str, bool] = field(default_factory=dict)
    """Known flags of errored omission notes, by note_key, for the strict majority."""
    clean_flags: dict[str, bool] = field(default_factory=dict)
    """Known flags of clean notes, by clean_key, for the strict majority."""

    @property
    def omit_discrimination(self) -> float | None:
        return _discrimination(self.omit_wins, self.omit_ties, self.omit_n)

    @property
    def change_discrimination(self) -> float | None:
        return _discrimination(self.change_wins, self.change_ties, self.change_n)

    @property
    def detection_rate(self) -> float | None:
        return _ratio(self.detection_flagged, self.detection_n)

    @property
    def clean_flag_rate(self) -> float | None:
        return _ratio(self.clean_flagged, self.clean_n)


FileResult = ReplicateResult
"""The earlier name, for one ungrouped set of rows."""


@dataclass
class GroupResult:
    """One group (file and run settings), per replicate and combined."""

    label: str
    key_values: dict[str, str]
    rows: int = 0
    parse_failures: int = 0
    replicates: dict[str, ReplicateResult] = field(default_factory=dict)
    majority_detection_n: int = 0
    majority_detection_flagged: int = 0
    majority_clean_n: int = 0
    majority_clean_flagged: int = 0

    @property
    def comparable(self) -> bool:
        version = self.key_values["dataset_version"]
        return not any(marker in version for marker in NOT_COMPARABLE)

    @property
    def omit_mean(self) -> float | None:
        return _mean(r.omit_discrimination for r in self.replicates.values())

    @property
    def change_mean(self) -> float | None:
        return _mean(r.change_discrimination for r in self.replicates.values())

    @property
    def majority_detection_rate(self) -> float | None:
        return _ratio(self.majority_detection_flagged, self.majority_detection_n)

    @property
    def majority_clean_rate(self) -> float | None:
        return _ratio(self.majority_clean_flagged, self.majority_clean_n)


def _discrimination(wins: int, ties: int, n: int) -> float | None:
    return (wins + 0.5 * ties) / n if n else None


def _ratio(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _mean(values: Iterable[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


def strict_majority(flags: Sequence[bool]) -> bool:
    """Flagged in more than half of the replicates whose flag is known."""
    return sum(flags) * 2 > len(flags)


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
    groups = [
        group
        for name, (_, rows) in zip(names, files, strict=True)
        for group in analyse(name, rows, types)
    ]
    print_report(groups, scored=len(files), skipped=skipped)
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


def _value(row: Mapping[str, Any], name: str) -> str:
    """A grouping value: missing and null fields stay distinct from every real value."""
    if name not in row:
        return MISSING
    value = row[name]
    return "(null)" if value is None else str(value)


def _replicate_order(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def analyse(
    label: str, rows: Sequence[dict[str, Any]], types: Mapping[str, str]
) -> list[GroupResult]:
    """Group a file's rows, then score each replicate and combine the replicates."""
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    failures: dict[tuple[str, ...], int] = {}
    for row in rows:
        key = tuple(_value(row, name) for name in GROUP_FIELDS)
        grouped.setdefault(key, [])
        if row.get("parse_failure") is True:
            failures[key] = failures.get(key, 0) + 1
        else:
            grouped[key].append(row)

    results = []
    for key, members in grouped.items():
        group = GroupResult(label, dict(zip(GROUP_FIELDS, key, strict=True)))
        group.rows, group.parse_failures = len(members), failures.get(key, 0)
        by_replicate: dict[str, list[dict[str, Any]]] = {}
        for row in members:
            by_replicate.setdefault(_value(row, "replicate"), []).append(row)
        for replicate in sorted(by_replicate, key=_replicate_order):
            group.replicates[replicate] = compute(by_replicate[replicate], types)
        _combine(group)
        results.append(group)
    return results


def _combine(group: GroupResult) -> None:
    """Strict-majority detection and clean flag rates across the group's replicates."""
    omission: dict[str, list[bool]] = {}
    clean: dict[str, list[bool]] = {}
    for replicate in group.replicates.values():
        for note, flag in replicate.omission_flags.items():
            omission.setdefault(note, []).append(flag)
        for note, flag in replicate.clean_flags.items():
            clean.setdefault(note, []).append(flag)
    group.majority_detection_n = len(omission)
    group.majority_detection_flagged = sum(strict_majority(f) for f in omission.values())
    group.majority_clean_n = len(clean)
    group.majority_clean_flagged = sum(strict_majority(f) for f in clean.values())


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


def compute(rows: Sequence[dict[str, Any]], types: Mapping[str, str]) -> ReplicateResult:
    """Score one replicate of one group (or any single set of rows)."""
    result = ReplicateResult()
    clean_rows = [r for r in rows if r.get("note_role") == "clean" or r.get("pair_id") is None]
    errored_rows = [
        r for r in rows if r.get("note_role") != "clean" and r.get("pair_id") is not None
    ]
    clean, clean_conflicts = _distinct(clean_rows, "clean_key")
    errored, errored_conflicts = _distinct(errored_rows, "note_key")
    result.conflicts = clean_conflicts + errored_conflicts

    for clean_key, note in clean.items():
        if isinstance(note.get("flagged"), bool):
            result.clean_n += 1
            result.clean_flagged += note["flagged"]
            result.clean_flags[clean_key] = note["flagged"]

    for note_key, note in errored.items():
        kind = types.get(note["pair_id"])
        if kind is not None:
            result.types_from_pairs += 1
        elif note.get("pair_type") in PAIR_TYPES:
            kind = note["pair_type"]
            result.types_from_rows += 1
        else:
            result.unknown_type += 1
            continue
        if kind == "omit" and isinstance(note.get("flagged"), bool):
            result.detection_n += 1
            result.detection_flagged += note["flagged"]
            result.omission_flags[note_key] = note["flagged"]
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
    return result


# Output


def _rate(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def _short(value: str) -> str:
    return value if len(value) <= MAX_VALUE else value[: MAX_VALUE - 3] + "..."


def _discrimination_line(
    name: str, mean: float | None, replicates: Mapping[str, ReplicateResult], omit: bool
) -> str:
    parts = [f"{name}: mean {_rate(mean)}"]
    for replicate, r in replicates.items():
        n, wins, ties = (
            (r.omit_n, r.omit_wins, r.omit_ties)
            if omit
            else (
                r.change_n,
                r.change_wins,
                r.change_ties,
            )
        )
        value = r.omit_discrimination if omit else r.change_discrimination
        parts.append(f"r{replicate} {_rate(value)} (n {n}, wins {wins}, ties {ties})")
    return " | ".join(parts)


def _rate_line(name: str, pairs: Sequence[tuple[str, int, int]], majority: tuple[int, int]) -> str:
    parts = [f"r{rep} {flagged}/{n} {_rate(_ratio(flagged, n))}" for rep, flagged, n in pairs]
    flagged, n = majority
    parts.append(f"majority {flagged}/{n} {_rate(_ratio(flagged, n))}")
    return f"{name}: " + " | ".join(parts)


def format_group(group: GroupResult) -> str:
    settings = " ".join(
        f"{'effort' if k == 'reasoning_effort' else 'dataset' if k == 'dataset_version' else k}="
        f"{_short(v)}"
        for k, v in group.key_values.items()
    )
    reps = group.replicates
    total = ReplicateResult()
    for r in reps.values():
        total.no_twin += r.no_twin
        total.missing_score += r.missing_score
        total.unknown_type += r.unknown_type
        total.conflicts += r.conflicts
        total.types_from_pairs += r.types_from_pairs
        total.types_from_rows += r.types_from_rows
    lines = [
        f"[{group.label}] {settings}",
        f"  rows {group.rows}, parse failures excluded {group.parse_failures}, "
        f"replicates {' '.join(reps)}, types from --pairs {total.types_from_pairs} "
        f"/ from rows {total.types_from_rows}",
        "  " + _discrimination_line("omission discrimination", group.omit_mean, reps, True),
        "  " + _discrimination_line("add/change discrimination", group.change_mean, reps, False),
        "  "
        + _rate_line(
            "detection",
            [(k, r.detection_flagged, r.detection_n) for k, r in reps.items()],
            (group.majority_detection_flagged, group.majority_detection_n),
        ),
        "  "
        + _rate_line(
            "clean flag rate",
            [(k, r.clean_flagged, r.clean_n) for k, r in reps.items()],
            (group.majority_clean_flagged, group.majority_clean_n),
        ),
        f"  not scored: no twin {total.no_twin}, missing score {total.missing_score}, "
        f"unknown type {total.unknown_type}, conflicts {total.conflicts}",
    ]
    return "\n".join(lines)


def print_report(groups: Sequence[GroupResult], *, scored: int, skipped: int) -> None:
    comparable = [g for g in groups if g.comparable]
    other = [g for g in groups if not g.comparable]
    for group in comparable:
        print(format_group(group) + "\n")
    if other:
        print("== Not comparable to published figures (smoke runs or the frozen master list) ==\n")
        for group in other:
            print(format_group(group) + "\n")
    print(
        f"{scored} file(s) scored, skipped {skipped} file(s) with no scored records; "
        f"{len(comparable)} comparable group(s), {len(other)} not comparable"
    )
    print(f"parse failures excluded in total: {sum(g.parse_failures for g in groups)}")


if __name__ == "__main__":
    raise SystemExit(main())
