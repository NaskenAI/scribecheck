"""Recompute the published OmissionBench numbers from released scored-records files.

Usage:
    python scripts/recompute_published.py <records-dir> --pairs <pairs/dataset_v2.json> [--summary]

Gate 1(a): validates the scoring code against the benchmark's released judgements, with no
API calls. Every *.json under <records-dir> with a "records" array of rows carrying note_key
is read. Each released file is one shard of a run, and a run is spread over many folders of
one family, judgements/judges/<family>/<YYYYMMDD-HHMMSS>-<commit>-r<N>/ (or, in the split's
flattened copies, "judgements__judges__<family>__<folder>__<file>.json"). A file's family
is that directory; a file outside judges/ takes its parent folder as its family.

Rows are grouped across files by family and GROUP_FIELDS (model, arm, cell, criterion,
format, k, tier, reasoning effort, dataset version), ignoring the folder's timestamp and
commit; a missing field is its own value, so groups never merge by accident. Within a group
and replicate, a note_key found in folders with different timestamps keeps only the rows
from the latest folder, and the others are counted as replaced; when any of its folders has
no timestamp, all its rows stay. Coverage (the distinct errored and clean notes of each
replicate) is counted next, then rows with parse_failure true are left out and counted.
A group is "complete" when every replicate covers COVERAGE for its dataset version,
"development complete" when every replicate covers exactly DEVELOPMENT_COVERAGE (never
counted as complete), and "partial coverage" otherwise. Within a group every metric is
computed per replicate:

- Paired discrimination, as the dataset card defines it: each errored note is compared with
  its clean twin (the clean row with the same clean_key in the same group and replicate). A
  win is an errored aggregate strictly lower than the clean one; a tie counts half. Omission
  and add-or-change pairs are reported separately. A pair's type comes from --pairs, or from
  the row's pair_type when its pair_id is not there.
- Detection rate: the share of errored omission notes flagged.
- Clean-twin flag rate: the share of distinct clean notes flagged.

Two published operating points are computed the same way, per replicate and as both
majorities below, with a parse failure counting as not flagged on the full denominator
(record/figures/extract_data.py:186-189 in the harness). They are named rules:

- route-two-below-10: flag a note when its aggregate is below 10 (FLAG_BELOW; the harness's
  w2_evalfull_analyze.py:44), for groups whose scores reach 10.
- pipeline-critical-absent: flag a note when any detail.missing_facts entry is "critical"
  and "absent" (w2_pipeline_replicates_analyze.py:99-108); without detail, when the split's
  n_absent_critical count is above 0. Only groups with either input get this rule.

Detection at matched false alarms follows the harness's sweep (extract_data.py:106-135),
per replicate, over the omission and clean notes that parsed and have a score: the
candidate thresholds are every observed score v and v + 1e-6, a note is flagged when its
score is strictly below the threshold, and for each k in MATCHED_K the chosen point has the
highest detection with at most k clean notes flagged, then the fewest clean notes flagged,
then the larger threshold. A tie that keeps a count from being reached exactly therefore
falls back to the best point with fewer. Scores with two or fewer distinct values give no
point. Across replicates each replicate keeps its own threshold, and the strict majority
treats a replicate with no point as not flagging.

Replicates are then combined as the dataset card does: discrimination is the mean over
replicates (each replicate's value is listed too), and detection and clean flag rates get a
strict-majority result, where a note counts as flagged if it is flagged in more than half of
the group's replicates; a replicate in which the note failed to parse, or has no known flag,
counts as not flagged, and a note seen only in failed rows still counts. Beside it is the
"known-only majority": more than half of the replicates whose flag for the note is known,
over the notes with at least one known flag.

Within one replicate, identical duplicate rows count once; a note whose duplicates disagree
on aggregate or flagged is left out and counted as a conflict. An errored note with no clean
twin, a null aggregate on either side, or an unknown type is counted, not scored. Groups
with no dataset version, or from smoke runs or the frozen master list (NOT_COMPARABLE), are
printed in a separate section. --summary prints one line per complete group instead (with
the matched points and rules indented below it), then the development-complete groups
under their own heading, then the comparable groups with partial coverage and their
coverage counts. Only numbers, folder names and
short group values are printed, never any other field, and no network call is made.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
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
NULL = "(null)"
MAX_VALUE = 60
# (errored notes, clean notes) each replicate must cover, by dataset version.
COVERAGE = {"factorial-v1": (495, 112), "factorial-v1-armconfirm": (151, 47)}
DEVELOPMENT_COVERAGE = {"factorial-v1": (265, 65)}
COMPLETE = "complete"
DEVELOPMENT_COMPLETE = "development complete"
PARTIAL = "partial coverage"
FLAG_BELOW = 10.0
MATCHED_K = (1, 2, 3)
FOLDER_TIMESTAMP = re.compile(r"(\d{8}-\d{6})-")


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


Counts = tuple[int, int, int, int]
"""(omission notes flagged, omission notes, clean notes flagged, clean notes)."""


@dataclass(frozen=True)
class OperatingPoint:
    """One replicate's threshold and the counts it gives."""

    threshold: float
    det_k: int
    n_err: int
    fa_k: int
    n_clean: int


@dataclass
class MatchedResult:
    """Detection at one matched false-alarm count: per replicate, and the strict majority."""

    per_replicate: dict[str, OperatingPoint | None] = field(default_factory=dict)
    majority: Counts = (0, 0, 0, 0)


@dataclass
class RuleResult:
    """One named flag rule: per replicate, and both majority readings."""

    per_replicate: dict[str, Counts] = field(default_factory=dict)
    majority: Counts = (0, 0, 0, 0)
    known_majority: Counts = (0, 0, 0, 0)


@dataclass
class GroupResult:
    """One group (file and run settings), per replicate and combined."""

    label: str
    """The family."""
    key_values: dict[str, str]
    files: int = 0
    rows: int = 0
    parse_failures: int = 0
    replaced: int = 0
    """Rows replaced by the same note_key in a later shard of the same replicate."""
    coverage: dict[str, tuple[int, int]] = field(default_factory=dict)
    """(distinct errored notes, distinct clean notes) per replicate, before parse failures."""
    replicates: dict[str, ReplicateResult] = field(default_factory=dict)
    matched: dict[int, MatchedResult] = field(default_factory=dict)
    """Detection at matched false alarms, by k."""
    rules: dict[str, RuleResult] = field(default_factory=dict)
    """The named flag rules that apply to the group."""
    replicate_count: int = 0
    """Distinct replicates in the group, counting those whose rows all failed to parse."""
    majority_detection_n: int = 0
    majority_detection_flagged: int = 0
    majority_clean_n: int = 0
    majority_clean_flagged: int = 0
    known_majority_detection_n: int = 0
    known_majority_detection_flagged: int = 0
    known_majority_clean_n: int = 0
    known_majority_clean_flagged: int = 0

    @property
    def comparable(self) -> bool:
        version = self.key_values["dataset_version"]
        if version in (MISSING, NULL):
            return False
        return not any(marker in version for marker in NOT_COMPARABLE)

    @property
    def coverage_label(self) -> str:
        version, covered = self.key_values["dataset_version"], list(self.coverage.values())
        if not covered or not self.comparable:
            return PARTIAL
        if version in COVERAGE:
            errored, clean = COVERAGE[version]
            if all(e >= errored and c >= clean for e, c in covered):
                return COMPLETE
        if version in DEVELOPMENT_COVERAGE:
            if all(c == DEVELOPMENT_COVERAGE[version] for c in covered):
                return DEVELOPMENT_COMPLETE
        return PARTIAL

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

    @property
    def known_majority_detection_rate(self) -> float | None:
        return _ratio(self.known_majority_detection_flagged, self.known_majority_detection_n)

    @property
    def known_majority_clean_rate(self) -> float | None:
        return _ratio(self.known_majority_clean_flagged, self.known_majority_clean_n)


def _discrimination(wins: int, ties: int, n: int) -> float | None:
    return (wins + 0.5 * ties) / n if n else None


def _ratio(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _mean(values: Iterable[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


def strict_majority(flags: Sequence[bool], replicates: int | None = None) -> bool:
    """Flagged in more than half of the replicates.

    With `replicates`, the count is over that many replicates and any without a known flag
    count as not flagged; without it, over the replicates whose flag is known.
    """
    total = len(flags) if replicates is None else replicates
    if total < len(flags):
        raise ValueError("more known flags than replicates")
    return sum(flags) * 2 > total


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("records_dir", type=Path)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument(
        "--summary", action="store_true", help="one line per complete group, numbers only"
    )
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
    groups = analyse_files([(source(path), rows) for path, rows in files], types)
    if args.summary:
        print_summary(groups, scored=len(files), skipped=skipped)
    else:
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


@dataclass(frozen=True)
class Source:
    """Where a file sits: its family, and its folder's timestamp when the name has one."""

    family: str
    timestamp: str | None


def source(path: Path) -> Source:
    """judges/<family>/<folder>/<file>, in a real path or a flattened "a__b__c.json" name."""
    parts = [*path.parent.parts, *path.name.split("__")]
    judges = [i for i, part in enumerate(parts) if part == "judges" and i + 3 < len(parts)]
    if not judges:
        return Source(parts[-2] if len(parts) >= 2 else path.stem, None)
    family, folder = parts[judges[-1] + 1], parts[judges[-1] + 2]
    match = FOLDER_TIMESTAMP.match(folder)
    return Source(family, match.group(1) if match else None)


def _value(row: Mapping[str, Any], name: str) -> str:
    """A grouping value: missing and null fields stay distinct from every real value."""
    if name not in row:
        return MISSING
    value = row[name]
    return NULL if value is None else str(value)


def _replicate_order(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def analyse(
    label: str, rows: Sequence[dict[str, Any]], types: Mapping[str, str]
) -> list[GroupResult]:
    """Group one set of rows, as if it were a single file of family `label`."""
    return analyse_files([(Source(label, None), rows)], types)


def analyse_files(
    files: Sequence[tuple[Source, Sequence[dict[str, Any]]]], types: Mapping[str, str]
) -> list[GroupResult]:
    """Group rows across files, then score each replicate and combine the replicates."""
    Tagged = tuple[int, str | None, dict[str, Any]]  # (file index, folder timestamp, row)
    grouped: dict[tuple[str, ...], dict[str, list[Tagged]]] = {}
    for index, (where, rows) in enumerate(files):
        for row in rows:
            key = (where.family, *(_value(row, name) for name in GROUP_FIELDS))
            replicates = grouped.setdefault(key, {})
            replicates.setdefault(_value(row, "replicate"), []).append(
                (index, where.timestamp, row)
            )

    results = []
    for key, replicates in grouped.items():
        group = GroupResult(key[0], dict(zip(GROUP_FIELDS, key[1:], strict=True)))
        group.files = len({index for tagged in replicates.values() for index, _, _ in tagged})
        group.replicate_count = len(replicates)
        failed: list[dict[str, Any]] = []
        notes: dict[str, Notes] = {}
        for replicate in sorted(replicates, key=_replicate_order):
            rows, replaced = _latest_shards(replicates[replicate])
            group.replaced += replaced
            group.coverage[replicate] = _coverage(rows)
            notes[replicate] = _replicate_notes(rows, types)
            kept = [row for row in rows if row.get("parse_failure") is not True]
            failed += [row for row in rows if row.get("parse_failure") is True]
            group.rows += len(kept)
            if kept:
                group.replicates[replicate] = compute(kept, types)
        group.parse_failures = len(failed)
        _combine(group, failed, types)
        _operating_points(group, notes)
        results.append(group)
    return results


Notes = tuple[dict[str, "dict[str, Any] | None"], dict[str, "dict[str, Any] | None"]]
"""One replicate's omission notes (by note_key) and clean notes (by clean_key), each with its
row, or None when the note's duplicate rows disagree."""


def _replicate_notes(rows: Iterable[dict[str, Any]], types: Mapping[str, str]) -> Notes:
    omission: dict[str, dict[str, Any] | None] = {}
    clean: dict[str, dict[str, Any] | None] = {}
    for row in rows:
        kind = _note_kind(row, types)
        if kind is None:
            continue
        side = clean if kind[0] == "clean" else omission
        if kind[1] in side and side[kind[1]] != row:
            side[kind[1]] = None
        else:
            side.setdefault(kind[1], row)
    return omission, clean


def _score_of(row: Mapping[str, Any] | None) -> float | None:
    """The row's aggregate; None for a parse failure, a missing row or a non-number."""
    if row is None or row.get("parse_failure") is True:
        return None
    return _score(row.get("aggregate"))


def matched_threshold(
    err_scores: Sequence[float], clean_scores: Sequence[float], k: int
) -> OperatingPoint | None:
    """The harness's best point with at most k clean notes flagged (see the module notes)."""
    values = sorted({round(v, 9) for v in [*err_scores, *clean_scores]})
    if not err_scores or not clean_scores or len(values) <= 2:
        return None
    best: tuple[int, int, float] | None = None
    for value in values:
        for threshold in (value, value + 1e-6):
            fa_k = sum(1 for v in clean_scores if v < threshold)
            if fa_k > k:
                continue
            key = (sum(1 for v in err_scores if v < threshold), -fa_k, threshold)
            if best is None or key > best:
                best = key
    if best is None:
        return None
    det_k, minus_fa, threshold = best
    return OperatingPoint(round(threshold, 6), det_k, len(err_scores), -minus_fa, len(clean_scores))


def below_10(row: Mapping[str, Any]) -> bool | None:
    score = _score_of(row)
    return None if score is None else score < FLAG_BELOW


def critical_absent(row: Mapping[str, Any]) -> bool | None:
    """Any detail.missing_facts entry critical and absent; else the split's count; else None."""
    detail = row.get("detail")
    facts = detail.get("missing_facts") if isinstance(detail, Mapping) else None
    if isinstance(facts, list):
        return any(
            isinstance(f, Mapping)
            and f.get("severity") == "critical"
            and f.get("verdict") == "absent"
            for f in facts
        )
    count = row.get("n_absent_critical")
    if isinstance(count, int) and not isinstance(count, bool):
        return count > 0
    return None


RULES: dict[str, Callable[[Mapping[str, Any]], bool | None]] = {
    "route-two-below-10": below_10,
    "pipeline-critical-absent": critical_absent,
}


def _rule_applies(name: str, notes: Mapping[str, Notes]) -> bool:
    rows = [r for o, c in notes.values() for r in [*o.values(), *c.values()] if r is not None]
    if name == "route-two-below-10":
        return any((s := _score_of(r)) is not None and s >= FLAG_BELOW for r in rows)
    return any(r.get("parse_failure") is not True and RULES[name](r) is not None for r in rows)


FlagOf = Callable[[str, Mapping[str, Any] | None], bool | None]
"""A note's flag in one replicate, from the replicate and the note's row; None if unknown."""


def _apply(
    notes: Mapping[str, Notes],
    flag_of: FlagOf,
    replicates: int,
) -> RuleResult:
    """Per-replicate counts (unknown = not flagged, full denominator) and both majorities."""
    result = RuleResult()
    votes: tuple[dict[str, list[bool]], dict[str, list[bool]]] = ({}, {})
    for replicate, sides in notes.items():
        counts: list[int] = []
        for side, side_votes in zip(sides, votes, strict=True):
            flags = {key: flag_of(replicate, row) for key, row in side.items()}
            counts += [sum(f is True for f in flags.values()), len(flags)]
            for key, flag in flags.items():
                known = side_votes.setdefault(key, [])
                if flag is not None:
                    known.append(flag)
        result.per_replicate[replicate] = (counts[0], counts[1], counts[2], counts[3])
    om, cl = votes
    result.majority = (
        sum(strict_majority(v, replicates) for v in om.values()),
        len(om),
        sum(strict_majority(v, replicates) for v in cl.values()),
        len(cl),
    )
    om_known, cl_known = [v for v in om.values() if v], [v for v in cl.values() if v]
    result.known_majority = (
        sum(strict_majority(v) for v in om_known),
        len(om_known),
        sum(strict_majority(v) for v in cl_known),
        len(cl_known),
    )
    return result


def _below_threshold(points: Mapping[str, OperatingPoint | None]) -> FlagOf:
    def flag_of(replicate: str, row: Mapping[str, Any] | None) -> bool | None:
        point, score = points[replicate], _score_of(row)
        return None if point is None or score is None else score < point.threshold

    return flag_of


def _by_rule(rule: Callable[[Mapping[str, Any]], bool | None]) -> FlagOf:
    def flag_of(_: str, row: Mapping[str, Any] | None) -> bool | None:
        if row is None or row.get("parse_failure") is True:
            return None
        return rule(row)

    return flag_of


def _operating_points(group: GroupResult, notes: Mapping[str, Notes]) -> None:
    """Matched false-alarm points and the named rules that apply to the group."""
    for k in MATCHED_K:
        points = {
            replicate: matched_threshold(
                [s for r in omission.values() if (s := _score_of(r)) is not None],
                [s for r in clean.values() if (s := _score_of(r)) is not None],
                k,
            )
            for replicate, (omission, clean) in notes.items()
        }
        majority = _apply(notes, _below_threshold(points), group.replicate_count).majority
        group.matched[k] = MatchedResult(points, majority)
    for name, rule in RULES.items():
        if _rule_applies(name, notes):
            group.rules[name] = _apply(notes, _by_rule(rule), group.replicate_count)


def _latest_shards(
    tagged: Sequence[tuple[int, str | None, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], int]:
    """The rows left once each note_key keeps only its latest folder's rows; and how many went."""
    stamps: dict[str, set[str | None]] = {}
    for _, timestamp, row in tagged:
        if isinstance(row.get("note_key"), str):
            stamps.setdefault(row["note_key"], set()).add(timestamp)
    kept, replaced = [], 0
    for _, timestamp, row in tagged:
        seen = stamps.get(row.get("note_key"))  # type: ignore[arg-type]
        if seen and None not in seen and timestamp != max(seen):  # type: ignore[type-var]
            replaced += 1
        else:
            kept.append(row)
    return kept, replaced


def _is_clean(row: Mapping[str, Any]) -> bool:
    return row.get("note_role") == "clean" or row.get("pair_id") is None


def _coverage(rows: Iterable[Mapping[str, Any]]) -> tuple[int, int]:
    """(distinct errored note_keys, distinct clean clean_keys)."""
    errored, clean = set(), set()
    for row in rows:
        if _is_clean(row):
            if isinstance(row.get("clean_key"), str):
                clean.add(row["clean_key"])
        elif isinstance(row.get("note_key"), str):
            errored.add(row["note_key"])
    return len(errored), len(clean)


def _note_kind(row: Mapping[str, Any], types: Mapping[str, str]) -> tuple[str, str] | None:
    """("clean", clean_key) or ("omit", note_key) for a row, as compute() classes it."""
    if _is_clean(row):
        key = row.get("clean_key")
        return ("clean", key) if isinstance(key, str) else None
    kind = types.get(row["pair_id"])
    if kind is None and row.get("pair_type") in PAIR_TYPES:
        kind = row["pair_type"]
    key = row.get("note_key")
    return ("omit", key) if kind == "omit" and isinstance(key, str) else None


def _combine(
    group: GroupResult, failed: Sequence[Mapping[str, Any]], types: Mapping[str, str]
) -> None:
    """Strict-majority detection and clean flag rates across the group's replicates."""
    omission: dict[str, list[bool]] = {}
    clean: dict[str, list[bool]] = {}
    for replicate in group.replicates.values():
        for note, flag in replicate.omission_flags.items():
            omission.setdefault(note, []).append(flag)
        for note, flag in replicate.clean_flags.items():
            clean.setdefault(note, []).append(flag)
    group.known_majority_detection_n = len(omission)
    group.known_majority_detection_flagged = sum(strict_majority(f) for f in omission.values())
    group.known_majority_clean_n = len(clean)
    group.known_majority_clean_flagged = sum(strict_majority(f) for f in clean.values())

    # Notes seen only in failed rows join the primary count, never flagged.
    every = {"omit": dict(omission), "clean": dict(clean)}
    for row in failed:
        kind = _note_kind(row, types)
        if kind is not None:
            every[kind[0]].setdefault(kind[1], [])
    n = group.replicate_count
    group.majority_detection_n = len(every["omit"])
    group.majority_detection_flagged = sum(strict_majority(f, n) for f in every["omit"].values())
    group.majority_clean_n = len(every["clean"])
    group.majority_clean_flagged = sum(strict_majority(f, n) for f in every["clean"].values())


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
    clean_rows = [r for r in rows if _is_clean(r)]
    errored_rows = [r for r in rows if not _is_clean(r)]
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


def _rate_line(
    name: str,
    pairs: Sequence[tuple[str, int, int]],
    majority: tuple[int, int],
    known_majority: tuple[int, int],
) -> str:
    parts = [f"r{rep} {flagged}/{n} {_rate(_ratio(flagged, n))}" for rep, flagged, n in pairs]
    for label, (flagged, n) in (("majority", majority), ("known-only majority", known_majority)):
        parts.append(f"{label} {flagged}/{n} {_rate(_ratio(flagged, n))}")
    return f"{name}: " + " | ".join(parts)


def _settings(group: GroupResult, skip: tuple[str, ...] = ()) -> str:
    return " ".join(
        f"{'effort' if k == 'reasoning_effort' else 'dataset' if k == 'dataset_version' else k}="
        f"{_short(v)}"
        for k, v in group.key_values.items()
        if k not in skip
    )


def _coverage_text(group: GroupResult) -> str:
    return " | ".join(
        f"r{rep} errored {errored} clean {clean}"
        for rep, (errored, clean) in group.coverage.items()
    )


def _counts(counts: Counts) -> str:
    return f"det {counts[0]}/{counts[1]} fa {counts[2]}/{counts[3]}"


def _threshold(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def operating_lines(group: GroupResult) -> list[str]:
    """The matched false-alarm points and named rules, one line each, numbers only."""
    lines = []
    for k, matched in group.matched.items():
        parts = [
            f"r{rep} –"
            if point is None
            else f"r{rep} det {point.det_k}/{point.n_err} fa {point.fa_k}/{point.n_clean} "
            f"(t {_threshold(point.threshold)})"
            for rep, point in matched.per_replicate.items()
        ]
        reached = any(point is not None for point in matched.per_replicate.values())
        parts.append(f"majority {_counts(matched.majority)}" if reached else "majority –")
        lines.append(f"matched false alarms k{k}: " + " | ".join(parts))
    for name, rule in group.rules.items():
        parts = [f"r{rep} {_counts(c)}" for rep, c in rule.per_replicate.items()]
        parts += [f"majority {_counts(rule.majority)}"]
        parts += [f"known-only majority {_counts(rule.known_majority)}"]
        lines.append(f"rule {name}: " + " | ".join(parts))
    return lines


def format_group(group: GroupResult) -> str:
    settings = _settings(group)
    reps = group.replicates
    total = ReplicateResult()
    for r in reps.values():
        total.no_twin += r.no_twin
        total.missing_score += r.missing_score
        total.unknown_type += r.unknown_type
        total.conflicts += r.conflicts
        total.types_from_pairs += r.types_from_pairs
        total.types_from_rows += r.types_from_rows
    coverage = _coverage_text(group)
    lines = [
        f"[{group.label}] {settings} — {group.coverage_label}",
        f"  files {group.files}, replaced by later shards {group.replaced}, coverage: {coverage}",
        f"  rows {group.rows}, parse failures excluded {group.parse_failures}, "
        f"replicates {' '.join(reps)} (of {group.replicate_count}), "
        f"types from --pairs {total.types_from_pairs} "
        f"/ from rows {total.types_from_rows}",
        "  " + _discrimination_line("omission discrimination", group.omit_mean, reps, True),
        "  " + _discrimination_line("add/change discrimination", group.change_mean, reps, False),
        "  "
        + _rate_line(
            "detection",
            [(k, r.detection_flagged, r.detection_n) for k, r in reps.items()],
            (group.majority_detection_flagged, group.majority_detection_n),
            (group.known_majority_detection_flagged, group.known_majority_detection_n),
        ),
        "  "
        + _rate_line(
            "clean flag rate",
            [(k, r.clean_flagged, r.clean_n) for k, r in reps.items()],
            (group.majority_clean_flagged, group.majority_clean_n),
            (group.known_majority_clean_flagged, group.known_majority_clean_n),
        ),
        f"  not scored: no twin {total.no_twin}, missing score {total.missing_score}, "
        f"unknown type {total.unknown_type}, conflicts {total.conflicts}",
        *(f"  {line}" for line in operating_lines(group)),
    ]
    return "\n".join(lines)


def summary_line(group: GroupResult) -> str:
    """One line of numbers and labels for a group."""
    reps = group.replicates

    def per_replicate(name: str, values: Iterable[tuple[str, float | None]]) -> str:
        return f"{name} " + " ".join(f"r{rep} {_rate(value)}" for rep, value in values)

    parts = [
        f"{_short(group.key_values['model'])} family={_short(group.label)} "
        + _settings(group, skip=("model",)),
        per_replicate("omit", ((k, r.omit_discrimination) for k, r in reps.items())),
        per_replicate("add/change", ((k, r.change_discrimination) for k, r in reps.items())),
        per_replicate("detection", ((k, r.detection_rate) for k, r in reps.items())),
        per_replicate("clean", ((k, r.clean_flag_rate) for k, r in reps.items())),
        f"majority detection {_rate(group.majority_detection_rate)} "
        f"clean {_rate(group.majority_clean_rate)}",
        f"known-only majority detection {_rate(group.known_majority_detection_rate)} "
        f"clean {_rate(group.known_majority_clean_rate)}",
    ]
    return " | ".join(parts)


def print_summary(groups: Sequence[GroupResult], *, scored: int, skipped: int) -> None:
    for label, heading in (
        (COMPLETE, None),
        (DEVELOPMENT_COMPLETE, "Development complete — not comparable to published figures"),
    ):
        chosen = [g for g in groups if g.comparable and g.coverage_label == label]
        if heading:
            print(f"\n== {heading} ==")
        for group in chosen:
            print(summary_line(group))
            for line in operating_lines(group):
                print(f"    {line}")
        if not chosen:
            print(f"(no {label} groups)")
    print("\n== Partial coverage ==")
    partial_groups = [g for g in groups if g.comparable and g.coverage_label == PARTIAL]
    for group in partial_groups:
        print(
            f"{_short(group.key_values['model'])} family={_short(group.label)} "
            f"{_settings(group, skip=('model',))} | coverage {_coverage_text(group)} "
            f"| parse failures excluded {group.parse_failures} "
            f"| replaced by later shards {group.replaced}"
        )
    if not partial_groups:
        print("(no comparable groups with partial coverage)")
    partial = sum(g.coverage_label == PARTIAL for g in groups)
    print(
        f"\n{scored} file(s) scored, skipped {skipped} file(s) with no scored records; "
        f"{len(groups)} group(s), {partial} with partial coverage or not comparable"
    )


def print_report(groups: Sequence[GroupResult], *, scored: int, skipped: int) -> None:
    comparable = [g for g in groups if g.comparable]
    other = [g for g in groups if not g.comparable]
    for group in comparable:
        print(format_group(group) + "\n")
    if other:
        print(
            "== Not comparable to published figures "
            "(no dataset version, smoke runs or the frozen master list) ==\n"
        )
        for group in other:
            print(format_group(group) + "\n")
    print(
        f"{scored} file(s) scored, skipped {skipped} file(s) with no scored records; "
        f"{len(comparable)} comparable group(s), {len(other)} not comparable"
    )
    print(f"parse failures excluded in total: {sum(g.parse_failures for g in groups)}")


if __name__ == "__main__":
    raise SystemExit(main())
