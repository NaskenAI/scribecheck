"""Gate 1(b): compare a new run of the two published methods with the released records.

Usage:
    python scripts/compare_gate1b.py spec --released data/omissionbench/judge_scores \
        --out runs/gate1b/subset_spec.json [--consultations KEY ...]
    python scripts/compare_gate1b.py compare --released data/omissionbench/judge_scores \
        --spec runs/gate1b/subset_spec.json [--new-power STORE.jsonl] [--new-b3 STORE.jsonl]

`spec` writes the harness's --subset-spec file: the note_keys of the chosen consultations
(GATE1B_CONSULTATIONS unless --consultations is given) that every released reference set
holds. `compare` reads the new run's record stores (the harness's JSONL) and the released
records on the same notes. The released reference sets are, on dataset factorial-v1 and
openai/gpt-5.4: power-gepa-high replicates 1 and 3 (family w2-power) for route two, and
B3 replicate 1 (family w2-pipeline) for route one.

Route two (power-gepa-high, the route-two-below-10 rule from recompute_published.py):
per-note agreement of the new run with released r1 and with released r3, over the notes
whose released flag is known; a note the new run failed or lacks counts as a disagreement.
The reference is released r1 against r3 on the same notes. It passes when both agreements
reach the pass mark, ROUTE_TWO_MARK or the released r1-versus-r3 agreement if that is lower.
The aggregate difference against r1 is reported too.

Route one (B3, the pipeline-critical-absent rule): the new run was made with fresh fact
extraction, because the released runs' fact lists were not released, so it is not gated on
per-note agreement. It passes when every note in the spec has a valid new record (present,
not a parse failure) and its detection count on omission notes and false-alarm count on
clean notes are each within DETECTION_MARGIN and FALSE_ALARM_MARGIN of released r1 on the
same notes. Per-note agreement and the n_absent_critical difference are reported for
information; agreement below INVESTIGATE_BELOW means investigate before M4. A B3 row that
parsed but has no graded missing fact is not flagged, and a released row with no
n_absent_critical count (the split writes none when nothing graded is missing) counts 0.

Only counts, percentages and note_keys are printed, never any other field, and no network
call is made. Exit status 0 means every compared method passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute_published import (  # noqa: E402
    _records,
    _score_of,
    below_10,
    critical_absent,
    source,
)

GATE1B_CONSULTATIONS = (
    "aci|D2N090",
    "aci|D2N156",
    "aci|D2N177",
    "authored|family_history_attribution",
    "authored|ha_sudden_thunderclap",
    "primock|day1_consultation01",
    "primock|day4_consultation10",
    "primock|day5_consultation07",
    "primock|day5_consultation11",
    "trapblind|tb_lipids_results_review",
)
MODEL = "openai/gpt-5.4"
DATASET_VERSION = "factorial-v1"
# name -> (family, arm, replicate)
REFERENCES = {
    "power r1": ("w2-power", "power-gepa-high", "1"),
    "power r3": ("w2-power", "power-gepa-high", "3"),
    "b3 r1": ("w2-pipeline", "B3", "1"),
}
ROUTE_TWO_MARK = 0.90
DETECTION_MARGIN = 4
FALSE_ALARM_MARGIN = 1
INVESTIGATE_BELOW = 0.80

Rows = dict[str, "dict[str, Any] | None"]
"""note_key -> row, or None when the note's duplicate rows disagree."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    spec = commands.add_parser("spec", help="write the harness's --subset-spec file")
    spec.add_argument("--released", type=Path, required=True)
    spec.add_argument("--out", type=Path, required=True)
    spec.add_argument("--consultations", nargs="+", default=list(GATE1B_CONSULTATIONS))
    compare = commands.add_parser("compare", help="compare a new run with the released records")
    compare.add_argument("--released", type=Path, required=True)
    compare.add_argument("--spec", type=Path, required=True)
    compare.add_argument("--new-power", type=Path)
    compare.add_argument("--new-b3", type=Path)
    args = parser.parse_args(argv)

    released = released_rows(args.released)
    if args.command == "spec":
        keys = spec_keys(released, args.consultations)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        document = {"note_keys": keys, "consultations": list(args.consultations)}
        args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        print(f"{len(keys)} note(s) over {len(args.consultations)} consultation(s) -> {args.out}")
        return 0

    if args.new_power is None and args.new_b3 is None:
        parser.error("compare needs --new-power, --new-b3 or both")
    keys = sorted(json.loads(args.spec.read_text(encoding="utf-8"))["note_keys"])
    passed = True
    if args.new_power is not None:
        new = read_store(args.new_power, lambda r: r.get("arm") == "power-gepa-high")
        lines, ok = route_two(new, released["power r1"], released["power r3"], keys)
        print("\n".join(lines) + "\n")
        passed &= ok
    if args.new_b3 is not None:
        new = read_store(args.new_b3, lambda r: _arm(r) == "B3")
        lines, ok = route_one(new, released["b3 r1"], keys)
        print("\n".join(lines))
        passed &= ok
    return 0 if passed else 1


def _arm(row: Mapping[str, Any]) -> object:
    return row.get("tier") or row.get("cell") or row.get("arm")


def _add(rows: Rows, row: dict[str, Any]) -> None:
    key = row.get("note_key")
    if not isinstance(key, str):
        return
    if key in rows and rows[key] != row:
        rows[key] = None
    else:
        rows.setdefault(key, row)


def released_rows(folder: Path) -> dict[str, Rows]:
    """Each reference set's rows by note_key, from the released scored-records files."""
    out: dict[str, Rows] = {name: {} for name in REFERENCES}
    for path in sorted(folder.rglob("*.json")):
        family = source(path).family
        for row in _records(path):
            if row.get("model") != MODEL or row.get("dataset_version") != DATASET_VERSION:
                continue
            for name, (want_family, arm, replicate) in REFERENCES.items():
                if (
                    family == want_family
                    and _arm(row) == arm
                    and str(row.get("replicate")) == replicate
                ):
                    _add(out[name], row)
    return out


def read_store(path: Path, keep: Callable[[Mapping[str, Any]], bool]) -> Rows:
    """A harness record store (JSONL): the kept rows by note_key."""
    rows: Rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn final line from a killed process
        if isinstance(row, dict) and keep(row):
            _add(rows, row)
    return rows


def spec_keys(released: Mapping[str, Rows], consultations: Iterable[str]) -> list[str]:
    """The note_keys of these consultations that every reference set holds."""
    wanted = set(consultations)
    sets = [
        {
            key
            for key, row in rows.items()
            if row is not None and f"{row.get('stratum')}|{row.get('consultation')}" in wanted
        }
        for rows in released.values()
    ]
    return sorted(set.intersection(*sets))


# Flags


def route_two_flag(row: Mapping[str, Any] | None) -> bool | None:
    return None if row is None else below_10(row)


def route_one_flag(row: Mapping[str, Any] | None) -> bool | None:
    """Critical fact judged absent; a parsed row that cannot say is not flagged."""
    if row is None or row.get("parse_failure") is True:
        return None
    return critical_absent(row) is True


def route_one_count(row: Mapping[str, Any] | None) -> int | None:
    """Facts graded critical and judged absent: from detail, else the split's count, else 0."""
    if row is None or row.get("parse_failure") is True:
        return None
    detail = row.get("detail")
    facts = detail.get("missing_facts") if isinstance(detail, Mapping) else None
    if isinstance(facts, list):
        return sum(
            1
            for f in facts
            if isinstance(f, Mapping)
            and f.get("severity") == "critical"
            and f.get("verdict") == "absent"
        )
    count = row.get("n_absent_critical")
    return count if isinstance(count, int) and not isinstance(count, bool) else 0


# Comparisons


@dataclass
class Agreement:
    agree: int = 0
    total: int = 0
    disagree: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.agree / self.total if self.total else 0.0

    def text(self) -> str:
        line = f"{self.agree}/{self.total} ({100 * self.rate:.1f}%)"
        return line + (f" | disagree: {', '.join(self.disagree)}" if self.disagree else "")


def agreement(
    new: Rows,
    reference: Rows,
    keys: Iterable[str],
    flag: Callable[[Mapping[str, Any] | None], bool | None],
    *,
    both_known: bool = False,
) -> Agreement:
    """Over notes whose reference flag is known; an unknown new flag is a disagreement
    (with both_known, notes where either flag is unknown are left out instead)."""
    result = Agreement()
    for key in keys:
        theirs, ours = flag(reference.get(key)), flag(new.get(key))
        if theirs is None or (both_known and ours is None):
            continue
        result.total += 1
        if ours == theirs:
            result.agree += 1
        else:
            result.disagree.append(key)
    return result


def _differences(
    new: Rows, reference: Rows, keys: Iterable[str], value: Callable[[Any], float | None]
) -> str:
    pairs = [
        (a, b)
        for key in keys
        if (a := value(new.get(key))) is not None and (b := value(reference.get(key))) is not None
    ]
    if not pairs:
        return "no notes with a value on both sides"
    diffs = [abs(a - b) for a, b in pairs]
    equal = sum(d == 0 for d in diffs)
    return (
        f"equal {equal}/{len(diffs)}, mean |difference| {sum(diffs) / len(diffs):.3f}, "
        f"max {max(diffs):g}"
    )


def route_two_passes(*, new_r1: float, new_r3: float, released: float) -> tuple[float, bool]:
    """(pass mark, passed): both agreements must reach 90%, or the released r1-vs-r3
    agreement if that is lower."""
    mark = min(ROUTE_TWO_MARK, released)
    return mark, new_r1 >= mark and new_r3 >= mark


def route_two(new: Rows, r1: Rows, r3: Rows, keys: Sequence[str]) -> tuple[list[str], bool]:
    vs_r1 = agreement(new, r1, keys, route_two_flag)
    vs_r3 = agreement(new, r3, keys, route_two_flag)
    reference = agreement(r1, r3, keys, route_two_flag, both_known=True)
    mark, passed = route_two_passes(new_r1=vs_r1.rate, new_r3=vs_r3.rate, released=reference.rate)
    return [
        f"route two (power-gepa-high, route-two-below-10), {len(keys)} notes in the spec",
        f"  new vs released r1: agree {vs_r1.text()}",
        f"  new vs released r3: agree {vs_r3.text()}",
        f"  released r1 vs r3: agree {reference.text()}",
        f"  aggregate, new vs released r1: {_differences(new, r1, keys, _score_of)}",
        f"  pass mark: {100 * mark:.1f}% (90%, or the released r1-vs-r3 agreement if lower)",
        f"route two: {'PASS' if passed else 'FAIL'}",
    ], passed


def _counts(rows: Rows, keys: Sequence[str]) -> tuple[int, int]:
    return sum(route_one_flag(rows.get(k)) is True for k in keys), len(keys)


def route_one(new: Rows, r1: Rows, keys: Sequence[str]) -> tuple[list[str], bool]:
    invalid = [k for k in keys if route_one_flag(new.get(k)) is None]
    kinds = {k: (r1.get(k) or new.get(k) or {}).get("pair_type") for k in keys}
    roles = {k: (r1.get(k) or new.get(k) or {}).get("note_role") for k in keys}
    omissions = [k for k in keys if kinds[k] == "omit"]
    clean = [k for k in keys if roles[k] == "clean"]
    (new_det, n_om), (old_det, _) = _counts(new, omissions), _counts(r1, omissions)
    (new_fa, n_cl), (old_fa, _) = _counts(new, clean), _counts(r1, clean)
    det_ok = abs(new_det - old_det) <= DETECTION_MARGIN
    fa_ok = abs(new_fa - old_fa) <= FALSE_ALARM_MARGIN
    passed = not invalid and det_ok and fa_ok
    vs_r1 = agreement(new, r1, keys, route_one_flag)
    valid = f"  valid new records: {len(keys) - len(invalid)}/{len(keys)}"
    if invalid:
        valid += f" | missing or failed: {', '.join(invalid)}"
    return [
        f"route one (B3, pipeline-critical-absent), {len(keys)} notes in the spec",
        valid,
        f"  detection on omission notes: new {new_det}/{n_om}, released r1 {old_det}/{n_om}, "
        f"difference {abs(new_det - old_det)} (margin {DETECTION_MARGIN})",
        f"  false alarms on clean notes: new {new_fa}/{n_cl}, released r1 {old_fa}/{n_cl}, "
        f"difference {abs(new_fa - old_fa)} (margin {FALSE_ALARM_MARGIN})",
        f"  per-note agreement (information only): {vs_r1.text()}",
        "  "
        + (
            f"below {100 * INVESTIGATE_BELOW:.0f}%: investigate before M4"
            if vs_r1.rate < INVESTIGATE_BELOW
            else f"{100 * INVESTIGATE_BELOW:.0f}% or above"
        ),
        f"  n_absent_critical, new vs released r1: {_differences(new, r1, keys, route_one_count)}",
        f"route one: {'PASS' if passed else 'FAIL'}",
    ], passed


if __name__ == "__main__":
    raise SystemExit(main())
