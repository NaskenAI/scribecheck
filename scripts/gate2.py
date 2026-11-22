"""Gate 2, Claude arm: held-out specs, run checks, fact judging, scoring and the spot-check.

Usage (only from docs/runbooks/gate2_claude.sh, run by a person in a normal terminal):
    python scripts/gate2.py specs --pairs <pairs> --ids <official ids> --out <dir>
    python scripts/gate2.py check-run --run <run> --pairs <pairs> --spec <spec> \
        --manifest <manifest>
    python scripts/gate2.py judge --runs <r1> <r2> <r3> --pairs <pairs> --spec <spec> \
        --manifest <manifest> --raw <raw folder> --config <judge config> --cache <cache>
    python scripts/gate2.py score --runs <r1> <r2> <r3> --supplementary-run <s1> \
        --pairs <pairs> --primary-spec <spec> --supplementary-spec <spec> \
        --primary-manifest <manifest> --supplementary-manifest <manifest> --raw <raw folder> \
        --judgements <cache> --actions <cache> --commit <sha> --prereg <sha> --out <file>
    python scripts/gate2.py sample --runs <r1> <r2> <r3> --pairs <pairs> --spec <spec> \
        --manifest <manifest> --judgements <cache> --out <csv>

Every path comes from the arguments; the script names no folder of its own. The analysis
follows docs/evidence/gate2_preregistration.md:

- `specs` writes primary_spec.json (the official 151 errored notes and the 47 clean twins)
  and supplementary_spec.json (the other 79 errored notes and the same 47 clean twins),
  and stops, writing nothing, unless the counts are EXPECTED.
- `check-run` says whether a run is usable: it is discarded (exit 3) when it reached its cost
  cap, or when more than MAX_FAILED_SHARE of its notes have no report or a failed Pass A.
- `judge` asks fact_judge.py, for each run's omission notes with any Pass A omission flag
  and for each route-one (B3) replicate's notes flagged by its rule, whether a flag states
  the removed fact (fact texts only; paid calls).
- `score` writes counts only: the primary endpoints by strict majority of the three runs
  (a note in which Pass A failed, or that has no report, counts as not flagged in that run),
  with Wilson 95% intervals; correct detection; the fixed low-false-alarm points
  (FIXED_POINTS, chosen on development data); the released gpt-5.4 baselines recomputed on
  the same notes; exact McNemar tests with Newcombe's paired interval; route one's correct
  detection; the plan-action subset; and the supplementary run.
- `sample` writes the held-out spot-check: SAMPLE_N omission notes the majority flags,
  drawn with SAMPLE_SEED, with the removed fact and each run's counted flags, for a person.

Only counts, rates and intervals are printed or written by `score`; never pair ids,
transcript, note or fact text. `sample`'s sheet holds fact text and belongs in the
protected folder, sealed with the run outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import action_classify  # noqa: E402
import fact_judge  # noqa: E402
import recompute_published as recompute  # noqa: E402
from compare_gate1b import _add, _arm, route_one_flag, route_two_flag  # noqa: E402
from score_run import (  # noqa: E402
    Grade,
    Note,
    _pairs,
    build_notes,
    confirmed_absent,
    decision_flags,
    grade,
    load_reports,
    model_names,
    removed_facts,
    spec_cells,
)

from scribecheck.config import load_config  # noqa: E402
from scribecheck.errors import InputError  # noqa: E402
from scribecheck.flags import Flag  # noqa: E402
from scribecheck.model import create_budget, create_client  # noqa: E402

EXPECTED = {
    "held-out pairs": 230,
    "official pairs": 151,
    "official omission pairs": 131,
    "official add or change pairs": 20,
    "supplementary pairs": 79,
    "clean twins": 47,
}
RUNS = 3
MAX_FAILED_SHARE = 0.02
FIXED_POINTS: dict[str, Grade] = {
    # The graded-score thresholds chosen on the v4 full development run
    # (runs/m3s4/claude-medium-v4-full) at 1/65 and 2/65 false alarms.
    "1/65": (0, 1, 0, 1, 2, 2),
    "2/65": (0, 1, 0, 1, 1, 2),
}
MODEL = "openai/gpt-5.4"
DATASET_VERSION = "factorial-v1-armconfirm"
REPLICATES = ("1", "2", "3")
# name -> (family, arm, flag rule)
BASELINES: dict[str, tuple[str, str, Callable[[Mapping[str, Any] | None], bool | None]]] = {
    "route one (B3)": ("w2-pipeline", "B3", route_one_flag),
    "route two (power-gepa-high)": ("w2-power", "power-gepa-high", route_two_flag),
}
ROUTE_ONE = "route one (B3)"
PUBLISHED = {ROUTE_ONE: (28, 1), "route two (power-gepa-high)": (42, 1)}
"""The dataset card's held-out strict majorities: (omission notes flagged of 131, clean
notes flagged of 47)."""
LABEL = "role-played benchmark data (OmissionBench held-out subset)"
SAMPLE_N = 30
SAMPLE_SEED = 20261123
Z = 1.959963984540054
COST_CAP = "cost_cap_reached"


# statistics


def wilson(k: int, n: int) -> tuple[float, float]:
    """The Wilson score 95% interval for k of n."""
    if n == 0:
        return (0.0, 1.0)
    p, z2 = k / n, Z * Z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """The exact two-sided McNemar p value for b and c discordant notes."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (1 << n)
    return min(1.0, 2 * tail)


def newcombe_paired(both: int, b: int, c: int, neither: int) -> tuple[float, float, float]:
    """The difference in rates (first minus second method) for paired data, with Newcombe's
    square-and-add 95% interval: Wilson limits for each marginal rate and the uncorrected
    phi (Newcombe 1998, method 10). `b`: first method only; `c`: second only."""
    n = both + b + c + neither
    if n == 0:
        return (0.0, -1.0, 1.0)
    p1, p2 = (both + b) / n, (both + c) / n
    l1, u1 = wilson(both + b, n)
    l2, u2 = wilson(both + c, n)
    margins = (both + b) * (c + neither) * (both + c) * (b + neither)
    phi = (both * neither - b * c) / math.sqrt(margins) if margins else 0.0
    d = p1 - p2
    low = (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2
    high = (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2
    return (d, max(-1.0, d - math.sqrt(max(low, 0.0))), min(1.0, d + math.sqrt(max(high, 0.0))))


def strict_majority(votes: Sequence[bool], total: int = RUNS) -> bool:
    return sum(votes) * 2 > total


def _rate(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else f"{k}/0"


def _with_ci(k: int, n: int) -> str:
    low, high = wilson(k, n)
    return f"{_rate(k, n)}, Wilson 95% {100 * low:.1f}% to {100 * high:.1f}%"


# specs


def build_specs(
    pairs: Sequence[Mapping[str, Any]],
    official: set[str],
    expected: Mapping[str, int] | None = None,
) -> tuple[list[str], list[str], dict[str, int]]:
    """(primary note keys, supplementary note keys, counts); ValueError unless the counts
    are `expected` (default EXPECTED)."""
    expected = EXPECTED if expected is None else expected
    errored = [p for p in pairs if isinstance(p.get("pair_id"), str)]
    known = {str(p["pair_id"]) for p in errored}
    chosen = [p for p in errored if p["pair_id"] in official]
    others = [p for p in errored if p["pair_id"] not in official]
    clean = sorted(k for keys in spec_cells(pairs)["clean"].values() for k in keys)
    counts = {
        "held-out pairs": len(errored),
        "official pairs": len(chosen),
        "official omission pairs": sum(p.get("type") == "omit" for p in chosen),
        "official add or change pairs": sum(p.get("type") in ("add", "change") for p in chosen),
        "supplementary pairs": len(others),
        "clean twins": len(clean),
    }
    problems = [
        f"{name} {counts[name]}, expected {want}"
        for name, want in expected.items()
        if counts[name] != want
    ]
    if official - known:
        problems.append(f"{len(official - known)} official ids not in the pairs file")
    if problems:
        raise ValueError("; ".join(problems))
    primary = sorted(f"{p['pair_id']}|err" for p in chosen) + clean
    supplementary = sorted(f"{p['pair_id']}|err" for p in others) + clean
    return primary, supplementary, counts


def run_specs(args: argparse.Namespace) -> int:
    try:
        primary, supplementary, counts = build_specs(
            _pairs(args.pairs), action_classify.load_ids(args.ids)
        )
    except ValueError as exc:  # counts only (build_specs) or a file's shape (load_ids)
        print(f"specs: stopped, nothing written: {exc}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    for name, keys in (("primary_spec.json", primary), ("supplementary_spec.json", supplementary)):
        (args.out / name).write_text(
            json.dumps({"note_keys": keys}, indent=1) + "\n", encoding="utf-8"
        )
    print("specs: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"specs: primary {len(primary)} notes, supplementary {len(supplementary)} notes")
    return 0


# runs


def _notes(
    run: Path, pairs: Sequence[Mapping[str, Any]], spec: Path, manifest: Path | None
) -> list[Note]:
    keys = sorted(json.loads(spec.read_text(encoding="utf-8"))["note_keys"])
    return build_notes(keys, pairs, load_reports(run, manifest)[0])


@dataclass(frozen=True)
class RunCheck:
    notes: int
    missing: int
    failed: int
    cost_cap: bool
    cost_usd: float | None

    @property
    def discard(self) -> bool:
        return self.cost_cap or (self.missing + self.failed) > MAX_FAILED_SHARE * self.notes


def check(notes: Sequence[Note]) -> RunCheck:
    reports = [n.report for n in notes if n.report is not None]
    reasons = {
        reason
        for report in reports
        for item in report.get("unchecked", [])
        for reason in item.get("reasons", [])
    }
    costs = [(r.get("usage") or {}).get("cost_usd") for r in reports]
    known = [float(c) for c in costs if c is not None]
    return RunCheck(
        notes=len(notes),
        missing=sum(n.report is None for n in notes),
        failed=sum(bool(n.failed) for n in notes),
        cost_cap=COST_CAP in reasons,
        cost_usd=sum(known) if known else None,
    )


def _check_line(name: str, c: RunCheck) -> str:
    cost = f"${c.cost_usd:.2f}" if c.cost_usd is not None else "cost unknown"
    return (
        f"{name}: {c.notes} notes, {c.missing} without a report, {c.failed} with Pass A failed, "
        f"cost cap reached: {'yes' if c.cost_cap else 'no'}, {cost}"
    )


def run_check(args: argparse.Namespace) -> int:
    c = check(_notes(args.run, _pairs(args.pairs), args.spec, args.manifest))
    verdict = "DISCARD (repeat this run in full)" if c.discard else "usable"
    print(f"check-run: {_check_line(args.run.name, c)}: {verdict}")
    return 3 if c.discard else 0


# released baselines


Released = dict[str, dict[str, dict[str, dict[str, Any] | None]]]
"""baseline name -> replicate -> note_key -> row (None when duplicate rows disagree)."""


def load_released(raw: Path) -> Released:
    """The released held-out rows of each baseline, by replicate and note_key: gpt-5.4,
    dataset version DATASET_VERSION, the baseline's family and arm. Within a replicate a
    note keeps only its latest shard folder's rows, as Gate 1(a) does."""
    tagged: dict[tuple[str, str], list[recompute.Tagged]] = {}
    for index, path in enumerate(sorted(raw.rglob("*.json"))):
        where = recompute.source(path)
        for name, (family, arm, _) in BASELINES.items():
            if where.family != family:
                continue
            for row in recompute._records(path):
                if (
                    row.get("model") == MODEL
                    and row.get("dataset_version") == DATASET_VERSION
                    and _arm(row) == arm
                ):
                    replicate = str(row.get("replicate"))
                    tagged.setdefault((name, replicate), []).append((index, where.timestamp, row))
    out: Released = {name: {r: {} for r in REPLICATES} for name in BASELINES}
    for (name, replicate), rows in tagged.items():
        if replicate not in REPLICATES:
            continue
        kept, _ = recompute._latest_shards(rows)
        for row in kept:
            _add(out[name][replicate], row)
    return out


def baseline_flags(released: Released, name: str, keys: Sequence[str]) -> dict[str, list[bool]]:
    """Each note's flag in each replicate under the baseline's rule; a missing, failed or
    conflicting row counts as not flagged."""
    rule = BASELINES[name][2]
    return {key: [rule(released[name][r].get(key)) is True for r in REPLICATES] for key in keys}


def route_one_facts(row: Mapping[str, Any] | None) -> list[str]:
    """The facts route one's rule counts on a note: graded critical and judged absent."""
    detail = row.get("detail") if row is not None else None
    facts = detail.get("missing_facts") if isinstance(detail, Mapping) else None
    if not isinstance(facts, list):
        return []
    return [
        " ".join(str(f["fact"]).split())
        for f in facts
        if isinstance(f, Mapping)
        and f.get("severity") == "critical"
        and f.get("verdict") == "absent"
        and isinstance(f.get("fact"), str)
    ]


def route_one_items(
    released: Released, notes: Sequence[Note]
) -> dict[str, dict[str, fact_judge.Item | None]]:
    """Per replicate, each omission note route one's rule flags: the judge item for its
    flagged facts (ids B1...), or None when the row holds no fact text."""
    out: dict[str, dict[str, fact_judge.Item | None]] = {}
    for r in REPLICATES:
        rows = released[ROUTE_ONE][r]
        out[r] = {}
        for n in notes:
            row = rows.get(n.key)
            if n.kind != "omission" or route_one_flag(row) is not True:
                continue
            facts = route_one_facts(row)
            out[r][n.key] = (
                fact_judge.Item(
                    f"B3 r{r}|{n.pair.get('pair_id')}",
                    tuple(removed_facts(n.pair)),
                    (),
                    tuple((f"B{i}", text) for i, text in enumerate(facts, start=1)),
                )
                if facts
                else None
            )
    return out


# our runs


def usable(n: Note) -> bool:
    return n.report is not None and not n.failed


def graded_flags(n: Note) -> list[Flag]:
    """The flags behind a note's graded score, as score_run.py counts them."""
    return [*confirmed_absent(n), *n.info, *(f for f in n.pass_a if f.verdict == "partial")]


def our_item(run_index: int, n: Note) -> fact_judge.Item:
    assert n.report is not None
    return fact_judge.item(f"r{run_index}|{n.pair.get('pair_id')}", removed_facts(n.pair), n.report)


def has_flags(n: Note) -> bool:
    return usable(n) and bool(n.pass_a or n.info)


Point = Callable[[Note], list[Flag]]
"""An operating point: the counted flags that flag a note (none: not flagged)."""


def decision_point(n: Note) -> list[Flag]:
    return decision_flags(n) if usable(n) else []


def fixed_point(threshold: Grade) -> Point:
    def pick(n: Note) -> list[Flag]:
        return graded_flags(n) if usable(n) and grade(n) >= threshold else []

    return pick


@dataclass
class Votes:
    """Per note key, one vote per run: flagged, and flagged with a flag that states the
    removed fact (by the judge)."""

    flagged: dict[str, list[bool]]
    correct: dict[str, list[bool]]

    def majority(self, keys: Sequence[str]) -> tuple[int, int]:
        return (
            sum(strict_majority(self.flagged[k]) for k in keys),
            sum(strict_majority(self.correct[k]) for k in keys),
        )


def votes(runs: Sequence[Sequence[Note]], point: Point, cache: Mapping[str, Any] | None) -> Votes:
    out = Votes({}, {})
    for index, notes in enumerate(runs, start=1):
        for n in notes:
            flags = point(n)
            out.flagged.setdefault(n.key, []).append(bool(flags))
            named = False
            if flags and n.kind == "omission" and cache is not None:
                named = model_names(fact_judge.lookup(cache, our_item(index, n)), flags)
            out.correct.setdefault(n.key, []).append(named)
    return out


# judge


def run_judge(args: argparse.Namespace) -> int:
    """Judge our runs' flagged omission notes and route one's (paid calls: the runbook)."""
    try:
        config = load_config(args.config)
        client = create_client(config)
        budget = create_budget(config)
    except InputError as exc:
        print(f"judge: {exc}", file=sys.stderr)
        return 2
    pairs = _pairs(args.pairs)
    runs = [_notes(run, pairs, args.spec, args.manifest) for run in args.runs]
    items = [
        our_item(index, n)
        for index, notes in enumerate(runs, start=1)
        for n in notes
        if n.kind == "omission" and has_flags(n)
    ]
    ours = len(items)
    by_replicate = route_one_items(load_released(args.raw), runs[0])
    lacking = 0
    for by_note in by_replicate.values():
        items += [it for it in by_note.values() if it is not None]
        lacking += sum(it is None for it in by_note.values())
    cache = fact_judge.load_cache(args.cache)
    tag = f"{config.model} effort={config.effort} max_tokens={config.max_tokens}"
    counts = fact_judge.judge(items, client, budget, cache, tag, args.cache)
    spent = budget.spent_usd
    print(
        f"judge: {ours} items from our runs, {len(items) - ours} from route one "
        f"({lacking} flagged notes without fact text): "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    )
    print(
        f"judge: {budget.calls} calls, "
        + (f"${spent:.4f}" if spent is not None else "cost unknown")
    )
    return 0 if sum(counts[k] for k in ("cached", "judged")) == len(items) else 2


# score


def mcnemar_lines(
    ours: Mapping[str, bool], theirs: Mapping[str, bool], keys: Sequence[str]
) -> tuple[str, str]:
    both = sum(ours[k] and theirs[k] for k in keys)
    b = sum(ours[k] and not theirs[k] for k in keys)
    c = sum(theirs[k] and not ours[k] for k in keys)
    neither = len(keys) - both - b - c
    d, low, high = newcombe_paired(both, b, c, neither)
    return (
        f"both {both}, ours only {b}, theirs only {c}, neither {neither}; "
        f"exact McNemar p = {mcnemar_exact(b, c):.4f}",
        f"difference in detection {100 * d:+.1f} points, Newcombe 95% {100 * low:+.1f} to "
        f"{100 * high:+.1f}",
    )


def report(
    runs: Sequence[Sequence[Note]],
    supplementary: Sequence[Note],
    released: Released,
    cache: Mapping[str, Any] | None,
    actions: Mapping[str, str | None],
    header: Sequence[str] = (),
) -> list[str]:
    """The Gate 2 report: counts, rates and intervals only."""
    out: list[str] = [*header]
    say = out.append
    first = runs[0]
    omission = [n.key for n in first if n.kind == "omission"]
    clean = [n.key for n in first if n.kind == "clean"]
    pair_of = {n.key: str(n.pair.get("pair_id")) for n in first}
    n_o, n_c = len(omission), len(clean)

    say("== Runs (each must have reached no cost cap and failed on at most 2% of notes)")
    for index, notes in enumerate(runs, start=1):
        say(f"   {_check_line(f'run {index}', check(notes))}")
    say(f"   {_check_line('supplementary run', check(supplementary))}")

    decision = votes(runs, decision_point, cache)
    det, correct = decision.majority(omission)
    alarms, _ = decision.majority(clean)
    say("== Primary endpoints: strict majority of three runs, decision rule")
    say(f"   detection: {_with_ci(det, n_o)}")
    say(f"   false alarms: {_with_ci(alarms, n_c)}")
    for index in range(len(runs)):
        d_r = sum(decision.flagged[k][index] for k in omission)
        f_r = sum(decision.flagged[k][index] for k in clean)
        say(f"   run {index + 1}: detection {_rate(d_r, n_o)}, false alarms {_rate(f_r, n_c)}")

    say("== Secondary 1: correct detection (fact_judge.py; counted only in runs that flag")
    say("   the note; strict majority of three)")
    if cache is None:
        say("   no judgements given")
    else:
        missing = sum(
            1
            for index, notes in enumerate(runs, start=1)
            for n in notes
            if n.kind == "omission"
            and decision_point(n)
            and fact_judge.lookup(cache, our_item(index, n)) is None
        )
        say(f"   correct detection: {_with_ci(correct, n_o)}")
        for index in range(len(runs)):
            c_r = sum(decision.correct[k][index] for k in omission)
            say(f"   run {index + 1}: correct detection {_rate(c_r, n_o)}")
        say(f"   flagged notes without a valid judgement (counted as not named): {missing}")

    say("== Secondary 2: fixed low-false-alarm points (development thresholds, unchanged;")
    say("   per run, then strict majority of three)")
    points: dict[str, Votes] = {"decision rule": decision}
    for name, threshold in FIXED_POINTS.items():
        v = votes(runs, fixed_point(threshold), cache)
        points[f"fixed point {name}"] = v
        d_p, c_p = v.majority(omission)
        f_p, _ = v.majority(clean)
        say(f"   at {name}, grade >= {threshold}:")
        say(f"      detection {_with_ci(d_p, n_o)}")
        say(f"      false alarms {_with_ci(f_p, n_c)}")
        if cache is not None:
            say(f"      correct detection {_rate(c_p, n_o)}")
        for index in range(len(runs)):
            d_r = sum(v.flagged[k][index] for k in omission)
            f_r = sum(v.flagged[k][index] for k in clean)
            rates = f"detection {_rate(d_r, n_o)}, false alarms {_rate(f_r, n_c)}"
            say(f"      run {index + 1}: {rates}")

    say("== Secondary 3: the published gpt-5.4 held-out figures (descriptive: the model differs)")
    theirs: dict[str, dict[str, bool]] = {}
    alarms_of: dict[str, int] = {}
    mismatched: list[str] = []
    for name in BASELINES:
        flags = baseline_flags(released, name, [*omission, *clean])
        theirs[name] = {k: strict_majority(v) for k, v in flags.items()}
        d_b = sum(theirs[name][k] for k in omission)
        alarms_of[name] = sum(theirs[name][k] for k in clean)
        pub_d, pub_f = PUBLISHED[name]
        rows = [
            sum(released[name][r].get(k) is not None for k in [*omission, *clean])
            for r in REPLICATES
        ]
        same = (d_b, alarms_of[name]) == (pub_d, pub_f) and (n_o, n_c) == (131, 47)
        if not same:
            mismatched.append(name)
        say(
            f"   {name}: published {pub_d}/131 at {pub_f}/47; recomputed from the released rows "
            f"{_rate(d_b, n_o)} at {_rate(alarms_of[name], n_c)} "
            f"({'matches' if same else 'DOES NOT MATCH'}); rows per replicate "
            + ", ".join(f"r{r} {k}" for r, k in zip(REPLICATES, rows, strict=True))
        )

    say("== Secondary 4: paired comparison on the same omission notes (strict majorities). A")
    say("   gain may be claimed only at a point whose false alarms are no higher than the")
    say("   baseline's")
    for point_name, v in points.items():
        ours = {k: strict_majority(v.flagged[k]) for k in omission}
        f_ours, _ = v.majority(clean)
        for name in BASELINES:
            counts, interval = mcnemar_lines(ours, theirs[name], omission)
            allowed = "yes" if f_ours <= alarms_of[name] else "no"
            say(f"   {point_name} vs {name}: {counts}")
            say(f"      {interval}")
            say(
                f"      false alarms: ours {f_ours}/{n_c}, theirs {alarms_of[name]}/{n_c}; "
                f"no higher than the baseline's: {allowed}"
            )

    say("== Secondary 5: route one's correct detection, from its released fact text, judged")
    say("   the same way (strict majority of three replicates; a missing row is not flagged)")
    if cache is None:
        say("   no judgements given")
    else:
        items = route_one_items(released, first)
        named = {k: [False] * len(REPLICATES) for k in omission}
        no_text = no_judgement = 0
        for i, r in enumerate(REPLICATES):
            for key, it in items[r].items():
                if it is None:
                    no_text += 1
                    continue
                j = fact_judge.lookup(cache, it)
                if j is None:
                    no_judgement += 1
                elif j.flags.same:
                    named[key][i] = True
        maj = sum(strict_majority(named[k]) for k in omission)
        say(f"   correct detection: {_with_ci(maj, n_o)}")
        for i, r in enumerate(REPLICATES):
            named_r = sum(named[k][i] for k in omission)
            say(f"   replicate {r}: correct detection {_rate(named_r, n_o)}")
        say(
            f"   flagged notes without fact text: {no_text}; without a valid judgement: "
            f"{no_judgement} (both counted as not named)"
        )

    say("== Secondary 6: the plan-action subset (removed fact labelled by action_classify.py)")
    say(f"   classifier prompt sha256 {action_classify.prompt_hash()}")
    label = {k: actions.get(pair_of[k]) for k in omission}
    unlabelled = sum(v is None for v in label.values())
    subset = [k for k in omission if label[k] not in (None, action_classify.NOT_ACTION)]
    say(f"   omission notes without a valid label: {unlabelled}")
    d_s, c_s = decision.majority(subset)
    say(f"   plan-action subset: {len(subset)}/{n_o} omission notes")
    say(f"   detection {_with_ci(d_s, len(subset))}")
    if cache is not None:
        say(f"   correct detection {_with_ci(c_s, len(subset))}")
    say("   by category (detection; correct detection):")
    for name in action_classify.LABELS:
        members = [k for k in omission if label[k] == name]
        d_l, c_l = decision.majority(members)
        say(f"      {name}: {_rate(d_l, len(members))}; {_rate(c_l, len(members))}")

    say("== Secondary 7: the supplementary set (one run, decision rule)")
    s_om = [n for n in supplementary if n.kind == "omission"]
    s_cl = [n for n in supplementary if n.kind == "clean"]
    s_det = sum(bool(decision_point(n)) for n in s_om)
    s_fa = sum(bool(decision_point(n)) for n in s_cl)
    say(f"   detection: {_with_ci(s_det, len(s_om))}")
    say(f"   false alarms: {_with_ci(s_fa, len(s_cl))}")
    for kind in ("add", "change"):
        of_kind = [n for n in supplementary if n.kind == kind]
        if of_kind:
            say(
                f"   (information) {kind} notes with a counted Pass A flag: "
                f"{_rate(sum(bool(decision_point(n)) for n in of_kind), len(of_kind))}"
            )
    kinds = Counter(n.kind for n in first)
    say("== Primary set by kind: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    for kind in ("add", "change"):
        keys = [n.key for n in first if n.kind == kind]
        if keys:
            d_k, _ = decision.majority(keys)
            say(f"   (information) {kind} notes flagged by the majority: {_rate(d_k, len(keys))}")
    if mismatched:
        out.insert(
            len(header),
            "# WARNING: the released rows do not reproduce the published figures for "
            + " and ".join(mismatched)
            + " (Secondary 3): every comparison with them is in doubt; record a deviation",
        )
    return out


def run_score(args: argparse.Namespace) -> int:
    if len(args.runs) != RUNS:
        print(f"score: expected {RUNS} runs, got {len(args.runs)}", file=sys.stderr)
        return 2
    pairs = _pairs(args.pairs)
    runs = [_notes(run, pairs, args.primary_spec, args.primary_manifest) for run in args.runs]
    supplementary = _notes(
        args.supplementary_run, pairs, args.supplementary_spec, args.supplementary_manifest
    )
    cache = fact_judge.load_cache(args.judgements) if args.judgements is not None else None
    actions = action_classify.labels(args.actions, pairs)
    header = [
        "# Gate 2, Claude arm: counts only",
        f"# {LABEL}",
        f"# commit {args.commit}; pre-registration commit {args.prereg}",
        "# docs/evidence/gate2_preregistration.md defines every figure below",
    ]
    lines = report(runs, supplementary, load_released(args.raw), cache, actions, header)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


# sample


SAMPLE_COLUMNS = (
    "pair_id",
    "removed_fact",
    "runs_flagged",
    "judge_runs_correct",
    "judge_majority_correct",
    *(f"r{i}_counted_flags" for i in range(1, RUNS + 1)),
    *(f"r{i}_judge_flag_ids" for i in range(1, RUNS + 1)),
    "reviewer_majority_correct",
    "reviewer_comment",
)


def run_sample(args: argparse.Namespace) -> int:
    """SAMPLE_N omission notes the strict majority flags, drawn with the seed, with each
    run's counted flags and the judge's verdicts, for a person to check."""
    pairs = _pairs(args.pairs)
    runs = [_notes(run, pairs, args.spec, args.manifest) for run in args.runs]
    cache = fact_judge.load_cache(args.judgements)
    v = votes(runs, decision_point, cache)
    pool = sorted(
        n.key for n in runs[0] if n.kind == "omission" and strict_majority(v.flagged[n.key])
    )
    chosen = sorted(random.Random(args.seed).sample(pool, min(args.n, len(pool))))
    by_key = [{n.key: n for n in notes} for notes in runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SAMPLE_COLUMNS)
        for key in chosen:
            texts, ids = [], []
            for index, notes in enumerate(by_key, start=1):
                n = notes[key]
                flags = decision_point(n)
                texts.append(" | ".join(f"{f.id}: {f.fact}" for f in flags))
                j = fact_judge.lookup(cache, our_item(index, n)) if flags else None
                ids.append(" ".join(j.flags.ids) if j is not None else "")
            n0 = by_key[0][key]
            writer.writerow(
                [
                    n0.pair.get("pair_id"),
                    " / ".join(removed_facts(n0.pair)),
                    sum(v.flagged[key]),
                    sum(v.correct[key]),
                    "yes" if strict_majority(v.correct[key]) else "no",
                    *texts,
                    *ids,
                    "",
                    "",
                ]
            )
    print(
        f"sample: {len(chosen)} of {len(pool)} majority-flagged omission notes (seed {args.seed})"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("specs")
    sp.add_argument("--pairs", type=Path, required=True)
    sp.add_argument("--ids", type=Path, required=True)
    sp.add_argument("--out", type=Path, required=True)
    cr = sub.add_parser("check-run")
    cr.add_argument("--run", type=Path, required=True)
    jd = sub.add_parser("judge")
    jd.add_argument("--runs", type=Path, nargs=RUNS, required=True)
    jd.add_argument("--raw", type=Path, required=True)
    jd.add_argument("--config", type=Path, required=True)
    jd.add_argument("--cache", type=Path, required=True)
    sa = sub.add_parser("sample")
    sa.add_argument("--runs", type=Path, nargs=RUNS, required=True)
    sa.add_argument("--judgements", type=Path, required=True)
    sa.add_argument("--out", type=Path, required=True)
    sa.add_argument("--n", type=int, default=SAMPLE_N)
    sa.add_argument("--seed", type=int, default=SAMPLE_SEED)
    for p in (cr, jd, sa):
        p.add_argument("--pairs", type=Path, required=True)
        p.add_argument("--spec", type=Path, required=True)
        p.add_argument("--manifest", type=Path, required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--runs", type=Path, nargs=RUNS, required=True)
    sc.add_argument("--supplementary-run", type=Path, required=True)
    sc.add_argument("--pairs", type=Path, required=True)
    sc.add_argument("--primary-spec", type=Path, required=True)
    sc.add_argument("--supplementary-spec", type=Path, required=True)
    sc.add_argument("--primary-manifest", type=Path, required=True)
    sc.add_argument("--supplementary-manifest", type=Path, required=True)
    sc.add_argument("--raw", type=Path, required=True)
    sc.add_argument("--judgements", type=Path)
    sc.add_argument("--actions", type=Path, required=True)
    sc.add_argument("--commit", required=True)
    sc.add_argument("--prereg", required=True)
    sc.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    commands = {
        "specs": run_specs,
        "check-run": run_check,
        "judge": run_judge,
        "score": run_score,
        "sample": run_sample,
    }
    try:
        return commands[args.command](args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        # The exception's type only: a message could quote held-out text.
        print(f"gate2 {args.command}: stopped ({type(exc).__name__})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
