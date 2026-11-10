"""Score a scribecheck run on an OmissionBench development subset (M3 session 4).

Usage:
    python scripts/score_run.py prepare --pairs data/omissionbench/pairs/dataset_v2.json \
        --transcripts data/omissionbench/transcripts --spec runs/gate1b/subset_spec.json \
        --out runs/m3s4/cases
    python scripts/score_run.py score --run runs/m3s4/claude-medium \
        --pairs data/omissionbench/pairs/dataset_v2.json --spec runs/gate1b/subset_spec.json \
        [--released data/omissionbench/judge_scores] [--gate1b runs/gate1b/harness-results] \
        [--show-facts] [--match-threshold 0.5] [--partial-as-flags]
    python scripts/score_run.py spec --pairs data/omissionbench/pairs/dataset_v2.json \
        --out runs/m3s4/full_spec.json
    python scripts/score_run.py misses --run runs/m3s4/claude-high \
        --pairs data/omissionbench/pairs/dataset_v2.json --spec runs/gate1b/subset_spec.json \
        [--manifest runs/m3s4/cases/manifest.json] [--show-facts]

`prepare` writes the subset as a batch folder: `<case>.transcript.txt` and `<case>.note.md`
for each note key (the case id is the note key with "|" as "__"), plus `manifest.json`.
Errored notes are `<pair_id>|err`; a clean note is `<consultation>|clean|<first 12 hex
digits of the clean note's SHA-256>`.

`spec` writes every errored note and every distinct clean note in the pairs file as a spec
(all 65 development consultations: 265 errored and 65 clean notes).

`score` reads the run's `batch.json` (or every report JSON in the folder) and prints, for
Pass A omission flags only, as the spec's benchmark scoring says (other passes' warnings
are shown separately, for information):

- detection on errored omission notes and false alarms on clean notes, under the decision
  rule (a note is flagged when its Pass A omission flags alone would make it warn or fail)
  and as the k-and-c sweep (at least k Pass A omission flags at confidence c or higher);
- a provisional correct-fact rate for detected omission notes, by fuzzy match between the
  pair's removed fact and the flags' facts, one line per note (fact text only with
  --show-facts);
- per-note agreement with the released B3 r1 and power-gepa-high r1/r3 records, and with
  the Gate 1(b) runs, on the same notes;
- each reference method's detection and false alarms on the same notes, under its own rule
  (route one: a critical fact judged absent; route two: aggregate below 10);
- route one's own detection, correct detection and false alarms on the same notes, from
  the critical facts judged absent in the Gate 1(b) B3 records and in the released B3 r1
  development rows (their `baseline_facts`, once the split has been re-run), matched to
  the removed fact the same way as our flags. Route two (power-gepa-high) gives a score
  and no named omissions, so its correct detection is not applicable;
- calls, tokens, cost and wall-clock time per note, and each unchecked reason's count.

Partial verdicts are information, not flags, whether the run listed them in `info` or (a
run from before the change) among its flags; `--partial-as-flags` counts them as before.

`misses` follows each omission note's removed fact through Pass A, from the report's
`pass_a_trace` (`[report] pass_a_trace = true`): the extracted fact that best matches it
(with the score), the audit's core or contextual call and severity, the check's verdict
(with the note quote for present) and the second look, and classes it as detected, not
extracted, dropped by the audit, contextual and so never checked, judged present, judged
partial, rescued by the second look, or unchecked; counts per class and per trace. An
unmatched removed fact whose closest comparable fact scores at least 0.25 is a "weak match
(compound?)", as when it is buried in a longer compound fact. A run without a trace can
only be classed from its flags.

Only counts, IDs, labels and scores are printed (and, with --show-facts, the benchmark's
and the flags' fact strings), never transcript or note text. Development data only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fact_judge  # noqa: E402
from compare_gate1b import (  # noqa: E402
    Rows,
    _arm,
    read_store,
    released_rows,
    route_one_flag,
    route_two_flag,
)
from fact_judge import Judgement  # noqa: E402

from scribecheck.config import load_config  # noqa: E402
from scribecheck.decision import decide  # noqa: E402
from scribecheck.errors import InputError  # noqa: E402
from scribecheck.flags import Flag  # noqa: E402
from scribecheck.model import create_budget, create_client  # noqa: E402
from scribecheck.rules.lexicon import find_drugs  # noqa: E402

RANK = {"possible": 1, "confirmed": 2, "certain": 3}
LEVELS = ("possible", "confirmed")
SEVERITY_RANK = {"peripheral": 1, "supporting": 2, "critical": 3}
FLOORS = (("all", "peripheral"), ("supporting+", "supporting"), ("critical", "critical"))
"""Severity floors for the sweep: all severities, supporting or higher, critical only."""
KS = (1, 2, 3, 4, 5)
SAMPLE_SEED = 20261109
"""The spot-check sample's seed (`sample`), fixed so the sample can be drawn again."""
_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = frozenset(
    "a an the of to and or in on at for with by is are was were be been has have had "
    "patient pt he she they his her their it its this that as from".split()
)


def case_id(note_key: str) -> str:
    """The note key as a file-name-safe case id ("|" as "__"); `prepare` refuses a key
    that would not map back exactly, and `score` prefers the manifest's mapping."""
    return note_key.replace("|", "__")


def note_key(case: str) -> str:
    return case.replace("__", "|")


# prepare


def _pairs(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = data["pairs"] if isinstance(data, dict) else data
    return [p for p in pairs if isinstance(p, dict)]


def resolve(key: str, pairs: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], str, str]:
    """(the pair the note belongs to, the note text, its kind) for a note key."""
    if key.endswith("|err"):
        pair = next((p for p in pairs if p.get("pair_id") == key[: -len("|err")]), None)
        if pair is None:
            raise KeyError(f"no pair for note key {key}")
        kind = {"omit": "omission"}.get(str(pair.get("type")), str(pair.get("type")))
        return pair, str(pair["errored"]), kind
    if "|clean|" in key:
        consultation, digest = key.split("|clean|", 1)
        for pair in pairs:
            clean = pair.get("clean")
            if pair.get("key") == consultation and isinstance(clean, str):
                if hashlib.sha256(clean.encode("utf-8")).hexdigest()[:12] == digest:
                    return pair, clean, "clean"
        raise KeyError(f"no clean note matches note key {key}")
    raise KeyError(f"unrecognised note key {key}")


def prepare(pairs_path: Path, transcripts: Path, spec_path: Path, out: Path) -> int:
    pairs = _pairs(pairs_path)
    keys = sorted(json.loads(spec_path.read_text(encoding="utf-8"))["note_keys"])
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, Any]] = {}
    for key in keys:
        if note_key(case_id(key)) != key:
            raise ValueError(f"note key {key} does not map to a file name and back exactly")
        pair, note, kind = resolve(key, pairs)
        source = transcripts / str(pair["stratum"]) / f"{pair['id']}.txt"
        if not source.is_file():
            raise FileNotFoundError(f"no transcript for note key {key}")
        case = case_id(key)
        (out / f"{case}.transcript.txt").write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
        )
        (out / f"{case}.note.md").write_text(note, encoding="utf-8", newline="\n")
        manifest[case] = {
            "note_key": key,
            "pair_id": pair.get("pair_id") if kind != "clean" else None,
            "kind": kind,
        }
    (out / "manifest.json").write_text(
        json.dumps({"spec": spec_path.name, "cases": manifest}, indent=1) + "\n",
        encoding="utf-8",
    )
    kinds = Counter(m["kind"] for m in manifest.values())
    print(f"prepared {len(manifest)} notes in {out}: " + _counts(kinds))
    return 0


def spec_cells(pairs: Sequence[Mapping[str, Any]]) -> dict[str, dict[tuple[str, ...], list[str]]]:
    """Every errored and every distinct clean note's key, in the strata a sample keeps in
    proportion: errored notes by kind (omission, add, change), source stratum and removed
    fact severity; clean notes by source stratum."""
    cells: dict[str, dict[tuple[str, ...], set[str]]] = {"errored": {}, "clean": {}}
    for pair in pairs:
        stratum = str(pair.get("stratum"))
        if pair.get("pair_id"):
            cell = (str(pair.get("type")), stratum, str(pair.get("severity")))
            cells["errored"].setdefault(cell, set()).add(f"{pair['pair_id']}|err")
        clean = pair.get("clean")
        if isinstance(clean, str) and pair.get("key"):
            digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()[:12]
            cells["clean"].setdefault((stratum,), set()).add(f"{pair['key']}|clean|{digest}")
    return {group: {c: sorted(k) for c, k in by.items()} for group, by in cells.items()}


def allocate(sizes: Mapping[Any, int], n: int) -> dict[Any, int]:
    """`n` places shared over the cells in proportion to their sizes, by largest remainder
    (ties to the larger cell, then the cell's sort order), so the counts sum to `n`."""
    total = sum(sizes.values())
    if not total:
        return dict.fromkeys(sizes, 0)
    exact = {cell: n * size / total for cell, size in sizes.items()}
    out = {cell: int(value) for cell, value in exact.items()}
    order = sorted(sizes, key=lambda c: (-(exact[c] - out[c]), -sizes[c], str(c)))
    for cell in order[: n - sum(out.values())]:
        out[cell] += 1
    return out


def sample_keys(
    cells: Mapping[str, Mapping[tuple[str, ...], Sequence[str]]], n: int, seed: int
) -> list[str]:
    """A fixed-seed sample of `n` note keys: errored and clean notes in proportion to their
    numbers, each spread over its cells in proportion, drawn at random within each cell."""
    groups = {g: sum(len(k) for k in by.values()) for g, by in cells.items()}
    if n > sum(groups.values()):
        raise ValueError(f"a sample of {n} is larger than the {sum(groups.values())} notes")
    rng = random.Random(seed)
    keys: list[str] = []
    for group, places in sorted(allocate(groups, n).items()):
        by = cells[group]
        counts = allocate({cell: len(k) for cell, k in by.items()}, places)
        for cell in sorted(by):
            keys += rng.sample(list(by[cell]), counts[cell])
    return sorted(keys)


def write_spec(
    pairs_path: Path, out: Path, sample: int | None = None, seed: int | None = None
) -> int:
    """Every errored note and every distinct clean note in the pairs file, or a stratified
    fixed-seed sample of them (`sample` notes, with `seed`), as a spec for `prepare` and
    `score` (the development pairs file holds development data only)."""
    pairs = _pairs(pairs_path)
    cells = spec_cells(pairs)
    if sample is None:
        keys = {k for by in cells.values() for ks in by.values() for k in ks}
    else:
        if seed is None:
            raise ValueError("a sample needs a seed")
        keys = set(sample_keys(cells, sample, seed))
    errored = sum(1 for k in keys if k.endswith("|err"))
    consultations = sorted({str(p["key"]) for p in pairs if p.get("key") and _touches(p, keys)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"note_keys": sorted(keys), "consultations": consultations}
            | ({} if sample is None else {"sample": sample, "seed": seed}),
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {out}: {len(keys)} notes over {len(consultations)} consultations: "
        f"{errored} errored, {len(keys) - errored} clean"
    )
    return 0


def _touches(pair: Mapping[str, Any], keys: set[str]) -> bool:
    """Whether a pair's consultation has a note among `keys`."""
    return any(k == f"{pair.get('pair_id')}|err" or k.startswith(f"{pair['key']}|") for k in keys)


# score


@dataclass
class Note:
    key: str
    kind: str
    pair: Mapping[str, Any]
    report: Mapping[str, Any] | None
    pass_a: list[Flag] = field(default_factory=list)
    others: list[Flag] = field(default_factory=list)
    info: list[Flag] = field(default_factory=list)
    """Partial verdicts, as information (the report's `info` list, and partial flags in
    reports written before partial verdicts became information)."""

    @property
    def flagged(self) -> bool:
        """Pass A omission flags alone would make the note warn (or fail)."""
        if self.report is None:
            return False
        return decide(self.pass_a, mode=self.report.get("mode", "full")) != "pass"

    def count_at(self, level: str) -> int:
        return sum(1 for f in self.pass_a if RANK[f.confidence] >= RANK[level])

    @property
    def failed(self) -> list[str]:
        """Reasons Pass A could not finish for this note (an extraction, audit or presence
        check that failed): such a note is excluded from the rates, not counted as clean."""
        if self.report is None:
            return []
        return [
            str((item.get("reasons") or ["?"])[0])
            for item in self.report.get("unchecked", [])
            if item.get("category") == "pass_a"
        ]

    @property
    def critical_absent(self) -> bool:
        """Route one's rule: a critical Pass A omission judged absent, at confirmed or
        higher confidence (our confirmed second look standing in for its refute stage)."""
        return bool(route_one_flags(self))


# Operating points: which Pass A omission flags count, and how many are needed.


def decision_flags(note: Note) -> list[Flag]:
    """The flags that make the decision rule warn on their own."""
    return [
        f for f in note.pass_a if f.severity in ("critical", "supporting") or f.verdict == "partial"
    ]


def sweep_flags(note: Note, level: str, floor: str) -> list[Flag]:
    return [
        f
        for f in note.pass_a
        if RANK[f.confidence] >= RANK[level]
        and SEVERITY_RANK.get(f.severity, 0) >= SEVERITY_RANK[floor]
    ]


def route_one_flags(note: Note) -> list[Flag]:
    return [
        f
        for f in note.pass_a
        if f.severity == "critical" and f.verdict == "absent" and RANK[f.confidence] >= 2
    ]


def load_reports(
    run: Path, manifest: Path | None = None
) -> tuple[dict[str, Mapping[str, Any]], float | None, list[str]]:
    """Reports by note key, the run's wall-clock seconds, and the batch's problems."""
    mapping: dict[str, str] = {}
    if manifest is not None:
        cases = json.loads(manifest.read_text(encoding="utf-8")).get("cases", {})
        mapping = {str(case).casefold(): str(info["note_key"]) for case, info in cases.items()}

    def key_of(case: str) -> str:
        return mapping.get(case.casefold(), note_key(case))

    batch = run / "batch.json"
    reports: dict[str, Mapping[str, Any]] = {}
    if batch.is_file():
        data = json.loads(batch.read_text(encoding="utf-8"))
        for case in data.get("cases", []):
            reports[key_of(str(case["id"]))] = case["report"]
        elapsed = data.get("elapsed_seconds")
        return (
            reports,
            float(elapsed) if elapsed is not None else None,
            list(data.get("problems", [])),
        )
    for path in sorted(run.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "flags" in data and data.get("consultation_id"):
            reports[key_of(str(data["consultation_id"]))] = data
    return reports, None, []


def build_notes(
    keys: Sequence[str],
    pairs: Sequence[Mapping[str, Any]],
    reports: Mapping[str, Any],
    *,
    partial_as_flags: bool = False,
) -> list[Note]:
    """Each note with its flags. Partial verdicts are information unless `partial_as_flags`,
    whichever way the run reported them, so runs from before and after the change score
    alike."""
    notes = []
    for key in keys:
        pair, _, kind = resolve(key, pairs)
        report = reports.get(key)
        note = Note(key, kind, pair, report)
        raws = [*(report or {}).get("flags", []), *(report or {}).get("info", [])]
        for raw in raws:
            flag = Flag.model_validate(raw)
            if flag.type != "omission":
                note.others.append(flag)
            elif flag.verdict == "partial" and not partial_as_flags:
                note.info.append(flag)
            else:
                note.pass_a.append(flag)
        notes.append(note)
    return notes


def _rate(n: int, total: int) -> str:
    return f"{n}/{total} ({100 * n / total:.1f}%)" if total else f"{n}/0 (-)"


def _counts(counter: Mapping[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(counter.items())) or "none"


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.casefold()) if w not in _STOP]


def similarity(a: str, b: str) -> float:
    """The greatest of difflib's ratio, the content-word Jaccard overlap and, when the
    shorter side has at least two content words, the share of them the other contains (a
    flag may name the removed fact in fewer words; one word alone is too weak)."""
    ta, tb = _tokens(a), _tokens(b)
    ratio = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio() if ta and tb else 0.0
    sa, sb = set(ta), set(tb)
    jaccard = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
    shorter = min(len(sa), len(sb))
    contained = len(sa & sb) / shorter if shorter >= 2 else 0.0
    return max(ratio, jaccard, contained)


def removed_facts(pair: Mapping[str, Any]) -> list[str]:
    facts = [pair.get("fact")]
    target = pair.get("target")
    if isinstance(target, Mapping):
        facts.append(target.get("fact"))
    return [f for f in dict.fromkeys(facts) if isinstance(f, str) and f.strip()]


_GENERIC = frozenset(
    "mg mcg g ml l units unit iu tablet tablets puff puffs drop drops daily twice once three "
    "times day days week weeks month months year years night morning evening bd tds qds od "
    "prn nocte mane dose doses given taking takes take started start stopped continue "
    "history family mother father mild severe pain left right both".split()
)
_NEGATION = frozenset("no not denies denied without never none negative".split())
_NUMBER = re.compile(r"\d+(?:[./]\d+)*")


def comparable(a: str, b: str) -> bool:
    """Whether candidate fact `b` can name removed fact `a`. They must share a distinctive
    word or a drug, and agree on negation: shared units, frequencies or filler alone never
    make two facts the same. When `a` names drugs (by the lexicon's generic names, so a
    brand matches its generic), `b` must name one of them. When `a` gives numbers, `b` must
    give at least one of them and none that `a` lacks."""
    drugs_a = _drugs(a)
    shared_drug = bool(drugs_a & _drugs(b))
    if drugs_a and not shared_drug:
        return False
    ta, tb = set(_tokens(a)), set(_tokens(b))
    distinctive = {w for w in ta & tb if w not in _GENERIC and not w[0].isdigit()}
    if not distinctive and not shared_drug:
        return False
    na, nb = set(_NUMBER.findall(a)), set(_NUMBER.findall(b))
    if na and (not nb or not nb <= na):
        return False
    neg_a = bool(set(a.casefold().split()) & _NEGATION)
    neg_b = bool(set(b.casefold().split()) & _NEGATION)
    return neg_a == neg_b


def _drugs(text: str) -> set[str]:
    """The generic names of the drugs `text` names."""
    return {generic for match in find_drugs(text) for generic in match.generic}


def match_texts(pair: Mapping[str, Any], texts: Sequence[str]) -> tuple[float, int | None, str]:
    """(score, index into `texts`, removed fact) of the best-scoring text comparable with
    one of the pair's removed facts; (0.0, None, "") when none is comparable."""
    best: tuple[float, int | None, str] = (0.0, None, "")
    for fact in removed_facts(pair):
        for i, text in enumerate(texts):
            if comparable(fact, text) and (score := similarity(fact, text)) > best[0]:
                best = (score, i, fact)
    return best


def best_match(note: Note, flags: Iterable[Flag]) -> tuple[float, Flag | None, str]:
    """The best-scoring flag whose fact is comparable with a removed fact."""
    listed = list(flags)
    score, index, fact = match_texts(note.pair, [f.fact or "" for f in listed])
    return score, None if index is None else listed[index], fact


def route_one_facts(row: Mapping[str, Any] | None) -> list[str] | None:
    """The facts a route-one (B3) record graded critical and judged absent: from its
    detail.missing_facts (a harness record), else from the `baseline_facts` the split keeps
    for a released development row. None when the record holds neither (a released row from
    before the split kept them) or is missing or failed."""
    if row is None or row.get("parse_failure") is True:
        return None
    detail = row.get("detail")
    facts = detail.get("missing_facts") if isinstance(detail, Mapping) else None
    if not isinstance(facts, list):
        return baseline_flagged_facts(row)
    return [
        str(f.get("fact") or "")
        for f in facts
        if isinstance(f, Mapping)
        and f.get("severity") == "critical"
        and f.get("verdict") == "absent"
    ]


def load_judged(path: Path | None, notes: Iterable[Note]) -> dict[str, Judgement | None] | None:
    """Each omission note's valid cached judgement (None for one the cache lacks, or whose
    entry is stale or invalid), by note key; None when no cache is given."""
    if path is None:
        return None
    cache = fact_judge.load_cache(path)
    out: dict[str, Judgement | None] = {}
    for n in notes:
        if n.kind == "omission" and n.report is not None:
            it = fact_judge.item(str(n.pair.get("pair_id")), removed_facts(n.pair), n.report)
            out[n.key] = fact_judge.lookup(cache, it)
    return out


def model_names(judgement: Judgement | None, flags: Iterable[Flag]) -> bool:
    """Whether the judge named one of `flags` as stating the removed fact."""
    return judgement is not None and bool(set(judgement.flags.ids) & {f.id for f in flags})


def confirmed_absent(note: Note) -> list[Flag]:
    return [f for f in note.pass_a if f.verdict == "absent" and RANK[f.confidence] >= 2]


Grade = tuple[int, int, int, int, int, int]
NO_GRADE: Grade = (0, 0, 0, 0, 0, 0)


def grade(note: Note) -> Grade:
    """A note's graded score, compared in order: its confirmed absent flags the second look
    called clearly absent (none in a run before v5), then its confirmed absent critical
    flags, then supporting ones, then all confirmed absent flags, then critical partial
    verdicts, then all partial verdicts. Built from the report alone; NO_GRADE is never
    flagged."""
    absent = confirmed_absent(note)
    partials = [*note.info, *(f for f in note.pass_a if f.verdict == "partial")]
    return (
        sum(f.absence == "clear" for f in absent),
        sum(f.severity == "critical" for f in absent),
        sum(f.severity == "supporting" for f in absent),
        len(absent),
        sum(f.severity == "critical" for f in partials),
        len(partials),
    )


def matched_grade(
    omissions: Sequence[Note], cleans: Sequence[Note], alarms: int, total: int
) -> Grade | None:
    """The most lenient grade threshold whose false-alarm rate on `cleans` is at most
    alarms/total, or None when even the strictest passes it. The thresholds are the grades
    the notes have; a lower threshold never has fewer false alarms."""
    if total == 0:
        return None  # no reference rate to match
    grades = sorted({grade(n) for n in [*omissions, *cleans]} - {NO_GRADE}, reverse=True)
    best = None
    for g in grades:
        if sum(grade(n) >= g for n in cleans) * total <= alarms * len(cleans):
            best = g
    return best


MECHANISMS = ("partial_element", "core_kind", "never_drop", "clear", "probable")
"""v5's tagged changes: a partial verdict's missing element (`pass_a.partial_element`), a
fact the code made core or kept against the audit (`core_kind`, `never_drop` in the trace),
and the second look's grade of a confirmed absence (`clear`, `probable`)."""
PARTIAL_ELEMENT = "pass_a.partial_element"


def mechanisms_of(note: Note, flag: Flag) -> set[str]:
    """The v5 mechanisms behind one of a note's flags: by its rule for a partial element,
    by its trace fact's `forced` tags, and by its `absence`."""
    tags: set[str] = {flag.absence} if flag.absence else set()
    fact = flag.fact or ""
    if flag.rule_id == PARTIAL_ELEMENT:
        tags.add("partial_element")
        fact = fact.rsplit(": missing ", 1)[0]  # "<fact>: missing <element>"
    trace = (note.report or {}).get("pass_a_trace") or []
    # The checked copy of the fact, when two trace facts share its text
    steps = sorted((s for s in trace if s.get("fact") == fact), key=lambda s: not s.get("checked"))
    tags.update(str(t) for t in (steps[0] if steps else {}).get("forced") or [])
    return tags & set(MECHANISMS)


def score(args: argparse.Namespace) -> int:
    pairs = _pairs(args.pairs)
    keys = sorted(json.loads(args.spec.read_text(encoding="utf-8"))["note_keys"])
    reports, elapsed, problems = load_reports(args.run, args.manifest)
    notes = build_notes(keys, pairs, reports, partial_as_flags=args.partial_as_flags)
    missing = [n.key for n in notes if n.report is None]
    present = [n for n in notes if n.report is not None]
    failed = [n for n in present if n.failed]
    scored = [n for n in notes if not n.failed]
    omissions = [n for n in scored if n.kind == "omission"]
    cleans = [n for n in scored if n.kind == "clean"]
    extra = sorted(set(reports) - set(keys))
    out: list[str] = []
    say = out.append

    say(f"== Coverage: {len(present)}/{len(notes)} notes have a report")
    say(
        "   partial verdicts: "
        + ("counted as flags (--partial-as-flags)" if args.partial_as_flags else "information")
        + f"; {sum(len(n.info) for n in present)} listed as information"
    )
    kinds = Counter(n.kind for n in notes)
    say(f"   notes in the spec by kind: {_counts(kinds)}")
    for key in missing:
        say(f"   missing (counted as not flagged): {key}")
    for n in failed:
        say(f"   Pass A failed (excluded from the rates): {n.key} ({', '.join(n.failed)})")
    for key in extra:
        say(f"   report not in the spec (ignored): {key}")
    for problem in problems:
        say(f"   batch problem: {_label(problem)}")

    threshold = args.match_threshold

    def names_fact(note: Note, flags: Sequence[Flag]) -> bool:
        value, flag, _ = best_match(note, flags)
        return flag is not None and value >= threshold

    def point(pick: Callable[[Note], list[Flag]], k: int = 1) -> tuple[int, int, int]:
        """(detected, correctly detected, false alarms) at one operating point."""
        detected = correct = 0
        for n in omissions:
            flags = pick(n)
            if len(flags) >= k:
                detected += 1
                correct += names_fact(n, flags)
        alarms = sum(1 for n in cleans if len(pick(n)) >= k)
        return detected, correct, alarms

    say("== Decision rule (Pass A omission flags only)")
    det = [n for n in omissions if n.flagged]
    fa = [n for n in cleans if n.flagged]
    _, correct_det, _ = point(decision_flags)
    say(f"   detection on omission notes: {_rate(len(det), len(omissions))}")
    say(
        f"   correct detection (a counted flag names the removed fact): "
        f"{_rate(correct_det, len(omissions))}"
    )
    say(f"   false alarms on clean notes: {_rate(len(fa), len(cleans))}")
    for label, attr in (("trace", "residual_level"), ("removed-fact severity", "severity")):
        groups: dict[str, list[Note]] = {}
        for n in omissions:
            groups.setdefault(str(n.pair.get(attr)), []).append(n)
        detail = ", ".join(
            f"{g} {_rate(sum(x.flagged for x in members), len(members))}"
            for g, members in sorted(groups.items())
        )
        say(f"   detection by {label}: {detail}")
    for kind in ("add", "change"):
        members = [n for n in notes if n.kind == kind]
        if members:
            say(
                f"   (information) {kind} notes with a Pass A flag: "
                f"{_rate(sum(n.flagged for n in members), len(members))}"
            )
    others = Counter(f"{f.type}/{f.rule_id or f.category}" for n in present for f in n.others)
    say(f"   (information, not scored) other warnings: {_counts(others)}")
    clean_others = sum(1 for n in cleans if n.others)
    say(f"   (information) clean notes with any other warning: {clean_others}/{len(cleans)}")

    say("== Route one's rule: a critical fact judged absent, at confirmed confidence")
    r_det, r_correct, r_alarms = point(route_one_flags)
    say(f"   detection on omission notes: {_rate(r_det, len(omissions))}")
    say(f"   correct detection: {_rate(r_correct, len(omissions))}")
    say(f"   false alarms on clean notes: {_rate(r_alarms, len(cleans))}")
    for name, rows in _route_one_stores(args).items():
        say(f"== Route one ({name}) on the same notes: its own rule, a critical fact judged")
        say("   absent, from the fact text in its records")
        usable_rows = {n.key: route_one_facts(rows.get(n.key)) for n in [*omissions, *cleans]}
        if not any(v is not None for v in usable_rows.values()):
            say(f"   its records carry no fact text: {NO_SPLIT}")
            continue
        lost = [key for key, v in usable_rows.items() if v is None]
        if lost:
            say(f"   {len(lost)} notes without a usable record (counted as not flagged)")
        hits: list[tuple[Note, float, str, str]] = []
        for n in omissions:
            facts = usable_rows[n.key] or []
            if facts:
                value, index, fact = match_texts(n.pair, facts)
                hits.append((n, value, fact, facts[index] if index is not None else ""))
        right = sum(1 for _, value, _, text in hits if text and value >= threshold)
        alarms = sum(1 for n in cleans if usable_rows[n.key])
        say(f"   detection on omission notes: {_rate(len(hits), len(omissions))}")
        say(f"   correct detection: {_rate(right, len(omissions))}")
        say(f"   false alarms on clean notes: {_rate(alarms, len(cleans))}")
        rate = _rate(right, len(hits))
        say(f"   correct-fact rate: {rate} of detected notes name the removed fact")
        say("   per detected note (pair id, score, matched):")
        for n, value, fact, text in hits:
            hit = "yes" if text and value >= threshold else "no"
            say(f"   {n.pair.get('pair_id')}  {value:.2f}  {hit}")
            if args.show_facts and text:
                say(f"      removed fact:   {fact}")
                say(f"      route one fact: {text}")

    say("== k-and-c sweep: at least k Pass A omission flags at confidence c or higher and")
    say("   severity at or above the floor")
    say("   c          severity     k  detection          correct            false alarms")
    for level in LEVELS:
        for floor_label, floor in FLOORS:
            for k in KS:
                d, c, f = point(partial(sweep_flags, level=level, floor=floor), k)
                say(
                    f"   {level:<10} {floor_label:<12} {k}  {_rate(d, len(omissions)):<18} "
                    f"{_rate(c, len(omissions)):<18} {_rate(f, len(cleans))}"
                )

    say(f"== Provisional correct-fact rate (match threshold {args.match_threshold})")
    points: list[tuple[str, Callable[[Note], list[Flag]]]] = [
        ("decision rule", decision_flags),
        ("k=1, c=possible", partial(sweep_flags, level="possible", floor="peripheral")),
        ("k=1, c=confirmed", partial(sweep_flags, level="confirmed", floor="peripheral")),
        ("route one's rule", route_one_flags),
    ]
    for label, pick in points:
        detected, correct, _ = point(pick)
        say(f"   {label}: {_rate(correct, detected)} of detected notes name the removed fact")
    say("   per note at the decision rule (pair id, flag id, score, matched):")
    for n in det:
        value, flag, fact = best_match(n, n.pass_a)
        hit = "yes" if value >= args.match_threshold and flag is not None else "no"
        say(f"   {n.pair.get('pair_id')}  {flag.id if flag else '-'}  {value:.2f}  {hit}")
        if args.show_facts and flag is not None:
            say(f"      removed fact: {fact}")
            say(f"      flag fact:    {flag.fact}")

    judgements = load_judged(args.judgements, omissions)
    if judgements is not None:
        say("== Model-assisted correct detection: a judge (fact_judge.py) says whether a counted")
        say("   flag states the removed fact; fact texts only, never the transcript or note. The")
        say("   word matcher is beside it as a cross-check.")
        lacking = [n for n in omissions if judgements.get(n.key) is None]
        say(f"   omission notes without a valid judgement: {len(lacking)} (counted as not named)")
        for n in lacking:
            say(f"      no judgement: {n.pair.get('pair_id')}")
        for label, pick in points:
            words = model = 0
            for n in omissions:
                if flags := pick(n):
                    words += names_fact(n, flags)
                    model += model_names(judgements.get(n.key), flags)
            say(
                f"   {label}: correct detection by words {_rate(words, len(omissions))}, "
                f"by the model {_rate(model, len(omissions))}"
            )
        say("   where they differ at the decision rule (pair id, counted flags):")
        differ = 0
        for n in det:
            flags = decision_flags(n)
            by_words = names_fact(n, flags)
            by_model = model_names(judgements.get(n.key), flags)
            if by_words != by_model:
                differ += 1
                side = "words yes, model no" if by_words else "words no, model yes"
                say(f"   {side}: {n.pair.get('pair_id')}  {','.join(x.id for x in flags)}")
                if args.show_facts:
                    for fact in removed_facts(n.pair):
                        say(f"      removed fact: {fact}")
                    for x in flags:
                        say(f"      flag {x.id}:      {x.fact}")
        say(f"   detected notes where they differ: {differ}/{len(det)}")

    say("== v5 mechanisms: decision-rule flags by the change behind them. 'with': notes with")
    say("   one of its flags; 'only': notes flagged by its flags alone (what it adds, and what")
    say("   it costs on clean notes). core_kind and never_drop count the code's overrides of the")
    say("   audit only; what the prompt changes do shows only against a run before v5")
    tagged = {
        n.key: [(f, mechanisms_of(n, f)) for f in decision_flags(n)] for n in [*omissions, *cleans]
    }
    if not any(tags for flags in tagged.values() for _, tags in flags):
        say("   no flag carries a v5 tag (a run before v5)")
    else:
        for mechanism in MECHANISMS:
            for view in ("with", "only"):

                def picked(n: Note, m: str = mechanism, v: str = view) -> list[Flag]:
                    flags = tagged[n.key]
                    mine = [f for f, tags in flags if m in tags]
                    if v == "only" and len(mine) < len(flags):
                        return []
                    return mine

                marked = [n for n in omissions if picked(n)]
                words = sum(names_fact(n, picked(n)) for n in marked)
                line = (
                    f"   {mechanism:<16}{view:<5} detection {_rate(len(marked), len(omissions))}, "
                    f"correct (words) {_rate(words, len(omissions))}"
                )
                if judgements is not None:
                    model = sum(model_names(judgements.get(n.key), picked(n)) for n in marked)
                    line += f", correct (model) {_rate(model, len(omissions))}"
                alarms_m = sum(1 for n in cleans if picked(n))
                say(line + f", false alarms {_rate(alarms_m, len(cleans))}")

    references = _references(args)
    if references:
        say("== Reference methods on the same notes: detection and false alarms (their own")
        say("   rules; correct detection applies to route one only, from the fact text the")
        say("   split keeps for development rows; route two is score-only; notes without a")
        say("   usable record are left out of these rates)")
    baseline_correct: dict[str, str] = dict.fromkeys(
        (name for name in SCORE_ONLY if name in references), NOT_APPLICABLE
    )
    if args.released is not None and not split_carries_facts(args.released):
        baseline_correct.update(dict.fromkeys(BASELINES, f"correct detection: {NO_SPLIT}"))
    elif judgements is not None and args.judgements is not None:
        cache = fact_judge.load_cache(args.judgements)
        for name, by_note in baseline_items(args.released, omissions).items():
            named = sum(
                1
                for it in by_note.values()
                if it is not None
                and (j := fact_judge.lookup(cache, it)) is not None
                and j.flags.same
            )
            unjudged_b = sum(
                1 for it in by_note.values() if it is None or fact_judge.lookup(cache, it) is None
            )
            gap = f" ({unjudged_b} flagged notes without fact text or judgement)"
            baseline_correct[name] = f"correct (model) {_rate(named, len(omissions))}" + (
                gap if unjudged_b else ""
            )
        for name, count in too_long_notes(args.released, omissions).items():
            if count and name in baseline_correct:
                baseline_correct[name] += (
                    f"; {count} flagged notes with a too-long fact, never judged"
                )
    for name, ref in references.items():
        omit = [ref.get(n.key) for n in omissions]
        clean = [ref.get(n.key) for n in cleans]
        known_o, known_c = [v for v in omit if v is not None], [v for v in clean if v is not None]
        say(
            f"   {name}: detection {_rate(sum(known_o), len(known_o))}, "
            f"false alarms {_rate(sum(known_c), len(known_c))}"
        )
        if name in baseline_correct:
            say(f"      {baseline_correct[name]}")
        if len(known_o) < len(omit) or len(known_c) < len(clean):
            say(
                f"      ({len(omit) - len(known_o)} omission and {len(clean) - len(known_c)} "
                "clean notes without a usable record)"
            )

    say("== Graded note score: (clearly absent; confirmed absent critical, supporting, all;")
    say("   then critical partials, all partials), compared in order; at each reference's")
    say("   false-alarm rate, the most lenient grade threshold that does not pass it")
    if not references:
        say("   no reference records given (--released, --gate1b)")
    for name, ref in references.items():
        known_c = [v for v in (ref.get(n.key) for n in cleans) if v is not None]
        target = matched_grade(omissions, cleans, sum(known_c), len(known_c))
        if target is None:
            say(f"   at {name}'s {_rate(sum(known_c), len(known_c))}: no threshold qualifies")
            continue
        graded = [
            (n, [*confirmed_absent(n), *n.info, *(f for f in n.pass_a if f.verdict == "partial")])
            for n in omissions
            if grade(n) >= target
        ]
        words = sum(names_fact(n, flags) for n, flags in graded)
        say(f"   at {name}'s false alarms {sum(known_c)}/{len(known_c)}: grade >= {target}")
        line = (
            f"      detection {_rate(len(graded), len(omissions))}, correct (words) "
            f"{_rate(words, len(omissions))}"
        )
        if judgements is not None:
            model = sum(model_names(judgements.get(n.key), flags) for n, flags in graded)
            line += f", correct (model) {_rate(model, len(omissions))}"
        raised = sum(grade(n) >= target for n in cleans)
        say(line + f", false alarms {_rate(raised, len(cleans))}")
        if name in baseline_correct:
            say(f"      {name} itself: {baseline_correct[name]}")

    say("== Per-note agreement on omission and clean notes (information, not a gate)")
    say("   ours: the decision rule on Pass A omission flags; against B3 also 'critical")
    say("   absent' (route one's rule). A note we have no usable report for counts as a")
    say("   disagreement; a reference row that is unknown is left out.")
    judged = [n for n in notes if n.kind in ("omission", "clean")]
    usable = {n.key: n for n in judged if n.report is not None and not n.failed}
    if not references:
        say("   no reference records given (--released, --gate1b)")
    for name, ref in references.items():
        views: list[tuple[str, Callable[[Note], bool]]] = [("decision rule", _decision)]
        if "B3" in name:
            views.append(("critical absent", _critical))
        for view, rule in views:
            keyed = [n.key for n in judged if ref.get(n.key) is not None]
            agree = [key for key in keyed if key in usable and rule(usable[key]) == ref[key]]
            lost = [key for key in keyed if key not in usable]
            say(
                f"   with {name} ({view}): {_rate(len(agree), len(keyed))}"
                + (f"; {len(lost)} without a usable report" if lost else "")
            )
            for key in keyed:
                if key in usable and rule(usable[key]) != ref[key]:
                    ours_flag = "flagged" if rule(usable[key]) else "not"
                    theirs = "flagged" if ref[key] else "not"
                    say(f"      differs: {key} (ours {ours_flag}, {name} {theirs})")

    say("== Calls, tokens, cost and time")
    usages = [n.report.get("usage") or {} for n in present if n.report is not None]
    if usages:

        def mean(field_: str) -> float:
            return sum(float(u.get(field_) or 0) for u in usages) / len(usages)

        costs = [u.get("cost_usd") for u in usages]
        known = [float(c) for c in costs if c is not None]
        say(
            f"   per note (mean over {len(usages)}): calls {mean('calls'):.1f} own + "
            f"{mean('shared_calls'):.1f} shared; tokens in {mean('input_tokens'):.0f}, "
            f"out {mean('output_tokens'):.0f}, cache read {mean('cache_read_tokens'):.0f}, "
            f"cache write {mean('cache_write_tokens'):.0f}"
        )
        if not known:
            say("   cost: unknown (no [model.prices] in the run's config)")
        else:
            say(
                f"   cost: ${sum(known):.4f} in all, ${sum(known) / len(known):.4f} per note"
                + (
                    ""
                    if len(known) == len(costs)
                    else f" (known for {len(known)} of {len(costs)} notes)"
                )
            )
    if elapsed is not None and present:
        say(f"   wall clock: {elapsed:.0f} s in all, {elapsed / len(present):.1f} s per note")
    models = sorted(
        {
            str(version)
            for n in present
            if n.report is not None
            and (version := (n.report.get("model") or {}).get("version")) is not None
        }
    )
    say(f"   model ids reported: {', '.join(models) or '-'}")
    reasons = Counter(
        reason
        for n in present
        if n.report is not None
        for item in n.report.get("unchecked", [])
        for reason in item.get("reasons", [])[:1]
    )
    say(f"   unchecked items by primary reason: {_counts(reasons)}")
    print("\n".join(out))
    return 0


# misses

MISS_CLASSES = (
    "detected",
    "not extracted",
    "dropped by the audit",
    "contextual, never checked",
    "weak match (compound?)",
    "judged present",
    "judged partial",
    "rescued by the second look",
    "unchecked",
    "no matching flag",
)
"""Where an omission note's removed fact ended up. "detected": a confirmed absent flag names
it. "weak match (compound?)": no extracted fact reaches the match threshold, but one
comparable fact scores at least WEAK_MATCH, as when the removed fact is buried in a longer
compound fact. "no matching flag" is the only class a run without a Pass A trace can give
besides "detected" and "judged partial"."""
WEAK_MATCH = 0.25


def trace_match(
    pair: Mapping[str, Any], trace: Sequence[Mapping[str, Any]]
) -> tuple[float, Mapping[str, Any] | None]:
    """The best-scoring trace fact comparable with a removed fact, with its score."""
    value, index, _ = match_texts(pair, [str(step.get("fact") or "") for step in trace])
    return value, None if index is None else trace[index]


def classify_miss(step: Mapping[str, Any] | None) -> str:
    """A removed fact's class from its matched trace fact (None: nothing matched)."""
    if step is None or not step.get("evidence_verified", True):
        return "not extracted"
    if step.get("audit") == "dropped":
        return "dropped by the audit"
    if step.get("audit") == "failed" or (step.get("checked") and step.get("verdict") is None):
        return "unchecked"
    if step.get("note_quote_verified") is False:
        return "unchecked"  # its present or partial quote was not found in the note
    if not step.get("checked"):
        return "contextual, never checked"
    verdict, look = step.get("verdict"), step.get("second_look")
    if verdict == "present":
        return "judged present"
    if verdict == "partial":
        return "judged partial"
    if look == "rescued":
        return "rescued by the second look"
    if look == "confirmed":
        return "detected"
    return "unchecked"


def model_step(
    judgement: Judgement, trace: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    """The trace fact the judge named, preferring the one that went furthest (a confirmed
    second look, then a checked fact, then any); None when it named none."""
    named = [step for step in trace if str(step.get("id")) in set(judgement.facts.ids)]

    def reach(step: Mapping[str, Any]) -> int:
        return (step.get("second_look") == "confirmed") * 2 + bool(step.get("checked"))

    return max(named, key=reach, default=None)


def misses(args: argparse.Namespace) -> int:
    """Trace each omission note's removed fact through Pass A and classify where it went."""
    pairs = _pairs(args.pairs)
    keys = sorted(json.loads(args.spec.read_text(encoding="utf-8"))["note_keys"])
    reports, _, _ = load_reports(args.run, args.manifest)
    notes = [
        n
        for n in build_notes(keys, pairs, reports)
        if n.kind == "omission" and n.report is not None and not n.failed
    ]
    judgements = load_judged(args.judgements, notes)
    threshold = args.match_threshold
    traced = sum(1 for n in notes if n.report is not None and "pass_a_trace" in n.report)
    out: list[str] = []
    say = out.append
    say(f"== Removed facts through Pass A ({len(notes)} omission notes, {traced} with a trace)")
    if traced < len(notes):
        say("   a note without a Pass A trace ([report] pass_a_trace) can only be classed")
        say("   from its flags: detected, judged partial or no matching flag")
    if judgements is not None:
        say("   matched by the judge (fact_judge.py) where it has a valid judgement, else by")
        say("   words; '(words: ...)' gives the word matcher's class where it differs")
    say("   pair id, trace, removed-fact severity, word match score, class, then the stages:")
    say("   extracted / audit (core or contextual, severity) / check verdict / second look")
    classes: Counter[str] = Counter()
    by_trace: dict[str, Counter[str]] = {}
    differ: Counter[tuple[str, str]] = Counter()
    unjudged = untraced = 0
    for n in notes:
        assert n.report is not None
        value, step, closest, label, named = _word_class(n, threshold)
        judgement = judgements.get(n.key) if judgements is not None else None
        words = label
        trace = n.report.get("pass_a_trace")
        if judgements is not None and judgement is None:
            unjudged += 1
        elif judgements is not None and trace is None:
            untraced += 1
        elif judgement is not None and trace is not None:
            step, closest = model_step(judgement, trace), None
            label = classify_miss(step)
            confirmed = [f for f in [*n.pass_a, *n.info] if f.id in set(judgement.flags.ids)]
            if any(f.verdict == "absent" and RANK[f.confidence] >= 2 for f in confirmed):
                label = "detected"
            named = confirmed[0] if confirmed and step is None else None
            if words != label:
                differ[(words, label)] += 1
        classes[label] += 1
        level = str(n.pair.get("residual_level"))
        by_trace.setdefault(level, Counter())[label] += 1
        aside = f"  (words: {words})" if words != label else ""
        say(
            f"   {n.pair.get('pair_id')}  {level}  {n.pair.get('severity')}  {value:.2f}  "
            f"{label}{aside}"
        )
        if step is not None:
            core = {True: "core", False: "contextual", None: "-"}[step.get("core")]
            say(
                f"      extracted {step.get('id')} ({step.get('origin')}) / audit "
                f"{step.get('audit', '-')} ({core}, {step.get('severity') or '-'}) / check "
                f"{step.get('verdict') or ('not checked' if not step.get('checked') else '-')}"
                f" / second look {step.get('second_look') or '-'}"
            )
        elif named is not None:
            say(f"      flag {named.id}: {named.verdict}, {named.confidence}, {named.severity}")
        if args.show_facts:
            for fact in removed_facts(n.pair):
                say(f"      removed fact: {fact}")
            if step is not None:
                say(f"      our fact:     {step.get('fact')}")
                if step.get("note_quote"):
                    say(f'      note quote:   "{step.get("note_quote")}"')
            elif closest is not None and label == "weak match (compound?)":
                say(f"      closest fact: {closest.get('fact')}")
            elif named is not None:
                say(f"      flag fact:    {named.fact}")
    say("== Classes")
    for label in MISS_CLASSES:
        if classes[label]:
            say(f"   {label}: {classes[label]}")
    say("== Classes by trace")
    for level, counter in sorted(by_trace.items()):
        detail = ", ".join(f"{c} {counter[c]}" for c in MISS_CLASSES if counter[c])
        say(f"   {level} ({sum(counter.values())}): {detail}")
    if judgements is not None:
        say("== Word and model classes")
        say(f"   notes classed by words for want of a valid judgement: {unjudged}")
        if untraced:
            say(f"   notes classed from their flags for want of a Pass A trace: {untraced}")
        say(f"   notes whose class differs: {sum(differ.values())}")
        for (words, model), count in sorted(differ.items()):
            say(f"   words '{words}' -> model '{model}': {count}")
    print("\n".join(out))
    return 0


def _word_class(
    n: Note, threshold: float
) -> tuple[float, Mapping[str, Any] | None, Mapping[str, Any] | None, str, Flag | None]:
    """The word matcher's view of a removed fact: (score, matched trace fact, closest
    unmatched trace fact, class, the flag that names it)."""
    assert n.report is not None
    flag_score, flag, _ = best_match(n, [*n.pass_a, *n.info])
    named = flag if flag is not None and flag_score >= threshold else None
    trace = n.report.get("pass_a_trace")
    closest = None
    if trace is not None:
        value, step = trace_match(n.pair, trace)
        if step is not None and value < threshold:
            closest, step = step, None
        label = classify_miss(step)
        if label == "not extracted" and closest is not None and value >= WEAK_MATCH:
            label = "weak match (compound?)"
        if named is not None and named.verdict == "absent" and RANK[named.confidence] >= 2:
            label = "detected"  # a confirmed rules candidate names it
        return value, step, closest, label, named
    if named is None:
        label = "no matching flag"
    elif named.verdict == "partial":
        label = "judged partial"
    else:
        label = "detected" if RANK[named.confidence] >= 2 else "no matching flag"
    return flag_score, None, None, label, named


# judge, sample, severity


def _omission_notes(args: argparse.Namespace) -> list[Note]:
    """The run's scored omission notes (with a report, Pass A finished)."""
    pairs = _pairs(args.pairs)
    keys = sorted(json.loads(args.spec.read_text(encoding="utf-8"))["note_keys"])
    reports, _, _ = load_reports(args.run, args.manifest)
    return [
        n
        for n in build_notes(keys, pairs, reports)
        if n.kind == "omission" and n.report is not None and not n.failed
    ]


BASELINES = {"released B3 r1": "b3 r1"}
"""The released reference sets whose correct detection can be judged, by their label in
`score`: route one (B3), whose development rows carry `baseline_facts` once the split has
been re-run (it keeps them for development B3 rows of the w2-pipeline family only)."""
TOO_LONG = "[too_long]"
"""The split's stand-in for a fact longer than its limit (split_omissionbench.TOO_LONG):
never judged nor matched; `score` counts the flagged notes that have one."""
SCORE_ONLY = ("released power-gepa-high r1", "released power-gepa-high r3")
"""Route two's reference sets: a score per note and no named omissions (the harness's
w2_power.py keeps neither the omissions nor the response text), so correct detection does
not apply to them."""
NOT_APPLICABLE = "correct detection: not applicable: score-only method"
NO_SPLIT = (
    "no development row carries baseline_facts: re-run the split "
    "(scripts/split_omissionbench.py, docs/runbooks/m3s4.md)"
)


def baseline_flagged_facts(row: Mapping[str, Any] | None) -> list[str] | None:
    """The facts route one's own rule counts on a note, from the row's `baseline_facts`:
    those graded critical and judged absent. None when the row holds no fact text."""
    facts = None if row is None else row.get("baseline_facts")
    if not isinstance(facts, list):
        return None
    return [f for f in _critical_absent(facts) if f != TOO_LONG]


def _critical_absent(facts: list[Any]) -> list[str]:
    parts = [str(f).split(" | ", 2) for f in facts]
    return [p[2] for p in parts if len(p) == 3 and p[0] == "critical" and p[1] == "absent"]


def too_long_notes(released: Path | None, notes: Iterable[Note]) -> dict[str, int]:
    """For each baseline, how many omission notes its own rule flags have a flagged fact the
    split kept as TOO_LONG (that fact is never judged, so the note may count as not named)."""
    if released is None:
        return {}
    rows = released_rows(released)
    out: dict[str, int] = {}
    for name, store in BASELINES.items():
        out[name] = 0
        for n in notes:
            row = rows[store].get(n.key)
            facts = None if row is None else row.get("baseline_facts")
            if (
                n.kind == "omission"
                and route_one_flag(row) is True
                and isinstance(facts, list)
                and TOO_LONG in _critical_absent(facts)
            ):
                out[name] += 1
    return out


def split_carries_facts(released: Path) -> bool:
    """Whether any of the baselines' development rows carries `baseline_facts` (False until
    the split has been re-run since it learned to keep them)."""
    rows = released_rows(released)
    return any(
        isinstance(row, Mapping) and "baseline_facts" in row
        for store in BASELINES.values()
        for row in rows[store].values()
    )


def baseline_items(
    released: Path | None, notes: Iterable[Note]
) -> dict[str, dict[str, fact_judge.Item | None]]:
    """For each baseline, each omission note its own rule flags: the judge item for its
    flagged facts (ids B1...), or None when its row holds no fact text."""
    if released is None:
        return {}
    rows = released_rows(released)
    out: dict[str, dict[str, fact_judge.Item | None]] = {}
    for name, store in BASELINES.items():
        out[name] = {}
        for n in notes:
            row = rows[store].get(n.key)
            if n.kind != "omission" or route_one_flag(row) is not True:
                continue
            facts = baseline_flagged_facts(row)
            out[name][n.key] = (
                None
                if not facts
                else fact_judge.Item(
                    f"{name}|{n.pair.get('pair_id')}",
                    tuple(removed_facts(n.pair)),
                    (),
                    tuple((f"B{i}", text) for i, text in enumerate(facts, start=1)),
                )
            )
    return out


def run_judge(args: argparse.Namespace) -> int:
    """Judge every scored omission note the cache lacks (paid calls: the user runs this)."""
    cache_path = args.cache or args.run / "fact_judgements.json"
    if args.released is not None and not split_carries_facts(args.released):
        # Before any client is made: judging would send none of the baselines' facts.
        print(f"judge: {NO_SPLIT}; or leave out --released", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config)
        client = create_client(config)
        budget = create_budget(config)
    except InputError as exc:
        print(f"judge: {exc}", file=sys.stderr)
        return 2
    notes = _omission_notes(args)
    items = [
        fact_judge.item(str(n.pair.get("pair_id")), removed_facts(n.pair), n.report)
        for n in notes
        if n.report is not None
    ]
    baselines = baseline_items(args.released, notes)
    for name, by_note in baselines.items():
        found = [it for it in by_note.values() if it is not None]
        print(
            f"judge: {name}: {len(by_note)} omission notes flagged by its rule, "
            f"{len(found)} with fact text"
        )
        items += found
    cache = fact_judge.load_cache(cache_path)
    tag = f"{config.model} effort={config.effort} max_tokens={config.max_tokens}"
    counts = fact_judge.judge(items, client, budget, cache, tag, cache_path)
    spent = budget.spent_usd
    print(f"judge: {len(items)} items (our notes and baselines'): {_counts(counts)}")
    print(
        f"judge: {budget.calls} calls, "
        + (f"${spent:.4f}" if spent is not None else "cost unknown")
        + f"; cache {cache_path}"
    )
    return 0 if sum(counts[k] for k in ("cached", "judged")) == len(items) else 2


SAMPLE_COLUMNS = (
    "pair_id",
    "removed_fact",
    "model_same",
    "model_fact_ids",
    "model_facts",
    "model_flag_ids",
    "word_fact",
    "word_score",
    "word_same",
    "reviewer_same",
    "reviewer_comment",
)


def write_sample(args: argparse.Namespace) -> int:
    """A fixed-seed sample of judged omission notes for a person to spot-check: the removed
    fact, the facts the judge named and the word matcher's best fact (benchmark fact text:
    the file is git-ignored)."""
    notes = _omission_notes(args)
    judgements = load_judged(args.judgements, notes) or {}
    pool = sorted((n for n in notes if judgements.get(n.key) is not None), key=lambda n: n.key)
    chosen = sorted(
        random.Random(args.seed).sample(pool, min(args.n, len(pool))), key=lambda n: n.key
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SAMPLE_COLUMNS)
        for n in chosen:
            judgement = judgements[n.key]
            assert judgement is not None and n.report is not None
            trace = n.report.get("pass_a_trace") or []
            texts = {str(s.get("id")): str(s.get("fact") or "") for s in trace}
            value, step = trace_match(n.pair, trace)
            writer.writerow(
                [
                    n.pair.get("pair_id"),
                    " / ".join(removed_facts(n.pair)),
                    "yes" if judgement.facts.same else "no",
                    " ".join(judgement.facts.ids),
                    " | ".join(texts.get(i, "") for i in judgement.facts.ids),
                    " ".join(judgement.flags.ids),
                    step.get("fact") if step is not None else "",
                    f"{value:.2f}",
                    "yes" if step is not None and value >= args.match_threshold else "no",
                    "",
                    "",
                ]
            )
    print(
        f"wrote {args.out}: {len(chosen)} of {len(pool)} judged omission notes (seed {args.seed})"
    )
    return 0


GRADES = ("critical", "supporting", "peripheral")


def kappa(table: Mapping[tuple[str, str], int]) -> float | None:
    """Cohen's kappa for a square table keyed (ours, benchmark); None when undefined."""
    n = sum(table.values())
    if not n:
        return None
    observed = sum(table.get((g, g), 0) for g in GRADES) / n
    expected = sum(
        sum(table.get((g, b), 0) for b in GRADES) * sum(table.get((o, g), 0) for o in GRADES)
        for g in GRADES
    ) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def severity_table(
    notes: Sequence[Note], pick: Callable[[Note], Mapping[str, Any] | None]
) -> Counter[tuple[str, str]]:
    """(our audit severity, the benchmark's severity) for each removed fact `pick` finds in
    the trace and the audit graded."""
    table: Counter[tuple[str, str]] = Counter()
    for n in notes:
        step = pick(n)
        ours, theirs = (step or {}).get("severity"), n.pair.get("severity")
        if ours in GRADES and theirs in GRADES:
            table[(str(ours), str(theirs))] += 1
    return table


def severity_lines(label: str, table: Mapping[tuple[str, str], int]) -> list[str]:
    n = sum(table.values())
    k = kappa(table)
    over = sum(c for (o, b), c in table.items() if SEVERITY_RANK[o] > SEVERITY_RANK[b])
    under = sum(c for (o, b), c in table.items() if SEVERITY_RANK[o] < SEVERITY_RANK[b])
    lines = [
        f"### {label}",
        "",
        f"Removed facts matched and graded by the audit: {n}. Cohen's kappa: "
        + ("undefined" if k is None else f"{k:.2f}")
        + f". Exact agreement: {sum(table.get((g, g), 0) for g in GRADES)}/{n}; "
        f"we grade higher: {over}; we grade lower: {under}.",
        "",
        "| ours \\ benchmark | " + " | ".join(GRADES) + " | total |",
        "|---|" + "---|" * (len(GRADES) + 1),
    ]
    for o in GRADES:
        row = [table.get((o, b), 0) for b in GRADES]
        lines.append(f"| {o} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
    cols = [sum(table.get((o, b), 0) for o in GRADES) for b in GRADES]
    lines.append("| total | " + " | ".join(map(str, cols)) + f" | {n} |")
    return [*lines, ""]


def severity(args: argparse.Namespace) -> int:
    """Our audit's severity against the benchmark's for each removed fact we extracted, by
    the word matcher and, with --judgements, by the judge's match. Counts only."""
    notes = [n for n in _omission_notes(args) if (n.report or {}).get("pass_a_trace") is not None]
    threshold = args.match_threshold

    def by_words(n: Note) -> Mapping[str, Any] | None:
        value, step = trace_match(n.pair, (n.report or {}).get("pass_a_trace") or [])
        return step if step is not None and value >= threshold else None

    lines = [
        f"# Severity agreement: {args.run.name}",
        "",
        "Our audit's severity for each removed fact we extracted, against the benchmark's",
        f"severity, over {len(notes)} omission notes with a Pass A trace. Development data only;",
        "counts only. Each omission note counts once, so a fact removed from several notes of",
        "one consultation counts once per note. 'Higher' means our audit graded the fact more",
        "severe than the benchmark. Regenerate with `scripts/score_run.py severity`.",
        "",
        *severity_lines(f"Word matcher (score >= {threshold})", severity_table(notes, by_words)),
    ]
    judgements = load_judged(args.judgements, notes)
    if judgements is not None:

        def by_model(n: Note) -> Mapping[str, Any] | None:
            judgement = judgements.get(n.key)
            trace = (n.report or {}).get("pass_a_trace") or []
            return None if judgement is None else model_step(judgement, trace)

        lacking = sum(1 for n in notes if judgements.get(n.key) is None)
        lines += severity_lines(
            f"Model-assisted matcher (fact_judge.py; {lacking} notes without a valid judgement)",
            severity_table(notes, by_model),
        )
    text = "\n".join(lines).rstrip() + "\n"
    print(text, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


# stability: the same subset run more than once


def fleiss_kappa(ratings: Sequence[Sequence[bool]]) -> float | None:
    """Fleiss' kappa for yes/no ratings, one row per note and one entry per run; None when
    every rating is the same (agreement by chance is then total)."""
    if not ratings or len(ratings[0]) < 2:
        return None
    n = len(ratings[0])
    yes = [sum(row) for row in ratings]
    agree = sum((y * y + (n - y) * (n - y) - n) / (n * (n - 1)) for y in yes) / len(ratings)
    p_yes = sum(yes) / (n * len(ratings))
    chance = p_yes**2 + (1 - p_yes) ** 2
    return None if chance == 1 else (agree - chance) / (1 - chance)


def _spread(values: Sequence[int], total: int) -> str:
    mean = sum(values) / len(values)
    return (
        f"min {_rate(min(values), total)}, max {_rate(max(values), total)}, mean {mean:.1f}/{total}"
    )


def stability(args: argparse.Namespace) -> int:
    """Per-note agreement between runs of the same subset, and the spread of their rates.
    Notes a run has no report for, or failed on, are left out of every run's rates."""
    pairs = _pairs(args.pairs)
    keys = sorted(json.loads(args.spec.read_text(encoding="utf-8"))["note_keys"])
    runs = [build_notes(keys, pairs, load_reports(r, args.manifest)[0]) for r in args.runs]
    # build_notes keeps the spec's order, so a note is at the same index in every run.
    usable = {
        key
        for i, key in enumerate(keys)
        if all(run[i].report is not None and not run[i].failed for run in runs)
    }
    threshold = args.match_threshold
    out: list[str] = []
    say = out.append
    say(f"== Stability: {len(runs)} runs of the same {len(keys)} notes")
    for path in args.runs:
        say(f"   run: {path}")
    left_out = [k for k in keys if k not in usable]
    say(f"   notes without a report or with Pass A failed in some run (left out): {len(left_out)}")
    for key in left_out:
        say(f"      {key}")
    by_kind: dict[str, list[list[Note]]] = {}
    for run in runs:
        for kind in ("omission", "clean", "add", "change"):
            by_kind.setdefault(kind, []).append(
                [n for n in run if n.kind == kind and n.key in usable]
            )
    omissions, cleans = by_kind["omission"], by_kind["clean"]
    total_o, total_c = len(omissions[0]), len(cleans[0])

    def correct(n: Note) -> bool:
        """As `score` counts it: a decision-rule flag names the removed fact."""
        flags = decision_flags(n)
        value, flag, _ = best_match(n, flags)
        return bool(flags) and flag is not None and value >= threshold

    detected = [sum(n.flagged for n in run) for run in omissions]
    named = [sum(correct(n) for n in run) for run in omissions]
    alarms = [sum(n.flagged for n in run) for run in cleans]
    say("== Each run at the decision rule (correct detection by the word matcher)")
    for i, path in enumerate(args.runs):
        say(
            f"   {path.name}: detection {_rate(detected[i], total_o)}, correct "
            f"{_rate(named[i], total_o)}, false alarms {_rate(alarms[i], total_c)}"
        )
    say("== Spread over the runs")
    say(f"   detection: {_spread(detected, total_o)}")
    say(f"   correct detection: {_spread(named, total_o)}")
    say(f"   false alarms: {_spread(alarms, total_c)}")

    say("== Per-note agreement between runs (flagged or not at the decision rule)")
    ratings_all: list[list[bool]] = []
    unstable: list[tuple[str, int]] = []
    for kind in ("omission", "clean", "add", "change"):
        notes_by_run = by_kind[kind]
        if not notes_by_run[0]:
            continue
        ratings = [[run[i].flagged for run in notes_by_run] for i in range(len(notes_by_run[0]))]
        same = sum(1 for row in ratings if len(set(row)) == 1)
        pairs_agree = [
            sum(row[a] == row[b] for row in ratings)
            for a, b in itertools.combinations(range(len(runs)), 2)
        ]
        mean_pair = sum(pairs_agree) / len(pairs_agree) / len(ratings)
        kappa_k = fleiss_kappa(ratings)
        say(
            f"   {kind} notes: same in every run {_rate(same, len(ratings))}; mean pairwise "
            f"agreement {100 * mean_pair:.1f}%; Fleiss' kappa "
            + ("-" if kappa_k is None else f"{kappa_k:.2f}")
        )
        if kind in ("omission", "clean"):
            ratings_all += ratings
        unstable += [
            (n.key, sum(row))
            for n, row in zip(notes_by_run[0], ratings, strict=True)
            if len(set(row)) > 1
        ]
    kappa_all = fleiss_kappa(ratings_all)
    say(
        "   omission and clean notes together: Fleiss' kappa "
        + ("-" if kappa_all is None else f"{kappa_all:.2f}")
    )
    if omissions[0]:
        times = Counter(sum(correct(run[i]) for run in omissions) for i in range(len(omissions[0])))
        say(
            "   omission notes by the number of runs whose flag names the removed fact: "
            + ", ".join(f"{k} runs {times[k]}" for k in range(len(runs), -1, -1))
        )
    say(f"== Notes whose result differs between runs: {len(unstable)}")
    for key, flagged_in in sorted(unstable):
        say(f"   {key}: flagged in {flagged_in}/{len(runs)}")
    print("\n".join(out))
    return 0


def _decision(note: Note) -> bool:
    return note.flagged


def _critical(note: Note) -> bool:
    return note.critical_absent


def _label(problem: str) -> str:
    """A batch problem's file or case name only (never the rest of the message)."""
    return problem.split(":", 1)[0] if ":" in problem else "(a problem without a case name)"


def _route_one_stores(args: argparse.Namespace) -> dict[str, Rows]:
    """Route one's (B3) records with their fact lists: Gate 1(b)'s run, and the released r1
    (whose development rows hold fact text once the split has been re-run; the section says
    when they do not)."""
    stores: dict[str, Rows] = {}
    if args.gate1b is not None:
        b3 = next(args.gate1b.glob("gate1b-pipeline/_state/*B3*.jsonl"), None)
        if b3 is not None:
            stores["Gate 1(b) B3"] = read_store(b3, lambda r: _arm(r) == "B3")
    if args.released is not None:
        stores["released B3 r1"] = released_rows(args.released)["b3 r1"]
    return stores


def _references(args: argparse.Namespace) -> dict[str, dict[str, bool | None]]:
    refs: dict[str, dict[str, bool | None]] = {}
    if args.released is not None:
        released = released_rows(args.released)
        refs["released B3 r1"] = _flags(released["b3 r1"], route_one_flag)
        refs["released power-gepa-high r1"] = _flags(released["power r1"], route_two_flag)
        refs["released power-gepa-high r3"] = _flags(released["power r3"], route_two_flag)
    if args.gate1b is not None:
        b3 = next(args.gate1b.glob("gate1b-pipeline/_state/*B3*.jsonl"), None)
        power = next(args.gate1b.glob("gate1b-power/_state/power-gepa-high*.jsonl"), None)
        if b3 is not None:
            rows = read_store(b3, lambda r: _arm(r) == "B3")
            refs["Gate 1(b) B3"] = _flags(rows, route_one_flag)
        if power is not None:
            rows = read_store(power, lambda r: r.get("arm") == "power-gepa-high")
            refs["Gate 1(b) power-gepa-high"] = _flags(rows, route_two_flag)
    return refs


def _flags(rows: Rows, rule: Any) -> dict[str, bool | None]:
    return {key: rule(row) for key, row in rows.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="write the subset as a batch folder")
    prep.add_argument("--pairs", type=Path, required=True)
    prep.add_argument("--transcripts", type=Path, required=True)
    prep.add_argument("--spec", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    sc = commands.add_parser("score", help="score a run's reports")
    sc.add_argument("--run", type=Path, required=True)
    sc.add_argument("--pairs", type=Path, required=True)
    sc.add_argument("--spec", type=Path, required=True)
    sc.add_argument("--manifest", type=Path, help="prepare's manifest.json (exact case ids)")
    sc.add_argument("--released", type=Path)
    sc.add_argument("--gate1b", type=Path)
    sc.add_argument("--show-facts", action="store_true")
    sc.add_argument("--match-threshold", type=float, default=0.5)
    sc.add_argument(
        "--partial-as-flags",
        action="store_true",
        help="count partial verdicts as flags, as before they became information",
    )
    sc.add_argument("--judgements", type=Path, help="fact_judgements.json from `judge`")
    sp = commands.add_parser("spec", help="write a spec of every note in the pairs file")
    sp.add_argument("--pairs", type=Path, required=True)
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--sample", type=int, help="a stratified sample of this many notes")
    sp.add_argument("--seed", type=int, help="the sample's seed (required with --sample)")
    st = commands.add_parser("stability", help="agreement between runs of the same subset")
    st.add_argument("--runs", type=Path, nargs="+", required=True)
    st.add_argument("--pairs", type=Path, required=True)
    st.add_argument("--spec", type=Path, required=True)
    st.add_argument("--manifest", type=Path, help="prepare's manifest.json (exact case ids)")
    st.add_argument("--match-threshold", type=float, default=0.5)
    ms = commands.add_parser("misses", help="trace each removed fact through Pass A")
    ms.add_argument("--run", type=Path, required=True)
    ms.add_argument("--pairs", type=Path, required=True)
    ms.add_argument("--spec", type=Path, required=True)
    ms.add_argument("--manifest", type=Path, help="prepare's manifest.json (exact case ids)")
    ms.add_argument("--show-facts", action="store_true")
    ms.add_argument("--match-threshold", type=float, default=0.5)
    ms.add_argument("--judgements", type=Path, help="fact_judgements.json from `judge`")
    jd = commands.add_parser("judge", help="judge fact matches with a model (paid calls)")
    jd.add_argument("--config", type=Path, required=True)
    jd.add_argument("--cache", type=Path, help="default: <run>/fact_judgements.json")
    jd.add_argument(
        "--released", type=Path, help="also judge the released baselines' flagged facts"
    )
    sa = commands.add_parser("sample", help="write a fixed-seed spot-check sample")
    sa.add_argument("--judgements", type=Path, required=True)
    sa.add_argument("--out", type=Path, required=True)
    sa.add_argument("--n", type=int, default=30)
    sa.add_argument("--seed", type=int, default=SAMPLE_SEED)
    sa.add_argument("--match-threshold", type=float, default=0.5)
    sv = commands.add_parser("severity", help="our audit severity against the benchmark's")
    sv.add_argument("--judgements", type=Path)
    sv.add_argument("--out", type=Path, help="also write the result here (Markdown)")
    sv.add_argument("--match-threshold", type=float, default=0.5)
    for sub in (jd, sa, sv):
        sub.add_argument("--run", type=Path, required=True)
        sub.add_argument("--pairs", type=Path, required=True)
        sub.add_argument("--spec", type=Path, required=True)
        sub.add_argument("--manifest", type=Path, help="prepare's manifest.json (exact case ids)")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare(args.pairs, args.transcripts, args.spec, args.out)
    if args.command == "misses":
        return misses(args)
    if args.command == "judge":
        return run_judge(args)
    if args.command == "sample":
        return write_sample(args)
    if args.command == "severity":
        return severity(args)
    if args.command == "spec":
        if args.sample is not None and args.seed is None:
            parser.error("--sample needs --seed")
        if args.sample is not None and args.sample < 1:
            parser.error("--sample must be at least 1")
        return write_spec(args.pairs, args.out, args.sample, args.seed)
    if args.command == "stability":
        if len(args.runs) < 2:
            parser.error("stability needs at least two runs")
        return stability(args)
    return score(args)


if __name__ == "__main__":
    raise SystemExit(main())
