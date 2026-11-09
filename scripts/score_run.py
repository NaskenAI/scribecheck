"""Score a scribecheck run on an OmissionBench development subset (M3 session 4).

Usage:
    python scripts/score_run.py prepare --pairs data/omissionbench/pairs/dataset_v2.json \
        --transcripts data/omissionbench/transcripts --spec runs/gate1b/subset_spec.json \
        --out runs/m3s4/cases
    python scripts/score_run.py score --run runs/m3s4/claude-medium \
        --pairs data/omissionbench/pairs/dataset_v2.json --spec runs/gate1b/subset_spec.json \
        [--released data/omissionbench/judge_scores] [--gate1b runs/gate1b/harness-results] \
        [--show-facts] [--match-threshold 0.5] [--partial-as-flags]
    python scripts/score_run.py misses --run runs/m3s4/claude-high \
        --pairs data/omissionbench/pairs/dataset_v2.json --spec runs/gate1b/subset_spec.json \
        [--manifest runs/m3s4/cases/manifest.json] [--show-facts]

`prepare` writes the subset as a batch folder: `<case>.transcript.txt` and `<case>.note.md`
for each note key (the case id is the note key with "|" as "__"), plus `manifest.json`.
Errored notes are `<pair_id>|err`; a clean note is `<consultation>|clean|<first 12 hex
digits of the clean note's SHA-256>`.

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
- calls, tokens, cost and wall-clock time per note, and each unchecked reason's count.

Partial verdicts are information, not flags, whether the run listed them in `info` or (a
run from before the change) among its flags; `--partial-as-flags` counts them as before.

`misses` follows each omission note's removed fact through Pass A, from the report's
`pass_a_trace` (`[report] pass_a_trace = true`): the extracted fact that best matches it
(with the score), the audit's core or contextual call and severity, the check's verdict
(with the note quote for present) and the second look, and classes it as detected, not
extracted, dropped by the audit, contextual and so never checked, judged present, judged
partial, rescued by the second look, or unchecked; counts per class and per trace. A run
without a trace can only be classed from its flags.

Only counts, IDs, labels and scores are printed (and, with --show-facts, the benchmark's
and the flags' fact strings), never transcript or note text. Development data only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from compare_gate1b import (  # noqa: E402
    Rows,
    _arm,
    read_store,
    released_rows,
    route_one_flag,
    route_two_flag,
)

from scribecheck.decision import decide  # noqa: E402
from scribecheck.flags import Flag  # noqa: E402

RANK = {"possible": 1, "confirmed": 2, "certain": 3}
LEVELS = ("possible", "confirmed")
SEVERITY_RANK = {"peripheral": 1, "supporting": 2, "critical": 3}
FLOORS = (("all", "peripheral"), ("supporting+", "supporting"), ("critical", "critical"))
"""Severity floors for the sweep: all severities, supporting or higher, critical only."""
KS = (1, 2, 3, 4, 5)
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
    """A match must share a distinctive word, give the same numbers, and agree on
    negation: shared units, frequencies or filler alone never make two facts the same."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    distinctive = {w for w in ta & tb if w not in _GENERIC and not w[0].isdigit()}
    if not distinctive:
        return False
    na, nb = set(_NUMBER.findall(a)), set(_NUMBER.findall(b))
    if na and nb and na != nb:
        return False
    neg_a = bool(set(a.casefold().split()) & _NEGATION)
    neg_b = bool(set(b.casefold().split()) & _NEGATION)
    return neg_a == neg_b


def best_match(note: Note, flags: Iterable[Flag]) -> tuple[float, Flag | None, str]:
    """The best-scoring flag whose fact is comparable with a removed fact."""
    best: tuple[float, Flag | None, str] = (0.0, None, "")
    for fact in removed_facts(note.pair):
        for flag in flags:
            if not comparable(fact, flag.fact or ""):
                continue
            score = similarity(fact, flag.fact or "")
            if score > best[0]:
                best = (score, flag, fact)
    return best


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

    say("== Per-note agreement on omission and clean notes (information, not a gate)")
    say("   ours: the decision rule on Pass A omission flags; against B3 also 'critical")
    say("   absent' (route one's rule). A note we have no usable report for counts as a")
    say("   disagreement; a reference row that is unknown is left out.")
    judged = [n for n in notes if n.kind in ("omission", "clean")]
    usable = {n.key: n for n in judged if n.report is not None and not n.failed}
    references = _references(args)
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
    "judged present",
    "judged partial",
    "rescued by the second look",
    "unchecked",
    "no matching flag",
)
"""Where an omission note's removed fact ended up. "detected": a confirmed absent flag names
it. "no matching flag" is the only class a run without a Pass A trace can give besides
"detected" and "judged partial"."""


def trace_match(
    pair: Mapping[str, Any], trace: Sequence[Mapping[str, Any]], threshold: float
) -> tuple[float, Mapping[str, Any] | None]:
    """The best-scoring trace fact comparable with a removed fact, if it reaches
    `threshold`, and the best score among comparable facts either way."""
    best: tuple[float, Mapping[str, Any] | None] = (0.0, None)
    for fact in removed_facts(pair):
        for step in trace:
            text = str(step.get("fact") or "")
            if comparable(fact, text) and (value := similarity(fact, text)) > best[0]:
                best = (value, step)
    return best if best[0] >= threshold else (best[0], None)


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
    threshold = args.match_threshold
    traced = sum(1 for n in notes if n.report is not None and "pass_a_trace" in n.report)
    out: list[str] = []
    say = out.append
    say(f"== Removed facts through Pass A ({len(notes)} omission notes, {traced} with a trace)")
    if traced < len(notes):
        say("   a note without a Pass A trace ([report] pass_a_trace) can only be classed")
        say("   from its flags: detected, judged partial or no matching flag")
    say("   pair id, trace, removed-fact severity, match score, class, then the stages:")
    say("   extracted / audit (core or contextual, severity) / check verdict / second look")
    classes: Counter[str] = Counter()
    by_trace: dict[str, Counter[str]] = {}
    for n in notes:
        assert n.report is not None
        flag_score, flag, _ = best_match(n, [*n.pass_a, *n.info])
        named = flag if flag is not None and flag_score >= threshold else None
        trace = n.report.get("pass_a_trace")
        if trace is not None:
            value, step = trace_match(n.pair, trace, threshold)
            label = classify_miss(step)
            if named is not None and named.verdict == "absent" and RANK[named.confidence] >= 2:
                label = "detected"  # a confirmed rules candidate names it
        else:
            value, step = flag_score, None
            if named is None:
                label = "no matching flag"
            elif named.verdict == "partial":
                label = "judged partial"
            else:
                label = "detected" if RANK[named.confidence] >= 2 else "no matching flag"
        classes[label] += 1
        level = str(n.pair.get("residual_level"))
        by_trace.setdefault(level, Counter())[label] += 1
        say(f"   {n.pair.get('pair_id')}  {level}  {n.pair.get('severity')}  {value:.2f}  {label}")
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
    print("\n".join(out))
    return 0


def _decision(note: Note) -> bool:
    return note.flagged


def _critical(note: Note) -> bool:
    return note.critical_absent


def _label(problem: str) -> str:
    """A batch problem's file or case name only (never the rest of the message)."""
    return problem.split(":", 1)[0] if ":" in problem else "(a problem without a case name)"


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
    ms = commands.add_parser("misses", help="trace each removed fact through Pass A")
    ms.add_argument("--run", type=Path, required=True)
    ms.add_argument("--pairs", type=Path, required=True)
    ms.add_argument("--spec", type=Path, required=True)
    ms.add_argument("--manifest", type=Path, help="prepare's manifest.json (exact case ids)")
    ms.add_argument("--show-facts", action="store_true")
    ms.add_argument("--match-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare(args.pairs, args.transcripts, args.spec, args.out)
    if args.command == "misses":
        return misses(args)
    return score(args)


if __name__ == "__main__":
    raise SystemExit(main())
