"""Split a raw OmissionBench download into development data and the held-out subset.

Usage (run by a person, in a normal terminal):
    python scripts/split_omissionbench.py --raw <raw download dir> \
        --heldout-records <w2-pipeline scored-records JSON> \
        --heldout-out <dir> [--dev-out data/omissionbench]

Every path comes from the arguments; the script names no folder of its own. It reads the
dataset's own layout (pairs/dataset_v2.json, transcripts/, fact_sheets/, pairs/fact_sites.json,
pairs/factorial_severity.json, and score files under judgements/, validation/ and notes/).

Rules:

1. Held-out consultation keys ("stratum|consultation") come from the distinct pair_ids in
   --heldout-records. Clean rows (null pair_id) are not used.
2. Eval-set pairs whose key is held out go to --heldout-out; the other eval-set pairs go to
   --dev-out. Pairs outside the eval set go nowhere. Each side keeps _attribution, its pairs
   and scalar top-level values; other top-level keys are left out and listed.
3. Each side gets its pairs' transcripts (transcripts/<stratum>/<id>.txt) and every licence
   or attribution file (a name containing "licen" or "attribution" that is not the
   transcript of any pair, so a transcript named "..._attribution.txt" stays on its side).
4. Fact sheets, fact sites and severity entries go to --dev-out only when they can be tied to a
   development consultation with certainty; anything uncertain is left out and listed.
5. Nothing from judgements/, validation/ or notes/ is copied, except a reduced copy of each
   scored-records file in --dev-out/judge_scores/: rows for development consultations only,
   with the fields in SCORE_FIELDS (ids, scores and the run's settings, so replicates, arms
   and models can be told apart). Fields in NEVER_KEPT, which can carry text, are never
   kept. A kept value that could hold text aborts the run.
6. Before anything is written, the counts must match EXPECTED_COUNTS, and both output
   folders must be empty or absent and outside --raw; otherwise the run stops and writes
   nothing.

The script never prints note or transcript text, and makes no network calls.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The benchmark partitions held-out data by consultation, not by pair: every eval-set pair
# of a held-out consultation is held out, so no development pair shares a transcript or a
# clean note with the held-out subset. The 47 held-out consultations hold 230 eval-set
# pairs, of which the published held-out subset (the pairs in --heldout-records) is 151;
# all 230 go to --heldout-out. The remaining 265 eval-set pairs, over 65 consultations
# (one clean twin each), are the development data.
EXPECTED_COUNTS = {
    "development pairs": 265,
    "held-out pairs": 230,
    "held-out consultations": 47,
    "development clean twins": 65,
}
BREAKDOWN_FIELDS = ("type", "residual_level")
MAX_LABEL = 40
SCORE_FIELDS = (
    # ids and scores
    "pair_id", "note_key", "note_role", "clean_key", "stratum", "consultation", "aggregate",
    "flagged",
    # the run: replicate, model, arm and settings
    "replicate", "run_seed", "run_id", "model", "model_key", "role", "arm", "arm_family",
    "cell", "criterion", "format", "k", "k_impl", "tier", "reasoning_effort", "max_tokens",
    "dataset_version", "parse_failure",
    # the pair and the judgement, as short labels and counts
    "pair_type", "pair_class", "residual_level", "severity", "verdict", "n_facts", "n_full",
    "n_present", "n_partial", "n_absent",
)  # fmt: skip
# Score-file fields that can carry text; never copied, whatever SCORE_FIELDS says.
NEVER_KEPT = frozenset({"matched_key", "fact_key", "detail", "samples", "receipts", "totals"})
assert not NEVER_KEPT & set(SCORE_FIELDS), "a text-bearing field is listed in SCORE_FIELDS"
LOG_FOLDERS = ("judgements", "validation", "notes")
MAX_SCORE_STRING = 200
PAIRS_FILE = Path("pairs") / "dataset_v2.json"
SITES_FILE = Path("pairs") / "fact_sites.json"
SEVERITY_FILE = Path("pairs") / "factorial_severity.json"


class SplitError(Exception):
    """A check failed; the message holds counts or reasons, never dataset text."""


@dataclass
class Plan:
    """Everything to write, decided before anything is written."""

    dev_json: dict[Path, Any] = field(default_factory=dict)
    heldout_json: dict[Path, Any] = field(default_factory=dict)
    dev_copies: dict[Path, Path] = field(default_factory=dict)
    heldout_copies: dict[Path, Path] = field(default_factory=dict)
    left_out: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    breakdown: dict[str, dict[str, int]] = field(default_factory=dict)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--heldout-records", type=Path, required=True)
    parser.add_argument("--dev-out", type=Path, default=Path("data") / "omissionbench")
    parser.add_argument("--heldout-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.raw, args.heldout_records, args.dev_out, args.heldout_out)
    except SplitError as error:
        print(f"split_omissionbench: stopped, nothing written: {error}", file=sys.stderr)
        return 1
    write_plan(plan, args.dev_out, args.heldout_out)
    report(plan, args.dev_out, args.heldout_out)
    return 0


# Planning


def build_plan(raw: Path, records_file: Path, dev_out: Path, heldout_out: Path) -> Plan:
    _check_outputs(raw, dev_out, heldout_out)
    plan = Plan()
    data = _load_json(raw / PAIRS_FILE)
    if not isinstance(data, dict) or not isinstance(data.get("pairs"), list):
        raise SplitError(f"{PAIRS_FILE} has no top-level 'pairs' list")

    heldout_keys = _heldout_keys(records_file)
    eval_pairs = [p for p in data["pairs"] if isinstance(p, dict) and p.get("eval_set") is True]
    for pair in eval_pairs:
        _check_pair(pair)
    dev = [p for p in eval_pairs if p["key"] not in heldout_keys]
    heldout = [p for p in eval_pairs if p["key"] in heldout_keys]
    dev_keys = {p["key"] for p in dev}
    plan.counts = {
        "development pairs": len(dev),
        "held-out pairs": len(heldout),
        "held-out consultations": len(heldout_keys),
        "development clean twins": len(dev_keys),
    }
    if plan.counts != EXPECTED_COUNTS:
        lines = ", ".join(
            f"{name} {plan.counts[name]} (expected {expected})"
            for name, expected in EXPECTED_COUNTS.items()
        )
        raise SplitError(f"counts differ: {lines}")
    plan.breakdown = {name: _tally(p.get(name) for p in dev) for name in BREAKDOWN_FIELDS}
    missing = heldout_keys - {p["key"] for p in heldout}
    if missing:
        raise SplitError(f"{len(missing)} held-out consultation(s) have no eval-set pairs")

    top, dropped = _top_level(data)
    plan.dev_json[PAIRS_FILE] = {**top, "pairs": dev}
    plan.heldout_json[PAIRS_FILE] = {**top, "pairs": heldout}
    plan.left_out += [f"{PAIRS_FILE.as_posix()}: top-level key '{k}'" for k in dropped]

    _plan_transcripts(raw, dev, heldout, plan)
    transcripts = {
        Path("transcripts") / p["stratum"] / f"{p['id']}.txt"
        for p in data["pairs"]
        if isinstance(p, dict)
        and isinstance(p.get("stratum"), str)
        and isinstance(p.get("id"), str)
    }
    for path in _licence_files(raw, transcripts):
        relative = path.relative_to(raw)
        plan.dev_copies[relative] = path
        plan.heldout_copies[relative] = path

    every_key = {
        p["key"] for p in data["pairs"] if isinstance(p, dict) and isinstance(p.get("key"), str)
    }
    known = _KnownKeys(dev_keys, (every_key | heldout_keys) - dev_keys)
    _plan_fact_sheets(raw, dev, heldout, plan)
    _plan_fact_file(raw, SITES_FILE, known, plan, nested=None)
    _plan_fact_file(raw, SEVERITY_FILE, known, plan, nested="facts")
    _plan_scores(raw, records_file, known, plan)
    return plan


def _tally(values: Iterable[object]) -> dict[str, int]:
    """Counts per label; anything that is not a short label is counted, never shown."""
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            label = "(missing)"
        elif _is_scalar(value) and len(str(value)) <= MAX_LABEL:
            label = str(value)
        else:
            label = "(other)"
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _check_outputs(raw: Path, dev_out: Path, heldout_out: Path) -> None:
    raw_r, dev_r, held_r = raw.resolve(), dev_out.resolve(), heldout_out.resolve()
    for name, out in (("--dev-out", dev_r), ("--heldout-out", held_r)):
        if out == raw_r or raw_r in out.parents:
            raise SplitError(f"{name} is inside --raw")
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise SplitError(f"{name} exists and is not empty; remove it and run again")
    if dev_r == held_r or dev_r in held_r.parents or held_r in dev_r.parents:
        raise SplitError("--dev-out and --heldout-out overlap")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SplitError(f"missing file: {path}") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SplitError(f"not valid JSON: {path}") from error


def _key_of(pair_id: object) -> str | None:
    if not isinstance(pair_id, str) or pair_id.count("|") < 2:
        return None
    stratum, consultation = pair_id.split("|")[:2]
    return f"{stratum}|{consultation}" if stratum and consultation else None


def _heldout_keys(records_file: Path) -> set[str]:
    data = _load_json(records_file)
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise SplitError("--heldout-records has no top-level 'records' list")
    keys = set()
    for row in data["records"]:
        pair_id = row.get("pair_id") if isinstance(row, dict) else None
        if pair_id is None:
            continue
        key = _key_of(pair_id)
        if key is None:
            raise SplitError("--heldout-records has a pair_id that is not stratum|consultation|...")
        keys.add(key)
    return keys


def _check_pair(pair: dict[str, Any]) -> None:
    for name in ("pair_id", "key", "stratum", "id"):
        if not isinstance(pair.get(name), str) or not pair[name]:
            raise SplitError(f"an eval-set pair has no '{name}'")
    if _key_of(pair["pair_id"]) != pair["key"]:
        raise SplitError(f"pair {pair['pair_id']!r}: key does not match its pair_id")


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _top_level(data: dict[str, Any], keep: str = "pairs") -> tuple[dict[str, Any], list[str]]:
    """_attribution and scalar top-level values; the names of everything else left out."""
    top, dropped = {}, []
    for key, value in data.items():
        if key == keep:
            continue
        if key == "_attribution" or _is_scalar(value):
            top[key] = value
        else:
            dropped.append(key)
    return top, dropped


def _is_licence(path: Path) -> bool:
    name = path.name.casefold()
    return "licen" in name or "attribution" in name


def _licence_files(raw: Path, transcripts: set[Path]) -> list[Path]:
    """Licence and attribution files; a pair's transcript never counts, whatever its name."""
    found = [p for p in raw.iterdir() if p.is_file() and _is_licence(p)]
    folder = raw / "transcripts"
    if folder.is_dir():
        found += [
            p
            for p in folder.rglob("*")
            if p.is_file() and _is_licence(p) and p.relative_to(raw) not in transcripts
        ]
    return sorted(found)


def _plan_transcripts(
    raw: Path, dev: list[dict[str, Any]], heldout: list[dict[str, Any]], plan: Plan
) -> None:
    def paths(pairs: Iterable[dict[str, Any]]) -> set[Path]:
        return {Path("transcripts") / p["stratum"] / f"{p['id']}.txt" for p in pairs}

    dev_paths, heldout_paths = paths(dev), paths(heldout)
    if dev_paths & heldout_paths:
        raise SplitError(f"{len(dev_paths & heldout_paths)} transcript(s) needed by both sides")
    missing = [p for p in dev_paths | heldout_paths if not (raw / p).is_file()]
    if missing:
        raise SplitError(f"{len(missing)} transcript(s) missing, e.g. {sorted(missing)[0]}")
    plan.dev_copies.update({p: raw / p for p in dev_paths})
    plan.heldout_copies.update({p: raw / p for p in heldout_paths})


@dataclass(frozen=True)
class _KnownKeys:
    dev: set[str]
    other: set[str]

    def decide(self, identifier: str | None, entry: object) -> str | None:
        """The consultation key an entry belongs to, or None when it is not certain."""
        candidates = set()
        if identifier is not None:
            prefix = _key_of(identifier)
            if prefix in self.dev or prefix in self.other:
                candidates.add(prefix)
        if isinstance(entry, dict):
            if isinstance(entry.get("key"), str):
                candidates.add(entry["key"])
            if isinstance(entry.get("stratum"), str) and isinstance(entry.get("consultation"), str):
                candidates.add(f"{entry['stratum']}|{entry['consultation']}")
        if len(candidates) != 1:
            return None
        (key,) = candidates
        return key if key in self.dev or key in self.other else None


def _plan_fact_sheets(
    raw: Path, dev: list[dict[str, Any]], heldout: list[dict[str, Any]], plan: Plan
) -> None:
    folder = raw / "fact_sheets"
    if not folder.is_dir():
        return

    def names(pairs: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
        found = set()
        for p in pairs:
            found.add((p["stratum"], p["id"]))
            found.add((p["stratum"], p["key"].split("|")[1]))
        return found

    dev_names, heldout_names = names(dev), names(heldout)
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        relative = path.relative_to(folder)
        name = (relative.parent.as_posix(), relative.stem)
        if len(relative.parts) == 2 and name in dev_names and name not in heldout_names:
            plan.dev_copies[Path("fact_sheets") / relative] = path
        elif len(relative.parts) == 2 and name in heldout_names:
            continue
        else:
            plan.left_out.append(f"fact_sheets/{relative.as_posix()}")


def _plan_fact_file(
    raw: Path, relative: Path, known: _KnownKeys, plan: Plan, *, nested: str | None
) -> None:
    path = raw / relative
    if not path.is_file():
        return
    data = _load_json(path)
    container = data.get(nested) if nested and isinstance(data, dict) else data
    if not isinstance(container, (dict, list)):
        plan.left_out.append(f"{relative.as_posix()} (unrecognised structure)")
        return
    if isinstance(container, dict):
        kept_items = {}
        for identifier, entry in container.items():
            key = known.decide(identifier, entry)
            if key is None:
                plan.left_out.append(f"{relative.as_posix()}: {_short(identifier)}")
            elif key in known.dev:
                kept_items[identifier] = entry
        kept: Any = kept_items
    else:
        kept_list = []
        for index, entry in enumerate(container):
            key = known.decide(None, entry)
            if key is None:
                plan.left_out.append(f"{relative.as_posix()}: item {index}")
            elif key in known.dev:
                kept_list.append(entry)
        kept = kept_list
    if nested:
        top, dropped = _top_level(data, keep=nested)
        plan.left_out += [f"{relative.as_posix()}: top-level key '{k}'" for k in dropped]
        plan.dev_json[relative] = {**top, nested: kept}
    else:
        plan.dev_json[relative] = kept


def _short(identifier: object) -> str:
    """An identifier for the left-out list, cut short in case a key holds more than an id."""
    text = str(identifier)
    return text if len(text) <= 80 else text[:77] + "..."


def _score_files(raw: Path, records_file: Path) -> list[tuple[str, Path]]:
    """(output name, path) for every scored-records file."""
    found: dict[Path, str] = {}
    for folder in LOG_FOLDERS:
        base = raw / folder
        if base.is_dir():
            for path in sorted(base.rglob("*.json")):
                found[path.resolve()] = path.relative_to(raw).as_posix().replace("/", "__")
    resolved = records_file.resolve()
    if resolved not in found:
        found[resolved] = f"heldout_records__{records_file.name}"
    return sorted((name, path) for path, name in found.items())


def _plan_scores(raw: Path, records_file: Path, known: _KnownKeys, plan: Plan) -> None:
    for name, path in _score_files(raw, records_file):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        records = data.get("records") if isinstance(data, dict) else None
        if not isinstance(records, list) or not records:
            continue
        if not all(isinstance(r, dict) and "note_key" in r for r in records):
            continue
        kept, dropped = [], 0
        for row in records:
            reduced = {f: row.get(f) for f in SCORE_FIELDS}
            for f, value in reduced.items():
                if not _is_scalar(value) or (
                    isinstance(value, str) and len(value) > MAX_SCORE_STRING
                ):
                    raise SplitError(f"{name}: field '{f}' holds a value that could be text")
            # Clean rows have no pair_id; their clean_key is "stratum|consultation|clean|<id>".
            key = _key_of(row.get("pair_id")) or _key_of(row.get("clean_key"))
            if key is None and isinstance(row.get("stratum"), str):
                key = f"{row['stratum']}|{row.get('consultation')}"
            if key in known.dev:
                kept.append(reduced)
            else:
                dropped += 1
        plan.dev_json[Path("judge_scores") / name] = {"records": kept}
        plan.notes.append(f"judge_scores/{name}: kept {len(kept)} rows, dropped {dropped}")


# Writing and reporting


def write_plan(plan: Plan, dev_out: Path, heldout_out: Path) -> None:
    for out, documents, copies in (
        (dev_out, plan.dev_json, plan.dev_copies),
        (heldout_out, plan.heldout_json, plan.heldout_copies),
    ):
        for relative, document in documents.items():
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", "utf-8")
        for relative, source in copies.items():
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def report(plan: Plan, dev_out: Path, heldout_out: Path) -> None:
    for name, value in plan.counts.items():
        print(f"{name}: {value}")
    for name, counts in plan.breakdown.items():
        parts = ", ".join(f"{label} {count}" for label, count in counts.items())
        print(f"development pairs by {name}: {parts}")
    print(f"--dev-out: {len(plan.dev_json) + len(plan.dev_copies)} files in {dev_out}")
    print(
        f"--heldout-out: {len(plan.heldout_json) + len(plan.heldout_copies)} files in {heldout_out}"
    )
    for note in plan.notes:
        print(note)
    if plan.left_out:
        print(f"left out ({len(plan.left_out)}), not certain to be development-only:")
        for item in plan.left_out:
            print(f"  {item}")


if __name__ == "__main__":
    raise SystemExit(main())
