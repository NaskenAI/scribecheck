"""Classify each omission pair's removed fact as a plan action or not (Gate 2 preparation).

Usage:
    python scripts/action_classify.py hash
    python scripts/action_classify.py classify --pairs <pairs/dataset_v2.json> \
        --config docs/runbooks/action_classify.toml --cache <cache.json> [--ids <ids.json>]
    python scripts/action_classify.py sample --pairs <pairs/dataset_v2.json> \
        --cache <cache.json> --out docs/review/action_spot_check.csv [--n 20] [--seed N]

One structured call per omission pair asks a model to label the removed fact, from its text
only (never the transcript or the note), as one of LABELS: a referral, a test or
investigation, a follow-up appointment or timing, a medicine started, stopped, changed or
continued, or not an action. The plan-action subset is every pair labelled other than
NOT_ACTION. `hash` prints the SHA-256 of the prompt (PROMPT_VERSION, SYSTEM and TASK),
which the Gate 2 pre-registration records and the Gate 2 runbook checks.

Labels are cached one entry per pair, keyed by a SHA-256 of the prompt, the model with its
effort and token limit, and the fact text, so classifying again makes no calls, and a
changed fact, prompt or setting is classified afresh. `classify` labels every omission pair
in the pairs file (only those listed in --ids, when given). `sample` writes a fixed-seed
spot-check sheet spread over the labels, for a person to review (benchmark fact text: keep it
git-ignored). Only counts are printed, never fact text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from scribecheck.config import load_config
from scribecheck.errors import InputError
from scribecheck.model import create_budget, create_client
from scribecheck.model.base import Block, Message, ModelClient, ModelFailure, ModelRequest
from scribecheck.model.budget import Budget
from scribecheck.model.structured import complete

PROMPT_VERSION = "action_classify_v1"
LABELS = ("referral", "test", "follow_up", "medication_change", "not_action")
NOT_ACTION = "not_action"
SAMPLE_SEED = 20261123
SYSTEM = (
    "You classify one clinical fact that was removed from a consultation note. You see only "
    "the fact's text, never a transcript or a note. Answer from the text alone."
)
TASK = """REMOVED FACT:
{fact}

Is this fact a plan action from the consultation? Choose one label.
- "referral": a referral or onward request to another clinician, service or team.
- "test": a test or investigation to be done: a blood test, scan, swab, ECG, measurement or
  similar.
- "follow_up": a planned follow-up appointment or review, or when the patient is to be seen
  again.
- "medication_change": a medicine started, stopped, changed (drug, dose, form or frequency)
  or continued.
- "not_action": anything else: safety-net advice (what to do or when to seek help if
  something happens), lifestyle advice, history, symptoms, examination findings, results
  already known, the assessment or the diagnosis.
A conditional instruction to come back only if something happens is safety-net advice, so
"not_action"; a planned review is "follow_up". When the fact holds more than one action,
choose the first that applies in this order: "medication_change", "referral", "test",
"follow_up".
Return {{"label": ...}}."""


class Classification(BaseModel):
    label: Literal["referral", "test", "follow_up", "medication_change", "not_action"]


def prompt_hash() -> str:
    """The SHA-256 of the prompt: its version, system text, task template, labels and the
    output schema the model must follow."""
    schema = Classification.model_json_schema()
    payload = json.dumps(
        [PROMPT_VERSION, SYSTEM, TASK, LABELS, schema], ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def key(fact: str, model: str) -> str:
    payload = json.dumps([prompt_hash(), model, fact], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def removed_text(pair: Mapping[str, Any]) -> str:
    """The pair's removed fact (and its target's, when that differs), as one text."""
    facts = [pair.get("fact")]
    target = pair.get("target")
    if isinstance(target, Mapping):
        facts.append(target.get("fact"))
    return " / ".join(f for f in dict.fromkeys(facts) if isinstance(f, str) and f.strip())


def omission_pairs(
    pairs: Sequence[Mapping[str, Any]], ids: set[str] | None = None
) -> dict[str, str]:
    """{pair_id: removed fact text} for the omission pairs (those in `ids`, when given)."""
    return {
        str(p["pair_id"]): removed_text(p)
        for p in pairs
        if p.get("type") == "omit"
        and isinstance(p.get("pair_id"), str)
        and (ids is None or p["pair_id"] in ids)
    }


def load_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"prompt": PROMPT_VERSION, "entries": {}}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("entries", {})
    return data


def save_cache(path: Path, cache: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def lookup(
    cache: Mapping[str, Any], pair_id: str, fact: str, model: str | None = None
) -> str | None:
    """The cached, valid label for this pair and fact text (under `model`, or the model the
    cache was made with), or None when it is missing, stale or invalid."""
    entry = cache.get("entries", {}).get(pair_id)
    if not isinstance(entry, Mapping) or entry.get("label") not in LABELS:
        return None
    if entry.get("key") != key(fact, model or str(cache.get("model") or "")):
        return None
    return str(entry["label"])


def labels(path: Path, pairs: Sequence[Mapping[str, Any]]) -> dict[str, str | None]:
    """Each omission pair's valid cached label by pair_id (None when it has none)."""
    cache = load_cache(path)
    return {pid: lookup(cache, pid, fact) for pid, fact in omission_pairs(pairs).items()}


def classify(
    items: Mapping[str, str],
    client: ModelClient,
    budget: Budget,
    cache: dict[str, Any],
    model: str,
    path: Path,
) -> Counter[str]:
    """Label each (pair_id, fact) not already cached, saving the cache after every call.
    Returns counts: cached, classified, and each failure reason."""
    cache["prompt"], cache["prompt_sha256"], cache["model"] = PROMPT_VERSION, prompt_hash(), model
    entries: dict[str, Any] = cache.setdefault("entries", {})
    counts: Counter[str] = Counter()
    for pair_id, fact in items.items():
        if lookup(cache, pair_id, fact, model) is not None:
            counts["cached"] += 1
            continue
        request = ModelRequest(
            system=SYSTEM,
            messages=(Message("user", (Block(TASK.format(fact=fact)),)),),
            output=Classification,
            custom_id="classify",
        )
        result = complete(client, request, budget)
        if isinstance(result, ModelFailure):
            counts[result.reason] += 1
            entries[pair_id] = {"key": key(fact, model), "label": None, "error": result.reason}
        else:
            counts["classified"] += 1
            entries[pair_id] = {
                "key": key(fact, model),
                "label": result.parsed.label,
                "model_id": result.model_id,
            }
        save_cache(path, cache)
        if isinstance(result, ModelFailure) and result.reason == "cost_cap_reached":
            break
    return counts


def _pairs(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = data["pairs"] if isinstance(data, dict) else data
    return [p for p in pairs if isinstance(p, dict)]


def load_ids(path: Path) -> set[str]:
    """Pair ids from a JSON list, or a {"records": [...]} list, of ids or {"pair_id": id}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("records") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list of pair ids or a 'records' list")
    ids = {r.get("pair_id") if isinstance(r, dict) else r for r in rows}
    if not all(isinstance(i, str) for i in ids):
        raise ValueError(f"{path}: an entry without a string pair_id")
    return {str(i) for i in ids}


def _counts(counter: Mapping[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(counter.items())) or "none"


def run_classify(args: argparse.Namespace) -> int:
    """Classify every omission pair the cache lacks (paid calls: the user runs this)."""
    try:
        config = load_config(args.config)
        client = create_client(config)
        budget = create_budget(config)
    except InputError as exc:
        print(f"classify: {exc}", file=sys.stderr)
        return 2
    ids = load_ids(args.ids) if args.ids is not None else None
    items = omission_pairs(_pairs(args.pairs), ids)
    if ids is not None and len(items) != len(ids):
        print(f"classify: {len(ids) - len(items)} listed ids are not omission pairs (skipped)")
    cache = load_cache(args.cache)
    tag = f"{config.model} effort={config.effort} max_tokens={config.max_tokens}"
    counts = classify(items, client, budget, cache, tag, args.cache)
    spent = budget.spent_usd
    found = Counter(lookup(cache, pid, fact) or "no valid label" for pid, fact in items.items())
    print(f"classify: {len(items)} omission pairs: {_counts(counts)}")
    print(f"classify: labels: {_counts(found)}")
    print(
        f"classify: {budget.calls} calls, "
        + (f"${spent:.4f}" if spent is not None else "cost unknown")
        + f"; prompt sha256 {prompt_hash()}; cache {args.cache}"
    )
    return 0 if sum(counts[k] for k in ("cached", "classified")) == len(items) else 2


SAMPLE_COLUMNS = ("pair_id", "removed_fact", "label", "reviewer_label", "reviewer_comment")


def write_sample(args: argparse.Namespace) -> int:
    """A fixed-seed spot-check sheet: labels taken in turn (in LABELS order), a random
    labelled pair from each, until `n` are chosen or none are left."""
    items = omission_pairs(_pairs(args.pairs))
    cache = load_cache(args.cache)
    by_label: dict[str, list[str]] = {label: [] for label in LABELS}
    for pid, fact in sorted(items.items()):
        if (label := lookup(cache, pid, fact)) is not None:
            by_label[label].append(pid)
    rng = random.Random(args.seed)
    for pids in by_label.values():
        rng.shuffle(pids)
    chosen: list[tuple[str, str]] = []
    while len(chosen) < args.n and any(by_label.values()):
        for label in LABELS:
            if by_label[label] and len(chosen) < args.n:
                chosen.append((by_label[label].pop(), label))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SAMPLE_COLUMNS)
        for pid, label in sorted(chosen):
            writer.writerow([pid, items[pid], label, "", ""])
    spread = Counter(label for _, label in chosen)
    print(f"wrote {args.out}: {len(chosen)} pairs (seed {args.seed}): {_counts(spread)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hash", help="print the prompt's SHA-256")
    cl = sub.add_parser("classify", help="label the omission pairs (paid calls)")
    cl.add_argument("--pairs", type=Path, required=True)
    cl.add_argument("--config", type=Path, required=True)
    cl.add_argument("--cache", type=Path, required=True)
    cl.add_argument("--ids", type=Path, help="only these pair ids (JSON list or 'records')")
    sa = sub.add_parser("sample", help="write a spot-check sheet")
    sa.add_argument("--pairs", type=Path, required=True)
    sa.add_argument("--cache", type=Path, required=True)
    sa.add_argument("--out", type=Path, required=True)
    sa.add_argument("--n", type=int, default=20)
    sa.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args(argv)
    try:
        if args.command == "hash":
            print(prompt_hash())
            return 0
        if args.command == "classify":
            return run_classify(args)
        return write_sample(args)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        print(f"action_classify: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
