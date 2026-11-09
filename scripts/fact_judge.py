"""Model-assisted fact matching for score_run.py (M3 session 4).

For each omission note, one structured call asks a model whether any of the note's
extracted facts, and separately any of its Pass A omission flags (and partial verdicts),
states the same clinical fact as the benchmark's removed fact, with the ids that do. Only
fact texts are sent: never the transcript or the note. The judge is the same model family
as the system under test (Claude Sonnet 5.5), so its matches are checked against the word
matcher and by a person on a fixed-seed sample (`score_run.py sample`).

The judgements are cached in the run folder, one entry per note keyed by a SHA-256 of the
prompt text, the model with its effort and token limit, and every text sent, so re-scoring
makes no calls, and a changed fact list, prompt or setting is judged afresh. Development data
only.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from scribecheck.model.base import Block, Message, ModelClient, ModelFailure, ModelRequest
from scribecheck.model.budget import Budget
from scribecheck.model.structured import complete

PROMPT_VERSION = "fact_judge_v1"
SYSTEM = (
    "You compare short clinical fact statements. You are given a removed fact, a list of "
    "extracted facts and a list of flagged facts, each with an id. You never see a "
    "transcript or a note. Answer only from the statements' meaning."
)
TASK = """REMOVED FACT:
{removed}

EXTRACTED FACTS:
{facts}

FLAGGED FACTS:
{flags}

For each list, decide whether any entry states the same clinical fact as the removed fact.
- "same": true when at least one entry states it, in any wording: other words, an
  abbreviation, a more specific version, or a longer compound statement that contains it.
  Then "ids" lists every such entry's id, exactly as given.
- An entry does not state it when it names a different drug, dose, side, number, threshold,
  symptom, finding or instruction, when it is vaguer than the removed fact, or when it only
  shares the topic. Then "same" is false and "ids" is empty.
Return {{"facts": {{"same": ..., "ids": [...]}}, "flags": {{"same": ..., "ids": [...]}}}}."""


class Match(BaseModel):
    same: bool
    ids: list[str]


class Judgement(BaseModel):
    facts: Match
    flags: Match


@dataclass(frozen=True)
class Item:
    """One omission note to judge: the removed fact(s) and the candidates sent. `model` in
    `key` is the judge's model with its effort and token limit."""

    pair_id: str
    removed: tuple[str, ...]
    facts: tuple[tuple[str, str], ...]
    flags: tuple[tuple[str, str], ...]

    def key(self, model: str) -> str:
        payload = json.dumps(
            [PROMPT_VERSION, SYSTEM, TASK, model, self.removed, self.facts, self.flags],
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def task(self) -> str:
        def block(rows: Sequence[tuple[str, str]]) -> str:
            return "\n".join(f"{i}: {text}" for i, text in rows) or "(none)"

        return TASK.format(
            removed="\n".join(self.removed), facts=block(self.facts), flags=block(self.flags)
        )


def item(pair_id: str, removed: Sequence[str], report: Mapping[str, Any]) -> Item:
    """The note's removed fact(s), its trace facts, and its Pass A omission flags and
    partial verdicts, as sent to the judge."""
    facts = tuple(
        (str(step.get("id")), str(step.get("fact") or ""))
        for step in report.get("pass_a_trace") or []
    )
    flags = tuple(
        (str(raw.get("id")), str(raw.get("fact") or ""))
        for raw in [*report.get("flags", []), *report.get("info", [])]
        if raw.get("type") == "omission"
    )
    return Item(pair_id, tuple(removed), facts, flags)


def problem(judgement: Judgement, it: Item) -> str | None:
    """Why a judgement cannot be used (an id that was not sent, or "same" without ids), or
    None when it is valid."""
    for name, match, rows in (
        ("facts", judgement.facts, it.facts),
        ("flags", judgement.flags, it.flags),
    ):
        sent = {i for i, _ in rows}
        if any(i not in sent for i in match.ids):
            return f"{name}: an id that was not sent"
        if match.same != bool(match.ids):
            return f"{name}: 'same' disagrees with the ids"
    return None


def load_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"prompt": PROMPT_VERSION, "entries": {}}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("entries", {})
    return data


def save_cache(path: Path, cache: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def lookup(cache: Mapping[str, Any], it: Item, model: str | None = None) -> Judgement | None:
    """The cached, valid judgement for `it` (under `model`, or the model the cache was made
    with), or None when it is missing, stale or invalid."""
    entry = cache.get("entries", {}).get(it.pair_id)
    if not isinstance(entry, Mapping) or entry.get("judgement") is None:
        return None
    if entry.get("key") != it.key(model or str(cache.get("model") or "")):
        return None
    try:
        judgement = Judgement.model_validate(entry["judgement"])
    except ValidationError:
        return None
    return judgement if problem(judgement, it) is None else None


def judge(
    items: Sequence[Item],
    client: ModelClient,
    budget: Budget,
    cache: dict[str, Any],
    model: str,
    path: Path,
) -> Counter[str]:
    """Judge each item not already cached, saving the cache after every call. Returns
    counts: cached, judged, and each failure reason."""
    cache["prompt"], cache["model"] = PROMPT_VERSION, model
    entries: dict[str, Any] = cache.setdefault("entries", {})
    counts: Counter[str] = Counter()
    for it in items:
        if lookup(cache, it, model) is not None:
            counts["cached"] += 1
            continue
        request = ModelRequest(
            system=SYSTEM,
            messages=(Message("user", (Block(it.task()),)),),
            output=Judgement,
            custom_id="judge",
        )
        result = complete(client, request, budget)
        if isinstance(result, ModelFailure):
            counts[result.reason] += 1
            entries[it.pair_id] = {"key": it.key(model), "judgement": None, "error": result.reason}
        elif (why := problem(result.parsed, it)) is not None:
            counts["invalid ids"] += 1
            entries[it.pair_id] = {"key": it.key(model), "judgement": None, "error": why}
        else:
            counts["judged"] += 1
            entries[it.pair_id] = {
                "key": it.key(model),
                "judgement": result.parsed.model_dump(),
                "model_id": result.model_id,
            }
        save_cache(path, cache)
        if isinstance(result, ModelFailure) and result.reason == "cost_cap_reached":
            break
    return counts
