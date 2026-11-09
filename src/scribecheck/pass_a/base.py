"""Shared pieces of Pass A with the model: cases, facts, runners and usage attribution.

Both pipelines run staged over a list of cases, so one code path serves a single live
check (a list of one) and a folder run, where each stage is one Message Batch. A `Runner`
sends one stage's requests; a `Ledger` attributes each request's calls, tokens and cost to
the notes it served, splitting a request shared by several notes (an extraction on a
transcript they share) evenly between them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from scribecheck.flags import Flag, NoteEvidence, Severity, TranscriptEvidence, UncheckedItem
from scribecheck.model.base import Completion, ModelClient, ModelFailure, ModelRequest, Usage
from scribecheck.model.budget import Budget
from scribecheck.model.structured import BatchClient, complete, complete_batch
from scribecheck.note import Note
from scribecheck.transcript import Transcript

Result = Completion[Any] | ModelFailure


@dataclass
class Case:
    """One transcript and note, with the rules layer's findings for the second check."""

    id: str
    transcript: Transcript
    note: Note
    candidates: tuple[Flag, ...] = ()
    """Rules omission candidates, for Pass A's second check."""
    warnings: tuple[Flag, ...] = ()
    """Rules changed-value warnings, for Pass B's model check."""
    pass_b_items: tuple[UncheckedItem, ...] = ()
    """Pass B items the rules could not settle, for Pass B's model check."""


@dataclass
class Fact:
    """A fact the transcript establishes, as one pipeline holds it."""

    id: str
    text: str
    category: str = "other"
    severity: Severity = "supporting"
    core: bool = True
    evidence: TranscriptEvidence | None = None
    evidence_quote: str = ""
    origin: Literal["model", "rules"] = "model"
    forced: tuple[str, ...] = ()
    """Code rules that overrode the audit for this fact ("core_kind", "never_drop")."""


@dataclass
class Tally:
    """One note's model usage."""

    calls: int = 0
    shared_calls: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    model_ids: set[str] = field(default_factory=set)


@dataclass
class Outcome:
    """Pass A's findings for one note."""

    flags: list[Flag] = field(default_factory=list)
    unchecked: list[UncheckedItem] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    resolved_candidates: set[int] = field(default_factory=set)
    """Indexes into `Case.candidates` the model decided (confirmed or rejected)."""
    resolved_warnings: set[int] = field(default_factory=set)
    """Indexes into `Case.warnings` Pass B's model decided."""
    resolved_items: set[int] = field(default_factory=set)
    """Indexes into `Case.pass_b_items` Pass B's model decided."""
    not_verifiable: list[tuple[NoteEvidence, str]] = field(default_factory=list)
    """Pass C statements reported as not verifiable, with their kind."""
    trace: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Each extracted fact's path through the scribecheck pipeline, by fact id (the fields
    of `report.TraceFact`)."""
    tally: Tally = field(default_factory=Tally)


class Runner(Protocol):
    """Sends one stage's requests and returns each custom_id's result."""

    batch: bool

    def run(self, requests: Sequence[ModelRequest[Any]]) -> dict[str, Result]: ...


class LiveRunner:
    """One validated call after another."""

    batch = False

    def __init__(self, client: ModelClient, budget: Budget) -> None:
        self.client, self.budget = client, budget

    def run(self, requests: Sequence[ModelRequest[Any]]) -> dict[str, Result]:
        return {r.custom_id: complete(self.client, r, self.budget) for r in requests}


class BatchRunner:
    """Each stage as one Message Batch (with one correction batch for parse failures)."""

    batch = True

    def __init__(self, client: BatchClient, budget: Budget) -> None:
        self.client, self.budget = client, budget

    def run(self, requests: Sequence[ModelRequest[Any]]) -> dict[str, Result]:
        if not requests:
            return {}
        return dict(complete_batch(self.client, requests, self.budget))


class Ledger:
    """Attributes each request's usage to the notes it served."""

    def __init__(self, outcomes: dict[str, Outcome], budget: Budget, *, batch: bool) -> None:
        self.outcomes, self.budget, self.batch = outcomes, budget, batch
        self._n = 0

    def custom_id(self, stage: str) -> str:
        self._n += 1
        return f"{stage}-{self._n}"

    def record(self, result: Result, owners: Sequence[str]) -> None:
        if not owners:
            return
        usage = result.usage
        attempts = result.attempts
        prices = self.budget.prices
        cost = None if prices is None else prices.cost(usage, batch=self.batch)
        share = len(owners)
        for i, owner in enumerate(owners):
            tally = self.outcomes[owner].tally
            if share == 1:
                tally.calls += attempts
            else:
                tally.shared_calls += attempts
            tally.usage = tally.usage + _split(usage, share, first=i == 0)
            if cost is not None:
                tally.cost_usd = (tally.cost_usd or 0.0) + cost / share
            if isinstance(result, Completion) and result.model_id:
                tally.model_ids.add(result.model_id)


def _split(usage: Usage, parts: int, *, first: bool) -> Usage:
    """An even share of `usage`; the first share takes the remainders."""

    def part(value: int) -> int:
        return value // parts + (value % parts if first else 0)

    return Usage(
        part(usage.input_tokens),
        part(usage.output_tokens),
        part(usage.cache_write_5m_tokens),
        part(usage.cache_write_1h_tokens),
        part(usage.cache_read_tokens),
    )


def render_transcript(transcript: Transcript) -> str:
    """The transcript as numbered turns, so quotes can be traced to one turn."""
    lines = []
    for turn in transcript.turns:
        who = turn.label or turn.speaker
        lines.append(f"[T{turn.number}] {who}: {turn.text}")
    return "\n".join(lines)


def plain_transcript(transcript: Transcript) -> str:
    """The transcript as speaker-prefixed lines, as the OmissionBench harness passes it."""
    if transcript.text is not None:
        return transcript.text
    return "\n".join(f"{t.label or t.speaker}: {t.text}" for t in transcript.turns)


def facts_block(items: Iterable[tuple[str, str]]) -> str:
    return "\n".join(f"{fid}: {' '.join(text.split())}" for fid, text in items)
