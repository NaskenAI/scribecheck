"""Pass B with the model: changed facts the rules could not settle (scribecheck pipeline).

The model judges, per note: the rules' changed-value warnings; the Pass B items the rules
left unchecked because they could not align or compare them (including a drug found in
the note only, a possible substitution); and certainty changes, with the transcript's
hedged sentences (ConText "possible" cues) as hints. Each "changed" verdict, with both its
quotes verified, then gets a skeptical second check:

- upheld: a changed-fact flag at confirmed confidence, with the model's severity;
- not upheld: unchecked "second_check_rejected" (or, with `[report]
  show_rejected_as_possible`, a flag at possible confidence);
- a quote that cannot be verified after one re-ask: unchecked "quote_unverified".

A warning the model finds consistent is dropped, like a rejected rules candidate.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel

from scribecheck import prompts
from scribecheck.config import Config
from scribecheck.flags import Flag, NoteEvidence, Severity, TranscriptEvidence, UncheckedItem
from scribecheck.model.base import Block, Message, ModelFailure, ModelRequest
from scribecheck.pass_a.base import Case, Ledger, Outcome, Runner, facts_block, render_transcript
from scribecheck.pass_a.quotes_stage import QuoteFixer
from scribecheck.rules.context import find_cues

PROMPTS = "scribecheck_v5"
Kind = Literal[
    "certainty",
    "drug_substituted",
    "dose",
    "frequency",
    "route",
    "side",
    "negation",
    "allergy",
    "other",
]
SENT_REASONS = frozenset(
    {
        "unaligned",
        "transcript_inconsistent",
        "note_inconsistent",
        "mixed_temporality",
        "not_comparable",
        "ambiguous_value",
        "possible",
    }
)
"""Pass B items with these primary reasons go to the model."""
RULE_IDS: dict[str, str] = {
    "certainty": "certainty.changed",
    "drug_substituted": "drug.substituted",
    "negation": "negation.reversed",
}
MAX_HINTS = 20
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")


class _Change(BaseModel):
    id: str
    verdict: Literal["changed", "consistent"]
    kind: Kind = "other"
    item: str = ""
    transcript_value: str = ""
    note_value: str = ""
    transcript_quote: str = ""
    note_quote: str = ""
    severity: Severity = "supporting"


class Changes(BaseModel):
    findings: list[_Change]


class _Second(BaseModel):
    id: str
    upheld: bool
    transcript_quote: str = ""
    note_quote: str = ""


class SecondCheck(BaseModel):
    claims: list[_Second]


def sendable(item: UncheckedItem) -> bool:
    """A Pass B item the model should judge (a drug in the transcript only is Pass A's)."""
    if item.reason not in SENT_REASONS:
        return False
    if item.reason == "unaligned":
        return bool(item.note_evidence) and not item.transcript_evidence
    return True


def hints(case: Case) -> list[str]:
    """Transcript sentences with a ConText "possible" cue, the hints for certainty changes."""
    out: list[str] = []
    for turn in case.transcript.turns:
        for cue in find_cues(turn.text):
            if cue.attribute != "possible":
                continue
            for sentence in _SENTENCE.finditer(turn.text):
                if sentence.start() <= cue.start < sentence.end():
                    text = f"[T{turn.number}] {sentence.group().strip()}"
                    if text not in out:
                        out.append(text)
    return out[:MAX_HINTS]


def run(
    cases: Sequence[Case],
    config: Config,
    runner: Runner,
    ledger: Ledger,
    outcomes: dict[str, Outcome],
) -> None:
    requests, owners = [], {}
    listed: dict[str, dict[str, tuple[str, int]]] = {}
    for case in cases:
        entries: list[tuple[str, str]] = []
        ids: dict[str, tuple[str, int]] = {}
        for i, flag in enumerate(case.warnings):
            pid = f"w{i + 1}"
            ids[pid] = ("warning", i)
            entries.append((pid, _describe_warning(flag)))
        for i, item in enumerate(case.pass_b_items):
            if sendable(item):
                pid = f"p{i + 1}"
                ids[pid] = ("item", i)
                entries.append((pid, _describe_item(item)))
        listed[case.id] = ids
        task = prompts.render(
            prompts.load(PROMPTS, "changes"),
            facts_block=facts_block(entries) or "(none)",
            hints="\n".join(hints(case)) or "(none)",
        )
        request = _request(config, ledger, "changes", case, task, Changes)
        requests.append(request)
        owners[request.custom_id] = case
    found: dict[str, list[_Change]] = {}
    sources: dict[int, tuple[str, int]] = {}
    """The rules warning or item a "changed" verdict is about. It is settled (removed from
    the report) only by a final verdict; if the quotes or the second check fail, the rules
    finding stays as it was."""
    for cid, result in runner.run(requests).items():
        case = owners[cid]
        ledger.record(result, [case.id])
        if isinstance(result, ModelFailure):
            outcomes[case.id].unchecked.append(
                UncheckedItem(category="pass_b", item="changed facts", reasons=(result.reason,))
            )
            continue
        changes = []
        for entry in result.parsed.findings:
            known = listed[case.id].get(entry.id.strip().lower())
            if entry.verdict == "changed":
                changes.append(entry)
                if known is not None:
                    sources[id(entry)] = known
            elif known is not None:
                _settle(outcomes[case.id], known)  # consistent: the rules item is dropped
        found[case.id] = changes

    # Verify the first verdicts' quotes (one re-ask each).
    t_fixer = QuoteFixer(runner, ledger, config, "transcript")
    n_fixer = QuoteFixer(runner, ledger, config, "note")
    by_id = {c.id: c for c in cases}
    for case_id, changes in found.items():
        case = by_id[case_id]
        for n, change in enumerate(changes):
            label = change.item or change.kind
            t_fixer.want(case, f"b{n}", label, change.transcript_quote)
            n_fixer.want(case, f"b{n}", label, change.note_quote)
    t_fixer.resolve()
    n_fixer.resolve()

    verified: dict[str, list[tuple[_Change, TranscriptEvidence, NoteEvidence]]] = {}
    for case_id, changes in found.items():
        case = by_id[case_id]
        for n, change in enumerate(changes):
            t_ev, n_ev = t_fixer.evidence(case, f"b{n}"), n_fixer.evidence(case, f"b{n}")
            if isinstance(t_ev, TranscriptEvidence) and isinstance(n_ev, NoteEvidence):
                verified.setdefault(case_id, []).append((change, t_ev, n_ev))
            elif id(change) not in sources:
                outcomes[case_id].unchecked.append(
                    UncheckedItem(
                        category=change.kind,
                        item=(change.item or change.kind)[:200],
                        reasons=("quote_unverified",),
                    )
                )

    # The second check.
    requests, owners2 = [], {}
    for case_id, items in verified.items():
        case = by_id[case_id]
        claims = [
            (
                f"s{n + 1}",
                f"{c.kind} of {c.item or 'the item'}: the transcript gives "
                f"{c.transcript_value!r} but the note gives {c.note_value!r}",
            )
            for n, (c, _, _) in enumerate(items)
        ]
        task = prompts.render(
            prompts.load(PROMPTS, "changes_second"), facts_block=facts_block(claims)
        )
        request = _request(config, ledger, "changes_second", case, task, SecondCheck)
        requests.append(request)
        owners2[request.custom_id] = case_id
    for cid, result in runner.run(requests).items():
        case_id = owners2[cid]
        ledger.record(result, [case_id])
        items = verified[case_id]
        if isinstance(result, ModelFailure):
            for change, _, _ in items:
                if id(change) in sources:
                    continue  # the rules finding stays
                outcomes[case_id].unchecked.append(
                    UncheckedItem(
                        category=change.kind,
                        item=(change.item or change.kind)[:200],
                        reasons=(result.reason,),
                    )
                )
            continue
        upheld = {e.id.strip().lower(): e.upheld for e in result.parsed.claims}
        for n, (change, t_ev, n_ev) in enumerate(items):
            verdict = upheld.get(f"s{n + 1}")
            source = sources.get(id(change))
            if verdict is None:
                if source is None:
                    outcomes[case_id].unchecked.append(
                        UncheckedItem(
                            category=change.kind,
                            item=(change.item or change.kind)[:200],
                            reasons=("model_parse_failure",),
                        )
                    )
                continue  # no answer: a rules finding stays as it was
            if source is not None:
                _settle(outcomes[case_id], source)
            if verdict:
                outcomes[case_id].flags.append(_flag(change, t_ev, n_ev, "confirmed"))
            elif config.show_rejected_as_possible:
                outcomes[case_id].flags.append(_flag(change, t_ev, n_ev, "possible"))
            else:
                outcomes[case_id].unchecked.append(
                    UncheckedItem(
                        category=change.kind,
                        item=(change.item or change.kind)[:200],
                        reasons=("second_check_rejected",),
                        transcript_value=change.transcript_value or None,
                        note_value=change.note_value or None,
                        transcript_evidence=(t_ev,),
                        note_evidence=(n_ev,),
                    )
                )


def _settle(outcome: Outcome, source: tuple[str, int]) -> None:
    kind, index = source
    (outcome.resolved_warnings if kind == "warning" else outcome.resolved_items).add(index)


def _describe_warning(flag: Flag) -> str:
    quote = flag.transcript_evidence.text if flag.transcript_evidence else ""
    return (
        f"{flag.category}: the transcript gives {flag.transcript_value!r} ({quote!r}); "
        f"the note gives {flag.note_value!r}"
    )


def _describe_item(item: UncheckedItem) -> str:
    t = "; ".join(q.text for q in item.transcript_evidence) or "-"
    n = "; ".join(q.text for q in item.note_evidence) or "-"
    what = (
        "a drug in the note only (possible substitution)"
        if item.reason == "unaligned"
        else (f"{item.category} of {item.item}")
    )
    return f"{what}: transcript {t!r}; note {n!r} (rules could not settle: {item.reason})"


def _flag(
    change: _Change,
    t_ev: TranscriptEvidence,
    n_ev: NoteEvidence,
    confidence: Literal["confirmed", "possible"],
) -> Flag:
    return Flag(
        id="F0",
        type="changed_fact",
        category=change.kind,
        verdict="changed",
        severity=change.severity,
        confidence=confidence,
        origin="model",
        rule_id=RULE_IDS.get(change.kind, f"{change.kind}.changed"),
        second_check="model",
        fact=change.item or None,
        transcript_value=change.transcript_value or None,
        note_value=change.note_value or None,
        transcript_evidence=t_ev,
        note_evidence=n_ev,
    )


def _request(
    config: Config, ledger: Ledger, stage: str, case: Case, task: str, output: type[BaseModel]
) -> ModelRequest[Any]:
    settings = config.stages.get(f"scribecheck.{stage}")
    return ModelRequest(
        system=prompts.load(PROMPTS, "system"),
        messages=(
            Message(
                "user",
                (
                    Block("TRANSCRIPT:\n" + render_transcript(case.transcript), cache=True),
                    Block("NOTE:\n" + case.note.text, cache=True),
                    Block(task),
                ),
            ),
        ),
        output=output,
        max_tokens=settings.max_tokens if settings and settings.max_tokens else 8000,
        custom_id=ledger.custom_id(f"b{stage[:4]}"),
        model=settings.name if settings else None,
        effort=settings.effort if settings else None,
    )
