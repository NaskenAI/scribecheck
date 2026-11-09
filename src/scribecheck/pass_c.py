"""Pass C with the model: unsupported statements (scribecheck pipeline).

1. Split the note into atomic statements, each a verbatim note span with a kind (the call
   sees the note only); a span that cannot be found in the note, after one re-ask, is
   unchecked "quote_unverified".
2. Exemptions (`[pass_c] exempt`) are reported as not verifiable and never flagged: vitals,
   results and demographics; examination findings in a face-to-face visit or one with no
   modality (a telephone or video finding is the examination rule's); and history from
   the record, but only where the note itself marks it so ("per records", "per chart",
   "previously documented", "from EHR", or a statement in an imported or prefilled
   section). Any other history is checked like a clinical statement.
3. The rest are checked against the transcript, about ten per call: supported,
   contradicted, unsupported or not verifiable, with a severity for the last two kinds of
   problem. A contradiction needs a verified transcript quote.
4. Every contradicted or unsupported verdict gets a skeptical second check: upheld gives a
   flag at confirmed confidence; overturned with a verified transcript quote means the
   transcript supports it after all (no flag); otherwise the verdict is unchecked
   "second_check_rejected" (or a possible flag, with `[report] show_rejected_as_possible`).
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
from scribecheck.note import Note
from scribecheck.pass_a.base import (
    Case,
    Ledger,
    Outcome,
    Result,
    Runner,
    facts_block,
    render_transcript,
)
from scribecheck.pass_a.quotes_stage import QuoteFixer
from scribecheck.rules.examination import HANDS_ON

PROMPTS = "scribecheck_v5"
StatementKind = Literal[
    "examination", "vital", "result", "record_history", "demographic", "clinical"
]
_RECORD_CUE = re.compile(
    r"(?<![A-Za-z])(?:per\s+(?:the\s+)?(?:records?|chart|notes|EHR|EPR|GP\s+records?)|"
    r"previously\s+documented|documented\s+previously|from\s+(?:the\s+)?(?:EHR|EPR|records?)|"
    r"according\s+to\s+(?:the\s+)?(?:records?|notes|chart))(?![A-Za-z])",
    re.IGNORECASE,
)
_RECORD_SECTION = re.compile(
    r"(?<![A-Za-z])(?:imported|prefilled|pre-filled|from\s+(?:the\s+)?(?:record|EHR|EPR))"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)
"""Section headings that mark imported or prefilled content (a plain "Problem list" does
not: AI scribes write their own)."""


class _Statement(BaseModel):
    text: str
    kind: StatementKind = "clinical"


class Split(BaseModel):
    statements: list[_Statement]


class _Checked(BaseModel):
    id: str
    verdict: Literal["supported", "contradicted", "unsupported", "not_verifiable"]
    transcript_quote: str = ""
    severity: Severity = "supporting"


class Checked(BaseModel):
    statements: list[_Checked]


class _Second(BaseModel):
    id: str
    upheld: bool
    transcript_quote: str = ""


class SecondCheck(BaseModel):
    statements: list[_Second]


class _Item:
    def __init__(self, key: str, kind: str, evidence: NoteEvidence) -> None:
        self.key, self.kind, self.evidence = key, kind, evidence
        self.verdict = ""
        self.severity: Severity = "supporting"
        self.quote = ""


def from_record(note: Note, evidence: NoteEvidence) -> bool:
    """The note itself marks this statement as history from the record: a cue in the
    statement's own text, or an imported or prefilled section."""
    return bool(_RECORD_CUE.search(evidence.text) or _RECORD_SECTION.search(evidence.section or ""))


def exempt(case: Case, config: Config, kind: str, evidence: NoteEvidence) -> bool:
    if kind == "clinical" or kind not in config.exempt:
        return False
    if kind == "record_history":
        return from_record(case.note, evidence)
    if kind == "examination":
        if case.transcript.modality == "video":
            return not HANDS_ON.search(evidence.text)  # seen on video; touch is the rule's
        return case.transcript.modality != "telephone"
    return True


def run(
    cases: Sequence[Case],
    config: Config,
    runner: Runner,
    ledger: Ledger,
    outcomes: dict[str, Outcome],
) -> None:
    by_id = {c.id: c for c in cases}

    # 1. Split (note only) and anchor.
    requests, owners = [], {}
    for case in cases:
        task = prompts.load(PROMPTS, "split")
        request = _request(config, ledger, "split", [_note(case)], task, Split)
        requests.append(request)
        owners[request.custom_id] = case.id
    split: dict[str, list[_Statement]] = {}
    for cid, result in runner.run(requests).items():
        case_id = owners[cid]
        ledger.record(result, [case_id])
        if isinstance(result, ModelFailure):
            outcomes[case_id].unchecked.append(
                UncheckedItem(category="pass_c", item="statements", reasons=(result.reason,))
            )
        else:
            split[case_id] = result.parsed.statements
    anchors = QuoteFixer(runner, ledger, config, "note")
    for case_id, statements in split.items():
        for n, statement in enumerate(statements):
            anchors.want(by_id[case_id], f"c{n + 1}", statement.text, statement.text)
    anchors.resolve()

    to_check: dict[str, list[_Item]] = {}
    for case_id, statements in split.items():
        case = by_id[case_id]
        seen: set[tuple[int, int]] = set()
        for n, statement in enumerate(statements):
            found = anchors.evidence(case, f"c{n + 1}")
            if not isinstance(found, NoteEvidence):
                outcomes[case_id].unchecked.append(
                    UncheckedItem(
                        category="statement",
                        item=statement.text[:200],
                        reasons=("quote_unverified",),
                    )
                )
                continue
            if (found.start, found.end) in seen:
                continue
            seen.add((found.start, found.end))
            if statement.kind == "examination" and (
                case.transcript.modality == "telephone"
                or (case.transcript.modality == "video" and HANDS_ON.search(found.text))
            ):
                continue  # the examination rule's
            if exempt(case, config, statement.kind, found):
                outcomes[case_id].not_verifiable.append((found, statement.kind))
                continue
            to_check.setdefault(case_id, []).append(_Item(f"c{n + 1}", statement.kind, found))

    # 2. Check against the transcript, about ten statements per call.
    per_call = config.check_facts_per_call
    requests, owners_c = [], {}
    for case_id, items in to_check.items():
        case = by_id[case_id]
        for start in range(0, len(items), per_call):
            group = items[start : start + per_call]
            task = prompts.render(
                prompts.load(PROMPTS, "statements"),
                facts_block=facts_block((i.key, i.evidence.text) for i in group),
            )
            request = _request(config, ledger, "statements", _both(case), task, Checked)
            requests.append(request)
            owners_c[request.custom_id] = (case_id, group)
    flagged: dict[str, list[_Item]] = {}
    _collect(runner.run(requests), owners_c, ledger, outcomes, flagged)

    # 3. Contradictions need a verified transcript quote.
    t_fixer = QuoteFixer(runner, ledger, config, "transcript")
    for case_id, items in flagged.items():
        for item in items:
            if item.verdict == "contradicted":
                t_fixer.want(by_id[case_id], item.key, item.evidence.text, item.quote)
    t_fixer.resolve()
    for case_id, items in list(flagged.items()):
        kept = []
        for item in items:
            if item.verdict == "contradicted" and not isinstance(
                t_fixer.evidence(by_id[case_id], item.key), TranscriptEvidence
            ):
                outcomes[case_id].unchecked.append(_unchecked(item, "quote_unverified"))
            else:
                kept.append(item)
        flagged[case_id] = kept

    # 4. Second check.
    requests, owners_s = [], {}
    for case_id, items in flagged.items():
        if not items:
            continue
        case = by_id[case_id]
        task = prompts.render(
            prompts.load(PROMPTS, "statements_second"),
            facts_block=facts_block((i.key, f"[{i.verdict}] {i.evidence.text}") for i in items),
        )
        request = _request(config, ledger, "statements_second", _both(case), task, SecondCheck)
        requests.append(request)
        owners_s[request.custom_id] = case_id
    seconds: dict[str, dict[str, _Second]] = {}
    failures: dict[str, ModelFailure] = {}
    for cid, result in runner.run(requests).items():
        case_id = owners_s[cid]
        ledger.record(result, [case_id])
        if isinstance(result, ModelFailure):
            failures[case_id] = result
        else:
            seconds[case_id] = {e.id.strip().lower(): e for e in result.parsed.statements}
    support = QuoteFixer(runner, ledger, config, "transcript")
    for case_id, entries in seconds.items():
        for key, entry in entries.items():
            if not entry.upheld and entry.transcript_quote.strip():
                support.want(by_id[case_id], key, key, entry.transcript_quote)
    support.resolve()

    for case_id, items in flagged.items():
        case = by_id[case_id]
        outcome = outcomes[case_id]
        for item in items:
            t_ev = t_fixer.evidence(case, item.key) if item.verdict == "contradicted" else None
            failure = failures.get(case_id)
            if failure is not None:
                outcome.unchecked.append(_unchecked(item, failure.reason))
                continue
            second = seconds.get(case_id, {}).get(item.key)
            transcript = t_ev if isinstance(t_ev, TranscriptEvidence) else None
            if second is not None and second.upheld:
                outcome.flags.append(_flag(item, transcript, "confirmed"))
            elif second is not None and isinstance(
                support.evidence(case, item.key), TranscriptEvidence
            ):
                continue  # the transcript supports it after all
            elif config.show_rejected_as_possible:
                outcome.flags.append(_flag(item, transcript, "possible"))
            else:
                outcome.unchecked.append(
                    _unchecked(item, "second_check_rejected" if second else "model_parse_failure")
                )


def _collect(
    results: dict[str, Result],
    owners: dict[str, tuple[str, list[_Item]]],
    ledger: Ledger,
    outcomes: dict[str, Outcome],
    flagged: dict[str, list[_Item]],
) -> None:
    for cid, result in results.items():
        case_id, group = owners[cid]
        ledger.record(result, [case_id])
        if isinstance(result, ModelFailure):
            for item in group:
                outcomes[case_id].unchecked.append(_unchecked(item, result.reason))
            continue
        verdicts = {e.id.strip().lower(): e for e in result.parsed.statements}
        for item in group:
            entry = verdicts.get(item.key)
            if entry is None:
                outcomes[case_id].unchecked.append(_unchecked(item, "model_parse_failure"))
            elif entry.verdict == "not_verifiable":
                outcomes[case_id].not_verifiable.append((item.evidence, item.kind))
            elif entry.verdict in ("contradicted", "unsupported"):
                item.verdict, item.severity, item.quote = (
                    entry.verdict,
                    entry.severity,
                    entry.transcript_quote,
                )
                flagged.setdefault(case_id, []).append(item)


def _flag(
    item: _Item, transcript: TranscriptEvidence | None, confidence: Literal["confirmed", "possible"]
) -> Flag:
    verdict: Literal["contradicted", "unsupported"] = (
        "contradicted" if item.verdict == "contradicted" else "unsupported"
    )
    return Flag(
        id="F0",
        type="unsupported",
        category="statement",
        verdict=verdict,
        severity=item.severity,
        confidence=confidence,
        origin="model",
        rule_id=f"statement.{verdict}",
        second_check="model",
        fact=item.evidence.text,
        transcript_evidence=transcript,
        note_evidence=item.evidence,
    )


def _unchecked(item: _Item, reason: Any) -> UncheckedItem:
    return UncheckedItem(
        category="statement",
        item=item.evidence.text[:200],
        reasons=(reason,),
        note_evidence=(item.evidence,),
    )


def _note(case: Case) -> Block:
    return Block("NOTE:\n" + case.note.text, cache=True)


def _both(case: Case) -> list[Block]:
    return [
        Block("TRANSCRIPT:\n" + render_transcript(case.transcript), cache=True),
        _note(case),
    ]


def _request(
    config: Config,
    ledger: Ledger,
    stage: str,
    blocks: list[Block],
    task: str,
    output: type[BaseModel],
) -> ModelRequest[Any]:
    settings = config.stages.get(f"scribecheck.{stage}")
    return ModelRequest(
        system=prompts.load(PROMPTS, "system"),
        messages=(Message("user", (*blocks, Block(task))),),
        output=output,
        max_tokens=settings.max_tokens if settings and settings.max_tokens else 8000,
        custom_id=ledger.custom_id(f"c{stage[:4]}"),
        model=settings.name if settings else None,
        effort=settings.effort if settings else None,
    )
