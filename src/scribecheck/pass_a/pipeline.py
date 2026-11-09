"""scribecheck's own Pass A pipeline (docs/spec.md, "Pass A: omissions").

1. Extract facts from the transcript only, once per distinct transcript in a run, in
   overlapping chunks when it is long; every fact's evidence quote must be found in the
   transcript (re-asked once), or the fact is dropped as unsupported.
2. Merge in the rules layer's structured facts (the patient's current medicines and
   allergies), unless a model fact already names them.
3. Audit: drop unsupported or duplicate facts, mark core or contextual, grade severity with
   Fox et al.'s rubric.
4. Check the facts the standard requires against the whole note, about ten per call, as
   present / partial / absent; present and partial must quote the note.
5. Second look at absent facts, which may rescue them only with a verified note quote.
6. Rules candidates (omissions and changed values) go to the same quote-verified second
   check; a confirmed one takes the model's severity grade, a rejected one is dropped.

Quotes are verified with `quotes.find_*` and re-asked once; an item whose quote still
cannot be found is reported as unchecked ("quote_unverified"). The transcript (then the
note) is sent first in cache-marked blocks, so one consultation's calls share a prefix.
Extraction and audit requests are checked for note text before they are sent.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel

from scribecheck import prompts
from scribecheck.config import Config
from scribecheck.flags import Flag, NoteEvidence, Severity, TranscriptEvidence, UncheckedItem
from scribecheck.model.base import Block, Message, ModelFailure, ModelRequest
from scribecheck.note import parse_note
from scribecheck.pass_a.base import (
    Case,
    Fact,
    Ledger,
    Outcome,
    Result,
    Runner,
    facts_block,
    render_transcript,
)
from scribecheck.pass_a.blind import assert_blind
from scribecheck.pass_a.quotes_stage import QuoteFixer
from scribecheck.rules.omission import structured_omissions
from scribecheck.transcript import Transcript

PROMPTS = "scribecheck_v2"
Category = Literal[
    "presenting_complaint",
    "symptom",
    "history",
    "family_social",
    "medicine",
    "allergy",
    "examination",
    "investigation",
    "assessment",
    "plan",
    "follow_up",
]
SECTION: dict[str, str] = {
    "presenting_complaint": "History",
    "symptom": "History",
    "history": "History",
    "family_social": "History",
    "medicine": "Medications",
    "allergy": "Allergies",
    "examination": "Examination",
    "investigation": "Results",
    "assessment": "Assessment",
    "plan": "Plan",
    "follow_up": "Plan",
}
MAX_TOKENS = {
    "extract": 16000,
    "audit": 16000,
    "check": 8000,
    "second_look": 8000,
    "candidate": 8000,
}
_PLACEHOLDER_NOTE = parse_note("(no note)")
"""Run against the rules' omission screen, every item the transcript establishes becomes a
candidate: the transcript-only structured facts, with no note text involved."""


class _Extracted(BaseModel):
    fact: str
    category: Category
    attributes: list[Literal["negated", "family", "hypothetical", "historical"]] = []
    evidence: str


class Extraction(BaseModel):
    facts: list[_Extracted]


class _Audited(BaseModel):
    id: str
    keep: bool
    core: bool
    severity: Severity


class Audit(BaseModel):
    facts: list[_Audited]


class _Checked(BaseModel):
    id: str
    verdict: Literal["present", "partial", "absent"]
    note_quote: str = ""


class Checked(BaseModel):
    verdicts: list[_Checked]


class _Looked(BaseModel):
    id: str
    found: bool
    note_quote: str = ""
    transcript_quote: str = ""


class Looked(BaseModel):
    facts: list[_Looked]


class _Judged(BaseModel):
    id: str
    confirmed: bool
    transcript_quote: str = ""
    note_quote: str = ""
    severity: Severity = "supporting"


class Judged(BaseModel):
    findings: list[_Judged]


def run(
    cases: Sequence[Case],
    config: Config,
    runner: Runner,
    ledger: Ledger,
    outcomes: dict[str, Outcome],
) -> None:
    """The scribecheck pipeline over `cases`, writing each note's findings into `outcomes`."""
    stages = _Stages(config, ledger)
    groups: dict[str, list[Case]] = {}
    for case in cases:
        key = hashlib.sha256(render_transcript(case.transcript).encode()).hexdigest()
        groups.setdefault(key, []).append(case)

    facts = _extract(groups, config, stages, runner, ledger, outcomes)
    audited = _audit(groups, facts, stages, runner, ledger, outcomes)
    for key, members in groups.items():
        for case in members:
            outcomes[case.id].facts = list(audited.get(key, []))
            _trace_extraction(outcomes[case.id], key, facts, audited, stages)
    to_check = {
        case.id: [
            f
            for f in audited.get(key, [])
            if f.core and (config.standard == "strict" or f.severity != "peripheral")
        ]
        for key, members in groups.items()
        for case in members
    }
    for case in cases:
        for fact in to_check[case.id]:
            outcomes[case.id].trace[fact.id]["checked"] = True
    absent = _check(cases, to_check, config, stages, runner, ledger, outcomes)
    _second_look_and_candidates(cases, absent, config, stages, runner, ledger, outcomes)


def _trace_extraction(
    outcome: Outcome,
    key: str,
    facts: dict[str, list[Fact]],
    audited: dict[str, list[Fact]],
    stages: _Stages,
) -> None:
    """Start each fact's trace: extraction and audit (the later stages fill in the rest)."""
    for i, (text, category) in enumerate(stages.unsupported.get(key, []), start=1):
        outcome.trace[f"u{i}"] = {
            "id": f"u{i}",
            "fact": text,
            "category": category,
            "origin": "model",
            "evidence_verified": False,
        }
    kept = {f.id for f in audited.get(key, [])}
    graded = stages.audited.get(key)
    for fact in facts.get(key, []):
        if graded is None:
            audit = "failed"
        elif fact.id not in graded:
            audit = "not_audited"
        else:
            audit = "kept" if fact.id in kept else "dropped"
        outcome.trace[fact.id] = {
            "id": fact.id,
            "fact": fact.text,
            "category": fact.category,
            "origin": fact.origin,
            "evidence_verified": True,
            "audit": audit,
            "core": fact.core if audit != "failed" else None,
            "severity": fact.severity if audit != "failed" else None,
            "checked": False,
        }


# 1-2. Extraction, evidence and rules facts


def _chunks(transcript: Transcript, config: Config) -> list[str]:
    lines = render_transcript(transcript).split("\n")
    if sum(len(line) + 1 for line in lines) <= config.chunk_chars:
        return ["\n".join(lines)]
    chunks: list[str] = []
    start = 0
    while start < len(lines):
        size, end = 0, start
        while end < len(lines) and (
            end == start or size + len(lines[end]) + 1 <= config.chunk_chars
        ):
            size += len(lines[end]) + 1
            end += 1
        chunks.append("\n".join(lines[start:end]))
        if end >= len(lines):
            break
        start = max(end - config.chunk_overlap_turns, start + 1)
    return chunks


def _extract(
    groups: dict[str, list[Case]],
    config: Config,
    stages: _Stages,
    runner: Runner,
    ledger: Ledger,
    outcomes: dict[str, Outcome],
) -> dict[str, list[Fact]]:
    requests, owners = [], {}
    for key, members in groups.items():
        chunks = _chunks(members[0].transcript, config)
        for i, chunk in enumerate(chunks):
            note = (
                ""
                if len(chunks) == 1
                else (
                    f" (PART {i + 1} OF {len(chunks)} of a long transcript; neighbouring parts "
                    "overlap by a few turns)"
                )
            )
            title = (
                "TRANSCRIPT" if len(chunks) == 1 else f"TRANSCRIPT, PART {i + 1} OF {len(chunks)}"
            )
            blocks = [Block(f"{title}:\n{chunk}", cache=True)]
            task = prompts.render(prompts.load(PROMPTS, "extract"), chunk_note=note)
            request = stages.request("extract", blocks, task, Extraction)
            _assert_blind(request, members, "extract")
            requests.append(request)
            owners[request.custom_id] = key
    found: dict[str, list[_Extracted]] = {}
    failed: dict[str, list[ModelFailure]] = {}
    for cid, result in runner.run(requests).items():
        key = owners[cid]
        ledger.record(result, [c.id for c in groups[key]])
        if isinstance(result, ModelFailure):
            failed.setdefault(key, []).append(result)
        else:
            found.setdefault(key, []).extend(result.parsed.facts)

    fixer = QuoteFixer(runner, ledger, config, "transcript")
    staged: dict[str, list[Fact]] = {}
    for key, members in groups.items():
        for failure in failed.get(key, []):
            # A failed chunk is reported; facts from the other chunks are still checked.
            for case in members:
                outcomes[case.id].unchecked.append(
                    UncheckedItem(category="pass_a", item="extraction", reasons=(failure.reason,))
                )
        if key not in found:
            continue
        owners_ = tuple(c.id for c in members)
        facts = []
        for i, item in enumerate(found.get(key, []), start=1):
            fact = Fact(f"x{i}", item.fact, item.category, evidence_quote=item.evidence)
            fixer.want(members[0], fact.id, fact.text, fact.evidence_quote, owners_)
            facts.append(fact)
        staged[key] = facts
    fixer.resolve()

    out: dict[str, list[Fact]] = {}
    for key, facts in staged.items():
        members = groups[key]
        kept: list[Fact] = []
        for fact in facts:
            evidence = fixer.evidence(members[0], fact.id)
            if not isinstance(evidence, TranscriptEvidence):
                # unsupported by the transcript: dropped, as the audit would
                stages.unsupported.setdefault(key, []).append((fact.text, fact.category))
                continue
            fact.evidence = evidence
            if not any(_same(fact, other) for other in kept):
                kept.append(fact)
        kept += _rules_facts(members[0], kept, config)
        for i, fact in enumerate(kept, start=1):
            fact.id = f"f{i}"
        out[key] = kept
    return out


def _same(a: Fact, b: Fact) -> bool:
    """Duplicates across chunks: the same normalised text or overlapping evidence."""
    if _norm(a.text) == _norm(b.text):
        return True
    ea, eb = a.evidence, b.evidence
    return (
        ea is not None
        and eb is not None
        and ea.turn == eb.turn
        and ea.start < eb.end
        and eb.start < ea.end
        and _norm(a.text)[:20] == _norm(b.text)[:20]
    )


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _rules_facts(case: Case, facts: list[Fact], config: Config) -> list[Fact]:
    """The transcript's medicines and allergies from the rules layer, unless already named."""
    screen = structured_omissions(case.transcript, _PLACEHOLDER_NOTE, brands=config.brands or None)
    added = []
    for candidate in screen.candidates:
        if candidate.category not in ("medicine", "allergy") or candidate.fact is None:
            continue
        name = _item_name(candidate.fact)
        if any(_names(name, f.text) for f in facts):
            continue
        text = (
            f"Allergic to {name}" if candidate.category == "allergy" else f"Takes {candidate.fact}"
        )
        added.append(
            Fact(
                "r",
                text,
                candidate.category,
                evidence=candidate.transcript_evidence,
                evidence_quote=candidate.transcript_evidence.text
                if candidate.transcript_evidence
                else "",
                origin="rules",
            )
        )
    return added


def _item_name(fact: str) -> str:
    """The drug or allergen a rules finding names, normalised."""
    text = fact.removeprefix("allergy: ").replace("(planned)", "")
    return _norm(text.split(" / ")[0].split(" ")[0] if text else "")


# 3. Audit


def _audit(
    groups: dict[str, list[Case]],
    facts: dict[str, list[Fact]],
    stages: _Stages,
    runner: Runner,
    ledger: Ledger,
    outcomes: dict[str, Outcome],
) -> dict[str, list[Fact]]:
    requests, owners = [], {}
    for key, members in groups.items():
        if not facts.get(key):
            continue
        blocks = [Block("TRANSCRIPT:\n" + render_transcript(members[0].transcript), cache=True)]
        task = prompts.render(
            prompts.load(PROMPTS, "audit"),
            facts_block=facts_block((f.id, f.text) for f in facts[key]),
        )
        request = stages.request("audit", blocks, task, Audit)
        _assert_blind(request, members, "audit")
        requests.append(request)
        owners[request.custom_id] = key
    out: dict[str, list[Fact]] = {}
    for cid, result in runner.run(requests).items():
        key = owners[cid]
        ledger.record(result, [c.id for c in groups[key]])
        if isinstance(result, ModelFailure):
            for case in groups[key]:
                outcomes[case.id].unchecked.append(
                    UncheckedItem(category="pass_a", item="audit", reasons=(result.reason,))
                )
            continue
        graded = {entry.id.strip().lower(): entry for entry in result.parsed.facts}
        stages.audited[key] = set(graded)
        kept = []
        for fact in facts[key]:
            entry = graded.get(fact.id)
            if entry is None:  # not audited: kept as a core supporting fact
                kept.append(fact)
                continue
            if not entry.keep:
                continue
            fact.core, fact.severity = entry.core, entry.severity
            kept.append(fact)
        out[key] = kept
    return out


# 4. Check


def _check(
    cases: Sequence[Case],
    to_check: dict[str, list[Fact]],
    config: Config,
    stages: _Stages,
    runner: Runner,
    ledger: Ledger,
    outcomes: dict[str, Outcome],
) -> dict[str, list[Fact]]:
    """Present/partial/absent per fact; returns each note's absent facts."""
    per_call = config.check_facts_per_call

    def requests_for(
        subsets: dict[str, list[Fact]],
    ) -> tuple[list[ModelRequest[Any]], dict[str, tuple[str, list[Fact]]]]:
        requests, owners = [], {}
        for case in cases:
            facts = subsets.get(case.id, [])
            for start in range(0, len(facts), per_call):
                group = facts[start : start + per_call]
                task = prompts.render(
                    prompts.load(PROMPTS, "check"),
                    facts_block=facts_block((f.id, f.text) for f in group),
                )
                request = stages.request("check", _both(case), task, Checked)
                requests.append(request)
                owners[request.custom_id] = (case.id, group)
        return requests, owners

    verdicts: dict[str, dict[str, _Checked]] = {c.id: {} for c in cases}
    failures: dict[tuple[str, str], ModelFailure] = {}

    def collect(results: dict[str, Result], owners: dict[str, tuple[str, list[Fact]]]) -> None:
        for cid, result in results.items():
            case_id, group = owners[cid]
            ledger.record(result, [case_id])
            if isinstance(result, ModelFailure):
                for fact in group:
                    failures[(case_id, fact.id)] = result
                continue
            wanted = {f.id for f in group}
            for entry in result.parsed.verdicts:
                fid = entry.id.strip().lower()
                if fid in wanted:
                    verdicts[case_id][fid] = entry
                    failures.pop((case_id, fid), None)

    collect(*_run(runner, *requests_for(to_check)))
    missing = {
        c.id: [f for f in to_check.get(c.id, []) if f.id not in verdicts[c.id]] for c in cases
    }
    collect(*_run(runner, *requests_for({k: v for k, v in missing.items() if v})))

    fixer = QuoteFixer(runner, ledger, config, "note")
    by_case = {c.id: c for c in cases}
    for case in cases:
        for fact in to_check.get(case.id, []):
            entry = verdicts[case.id].get(fact.id)
            if entry is not None and entry.verdict != "absent":
                fixer.want(case, fact.id, fact.text, entry.note_quote)
    fixer.resolve()

    absent: dict[str, list[Fact]] = {}
    for case in cases:
        outcome = outcomes[case.id]
        for fact in to_check.get(case.id, []):
            entry = verdicts[case.id].get(fact.id)
            if entry is None:
                failure = failures.get((case.id, fact.id))
                outcome.unchecked.append(
                    _unchecked(fact, failure.reason if failure else "model_parse_failure")
                )
                continue
            outcome.trace[fact.id]["verdict"] = entry.verdict
            if entry.verdict == "absent":
                absent.setdefault(case.id, []).append(fact)
                continue
            note_quote = fixer.evidence(by_case[case.id], fact.id)
            if not isinstance(note_quote, NoteEvidence):
                outcome.unchecked.append(_unchecked(fact, "quote_unverified"))
                continue
            outcome.trace[fact.id]["note_quote"] = note_quote.text
            if entry.verdict == "partial":
                outcome.flags.append(_flag(fact, "partial", "possible", note_quote))
    return absent


# 5-6. Second look and rules candidates, in one round


def _second_look_and_candidates(
    cases: Sequence[Case],
    absent: dict[str, list[Fact]],
    config: Config,
    stages: _Stages,
    runner: Runner,
    ledger: Ledger,
    outcomes: dict[str, Outcome],
) -> None:
    requests, owners = [], {}
    for case in cases:
        facts = absent.get(case.id, [])
        if facts:
            task = prompts.render(
                prompts.load(PROMPTS, "second_look"),
                facts_block=facts_block((f.id, f.text) for f in facts),
            )
            request = stages.request("second_look", _both(case), task, Looked)
            requests.append(request)
            owners[request.custom_id] = ("look", case.id)
        if case.candidates:
            task = prompts.render(
                prompts.load(PROMPTS, "candidate"),
                facts_block=facts_block(
                    (f"c{i + 1}", _describe(flag)) for i, flag in enumerate(case.candidates)
                ),
            )
            request = stages.request("candidate", _both(case), task, Judged)
            requests.append(request)
            owners[request.custom_id] = ("candidate", case.id)

    looked: dict[str, dict[str, _Looked]] = {}
    judged: dict[str, dict[str, _Judged]] = {}
    failed: dict[tuple[str, str], ModelFailure] = {}
    for cid, result in runner.run(requests).items():
        kind, case_id = owners[cid]
        ledger.record(result, [case_id])
        if isinstance(result, ModelFailure):
            failed[(kind, case_id)] = result
        elif kind == "look":
            looked[case_id] = {e.id.strip().lower(): e for e in result.parsed.facts}
        else:
            judged[case_id] = {e.id.strip().lower(): e for e in result.parsed.findings}

    note_fixer = QuoteFixer(runner, ledger, config, "note")
    transcript_fixer = QuoteFixer(runner, ledger, config, "transcript")
    for case in cases:
        for fact in absent.get(case.id, []):
            entry = looked.get(case.id, {}).get(fact.id)
            if entry is not None and entry.found:
                note_fixer.want(case, f"look:{fact.id}", fact.text, entry.note_quote)
        for i, flag in enumerate(case.candidates):
            entry_c = judged.get(case.id, {}).get(f"c{i + 1}")
            if entry_c is None or not entry_c.confirmed:
                continue
            item = flag.fact or flag.category
            transcript_fixer.want(case, f"cand:{i}", item, entry_c.transcript_quote)
            if flag.type == "changed_fact":
                note_fixer.want(case, f"cand:{i}", item, entry_c.note_quote)
    note_fixer.resolve()
    transcript_fixer.resolve()

    for case in cases:
        outcome = outcomes[case.id]
        for fact in absent.get(case.id, []):
            failure = failed.get(("look", case.id))
            entry = looked.get(case.id, {}).get(fact.id)
            step = outcome.trace[fact.id]
            if failure is not None or entry is None:
                outcome.unchecked.append(
                    _unchecked(fact, failure.reason if failure else "model_parse_failure")
                )
                step["second_look"] = "failed"
            elif entry.found:
                rescued = note_fixer.evidence(case, f"look:{fact.id}")
                if not isinstance(rescued, NoteEvidence):
                    outcome.unchecked.append(_unchecked(fact, "quote_unverified"))
                    step["second_look"] = "unverified"
                else:  # rescued: the note carries it after all
                    step["second_look"] = "rescued"
                    step["note_quote"] = rescued.text
            else:
                outcome.flags.append(_flag(fact, "absent", "confirmed", None))
                step["second_look"] = "confirmed"
        confirmed_names: list[str] = []
        from_candidates: set[int] = set()
        for i, flag in enumerate(case.candidates):
            if ("candidate", case.id) in failed:
                break  # the rules warning stands, unconfirmed
            entry_c = judged.get(case.id, {}).get(f"c{i + 1}")
            if entry_c is None:
                continue
            outcome.resolved_candidates.add(i)
            if not entry_c.confirmed:
                continue
            t_found = transcript_fixer.evidence(case, f"cand:{i}")
            n_found = note_fixer.evidence(case, f"cand:{i}")
            if not isinstance(t_found, TranscriptEvidence) or (
                flag.type == "changed_fact" and not isinstance(n_found, NoteEvidence)
            ):
                outcome.unchecked.append(
                    UncheckedItem(
                        category=flag.category,
                        item=flag.fact or flag.category,
                        reasons=("quote_unverified",),
                    )
                )
                continue
            confirmed = flag.model_copy(
                update={
                    "confidence": "confirmed",
                    "severity": entry_c.severity,
                    "second_check": "model",
                    "transcript_evidence": t_found,
                    "note_evidence": n_found if flag.type == "changed_fact" else None,
                }
            )
            outcome.flags.append(confirmed)
            from_candidates.add(id(confirmed))
            if flag.type == "omission" and flag.fact:
                confirmed_names.append(_item_name(flag.fact))
        if confirmed_names:  # one discrepancy, one flag: the rules candidate wins
            outcome.flags = [
                f
                for f in outcome.flags
                if id(f) in from_candidates
                or f.category not in ("medicine", "allergy")
                or not any(_names(n, f.fact or "") for n in confirmed_names)
            ]


def _names(name: str, text: str) -> bool:
    """`text` names the drug or allergen `name` as a whole word."""
    return bool(name) and re.search(rf"\b{re.escape(name)}\b", _norm(text)) is not None


def _describe(flag: Flag) -> str:
    quote = flag.transcript_evidence.text if flag.transcript_evidence else ""
    if flag.type == "changed_fact":
        return (
            f"changed {flag.category}: the transcript gives {flag.transcript_value!r} "
            f"({quote!r}) but the note gives {flag.note_value!r}"
        )
    return f"missing {flag.category}: {flag.fact} (transcript: {quote!r})"


# Shared


def _both(case: Case) -> list[Block]:
    return [
        Block("TRANSCRIPT:\n" + render_transcript(case.transcript), cache=True),
        Block("NOTE:\n" + case.note.text, cache=True),
    ]


def _run(
    runner: Runner, requests: list[ModelRequest[Any]], owners: dict[str, Any]
) -> tuple[dict[str, Result], dict[str, Any]]:
    return runner.run(requests), owners


def _assert_blind(request: ModelRequest[Any], members: Sequence[Case], stage: str) -> None:
    text = request.system + "\n".join(b.text for m in request.messages for b in m.blocks)
    assert_blind(text, [c.note.text for c in members], f"scribecheck {stage}")


def _flag(
    fact: Fact,
    verdict: Literal["absent", "partial"],
    confidence: Literal["confirmed", "possible"],
    note: NoteEvidence | None,
) -> Flag:
    return Flag(
        id="F0",
        type="omission",
        category=fact.category,
        verdict=verdict,
        severity=fact.severity,
        confidence=confidence,
        origin=fact.origin,
        rule_id=f"{fact.category}.missing" if fact.origin == "rules" else "pass_a.missing",
        second_check="model" if verdict == "absent" else None,
        fact=fact.text,
        transcript_evidence=fact.evidence,
        note_evidence=note,
        expected_section=SECTION.get(fact.category),
    )


def _unchecked(fact: Fact, reason: Any) -> UncheckedItem:
    return UncheckedItem(
        category=fact.category,
        item=fact.text[:200],
        reasons=(reason,),
        transcript_evidence=(fact.evidence,) if fact.evidence else (),
    )


class _Stages:
    def __init__(self, config: Config, ledger: Ledger) -> None:
        self.config, self.ledger = config, ledger
        self.unsupported: dict[str, list[tuple[str, str]]] = {}
        """Per transcript: facts dropped because their evidence quote was not found."""
        self.audited: dict[str, set[str]] = {}
        """Per transcript: the fact ids the audit graded (absent when the audit failed)."""

    def request(
        self, stage: str, blocks: list[Block], task: str, output: type[BaseModel]
    ) -> ModelRequest[Any]:
        settings = self.config.stages.get(f"scribecheck.{stage}")
        return ModelRequest(
            system=prompts.load(PROMPTS, "system"),
            messages=(Message("user", (*blocks, Block(task))),),
            output=output,
            max_tokens=(
                settings.max_tokens if settings and settings.max_tokens else MAX_TOKENS[stage]
            ),
            custom_id=self.ledger.custom_id(f"sc{stage[:4]}"),
            model=settings.name if settings else None,
            effort=settings.effort if settings else None,
        )
