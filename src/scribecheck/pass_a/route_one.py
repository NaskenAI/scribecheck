"""Route one: Fox et al.'s per-fact pipeline, reproduced from the OmissionBench harness.

Stages, prompts and parsing follow `judges/w2_pipeline.py` (tier B3) of the harness
(Composo, MIT; prompts copied unmodified into prompts/route_one/, see its README):

1. extract: facts from the transcript only (`pipeline_extract`), once per transcript;
2. audit: drop unsupported, duplicate or not-note-bearing facts and grade severity with
   the study's rubric (`pipeline_audit`), once per transcript;
3. check: full / partial / absent per fact, keyed by id, with the whole note
   (`pipeline_check_tri`); ids that come back missing get ONE targeted retry, and a note
   with ids still unanswered is a parse failure;
4. refute: facts judged absent go to a skeptical reviewer (`pipeline_refute`), whose rescue
   counts only if its quote is in the note (`quote_in_note`).

As published, replies are not schema-constrained (prompt-only JSON, parsed leniently) and
"present" verdicts carry no quote. Route one does not use the rules layer.

Quote checks never change route one's verdicts. Each flag looks for its transcript quote
(the extraction's evidence, re-asked once if not found); a flag whose quote still cannot be
found stays a flag, at possible confidence, with `evidence_verified` false. Scoring of route
one uses its verdicts unchanged.

Confidence: an absence the refuter saw and did not rescue, with a verified transcript quote,
is "confirmed"; a partial verdict, an absence whose refute call failed, or a flag without a
verified quote is "possible".
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Any

from pydantic import BaseModel, PrivateAttr, RootModel, model_validator

from scribecheck import prompts
from scribecheck.config import Config
from scribecheck.flags import Flag, Severity, TranscriptEvidence, UncheckedItem
from scribecheck.model.base import Block, Completion, Message, ModelFailure, ModelRequest
from scribecheck.pass_a.base import (
    Case,
    Fact,
    Ledger,
    Outcome,
    Runner,
    plain_transcript,
)
from scribecheck.pass_a.blind import assert_blind
from scribecheck.pass_a.quotes_stage import QuoteFixer

PROMPTS = "route_one"
SEVERITIES: tuple[Severity, ...] = ("critical", "supporting", "peripheral")
EXTRACT_MAX_TOKENS = 12000
AUDIT_MAX_TOKENS = 24000
REFUTE_MAX_TOKENS = 16000
DEFAULT_EFFORT = {"extract": "medium", "audit": "high", "check": "medium", "refute": "high"}
"""The harness's pinned effort per stage, used unless the config sets one."""
HARNESS_TEMPERATURE = 1.0
"""The harness's sampling temperature (`W.judge(..., temperature=1.0)`); sent to
OpenAI-compatible endpoints only, since the Anthropic API accepts none."""


def facts_block(facts: Iterable[tuple[str, str]]) -> str:
    """The numbered fact list, formatted exactly as the harness's `facts_block`."""
    return "\n".join(f"{fid}: {str(text).strip()}" for fid, text in facts)


# Parsers, ported from the harness (judges/w2_pipeline.py), operating on parsed JSON.


def parse_facts(j: Any) -> list[dict[str, str]] | None:
    """{"facts": [{"id","fact","evidence"}]} -> a normalised fact list, or None."""
    if not isinstance(j, dict):
        return None
    raw = j.get("facts")
    if not isinstance(raw, list) or not raw:
        return None
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            item = {"fact": item}
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact") or "").strip()
        if not fact:
            continue
        fid = str(item.get("id") or "").strip().lower() or f"f{len(out) + 1}"
        if fid in seen:
            fid = f"{fid}_{len(out) + 1}"
        seen.add(fid)
        out.append(
            {"id": fid, "fact": fact, "evidence": str(item.get("evidence") or "").strip()[:400]}
        )
    return out or None


def parse_audit(j: Any, candidate_ids: Sequence[str]) -> list[dict[str, str]] | None:
    """{"facts": [{"id","fact","severity"}], "dropped": [...]} -> kept facts, or None.

    Ids not in the candidate list, or repeated, are dropped; an unknown severity defaults
    to "supporting", as in the harness.
    """
    if not isinstance(j, dict):
        return None
    raw = j.get("facts")
    if not isinstance(raw, list):
        return None
    want = {i.lower() for i in candidate_ids}
    kept: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("id") or "").strip().lower()
        fact = str(item.get("fact") or "").strip()
        if fid not in want or fid in seen or not fact:
            continue
        seen.add(fid)
        severity = str(item.get("severity") or "").strip().lower()
        if severity not in SEVERITIES:
            severity = "supporting"
        kept.append({"id": fid, "fact": fact, "severity": severity})
    return kept or None


_SYNONYMS = {
    "true": "present",
    "yes": "present",
    "covered": "present",
    "captured": "present",
    "false": "absent",
    "no": "absent",
    "missing": "absent",
    "not covered": "absent",
    "complete": "full",
    "fully": "full",
    "partially": "partial",
    "partial_": "partial",
}


def parse_keyed_verdicts(
    j: Any, want_ids: Sequence[str], allowed: Sequence[str]
) -> dict[str, str] | None:
    """{"verdicts": {"f1": "full", ...}} -> {id: label} for the ids asked about, or None."""
    if not isinstance(j, dict):
        return None
    v = j.get("verdicts")
    if not isinstance(v, dict):
        v = j.get("coverage") if isinstance(j.get("coverage"), dict) else None
    if not isinstance(v, dict):
        v = j
    want = {str(i).lower() for i in want_ids}
    out: dict[str, str] = {}
    assert isinstance(v, dict)
    for key_raw, val in v.items():
        key = str(key_raw).strip().lower()
        if key not in want:
            continue
        if isinstance(val, bool):
            label = "present" if val else "absent"
        elif isinstance(val, dict):
            label = str(val.get("verdict") or val.get("label") or "").strip().lower()
        else:
            label = str(val).strip().lower()
        label = _SYNONYMS.get(label, label)
        if label == "present" and "present" not in allowed:
            label = "full"
        if label == "full" and "full" not in allowed:
            label = "present"
        if label == "partial" and "partial" not in allowed:
            label = "present"
        if label in allowed:
            out[key] = label
    return out or None


def parse_refute(j: Any, want_ids: Sequence[str]) -> dict[str, tuple[str, str]] | None:
    """{"verdicts": {"f3": {"found": "full", "quote": "..."}}} -> {id: (found, quote)}."""
    if not isinstance(j, dict):
        return None
    v: dict[Any, Any] = j["verdicts"] if isinstance(j.get("verdicts"), dict) else j
    want = {str(i).lower() for i in want_ids}
    out: dict[str, tuple[str, str]] = {}
    for key_raw, val in v.items():
        key = str(key_raw).strip().lower()
        if key not in want:
            continue
        if isinstance(val, str):
            found, quote = val.strip().lower(), ""
        elif isinstance(val, dict):
            found = str(val.get("found") or val.get("verdict") or "").strip().lower()
            quote = str(val.get("quote") or "").strip()
        else:
            continue
        if found in ("yes", "true", "present"):
            found = "full"
        if found in ("false", "absent", "none", "not found"):
            found = "no"
        if found in ("full", "partial", "no"):
            out[key] = (found, quote)
    return out or None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def quote_in_note(quote: str, note_text: str, min_chars: int = 8) -> bool:
    """A refute rescue counts only if its quote really is in the note (normalised)."""
    q, n = _norm(quote), _norm(note_text)
    if len(q) < min_chars:
        return False
    if q in n:
        return True
    head = q[:60]
    return len(head) >= min_chars and head in n


def check_budget(n_facts: int, effort: str | None, per_fact: int = 16) -> int:
    body = 512 + per_fact * max(1, n_facts)
    if effort in (None, "none"):
        return int(min(4096, max(1024, body)))
    return int(min(9000, max(3000, 2500 + body)))


def refute_budget(n_flagged: int, effort: str | None) -> int:
    body = 1024 + 90 * max(1, n_flagged)
    if effort in (None, "none"):
        return int(min(4096, max(1024, body)))
    return int(min(REFUTE_MAX_TOKENS, max(6000, 4000 + body)))


# Output models: the parsed JSON, checked by the harness's parser so that an unusable reply
# is retried once (the harness's pre-registered single retry).


def _output(parser: Callable[[Any], Any]) -> type[BaseModel]:
    class Parsed(RootModel[Any]):
        _value: Any = PrivateAttr(default=None)

        @model_validator(mode="after")
        def _parse(self) -> Parsed:
            value = parser(self.root)
            if value is None:
                raise ValueError("the reply has no usable content in the expected shape")
            self._value = value
            return self

    return Parsed


def _value(result: Completion[Any]) -> Any:
    return result.parsed._value


# The pipeline


def run(
    cases: Sequence[Case],
    config: Config,
    runner: Runner,
    ledger: Ledger,
    outcomes: dict[str, Outcome],
) -> None:
    """Route one over `cases`, writing each note's findings into `outcomes`."""
    stage = _Stages(config)
    groups: dict[str, list[Case]] = {}
    for case in cases:
        key = hashlib.sha256(plain_transcript(case.transcript).encode()).hexdigest()
        groups.setdefault(key, []).append(case)

    # 1. Extract, once per transcript.
    requests, owners = [], {}
    for key, members in groups.items():
        text = plain_transcript(members[0].transcript)
        prompt = prompts.render(prompts.load(PROMPTS, "pipeline_extract"), transcript=text)
        assert_blind(prompt, [c.note.text for c in members], "route_one extract")
        request = stage.request("extract", prompt, _output(parse_facts), ledger, EXTRACT_MAX_TOKENS)
        requests.append(request)
        owners[request.custom_id] = (key, members)
    facts: dict[str, list[dict[str, str]]] = {}
    for cid, result in runner.run(requests).items():
        key, members = owners[cid]
        ledger.record(result, [c.id for c in members])
        if isinstance(result, ModelFailure):
            for c in members:
                outcomes[c.id].unchecked.append(_stage_failure("extraction", result))
            continue
        facts[key] = _value(result)

    # 2. Audit, once per transcript.
    requests, owners = [], {}
    for key, members in groups.items():
        if key not in facts:
            continue
        text = plain_transcript(members[0].transcript)
        block = facts_block((f["id"], f["fact"]) for f in facts[key])
        prompt = prompts.render(
            prompts.load(PROMPTS, "pipeline_audit"), transcript=text, facts_block=block
        )
        assert_blind(prompt, [c.note.text for c in members], "route_one audit")
        ids = [f["id"] for f in facts[key]]
        request = stage.request(
            "audit",
            prompt,
            _output(partial(parse_audit, candidate_ids=ids)),
            ledger,
            AUDIT_MAX_TOKENS,
        )
        requests.append(request)
        owners[request.custom_id] = (key, members)
    audited: dict[str, list[Fact]] = {}
    for cid, result in runner.run(requests).items():
        key, members = owners[cid]
        ledger.record(result, [c.id for c in members])
        if isinstance(result, ModelFailure):
            for c in members:
                outcomes[c.id].unchecked.append(_stage_failure("audit", result))
            continue
        quoted = {f["id"]: f["evidence"] for f in facts[key]}
        audited[key] = [
            Fact(f["id"], f["fact"], "fact", f["severity"], evidence_quote=quoted.get(f["id"], ""))
            for f in _value(result)
        ]

    # 3. Check, per note, with one targeted retry for missing ids.
    per_case: dict[str, list[Fact]] = {}
    for key, members in groups.items():
        for c in members:
            if key in audited:
                per_case[c.id] = audited[key]
                outcomes[c.id].facts = list(audited[key])
    verdicts = _check(cases, per_case, stage, runner, ledger, outcomes)

    # 4. Refute the absences.
    refuted, refute_failed = _refute(cases, per_case, verdicts, stage, runner, ledger)

    # 5. Flags, each with a transcript quote.
    fixer = QuoteFixer(runner, ledger, config)
    pending: list[tuple[Case, Fact, str]] = []
    for case in cases:
        for fact in per_case.get(case.id, []):
            label = verdicts.get(case.id, {}).get(fact.id)
            if label in ("partial", "absent"):
                pending.append((case, fact, label))
                fixer.want(case, fact.id, fact.text, fact.evidence_quote)
    fixer.resolve()
    for case, fact, label in pending:
        found = fixer.evidence(case, fact.id)
        evidence = found if isinstance(found, TranscriptEvidence) else None
        confirmed = (
            evidence is not None
            and label == "absent"
            and fact.id in refuted.get(case.id, set())
            and case.id not in refute_failed
        )
        outcomes[case.id].flags.append(
            Flag(
                id="F0",
                type="omission",
                category="fact",
                verdict="absent" if label == "absent" else "partial",
                severity=fact.severity,
                confidence="confirmed" if confirmed else "possible",
                origin="model",
                rule_id="route_one.missing",
                second_check=(
                    "refute" if label == "absent" and not refute_failed.get(case.id) else None
                ),
                fact=fact.text,
                transcript_evidence=evidence,
                evidence_verified=None if evidence is not None else False,
            )
        )


def _check(
    cases: Sequence[Case],
    per_case: dict[str, list[Fact]],
    stage: _Stages,
    runner: Runner,
    ledger: Ledger,
    outcomes: dict[str, Outcome],
) -> dict[str, dict[str, str]]:
    allowed = ("full", "partial", "absent")
    effort = stage.effort("check")

    def requests_for(
        subsets: dict[str, list[Fact]],
    ) -> tuple[list[ModelRequest[Any]], dict[str, str]]:
        requests, owners = [], {}
        for case in cases:
            subset = subsets.get(case.id)
            if not subset:
                continue
            ids = [f.id for f in subset]
            prompt = prompts.render(
                prompts.load(PROMPTS, "pipeline_check_tri"),
                note=case.note.text,
                facts_block=facts_block((f.id, f.text) for f in subset),
            )
            request = stage.request(
                "check",
                prompt,
                _output(partial(parse_keyed_verdicts, want_ids=ids, allowed=allowed)),
                ledger,
                check_budget(len(ids), effort),
            )
            requests.append(request)
            owners[request.custom_id] = case.id
        return requests, owners

    verdicts: dict[str, dict[str, str]] = {c.id: {} for c in cases}
    failed: dict[str, ModelFailure] = {}
    requests, owners = requests_for(per_case)
    for cid, result in runner.run(requests).items():
        owner = owners[cid]
        ledger.record(result, [owner])
        if isinstance(result, ModelFailure) and result.reason != "model_parse_failure":
            failed[owner] = result
        elif isinstance(result, Completion):
            verdicts[owner].update(_value(result))
        # An unparseable check counts as no verdicts, so every id gets the targeted retry
        # (the harness: `verdicts = check(ids, 0) or {}`).
    missing = {
        c.id: [f for f in per_case.get(c.id, []) if f.id not in verdicts[c.id]]
        for c in cases
        if c.id not in failed
    }
    requests, owners = requests_for({k: v for k, v in missing.items() if v})
    for cid, result in runner.run(requests).items():
        owner = owners[cid]
        ledger.record(result, [owner])
        if isinstance(result, Completion):
            want = {f.id for f in missing[owner]}
            verdicts[owner].update({k: v for k, v in _value(result).items() if k in want})
    for case in cases:
        facts = per_case.get(case.id, [])
        unanswered = [f for f in facts if f.id not in verdicts[case.id]]
        if case.id in failed or unanswered:
            # The harness's contract: every id answered, or the note is a parse failure.
            failure = failed.get(case.id)
            reason = failure.reason if failure else "model_parse_failure"
            outcomes[case.id].unchecked.append(
                UncheckedItem(category="pass_a", item="presence check", reasons=(reason,))
            )
            verdicts[case.id] = {}
    return verdicts


def _refute(
    cases: Sequence[Case],
    per_case: dict[str, list[Fact]],
    verdicts: dict[str, dict[str, str]],
    stage: _Stages,
    runner: Runner,
    ledger: Ledger,
) -> tuple[dict[str, set[str]], dict[str, bool]]:
    """Run the refuter; rescued facts become "full". Returns, per note, the absences the
    refuter saw and did not rescue, and the notes whose refute call failed."""
    effort = stage.effort("refute")
    requests, owners, flagged = [], {}, {}
    for case in cases:
        absent = [f for f in per_case.get(case.id, []) if verdicts[case.id].get(f.id) == "absent"]
        if not absent:
            continue
        ids = [f.id for f in absent]
        flagged[case.id] = ids
        prompt = prompts.render(
            prompts.load(PROMPTS, "pipeline_refute"),
            note=case.note.text,
            facts_block=facts_block((f.id, f.text) for f in absent),
        )
        request = stage.request(
            "refute",
            prompt,
            _output(partial(parse_refute, want_ids=ids)),
            ledger,
            refute_budget(len(ids), effort),
        )
        requests.append(request)
        owners[request.custom_id] = case.id
    upheld: dict[str, set[str]] = {}
    failed: dict[str, bool] = {}
    for cid, result in runner.run(requests).items():
        owner = owners[cid]
        ledger.record(result, [owner])
        case = next(c for c in cases if c.id == owner)
        if isinstance(result, ModelFailure):
            # As in the harness: a dead refuter degrades to the pre-refute verdicts.
            failed[owner] = True
            continue
        got: dict[str, tuple[str, str]] = _value(result)
        stays = set(flagged[owner])
        for fid, (found, quote) in got.items():
            if found != "no" and quote_in_note(quote, case.note.text):
                verdicts[owner][fid] = found
                stays.discard(fid)
        upheld[owner] = stays
    return upheld, failed


class _Stages:
    """Per-stage model, effort and token budget from config."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def effort(self, stage: str) -> str | None:
        settings = self.config.stages.get(f"route_one.{stage}")
        if settings is not None and settings.effort is not None:
            return settings.effort
        return DEFAULT_EFFORT[stage]

    def request(
        self, stage: str, prompt: str, output: type[BaseModel], ledger: Ledger, max_tokens: int
    ) -> ModelRequest[Any]:
        settings = self.config.stages.get(f"route_one.{stage}")
        limit = settings.max_tokens if settings and settings.max_tokens else max_tokens
        return ModelRequest(
            system="",
            messages=(Message("user", (Block(prompt),)),),
            output=output,
            max_tokens=limit,
            custom_id=ledger.custom_id(f"r1{stage}"),
            json_mode="prompt",
            model=settings.name if settings else None,
            effort=self.effort(stage),
            temperature=HARNESS_TEMPERATURE,
        )


def _stage_failure(stage: str, failure: ModelFailure) -> UncheckedItem:
    return UncheckedItem(category="pass_a", item=stage, reasons=(failure.reason,))
