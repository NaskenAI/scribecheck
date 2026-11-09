"""The versioned JSON report and its Markdown rendering.

The schema follows docs/spec.md ("Data model and report format"), plus an `unchecked`
list: the items the rules could not decide, each with its reason codes. A report holds
evidence quotes only, never the full transcript or note.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from scribecheck.config import Mode, Standard
from scribecheck.decision import Decision
from scribecheck.flags import Flag, NoteEvidence, TranscriptEvidence, UncheckedItem

SCHEMA_VERSION = "1.0"

_FLAG_KEYS = (
    "id",
    "type",
    "category",
    "verdict",
    "severity",
    "confidence",
    "origin",
    "rule_id",
    "second_check",
    "model_version",
    "fact",
    "transcript_value",
    "note_value",
    "transcript_evidence",
    "note_evidence",
    "evidence_verified",
    "expected_section",
)
_ALWAYS = frozenset({"transcript_evidence", "note_evidence"})
"""Flag keys written even when null, as in the spec's example."""

RULES_MODE_NOTE = (
    "Rules mode: findings are warnings at possible confidence and never fail a build. "
    "It checks omissions of medicines, doses, frequencies, allergies and sides, changed "
    "doses, frequencies, routes, sides and symptom negation, and examination findings in "
    "telephone or video consultations; every other check needs the model."
)


class ModelInfo(BaseModel):
    """The model behind a full-mode report."""

    model_config = ConfigDict(frozen=True)

    name: str
    """The model requested."""
    version: str | None = None
    """The model id the provider reported."""
    endpoint_host: str | None = None
    provider: str | None = None
    effort: str | None = None
    temperature: float | None = None
    """None where none was sent (the Anthropic API accepts none)."""
    seed: int | None = None


class UsageInfo(BaseModel):
    """Token usage for a report's model calls, and their cost when prices are configured."""

    model_config = ConfigDict(frozen=True)

    calls: int = 0
    """Calls made for this note alone."""
    shared_calls: int = 0
    """Calls shared with other notes of the same transcript (extraction, audit); their
    tokens and cost are split evenly between those notes."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float | None = None


class Counts(BaseModel):
    """Findings by kind."""

    model_config = ConfigDict(frozen=True)

    omission: int = 0
    changed_fact: int = 0
    unsupported: int = 0
    not_verifiable: int = 0
    unchecked: int = 0


class PassAInfo(BaseModel):
    """Which Pass A configuration and prompts produced a full-mode report."""

    model_config = ConfigDict(frozen=True)

    pipeline: str
    prompts: str
    batch: bool = False
    unverified_evidence: int = 0
    """Flags kept although their transcript quote could not be found (route one only)."""


class NotVerifiable(NoteEvidence):
    """A note statement the transcript could not be expected to support (Pass C)."""

    kind: str


class Report(BaseModel):
    """One consultation's report."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    consultation_id: str | None
    mode: Mode
    standard: Standard
    model: ModelInfo | None = None
    decision: Decision
    counts: Counts
    flags: tuple[Flag, ...] = ()
    unchecked: tuple[UncheckedItem, ...] = ()
    usage: UsageInfo | None = None
    """Full mode only."""
    pass_a: PassAInfo | None = None
    """Full mode only."""
    passes: dict[str, str] | None = None
    """Full mode only: what each pass ran as (a pipeline or "model", "rules", or "off")."""
    not_verifiable: tuple[NotVerifiable, ...] = ()
    """Full mode only: statements reported as not verifiable."""

    def to_dict(self) -> dict[str, Any]:
        """The report as JSON-ready data, keys in the spec's order."""
        data = {
            "schema_version": self.schema_version,
            "consultation_id": self.consultation_id,
            "mode": self.mode,
            "standard": self.standard,
            "model": None if self.model is None else self.model.model_dump(exclude_none=True),
            "decision": self.decision,
            "counts": self.counts.model_dump(),
            "flags": [_flag_dict(f) for f in self.flags],
            "unchecked": [_unchecked_dict(u) for u in self.unchecked],
        }
        if self.usage is not None:
            data["usage"] = self.usage.model_dump()
        if self.pass_a is not None:
            data["pass_a"] = self.pass_a.model_dump()
        if self.passes is not None:
            data["passes"] = dict(self.passes)
        if self.not_verifiable:
            data["not_verifiable"] = [n.model_dump() for n in self.not_verifiable]
        return data

    def to_json(self) -> str:
        return _dumps(self.to_dict())


@dataclass(frozen=True)
class Case:
    """One report in a batch, with the pair's id and its files relative to the folder."""

    id: str
    transcript: str
    note: str
    report: Report


def count(
    flags: Iterable[Flag], unchecked: Sequence[UncheckedItem], *, not_verifiable: int = 0
) -> Counts:
    """Counts for a report's flags, unchecked items and not-verifiable statements."""
    by_type = Counter(f.type for f in flags)
    return Counts(
        omission=by_type["omission"],
        changed_fact=by_type["changed_fact"],
        unsupported=by_type["unsupported"],
        not_verifiable=not_verifiable,
        unchecked=len(unchecked),
    )


def batch_json(cases: Sequence[Case], problems: Sequence[str]) -> str:
    """Several reports with their files, and the problems met pairing or reading files."""
    return _dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "cases": [
                {
                    "id": c.id,
                    "transcript": c.transcript,
                    "note": c.note,
                    "report": c.report.to_dict(),
                }
                for c in cases
            ],
            "problems": list(problems),
        }
    )


def _dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _flag_dict(flag: Flag) -> dict[str, Any]:
    data = flag.model_dump(mode="json")
    return {k: data[k] for k in _FLAG_KEYS if data[k] is not None or k in _ALWAYS}


def _unchecked_dict(item: UncheckedItem) -> dict[str, Any]:
    data = item.model_dump(mode="json")
    return {k: v for k, v in data.items() if v is not None}


# Markdown


def render_markdown(report: Report, *, level: int = 1, case: Case | None = None) -> str:
    """The report as Markdown; `level` is the heading level of its title, and `case`
    (in a batch) names the pair and its files."""
    h = "#" * level
    title = report.consultation_id or "consultation"
    lines = [f"{h} scribecheck report: {_cell(title)}", ""]
    if case is not None:
        lines += [
            f"Case {_cell(case.id)}: `{_cell(case.transcript)}` and `{_cell(case.note)}`",
            "",
        ]
    lines += [
        f"**Decision: {report.decision}** (mode {report.mode}, standard {report.standard})",
        "",
    ]
    if report.mode == "rules":
        lines += [f"> {RULES_MODE_NOTE}", ""]
        if report.passes is not None:
            ran = ", ".join(f"Pass {k.upper()} {v}" for k, v in sorted(report.passes.items()))
            lines += [f"> Passes: {ran}. A pass that is off reports nothing.", ""]
    else:
        lines += [f"> {_full_mode_note(report)}", ""]
    c = report.counts
    lines += [
        "| Omissions | Changed facts | Unsupported | Not verifiable | Unchecked |",
        "| --- | --- | --- | --- | --- |",
        f"| {c.omission} | {c.changed_fact} | {c.unsupported} | {c.not_verifiable} "
        f"| {c.unchecked} |",
        "",
        f"{h}# Flags",
        "",
    ]
    if report.flags:
        lines += [
            "| ID | Type | Category | Severity | Confidence | Finding | Transcript evidence "
            "| Note evidence | Expected section |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        lines += [_flag_row(f) for f in report.flags]
    else:
        lines.append("No flags.")
    lines += ["", f"{h}# Unchecked items", ""]
    if report.unchecked:
        reasons = Counter(u.reason for u in report.unchecked)
        lines += ["| Primary reason | Items |", "| --- | --- |"]
        lines += [f"| {reason} | {n} |" for reason, n in sorted(reasons.items())]
        lines += [
            "",
            "| Category | Item | Reasons | Transcript evidence | Note evidence |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [_unchecked_row(u) for u in report.unchecked]
    else:
        lines.append("No unchecked items.")
    return "\n".join(lines) + "\n"


def _full_mode_note(report: Report) -> str:
    parts = ["Full mode: rules plus the model"]
    if report.passes is not None:
        ran = ", ".join(f"Pass {k.upper()} {v}" for k, v in sorted(report.passes.items()))
        parts[0] += f" ({ran})"
    if report.pass_a is not None:
        parts.append(f"prompts {report.pass_a.prompts}")
    if report.model is not None:
        parts.append(f"model {report.model.version or report.model.name}")
    if report.usage is not None:
        u = report.usage
        cost = "cost unknown" if u.cost_usd is None else f"about ${u.cost_usd:.4f}"
        parts.append(f"{u.calls} calls ({u.shared_calls} shared), {cost}")
    if report.pass_a is not None and report.pass_a.unverified_evidence:
        parts.append(f"{report.pass_a.unverified_evidence} flags without a verified quote")
    if report.counts.not_verifiable:
        parts.append(f"{report.counts.not_verifiable} statements not verifiable")
    return "; ".join(parts) + "."


def render_batch_markdown(
    cases: Sequence[Case], problems: Sequence[str], *, preamble: str | None = None
) -> str:
    """A summary table, any problems, then each report."""
    lines = ["# scribecheck batch report", ""]
    if preamble:
        lines += [preamble, ""]
    if cases:
        lines += [
            "| Case | Consultation | Decision | Omissions | Changed facts | Unsupported "
            "| Unchecked |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for case in cases:
            r, c = case.report, case.report.counts
            lines.append(
                f"| {_cell(case.id)} | {_cell(r.consultation_id or '-')} | {r.decision} "
                f"| {c.omission} | {c.changed_fact} | {c.unsupported} | {c.unchecked} |"
            )
    else:
        lines.append("No consultations were checked.")
    if problems:
        lines += ["", "## Problems", ""]
        lines += [f"- {_cell(p)}" for p in problems]
    text = "\n".join(lines) + "\n"
    for case in cases:
        text += "\n" + render_markdown(case.report, level=2, case=case)
    return text


def _flag_row(flag: Flag) -> str:
    if flag.fact is not None:
        finding = flag.fact
    elif flag.transcript_value is not None or flag.note_value is not None:
        finding = f"{flag.transcript_value or '-'} -> {flag.note_value or '-'}"
    else:
        finding = "-"
    cells = [
        flag.id,
        flag.type,
        flag.category,
        flag.severity,
        flag.confidence,
        finding,
        _t_quote(flag.transcript_evidence),
        _n_quote(flag.note_evidence),
        flag.expected_section or "-",
    ]
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _unchecked_row(item: UncheckedItem) -> str:
    cells = [
        item.category,
        item.item,
        ", ".join(item.reasons),
        "; ".join(_t_quote(q) for q in item.transcript_evidence) or "-",
        "; ".join(_n_quote(q) for q in item.note_evidence) or "-",
    ]
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _t_quote(quote: TranscriptEvidence | None) -> str:
    return "-" if quote is None else f'"{quote.text}" (turn {quote.turn})'


def _n_quote(quote: NoteEvidence | None) -> str:
    if quote is None:
        return "-"
    return f'"{quote.text}"' + (f" ({quote.section})" if quote.section else "")


def _cell(text: str) -> str:
    """Text safe inside a Markdown table cell: one line, pipes escaped."""
    return " ".join(text.split()).replace("\\", "\\\\").replace("|", "\\|")
