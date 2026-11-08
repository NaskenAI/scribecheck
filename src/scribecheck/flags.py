"""Flags, evidence and unchecked items, in the shape of the spec's report example.

Evidence is built only through `transcript_evidence` and `note_evidence`, which slice the
parsed source, so every quote is exactly the text at its offsets. Transcript offsets index
`Turn.text` of a turn numbered from 1; note offsets index `Note.text`. `verify_evidence`
re-checks a flag or unchecked item against its sources.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scribecheck.note import Note
from scribecheck.transcript import Transcript

FlagType = Literal["omission", "changed_fact", "unsupported"]
Verdict = Literal["absent", "partial", "changed", "contradicted", "unsupported"]
Severity = Literal["critical", "supporting", "peripheral"]
Confidence = Literal["certain", "confirmed", "possible"]
Origin = Literal["rules", "model"]
UncheckedReason = Literal[
    "certain_disabled_pending_allowlist",
    "unaligned",
    "planned_or_past",
    "unattributed_value",
    "ambiguous_value",
    "range_or_maximum",
    "tablet_count",
    "plural_site",
    "status_change",
    "negation_scope",
    "transcript_inconsistent",
    "note_inconsistent",
    "not_comparable",
    "mixed_temporality",
    "possible",
    "eye_abbreviation",
    "spelling_tolerance",
    "unreviewed_data",
    "unconfirmed_presence",
    "clinician_only",
    "no_speaker_labels",
    "question_unanswered",
    "model_parse_failure",
    "cost_cap_reached",
    "model_refusal",
    "model_error",
]
"""Why an item got no certain flag; see rules/changed.py for each."""


class TranscriptEvidence(BaseModel):
    """A quote from one transcript turn."""

    model_config = ConfigDict(frozen=True)

    text: str
    turn: int
    """The turn number, from 1."""
    start: int
    end: int


class NoteEvidence(BaseModel):
    """A quote from the note."""

    model_config = ConfigDict(frozen=True)

    text: str
    section: str | None
    """The heading of the section holding the quote, as written; None in the preamble."""
    start: int
    end: int


class Flag(BaseModel):
    """One reported discrepancy."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: FlagType
    category: str
    verdict: Verdict
    severity: Severity
    confidence: Confidence
    origin: Origin
    rule_id: str | None = None
    transcript_value: str | None = None
    note_value: str | None = None
    transcript_evidence: TranscriptEvidence | None = None
    note_evidence: NoteEvidence | None = None
    fact: str | None = None
    expected_section: str | None = None
    second_check: str | None = None
    model_version: str | None = None


class UncheckedItem(BaseModel):
    """An item the rules did not report as a certain flag, with the reasons and its mentions.

    `reasons[0]` is the primary reason. While certain flags are disabled (see
    rules/changed.py), every item also carries "certain_disabled_pending_allowlist", and an
    item that would have been a certain flag has it as its only reason, with the values it
    would have reported.
    """

    model_config = ConfigDict(frozen=True)

    category: str
    item: str
    reasons: tuple[UncheckedReason, ...] = Field(min_length=1)
    transcript_value: str | None = None
    note_value: str | None = None
    transcript_evidence: tuple[TranscriptEvidence, ...] = ()
    note_evidence: tuple[NoteEvidence, ...] = ()

    @property
    def reason(self) -> UncheckedReason:
        """The primary reason."""
        return self.reasons[0]


class ChangedFactResult(BaseModel):
    """Pass B's rules output: certain flags, and the items it left unchecked."""

    model_config = ConfigDict(frozen=True)

    flags: tuple[Flag, ...] = ()
    unchecked: tuple[UncheckedItem, ...] = ()


def transcript_evidence(
    transcript: Transcript, turn: int, start: int, end: int
) -> TranscriptEvidence:
    """Evidence for `start:end` within turn `turn` (from 1). Raises ValueError if outside it."""
    if not 1 <= turn <= len(transcript.turns):
        raise ValueError(f"turn {turn} out of range 1..{len(transcript.turns)}")
    text = transcript.turns[turn - 1].text
    if not 0 <= start < end <= len(text):
        raise ValueError(f"span ({start}, {end}) is not inside turn {turn}")
    return TranscriptEvidence(text=text[start:end], turn=turn, start=start, end=end)


def note_evidence(note: Note, start: int, end: int) -> NoteEvidence:
    """Evidence for `note.text[start:end]`, naming its section. Raises ValueError if outside."""
    if not 0 <= start < end <= len(note.text):
        raise ValueError(f"span ({start}, {end}) is not inside the note")
    section = next((s.heading for s in note.sections if s.start <= start < s.end), None)
    return NoteEvidence(text=note.text[start:end], section=section, start=start, end=end)


def verify_evidence(found: Flag | UncheckedItem, transcript: Transcript, note: Note) -> None:
    """Raise ValueError unless every quote in `found` slices back exactly from its source."""
    quotes_t: tuple[TranscriptEvidence, ...]
    quotes_n: tuple[NoteEvidence, ...]
    if isinstance(found, Flag):
        quotes_t = () if found.transcript_evidence is None else (found.transcript_evidence,)
        quotes_n = () if found.note_evidence is None else (found.note_evidence,)
    else:
        quotes_t, quotes_n = found.transcript_evidence, found.note_evidence
    for quote in quotes_t:
        if not 1 <= quote.turn <= len(transcript.turns) or (
            transcript.quote(quote.turn, quote.start, quote.end) != quote.text
        ):
            raise ValueError(f"transcript evidence {quote.text!r} does not match turn {quote.turn}")
    for quote_n in quotes_n:
        if note.text[quote_n.start : quote_n.end] != quote_n.text:
            raise ValueError(f"note evidence {quote_n.text!r} does not match the note")
