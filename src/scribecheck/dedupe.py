"""One discrepancy, one flag, across the three passes (docs/spec.md, "Checking pipeline").

A flag's pass is its type: omission (Pass A), changed_fact (Pass B), unsupported (Pass C).
Drugs are compared as the drug lexicon's ingredient sets found in a flag's text and
quotes, never as free-text words.

- Pass B wins over Pass C: a model statement flag whose note span lies mostly within a
  changed fact's note span is the same discrepancy, seen from the note's side. The
  examination rule's flags are never dropped this way.
- Pass A gives way to Pass B: an omission whose transcript span overlaps a changed fact's
  transcript span, or that names the same drug, is the aligned pair's other side.
- Within Pass B, two flags of the same kind about the same drug (or, without a drug, on
  overlapping spans) are one finding; within Pass C, two flags on overlapping note spans
  are (an examination-rule warning and a model verdict on the same sentence). The higher
  confidence is kept, the earlier one on a tie. Omissions are never merged with each
  other: a missing dose and a missing frequency of one drug often quote overlapping words.
"""

from __future__ import annotations

from collections.abc import Sequence

from scribecheck.flags import Flag, NoteEvidence, TranscriptEvidence
from scribecheck.rules.lexicon import find_drugs

_RANK = {"certain": 3, "confirmed": 2, "possible": 1}


def deduplicate(flags: Sequence[Flag]) -> list[Flag]:
    """`flags` with duplicates across and within passes removed, order kept."""
    changed = [f for f in flags if f.type == "changed_fact"]
    kept: list[Flag] = []
    for flag in flags:
        if (
            flag.type == "unsupported"
            and flag.category != "examination"
            and any(_mostly_within(flag.note_evidence, b.note_evidence) for b in changed)
        ):
            continue
        if flag.type == "omission" and any(_same_item(flag, b) for b in changed):
            continue
        kept.append(flag)
    out: list[Flag] = []
    for flag in kept:
        rival = next((f for f in out if _same_finding(flag, f)), None)
        if rival is None:
            out.append(flag)
        elif _RANK[flag.confidence] > _RANK[rival.confidence]:
            out[out.index(rival)] = flag
    return out


Evidence = TranscriptEvidence | NoteEvidence | None


def _spans_overlap(a: Evidence, b: Evidence) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, TranscriptEvidence) and isinstance(b, TranscriptEvidence) and a.turn != b.turn:
        return False
    return a.start < b.end and b.start < a.end


def _mostly_within(a: NoteEvidence | None, b: NoteEvidence | None) -> bool:
    """At least half of span `a` lies inside span `b`."""
    if a is None or b is None or a.end <= a.start:
        return False
    shared = min(a.end, b.end) - max(a.start, b.start)
    return shared * 2 >= a.end - a.start


def _drugs(flag: Flag) -> set[frozenset[str]]:
    """The drugs a flag is about: named in its fact or values, else in its quotes."""
    texts: list[str | None] = [
        flag.fact,
        flag.transcript_value,
        flag.note_value,
        flag.transcript_evidence.text if flag.transcript_evidence else None,
        flag.note_evidence.text if flag.note_evidence else None,
    ]
    own = _named(texts[:3])
    return own or _named(texts[3:])  # a quoted line can name other drugs: quotes are a fallback


def _named(texts: list[str | None]) -> set[frozenset[str]]:
    return {frozenset(d.generic) for text in texts if text for d in find_drugs(text) if not d.fuzzy}


def _same_finding(a: Flag, b: Flag) -> bool:
    if a.type != b.type or a.type == "omission":
        return False
    if a.type == "unsupported":
        return _spans_overlap(a.note_evidence, b.note_evidence)
    if a.category != b.category:
        return False
    drugs_a, drugs_b = _drugs(a), _drugs(b)
    if drugs_a and drugs_b:
        return bool(drugs_a & drugs_b)
    return _spans_overlap(a.note_evidence, b.note_evidence) or _spans_overlap(
        a.transcript_evidence, b.transcript_evidence
    )


def _same_item(omission: Flag, change: Flag) -> bool:
    if _spans_overlap(omission.transcript_evidence, change.transcript_evidence):
        return True
    named = {frozenset(d.generic) for d in find_drugs(omission.fact or "") if not d.fuzzy}
    return bool(named and named & _drugs(change))
