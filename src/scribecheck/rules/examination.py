"""Pass C's examination rule: examination findings a remote consultation cannot support.

When the transcript's modality is telephone, every examination finding in the note is
flagged as unsupported; for video, only hands-on findings (palpation, auscultation,
percussion, reflexes and the like). Without a modality, or for an in-person visit, the rule
does not run. Findings are the sentences of the note's Examination section (vitals
sections excepted) and the text after "O/E", "on examination" or "examination:" anywhere
in the note; a statement that no examination was done is not a finding.

Flags are warnings at possible confidence (rules-mode certain flags are disabled), with
the spec's rules-mode severity: critical.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from scribecheck.flags import Flag, note_evidence
from scribecheck.note import CanonicalSection, Note
from scribecheck.transcript import Transcript

RULE_ID = "examination.remote"

_PHRASE = re.compile(
    r"(?im)^[ \t]*(?:[-*•][ \t]*)?(?:O/E|on[ \t]+examination|examination|exam)(?![A-Za-z])"
    r"[ \t]*[:,-]?[ \t]*"
)
"""A trigger phrase at the start of a line (after an optional bullet), never mid-sentence."""
_SENTENCE = re.compile(r"(?:[^.;\n]|\.(?=\d))+")
"""Sentences end at ".", ";" or a line break, but not at a decimal point."""
_NOT_DONE = re.compile(
    r"(?<![A-Za-z])(?:not\s+examined|no\s+examination|unable\s+to\s+examine|"
    r"examination\s+not\s+(?:done|performed|possible)|telephone\s+consultation|"
    r"remote\s+consultation|deferred|postponed|declined|not\s+(?:done|performed|possible)|"
    r"to\s+be\s+examined|face[\s-]+to[\s-]+face)(?![A-Za-z])",
    re.IGNORECASE,
)
_EMPTY = frozenset({"none", "n/a", "na", "nil", "nad", "-"})
_PLAN = re.compile(
    r"^(?:(?:patient|pt|he|she|they|mum|dad|parent)\s+(?:was\s+|is\s+|has\s+been\s+)?)?"
    r"(?:review|refer\w*|follow|advise\w*|advice|start\w*|continue\w*|prescrib\w*|return|"
    r"safety|plan|book\w*|arrange\w*|discuss\w*|recheck|repeat|see|call|stop\w*|told|asked|"
    r"reassured|counsel+ed|given)(?![A-Za-z])",
    re.IGNORECASE,
)
"""A plan or advice sentence after a finding on the same line is not a finding."""
_VITAL = re.compile(
    r"^(?:home\s+|self[\s-]+reported\s+|reported\s+)?(?:BP|blood\s+pressure|HR|heart\s+rate|"
    r"pulse|temp\w*|T|RR|resp\w*\s+rate|sats?|SpO2|O2\s+sats?|oxygen\s+sat\w*|weight|height|"
    r"BMI|BM|glucose|peak\s+flow|PEFR)(?![A-Za-z])",
    re.IGNORECASE,
)
_REPORTED = re.compile(
    r"(?<![A-Za-z])(?:reported\s+by|patient[\s-]+reported|self[\s-]+reported|"
    r"home\s+(?:BP|readings?|monitor\w*)|per\s+patient)(?![A-Za-z])",
    re.IGNORECASE,
)
_SEP = r"[\s:,]+"
HANDS_ON = re.compile(
    rf"(?<![A-Za-z])(?:palpat\w*|tender\w*|auscultat\w*|percuss\w*|reflex\w*|"
    rf"(?:chest|lungs?){_SEP}(?:is{_SEP}|are{_SEP}|was{_SEP}|were{_SEP})?clear|"
    rf"breath{_SEP}sounds|heart{_SEP}sounds|murmurs?|crackles|crepitations?|rhonchi|wheeze|"
    rf"air{_SEP}entry|bowel{_SEP}sounds|guarding|rebound|"
    rf"abdomen{_SEP}(?:is{_SEP}|was{_SEP})?(?:soft|distended|rigid)|"
    rf"soft{_SEP}(?:and{_SEP})?non-?tender|pulses?\b|capillary{_SEP}refill|"
    rf"lymphadenopathy|lymph{_SEP}nodes|masses|mass{_SEP}felt|hepatomegaly|splenomegaly|"
    rf"organomegaly|effusion|warm{_SEP}to{_SEP}touch|ballot\w*|pitting|"
    rf"power(?:{_SEP}\d)?|tone|sensation|clonus|plantars?|Babinski|Romberg|"
    rf"(?:range{_SEP}of{_SEP}movement{_SEP})?(?:passive|resisted))(?![A-Za-z])",
    re.IGNORECASE,
)


def examination_flags(transcript: Transcript, note: Note) -> tuple[Flag, ...]:
    """Unsupported-examination warnings for a telephone or video consultation."""
    modality = transcript.modality
    if modality not in ("telephone", "video"):
        return ()
    flags = []
    for start, end in _findings(note):
        text = note.text[start:end]
        if modality == "video" and not HANDS_ON.search(text):
            continue
        flags.append(
            Flag(
                id="F0",
                type="unsupported",
                category="examination",
                verdict="unsupported",
                severity="critical",
                confidence="possible",
                origin="rules",
                rule_id=RULE_ID,
                fact=f"examination finding in a {modality} consultation",
                note_evidence=note_evidence(note, start, end),
            )
        )
    return tuple(flags)


def _findings(note: Note) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for section in note.sections:
        if (
            section.canonical is not CanonicalSection.EXAMINATION
            or "vital" in section.heading.lower()
        ):
            continue
        spans.extend(_sentences(note.text, section.body_start, section.body_end))
    for phrase in _PHRASE.finditer(note.text):
        line_end = note.text.find("\n", phrase.end())
        end = len(note.text) if line_end < 0 else line_end
        spans.extend(_sentences(note.text, phrase.end(), end))
    kept: list[tuple[int, int]] = []
    for span in sorted(set(spans)):
        if any(s < span[1] and span[0] < e for s, e in kept):
            continue  # already covered (an "O/E" line inside the Examination section)
        kept.append(span)
    return kept


def _sentences(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    for match in _SENTENCE.finditer(text, start, end):
        s, e = match.start(), match.end()
        while s < e and text[s] in " \t-*•:":
            s += 1
        while e > s and text[e - 1] in " \t":
            e -= 1
        chunk = text[s:e]
        if len(chunk) < 3 or not re.search(r"[A-Za-z]", chunk) or chunk.strip().lower() in _EMPTY:
            continue
        if _NOT_DONE.search(chunk) or _PLAN.match(chunk):
            continue
        if _VITAL.match(chunk) or _REPORTED.search(chunk):
            continue  # vitals and readings the patient reported are not examination findings
        if chunk.lower() in ("o/e", "on examination", "examination", "exam"):
            continue
        yield s, e
