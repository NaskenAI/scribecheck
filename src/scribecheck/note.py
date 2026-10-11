"""Note parsing: section detection for plain-text and Markdown notes.

Headings are Markdown headings (any level) or lines such as "Allergies:", including
ones with content on the same line ("Allergies: NKDA"). A line that is only a heading
name, either one not in the heading map ("Ideas and concerns:") or a known one without a
colon ("Plan"), must start the note or follow a blank line, so list lead-ins and items
inside a section are not split off. An unknown heading needs a colon, except in the
ACI style: a line of one to six capitalised words ("ENCOUNTER NOTE") that starts the
note or follows a blank line and is followed by text. All-capitals body phrases such as
"NKDA" or "NO KNOWN DRUG ALLERGIES" never count. Each heading maps to a
canonical section through a heading map. Every section keeps character offsets into
`Note.text`, so slicing with them returns the stored text exactly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from scribecheck._text import normalise_text, read_text_file, strip_span
from scribecheck.errors import InputError


class CanonicalSection(str, Enum):
    """The note sections that fact categories expect (see the spec's fact categories)."""

    HISTORY = "History"
    MEDICATIONS = "Medications"
    ALLERGIES = "Allergies"
    EXAMINATION = "Examination"
    RESULTS = "Results"
    ASSESSMENT = "Assessment"
    PLAN = "Plan"


def _defaults() -> dict[str, CanonicalSection]:
    names = {
        CanonicalSection.HISTORY: [
            "history",
            "hpc",
            "hpi",
            "history of presenting complaint",
            "presenting complaint",
            "subjective",
            "pmh",
            "past medical history",
            "social history",
            "family history",
            "chief complaint",
            "cc",
            "history of present illness",
            "review of systems",
            "ros",
            "medical history",
            "surgical history",
            "past surgical history",
        ],
        CanonicalSection.MEDICATIONS: [
            "medications",
            "medication",
            "meds",
            "current medications",
            "drug history",
            "dh",
        ],
        CanonicalSection.ALLERGIES: ["allergies", "allergy", "drug allergies"],
        CanonicalSection.EXAMINATION: [
            "examination",
            "exam",
            "o/e",
            "on examination",
            "objective",
            "physical examination",
            "physical exam",
            "pe",
            "vitals",
            "vital signs",
            "vitals reviewed",
        ],
        CanonicalSection.RESULTS: ["results", "investigations", "test results", "bloods", "labs"],
        CanonicalSection.ASSESSMENT: [
            "assessment",
            "impression",
            "diagnosis",
            "differential diagnosis",
        ],
        CanonicalSection.PLAN: [
            "plan",
            "management",
            "management plan",
            "follow-up",
            "follow up",
            "safety netting",
            # Combined sections hold mostly plan items (tests, referrals, follow-up).
            "assessment and plan",
            "assessment & plan",
            "a&p",
            "a/p",
            "instructions",
            "patient instructions",
        ],
    }
    return {name: section for section, headings in names.items() for name in headings}


DEFAULT_HEADING_MAP: Mapping[str, CanonicalSection] = _defaults()

# A heading that is not in the heading map must look like one: short, no sentence punctuation.
_MAX_UNKNOWN_HEADING_WORDS = 6
_SENTENCE_PUNCTUATION = frozenset(".?!")


class Section(BaseModel):
    """One note section, from its heading to the next heading."""

    model_config = ConfigDict(frozen=True)

    heading: str
    """The heading as written, without Markdown markers or the trailing colon."""
    canonical: CanonicalSection | None
    """The mapped canonical section, or None for a heading not in the heading map."""
    start: int
    end: int
    text: str
    """The whole section, heading included, with trailing whitespace trimmed."""
    heading_start: int
    heading_end: int
    body_start: int
    body_end: int
    body: str
    """The content after the heading, trimmed; may start on the heading's line."""


class Note(BaseModel):
    """A parsed clinical note."""

    model_config = ConfigDict(frozen=True)

    text: str
    """The normalised note text that every offset refers to."""
    preamble: str | None = None
    """Text before the first heading, trimmed; it belongs to no section."""
    preamble_start: int | None = None
    preamble_end: int | None = None
    sections: tuple[Section, ...] = ()

    def sections_for(self, canonical: CanonicalSection) -> tuple[Section, ...]:
        """All sections mapped to `canonical`, in note order."""
        return tuple(s for s in self.sections if s.canonical is canonical)


def normalise_heading(heading: str) -> str:
    """Heading-map key: case-folded, with runs of whitespace collapsed."""
    return " ".join(heading.split()).casefold()


def build_heading_map(
    user: Mapping[str, CanonicalSection | str] | None = None,
    *,
    include_defaults: bool = True,
) -> dict[str, CanonicalSection]:
    """Combine the user's heading map with the defaults; the user's entries win.

    Values may be `CanonicalSection` members or section names in any case ("plan").
    """
    by_name = {s.value.casefold(): s for s in CanonicalSection}
    heading_map = dict(DEFAULT_HEADING_MAP) if include_defaults else {}
    for heading, section in (user or {}).items():
        canonical = (
            section
            if isinstance(section, CanonicalSection)
            else by_name.get(str(section).strip().casefold())
        )
        if canonical is None:
            allowed = ", ".join(s.value for s in CanonicalSection)
            raise InputError(
                f"heading {heading!r} maps to unknown section {section!r}; expected one of: "
                f"{allowed}"
            )
        heading_map[normalise_heading(heading)] = canonical
    return heading_map


def parse_note(
    text: str,
    *,
    heading_map: Mapping[str, CanonicalSection | str] | None = None,
) -> Note:
    """Parse a plain-text or Markdown note into sections.

    `heading_map` replaces `DEFAULT_HEADING_MAP`; use `build_heading_map` to extend
    the defaults instead. Raises `InputError` for an empty note.
    """
    text = normalise_text(text)
    if not text.strip():
        raise InputError("note is empty")
    known = build_heading_map(heading_map, include_defaults=heading_map is None)

    headings: list[_Heading] = []
    position = 0
    after_blank = True
    lines = text.split("\n")
    for index, line in enumerate(lines):
        followed_by_text = index + 1 < len(lines) and bool(lines[index + 1].strip())
        heading = _match_heading(
            line, known, after_blank=after_blank, followed_by_text=followed_by_text
        )
        if heading is not None:
            headings.append(heading.shifted(position))
        position += len(line) + 1
        after_blank = not line.strip()

    preamble_end = headings[0].line_start if headings else len(text)
    p_start, p_end = strip_span(text, 0, preamble_end)
    has_preamble = p_start < p_end

    sections = []
    for i, heading in enumerate(headings):
        limit = headings[i + 1].line_start if i + 1 < len(headings) else len(text)
        start, end = strip_span(text, heading.line_start, limit)
        body_start, body_end = strip_span(text, heading.body_from, limit)
        if body_start == body_end:
            body_start = body_end = end
        name = text[heading.name_start : heading.name_end]
        sections.append(
            Section(
                heading=name,
                canonical=known.get(normalise_heading(name)),
                start=start,
                end=end,
                text=text[start:end],
                heading_start=heading.name_start,
                heading_end=heading.name_end,
                body_start=body_start,
                body_end=body_end,
                body=text[body_start:body_end],
            )
        )
    return Note(
        text=text,
        preamble=text[p_start:p_end] if has_preamble else None,
        preamble_start=p_start if has_preamble else None,
        preamble_end=p_end if has_preamble else None,
        sections=tuple(sections),
    )


def parse_note_file(
    path: str | Path,
    *,
    heading_map: Mapping[str, CanonicalSection | str] | None = None,
) -> Note:
    """Parse a note file (plain text or Markdown, UTF-8)."""
    path = Path(path)
    text = read_text_file(path)
    try:
        return parse_note(text, heading_map=heading_map)
    except InputError as exc:
        raise InputError(f"{path}: {exc}") from exc


# Heading detection

_MARKDOWN = re.compile(r"^[ \t]{0,3}#{1,6}(?:[ \t]+(?P<content>.*?))?(?:[ \t]+#+)?[ \t]*$")
# Bold forms first, so "**Plan:**" is not read as a plain "Name: rest" line.
_FORMS = (
    re.compile(r"^\*\*(?P<name>[^*:\n]+?)[ \t]*(?P<colon>:)[ \t]*\*\*[ \t]*(?P<after>.*)$"),
    re.compile(r"^\*\*(?P<name>[^*:\n]+?)[ \t]*\*\*[ \t]*(?P<colon>:)?[ \t]*(?P<after>.*)$"),
    re.compile(r"^(?P<name>[^:\n]+?)[ \t]*(?P<colon>:)[ \t]*(?P<after>.*)$"),
    re.compile(r"^(?P<name>.+)(?P<colon>:)?(?P<after>)$"),
)


class _Heading:
    """Offsets of a detected heading; relative to its line until shifted."""

    __slots__ = ("line_start", "name_start", "name_end", "body_from")

    def __init__(self, line_start: int, name_start: int, name_end: int, body_from: int) -> None:
        self.line_start = line_start
        self.name_start = name_start
        self.name_end = name_end
        self.body_from = body_from

    def shifted(self, offset: int) -> _Heading:
        return _Heading(
            self.line_start + offset,
            self.name_start + offset,
            self.name_end + offset,
            self.body_from + offset,
        )


def _looks_like_heading(name: str) -> bool:
    return (
        name[:1].isalpha()
        and len(name.split()) <= _MAX_UNKNOWN_HEADING_WORDS
        and not _SENTENCE_PUNCTUATION.intersection(name)
    )


_CAPS_HEADING = re.compile(r"[A-Z]+(?:[ \t]*[&/][ \t]*[A-Z]+|[ \t]+[A-Z]+){0,5}")
_CAPS_BODY = frozenset({"NKDA", "NKA", "NAD", "NIL", "NONE", "N/A"})


def _is_caps_heading(name: str) -> bool:
    """An ACI-style heading: one to six capitalised words, not a body phrase like NKDA."""
    return (
        _CAPS_HEADING.fullmatch(name) is not None
        and name not in _CAPS_BODY
        and not name.startswith("NO ")
    )


def _match_heading(
    line: str,
    known: Mapping[str, CanonicalSection],
    *,
    after_blank: bool,
    followed_by_text: bool = False,
) -> _Heading | None:
    markdown = _MARKDOWN.match(line)
    if markdown:
        if markdown.group("content") is None:
            return None
        content_start, content_end = markdown.span("content")
    else:
        content_start, content_end = strip_span(line, 0, len(line))
    if content_start == content_end:
        return None
    content = line[content_start:content_end]

    for form in _FORMS:
        match = form.match(content)
        if match:
            break
    else:  # pragma: no cover - the last form matches any non-empty content
        return None

    name_start, name_end = (content_start + i for i in match.span("name"))
    has_colon = match.group("colon") is not None
    after = match.group("after")
    is_known = normalise_heading(match.group("name")) in known

    is_bold = match.re in _FORMS[:2]
    if after and not has_colon:
        is_heading = False  # "**Bold** words" is emphasis, not a heading
    elif is_known:
        is_heading = has_colon or is_bold or after_blank
    else:
        name = match.group("name")
        is_heading = (
            not after
            and after_blank
            and (
                (has_colon and _looks_like_heading(name))
                or (not has_colon and followed_by_text and _is_caps_heading(name))
            )
        )

    if not is_heading:
        if not markdown:
            return None
        # Any Markdown heading counts; keep its whole text as the name.
        name_start, name_end = content_start, content_end
        if line[name_end - 1] == ":":
            name_end = strip_span(line, name_start, name_end - 1)[1]
        return _Heading(0, name_start, name_end, len(line))

    body_from = content_start + match.start("after") if after else len(line)
    return _Heading(0, name_start, name_end, body_from)
