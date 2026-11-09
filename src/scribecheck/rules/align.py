"""The patient's own current facts in a transcript or note, ready to align.

Each transcript turn is read on its own (offsets index `Turn.text`); the note is read as
one text (offsets index `Note.text`). Drugs, symptoms and sides come from the lexicons and
finders, and the ConText rules run once over all of them together. Family and
hypothetical mentions are dropped: they are someone else's, or not facts. Historical,
possible and negated mentions are kept with their attributes, so the comparison can
refuse to decide on them.

Drugs align by their set of US generic ingredients, so brands, international names and
combination products resolve through the lexicon. A dose, frequency or route belongs to
the nearest drug in its sentence (before or after it) when nothing separates them: no
comma, semicolon, colon or bracket, no conjunction, pronoun or determiner, and no other
drug. A value two drugs could claim is ambiguous. A route adjective ("oral", "topical")
counts only when followed by a dose form or by no word, so "oral thrush" is not a route.

Each drug mention also records what makes its values unsafe to compare:

- `planned_or_past`: it is in a question, or its sentence speaks of a planned change or
  past use (increase, start, stop, switch, titrate, used to, was on, ...); in a transcript,
  a drug named in a recounted past episode (an emergency visit, a hospital stay, "back
  then", "that time") is past use too, when the episode is in its sentence or the sentence
  before it, which may end the same speaker's previous turn;
- `range_or_max`: its sentence gives a range or a maximum ("2-3 times", "up to");
- a tablet count (`has_count`, `count`): a count unit ("one tablet", "2 puffs"), a bare
  number ("two metformin 500 mg tablets"), a multiplier ("x2") or a split ("cut in half").
  A count is kept apart from the strength.

Across a whole text, `orphan_values` are dose or frequency values in sentences with no
drug name ("Let's increase it to 1000 mg"), `orphan_changes` those with a planned-change
or past word, and `resolutions` are words that say a symptom has gone ("went away",
"resolved").

Sides align by body site (and quadrant). A plural site ("knees", "both knees") is
recorded separately, since it means both or unspecified. Symptoms align by lexicon term,
with a status: negated (by ConText, or by a "No" to a single-symptom question), present,
or "question" for a mention in a question nobody answered in a way the rules can bind. A
negation also records whether its cue sits right next to the symptom
(`negation_adjacent`). A transcript presence records who established it: the patient side
(patient, relative, carer, parent, or a bare "Yes" to a present-tense single-symptom
question), a clinician (clinician, nurse), or nobody identifiable (no speaker labels). A
patient's statement counts as the patient side only when its sentence says "I", "me" or
"my", or it answers the clinician's question just before it, one with no condition in it
("What brings you in?" "Chest pain."). Anything else is a `fragment`: a patient reading
back advice ("Come back if you get chest pain ..." "Chest pain, fainting ...") is not
reporting a symptom.

Limits: side mentions with no body site ("right-sided") are not aligned; negated drug
mentions give no values.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from scribecheck._text import strip_span
from scribecheck.note import Note
from scribecheck.rules._context import sentence_span
from scribecheck.rules._match import Match
from scribecheck.rules.context import Attribute, TargetContext, classify
from scribecheck.rules.dialogue import (
    DialogueAffirmation,
    DialogueNegation,
    bind_affirmations,
    bind_negations,
)
from scribecheck.rules.frequency import FrequencyMatch, find_frequencies
from scribecheck.rules.lexicon import DrugMatch, SymptomMatch, find_drugs, find_symptoms
from scribecheck.rules.numbers import find_numbers
from scribecheck.rules.route import RouteMatch, find_routes
from scribecheck.rules.side import _PLURAL, SideMatch, find_sides
from scribecheck.rules.units import DoseMatch, Unit, find_doses
from scribecheck.transcript import UNKNOWN_SPEAKER, Transcript

SymptomStatus = Literal["present", "negated", "question"]
Presence = Literal["patient", "fragment", "clinician", "unattributed"]
ValueKind = Literal["doses", "frequencies", "routes"]

PATIENT_SIDE = frozenset({"patient", "relative", "carer", "parent"})
"""Speakers whose statement that a symptom is present counts as the patient's."""
STAFF = frozenset({"clinician", "nurse"})
COUNT_UNITS = frozenset({Unit.TABLET, Unit.CAPSULE, Unit.PUFF, Unit.DROP})
"""Units that count items; every other unit is a strength."""

_DROPPED: frozenset[Attribute] = frozenset({"family", "hypothetical"})
_B = r"(?<![A-Za-z])"
_E = r"(?![A-Za-z])"


def _any(*phrases: str) -> re.Pattern[str]:
    words = sorted(phrases, key=len, reverse=True)
    alternation = "|".join(
        r"\s+".join(map(re.escape, p.split())).replace("'", "['’]") for p in words
    )
    return re.compile(rf"{_B}(?:{alternation}){_E}", re.IGNORECASE)


_ROUTE_ADJECTIVES = frozenset(
    {
        "oral",
        "intravenous",
        "intramuscular",
        "subcutaneous",
        "sublingual",
        "rectal",
        "topical",
        "inhaled",
        "nebulised",
        "nebulized",
    }
)
_DOSE_FORM = re.compile(
    r"[ \t]+(?:tablets?|tabs?|capsules?|caps?|solution|suspension|liquid|syrup|injections?|"
    r"spray|inhaler|drops|cream|gel|ointment)(?![A-Za-z])",
    re.IGNORECASE,
)
_NEXT_WORD = re.compile(r"[ \t]+[A-Za-z]")
# What separates a value from a drug: punctuation, conjunctions, pronouns, determiners.
_SEPARATOR = re.compile(
    r"[,;:()&]|"
    + _any(
        "and", "or", "plus", "with", "also", "then", "but", "while", "whereas", "which",
        "who", "that", "this", "these", "those", "my", "your", "his", "her", "their", "our",
        "the", "a", "an", "we", "i", "you", "he", "she", "it", "they", "is", "are",
    ).pattern,
    re.IGNORECASE,
)  # fmt: skip
_PLANNED = _any(
    "increase", "increased", "increasing", "decrease", "decreased", "decreasing", "reduce",
    "reduced", "reducing", "start", "starting", "stop", "stopping", "switch", "switched",
    "switching", "change", "changed", "changing", "titrate", "titrated", "titrating",
    "uptitrate", "go up", "going up", "gone up", "go down", "going down", "wean", "weaning",
    "taper", "tapering", "double", "doubled", "halve", "restart", "commence", "begin", "try",
    "trial", "consider", "will", "I'll", "we'll", "you'll", "going to", "let's", "let me",
    "prescribe", "prescribing", "put you on", "I'd like you to", "from tomorrow",
    "from now on", "next week",
)  # fmt: skip
"""A planned change: the value may be future, not current."""
_PAST = _any(
    "started", "stopped", "commenced", "discontinue", "discontinued", "cut down", "come off",
    "came off", "been off", "off", "used to", "was on", "were on", "was taking",
    "were taking", "took", "quit", "gave up", "given up", "until", "last year", "last month",
    "last week", "any more", "anymore", "no longer", "previously", "before",
)  # fmt: skip
"""Past or stopped use: not a current medicine."""
_EPISODE = _any(
    "in ER", "the ER", "A&E", "A and E", "emergency department", "emergency room",
    "casualty", "in hospital", "in the hospital", "hospital stay", "was admitted",
    "were admitted", "got admitted", "back then", "that time", "at the time", "once I had",
)  # fmt: skip
"""A recounted past episode: a drug given or withheld then is not a current medicine."""
_PLANNED_OR_PAST = re.compile(f"{_PLANNED.pattern}|{_PAST.pattern}", re.IGNORECASE)
_STOPPED = _any(
    "stopped it", "stopped them", "stopped taking", "quit it", "quit them", "quit taking",
    "gave it up", "gave them up", "gave up taking", "came off it", "came off them",
    "come off it", "come off them", "been off it", "off it", "off them", "don't take it",
    "don't take them", "no longer take", "not taking it", "not taking them",
)  # fmt: skip
"""A stop in a sentence that names no drug ("I stopped it") may refer to any drug."""
_QUESTION_START = re.compile(
    r"^\W*(?:(?:and|so|ok|okay|well|now|right)\b[\s,]*)*(?:"
    r"(?:are|is|was|were|do|does|did|have|has|had|can|could|should|would|will|shall)\s+"
    r"(?:you|i|we|he|she|it|they|there|your|my|this|that|the)\b|any\b)",
    re.IGNORECASE,
)
"""A sentence that opens like a question is one, even without its "?" (speech to text)."""
_THIRD_PERSON = _any("he", "she", "he's", "she's", "his", "him", "her", "they", "their", "them")
_YEAR = re.compile(r"(?:19|20)\d\d")
_RANGE_OR_MAX = re.compile(
    r"\d\s*(?:-|–|to|or)\s*\d|"
    + _any(
        "up to", "at most", "max", "maximum", "no more than", "not more than", "not exceeding"
    ).pattern
    + r"|"
    + _any(*(f"{a} {j} {b}" for a in ("one", "two", "three", "four") for j in ("or", "to")
             for b in ("two", "three", "four", "five", "six"))).pattern,
    re.IGNORECASE,
)  # fmt: skip
_MULTIPLIER = re.compile(
    r"(?<![A-Za-z])[x×]\s*\d|" + _any("half", "halve", "halves", "split", "quarter").pattern,
    re.IGNORECASE,
)
_DURATION = re.compile(
    r"\s*(?:seconds?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|o['’]clock|am|pm)\b",
    re.IGNORECASE,
)
_RESOLUTION = _any(
    "went away", "gone away", "has gone", "is gone", "it's gone", "resolved", "settled",
    "cleared", "cleared up", "stopped", "eased off", "no longer", "not any more", "any more",
    "anymore", "better now",
)  # fmt: skip
_FIRST_PERSON = _any("I", "I'm", "I've", "I'd", "I'll", "me", "my", "mine", "myself")
_CONDITION = _any(
    "if", "in case", "come back", "return if", "watch out", "look out", "should you",
    "should it", "would make you", "would you",
)  # fmt: skip
_ADJACENT_WORDS = frozenset({"any", "a", "an", "the", "some"})
"""Words that may sit between a negation cue and its symptom for the cue to be adjacent."""
_WEAK_NEGATIONS = frozenset({"without"})
_WORD = re.compile(r"[\w'’]+")
_PLURAL_PARTS = "|".join(sorted(map(re.escape, _PLURAL), key=len, reverse=True))
_PLURAL_SITE = re.compile(rf"{_B}(?:both[ \t]+)?(?P<part>{_PLURAL_PARTS}){_E}", re.IGNORECASE)


@dataclass(frozen=True)
class Span:
    """Where a quote sits: a transcript turn (from 1), or the note when `turn` is None."""

    turn: int | None
    start: int
    end: int


@dataclass(frozen=True)
class DrugFact:
    turn: int | None
    """The transcript turn (from 1), or None in the note."""
    match: DrugMatch
    attributes: frozenset[Attribute]
    doses: tuple[DoseMatch, ...] = ()
    """Strength doses attached to this mention; count units are in `count`."""
    frequencies: tuple[FrequencyMatch, ...] = ()
    routes: tuple[RouteMatch, ...] = ()
    ambiguous: frozenset[ValueKind] = frozenset()
    """Kinds of value another drug could equally claim."""
    planned_or_past: bool = False
    """In a question, or speaking of a planned change or past use (union of the next three)."""
    planned: bool = False
    past: bool = False
    question: bool = False
    third_person: bool = False
    """A patient's sentence about "he" or "she" with no "I", "me" or "my": someone else's."""
    range_or_max: bool = False
    has_count: bool = False
    count: Decimal | None = None
    """The tablet count when it is a single known number; None when unknown or absent."""

    @property
    def key(self) -> frozenset[str]:
        return frozenset(self.match.generic)


@dataclass(frozen=True)
class SideFact:
    turn: int | None
    match: SideMatch
    attributes: frozenset[Attribute]
    role: str | None = None
    """The speaker's role for a transcript mention; None in the note."""
    question: bool = False
    third_person: bool = False

    @property
    def key(self) -> tuple[str | None, str | None]:
        return self.match.site, self.match.qualifier


@dataclass(frozen=True)
class PluralSite:
    """A body part named in the plural ("knees", "both knees"): both sides, or unspecified."""

    span: Span
    site: str


@dataclass(frozen=True)
class SymptomFact:
    turn: int | None
    match: SymptomMatch
    attributes: frozenset[Attribute]
    status: SymptomStatus
    presence: Presence | None
    """Who established a transcript presence; None for other statuses and in the note."""
    evidence: Span
    """The quote that shows the status: the cue and the symptom, or the patient's answer."""
    negation_adjacent: bool = True
    """For a ConText negation, whether a strong cue sits right next to the symptom."""


@dataclass(frozen=True)
class Facts:
    drugs: tuple[DrugFact, ...] = ()
    sides: tuple[SideFact, ...] = ()
    symptoms: tuple[SymptomFact, ...] = ()
    plural_sites: tuple[PluralSite, ...] = ()
    orphan_values: tuple[Span, ...] = ()
    """Dose or frequency values in sentences that name no drug."""
    orphan_changes: tuple[Span, ...] = ()
    """Orphan values whose sentence speaks of a planned change or past use."""
    resolutions: tuple[Span, ...] = ()
    """Words that say a symptom has gone."""
    orphan_stops: tuple[Span, ...] = ()
    """Sentences that stop "it" without naming a drug ("I stopped it last week")."""


@dataclass
class _Collected:
    drugs: list[DrugFact] = field(default_factory=list)
    sides: list[SideFact] = field(default_factory=list)
    symptoms: list[SymptomFact] = field(default_factory=list)
    plural_sites: list[PluralSite] = field(default_factory=list)
    orphan_values: list[Span] = field(default_factory=list)
    orphan_changes: list[Span] = field(default_factory=list)
    resolutions: list[Span] = field(default_factory=list)
    orphan_stops: list[Span] = field(default_factory=list)

    def facts(self) -> Facts:
        return Facts(
            tuple(self.drugs),
            tuple(self.sides),
            tuple(self.symptoms),
            tuple(self.plural_sites),
            tuple(self.orphan_values),
            tuple(self.orphan_changes),
            tuple(self.resolutions),
            tuple(self.orphan_stops),
        )


@dataclass(frozen=True)
class _Speaker:
    """How to read symptom mentions in one unit of text."""

    transcript: Transcript | None
    """The transcript, for a turn; None for the note."""
    role: str | None = None


def transcript_facts(
    transcript: Transcript,
    *,
    brands: Mapping[str, Sequence[str]] | None = None,
    symptom_extra: Mapping[str, str] | None = None,
) -> Facts:
    """The patient's facts in every turn of `transcript`."""
    symptoms_by_turn = {
        turn.number: find_symptoms(turn.text, extra=symptom_extra) for turn in transcript.turns
    }
    targets = {n: [(m.start, m.end) for m in found] for n, found in symptoms_by_turn.items()}
    bound: list[tuple[SymptomStatus, DialogueNegation | DialogueAffirmation]] = [
        *(("negated", b) for b in bind_negations(transcript, targets)),
        *(("present", b) for b in bind_affirmations(transcript, targets)),
    ]
    answers: dict[tuple[int, int, int], tuple[SymptomStatus, Span]] = {}
    for status, binding in bound:
        answer_text = transcript.turns[binding.answer_turn - 1].text
        start, end = strip_span(answer_text, 0, len(answer_text))
        key = (binding.question_turn, binding.target.start, binding.target.end)
        answers[key] = (status, Span(binding.answer_turn, start, end))

    collected = _Collected()
    for i, turn in enumerate(transcript.turns):
        before = transcript.turns[i - 1] if i else None
        _unit_facts(
            collected,
            turn.text,
            turn.number,
            _Speaker(transcript, turn.speaker),
            symptoms_by_turn[turn.number],
            answers,
            brands,
            previous=before.text if before is not None and before.speaker == turn.speaker else "",
        )
    return collected.facts()


def note_facts(
    note: Note,
    *,
    brands: Mapping[str, Sequence[str]] | None = None,
    symptom_extra: Mapping[str, str] | None = None,
) -> Facts:
    """The patient's facts in `note`."""
    collected = _Collected()
    symptoms = find_symptoms(note.text, extra=symptom_extra)
    _unit_facts(collected, note.text, None, _Speaker(None), symptoms, {}, brands)
    return collected.facts()


def _in_episode(text: str, start: int, end: int, previous: str) -> bool:
    """Whether text[start:end], or the sentence before it, recounts a past episode."""
    if _EPISODE.search(text, start, end):
        return True
    head = text[:start].rstrip()
    if head:
        before_start, before_end = sentence_span(head, len(head) - 1, len(head))
        return bool(_EPISODE.search(head, before_start, before_end))
    tail = previous.rstrip()
    if not tail:
        return False
    before_start, before_end = sentence_span(tail, len(tail) - 1, len(tail))
    return bool(_EPISODE.search(tail, before_start, before_end))


def _unit_facts(
    out: _Collected,
    text: str,
    turn: int | None,
    speaker: _Speaker,
    symptoms: Sequence[SymptomMatch],
    answers: Mapping[tuple[int, int, int], tuple[SymptomStatus, Span]],
    brands: Mapping[str, Sequence[str]] | None,
    *,
    previous: str = "",
) -> None:
    """`previous` is the same speaker's turn just before this one, if any (transcripts
    only), for a past episode recounted across two turns."""
    drugs = find_drugs(text, brands=brands)
    sides = find_sides(text)
    plurals = [m for m in _PLURAL_SITE.finditer(text)]
    spans = [(m.start, m.end) for m in (*drugs, *symptoms, *sides)] + [m.span() for m in plurals]
    contexts = classify(text, spans) if spans else ()
    n_drugs, n_symptoms, n_sides = len(drugs), len(symptoms), len(sides)
    drug_contexts = contexts[:n_drugs]
    symptom_contexts = contexts[n_drugs : n_drugs + n_symptoms]
    side_contexts = contexts[n_drugs + n_symptoms : n_drugs + n_symptoms + n_sides]
    plural_contexts = contexts[n_drugs + n_symptoms + n_sides :]

    doses, frequencies = find_doses(text), find_frequencies(text)
    routes = [r for r in find_routes(text) if _is_route(text, r)]
    values: list[tuple[ValueKind, Match]] = [
        *(("doses", d) for d in doses),
        *(("frequencies", f) for f in frequencies),
        *(("routes", r) for r in routes),
    ]
    attached: dict[int, dict[ValueKind, list[Match]]] = {i: {} for i in range(n_drugs)}
    ambiguous: dict[int, set[ValueKind]] = {i: set() for i in range(n_drugs)}
    for kind, value in values:
        claimants = [i for i, drug in enumerate(drugs) if _claims(text, drug, value, drugs, values)]
        if len(claimants) == 1:
            attached[claimants[0]].setdefault(kind, []).append(value)
        else:
            for i in claimants:
                ambiguous[i].add(kind)
        if kind != "routes" and not any(_same_sentence(text, d, value) for d in drugs):
            span = Span(turn, *strip_span(text, *sentence_span(text, value.start, value.end)))
            out.orphan_values.append(span)
            if _PLANNED_OR_PAST.search(text, span.start, span.end):
                out.orphan_changes.append(span)

    for index, (drug, context) in enumerate(zip(drugs, drug_contexts, strict=True)):
        if context.attributes & _DROPPED:
            continue
        sentence_start, sentence_end = sentence_span(text, drug.start, drug.end)
        sentence = text[sentence_start:sentence_end]
        question = is_question(text, sentence_start, sentence_end)
        planned, past = bool(_PLANNED.search(sentence)), bool(_PAST.search(sentence))
        if speaker.transcript is not None and not past:
            past = _in_episode(text, sentence_start, sentence_end, previous)
        mine = attached[index]
        strengths = [
            d
            for d in mine.get("doses", [])
            if isinstance(d, DoseMatch) and d.unit not in COUNT_UNITS
        ]
        has_count, count = _count(
            text, sentence_start, sentence_end, (*doses, *frequencies, *routes, *drugs), mine
        )
        out.drugs.append(
            DrugFact(
                turn=turn,
                match=drug,
                attributes=context.attributes,
                doses=tuple(strengths),
                frequencies=tuple(
                    f for f in mine.get("frequencies", []) if isinstance(f, FrequencyMatch)
                ),
                routes=tuple(r for r in mine.get("routes", []) if isinstance(r, RouteMatch)),
                ambiguous=frozenset(ambiguous[index]),
                planned_or_past=question or planned or past,
                planned=planned,
                past=past,
                question=question,
                third_person=_third_person(speaker, sentence),
                range_or_max=bool(_RANGE_OR_MAX.search(sentence)),
                has_count=has_count,
                count=count,
            )
        )

    for side, context in zip(sides, side_contexts, strict=True):
        if context.attributes & _DROPPED:
            continue
        sentence_start, sentence_end = sentence_span(text, side.start, side.end)
        out.sides.append(
            SideFact(
                turn=turn,
                match=side,
                attributes=context.attributes,
                role=speaker.role if speaker.transcript is not None else None,
                question=is_question(text, sentence_start, sentence_end),
                third_person=_third_person(speaker, text[sentence_start:sentence_end]),
            )
        )
    out.plural_sites += [
        PluralSite(Span(turn, *m.span()), _PLURAL[m.group("part").lower()])
        for m, context in zip(plurals, plural_contexts, strict=True)
        if not context.attributes & _DROPPED
    ]
    out.symptoms += [
        _symptom(text, turn, speaker, symptom, context, answers)
        for symptom, context in zip(symptoms, symptom_contexts, strict=True)
        if not context.attributes & _DROPPED
    ]
    if speaker.transcript is not None:
        out.resolutions += [Span(turn, *m.span()) for m in _RESOLUTION.finditer(text)]
        for stop in _STOPPED.finditer(text):
            bounds = sentence_span(text, stop.start(), stop.end())
            if not any(sentence_span(text, d.start, d.end) == bounds for d in drugs):
                out.orphan_stops.append(Span(turn, *strip_span(text, *bounds)))


def is_question(text: str, start: int, end: int) -> bool:
    """A sentence ending in "?", or opening like a question ("Are you on ...")."""
    return text[end : end + 1] == "?" or bool(_QUESTION_START.match(text, start, end))


def _third_person(speaker: _Speaker, sentence: str) -> bool:
    """A patient talking about "he" or "she", with no "I", "me" or "my" in the sentence."""
    if speaker.transcript is None or speaker.role != "patient":
        return False
    return bool(_THIRD_PERSON.search(sentence)) and not _FIRST_PERSON.search(sentence)


def _same_sentence(text: str, a: Match, b: Match) -> bool:
    return sentence_span(text, a.start, a.end) == sentence_span(text, b.start, b.end)


def _claims(
    text: str,
    drug: DrugMatch,
    value: Match,
    drugs: Sequence[DrugMatch],
    values: Sequence[tuple[ValueKind, Match]],
) -> bool:
    """Whether `value` can belong to `drug`: same sentence, nothing separating them."""
    if not _same_sentence(text, drug, value):
        return False
    if drug.end <= value.start:
        gap_start, gap_end = drug.end, value.start
    elif value.end <= drug.start:
        gap_start, gap_end = value.end, drug.start
    else:
        return False
    if any(gap_start < other.start < gap_end for other in drugs):
        return False
    gap = list(text[gap_start:gap_end])
    for _, other in values:  # other values may sit between ("500 mg twice daily of ...")
        for i in range(max(gap_start, other.start), min(gap_end, other.end)):
            gap[i - gap_start] = " "
    return _SEPARATOR.search("".join(gap)) is None


def _count(
    text: str,
    start: int,
    end: int,
    matches: Sequence[Match],
    mine: Mapping[ValueKind, Sequence[Match]],
) -> tuple[bool, Decimal | None]:
    """Whether the sentence `start:end` gives a tablet count, and its value when single.

    `matches` are the doses, frequencies, routes and drug names in the text; numbers inside
    them are not counts.
    """
    counted: list[Decimal] = [
        d.value for d in mine.get("doses", []) if isinstance(d, DoseMatch) and d.unit in COUNT_UNITS
    ]
    unknown = not counted and any(
        isinstance(m, DoseMatch) and m.unit in COUNT_UNITS and start <= m.start < end
        for m in matches
    )
    covered = [(m.start, m.end) for m in matches]
    for number in find_numbers(text[start:end]):
        n_start, n_end = start + number.start, start + number.end
        if any(s <= n_start and n_end <= e for s, e in covered):
            continue
        if _DURATION.match(text, n_end) or _YEAR.fullmatch(text[n_start:n_end]):
            continue
        counted.append(number.value)
    if _MULTIPLIER.search(text, start, end):
        unknown = True
    if not counted and not unknown:
        return False, None
    return True, counted[0] if len(counted) == 1 and not unknown else None


def _is_route(text: str, route: RouteMatch) -> bool:
    """An adjective form names a route only before a dose form or no word at all."""
    if route.text.lower() not in _ROUTE_ADJECTIVES:
        return True
    if _DOSE_FORM.match(text, route.end):
        return True
    return _NEXT_WORD.match(text, route.end) is None


def _symptom(
    text: str,
    turn: int | None,
    speaker: _Speaker,
    symptom: SymptomMatch,
    context: TargetContext,
    answers: Mapping[tuple[int, int, int], tuple[SymptomStatus, Span]],
) -> SymptomFact:
    def fact(
        status: SymptomStatus, presence: Presence | None, evidence: Span, adjacent: bool = True
    ) -> SymptomFact:
        return SymptomFact(turn, symptom, context.attributes, status, presence, evidence, adjacent)

    mention = Span(turn, symptom.start, symptom.end)
    if speaker.transcript is not None and turn is not None:
        sentence_start, sentence_end = sentence_span(text, symptom.start, symptom.end)
        if is_question(text, sentence_start, sentence_end):
            # A question asserts nothing, unless its answer was bound.
            bound = answers.get((turn, symptom.start, symptom.end))
            if bound is None:
                return fact("question", None, mention)
            status, answer = bound
            return fact(status, "patient" if status == "present" else None, answer)
    if context.negated:
        cues = [c for c in context.cues if c.attribute == "negated"]
        start = min([symptom.start, *(c.start for c in cues)])
        end = max([symptom.end, *(c.end for c in cues)])
        adjacent = any(_adjacent(text, cue, symptom) for cue in cues)
        return fact("negated", None, Span(turn, start, end), adjacent)
    if speaker.transcript is None or turn is None:
        return fact("present", None, mention)
    sentence = text[slice(*sentence_span(text, symptom.start, symptom.end))]
    return fact("present", _presence(speaker, turn, sentence), mention)


def _adjacent(text: str, cue: Match, symptom: Match) -> bool:
    """A strong negation cue with at most determiners between it and the symptom."""
    if " ".join(cue.text.lower().split()) in _WEAK_NEGATIONS:
        return False
    gap = (
        text[cue.end : symptom.start] if cue.end <= symptom.start else text[symptom.end : cue.start]
    )
    return all(word.lower() in _ADJACENT_WORDS for word in _WORD.findall(gap))


def _presence(speaker: _Speaker, turn: int, sentence: str) -> Presence:
    transcript = speaker.transcript
    assert transcript is not None
    if not transcript.has_speaker_labels or speaker.role in (None, UNKNOWN_SPEAKER):
        return "unattributed"
    if speaker.role == "patient":
        if _FIRST_PERSON.search(sentence):
            return "patient"
        previous = transcript.turns[turn - 2] if turn > 1 else None
        asked = previous is not None and previous.speaker == "clinician"
        if asked and previous is not None and previous.text.rstrip().endswith("?"):
            if not _CONDITION.search(previous.text):
                return "patient"
        return "fragment"
    if speaker.role in PATIENT_SIDE:
        return "patient"
    if speaker.role in STAFF:
        return "clinician"
    return "unattributed"
