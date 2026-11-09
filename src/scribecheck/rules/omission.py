"""Rules-mode Pass A: structured omission candidates.

For each medicine, dose, frequency, allergy and side the transcript establishes for the
patient, check whether the note contains it. When it doesn't, emit a candidate omission at
"possible" confidence with origin "rules" and verified transcript evidence: in full mode
it goes to the model's quote-verified second check, and in rules mode it is a warning.
Severity follows the rules-mode table: a missing medicine, dose, frequency or side is
supporting, and a missing allergy is critical.

What the transcript establishes reuses rules/align.py and its guards. Only the patient's
own mentions count: not someone else's (including a patient's sentence about "he" or "she"
with no "I", "me" or "my"), not hypothetical, historical, possible or negated, and not in
a question (one ending in "?" or opening like one). A drug named in an allergy statement is
an allergen, not a medicine. A medicine also needs a mention that is not past or stopped
use and not a spelling-tolerant match; a stop that names no drug ("I stopped it") leaves
every medicine unchecked. A planned medicine ("we'll start ...", "I'm going to put you on
...") may give a candidate, marked "(planned)" with the Plan section expected; a current
mention outranks it, and planned values never give dose or frequency candidates. A dose or
frequency must be consistent across the transcript, with no ambiguous value, range or
maximum, mixed tablet count, or orphan value elsewhere. Allergies need the patient's own
current statement (not outgrown, not about someone else), and a person is never an
allergen. An explicit statement that the patient has no allergies ("NKDA", "no known drug
allergies"), by anyone and not in a question, gives an allergy-status candidate ("no known
allergies", or "no known drug allergies" for a drug-only denial; rule
`allergy.status_missing`) when the transcript names no allergen of the patient's at all
(even a possible or past one) and the note says nothing about allergies (no allergy word,
no allergy statement and no Allergies section with text). An item a guard stops is
reported as unchecked with its reason code.

What counts as present in the note: a medicine under any name and in any context
("metformin stopped"); a dose or frequency when the drug's own lines (up to the next drug
name or blank line) give any value of that kind, including a timing word for frequency,
because a different value is Pass B's business; when the drug itself is missing, only the
medicine candidate is emitted. An allergy is present when the note names the allergen
(allowing plural and spacing variants), as a drug with the same ingredients, or denies
allergies in a way that covers it: "No allergies" covers all, "NKDA" drug allergens and "no
known food allergies" food ones; a denial for someone else, or "not allergic to X", covers
nothing else. A side candidate needs the note to name the same body site without any side
(a side word on the same line counts as a side); if the note gives a side for the site,
names it in the plural, or does not name it at all, there is no side candidate.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from scribecheck.flags import (
    Flag,
    Severity,
    TranscriptEvidence,
    UncheckedItem,
    UncheckedReason,
    transcript_evidence,
)
from scribecheck.note import CanonicalSection, Note
from scribecheck.rules._context import sentence_span
from scribecheck.rules._match import Comparison
from scribecheck.rules.align import (
    PATIENT_SIDE,
    DrugFact,
    Facts,
    Span,
    is_question,
    note_facts,
    transcript_facts,
)
from scribecheck.rules.allergy import AllergyMatch, find_allergies
from scribecheck.rules.frequency import FrequencyMatch, compare_frequencies, find_frequencies
from scribecheck.rules.lexicon import find_drugs
from scribecheck.rules.side import _PLURAL, _SINGULAR
from scribecheck.rules.units import DoseMatch, Unit, compare_doses, find_doses
from scribecheck.transcript import Transcript, Turn

SEVERITY: Mapping[str, Severity] = {
    "medicine": "supporting",
    "dose": "supporting",
    "frequency": "supporting",
    "side": "supporting",
    "allergy": "critical",
}
STATUS_RULE = "allergy.status_missing"
"""The rule of an allergy-status candidate: the transcript says the patient has no
allergies ("no known allergies", or "no known drug allergies" for a drug-only denial), and
the note says nothing about allergies."""
_ALLERGY_WORD = re.compile(r"(?<![A-Za-z])(?:allerg\w*|nkda|nka|nkfa)(?![A-Za-z])", re.I)
SECTION = {
    "medicine": "Medications",
    "dose": "Medications",
    "frequency": "Medications",
    "allergy": "Allergies",
    "side": "History",
}
_NOT_CURRENT = frozenset({"historical", "possible", "negated"})
_WORD = re.compile(r"[a-z0-9]+")


class OmissionResult(BaseModel):
    """Candidate omissions, and the items the guards left unchecked."""

    model_config = ConfigDict(frozen=True)

    candidates: tuple[Flag, ...] = ()
    unchecked: tuple[UncheckedItem, ...] = ()


class _Out:
    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript
        self.candidates: list[Flag] = []
        self.unchecked: list[UncheckedItem] = []

    def quotes(self, spans: Iterable[Span]) -> tuple[TranscriptEvidence, ...]:
        return tuple(self.quote(s) for s in spans)

    def quote(self, span: Span) -> TranscriptEvidence:
        assert span.turn is not None
        return transcript_evidence(self.transcript, span.turn, span.start, span.end)

    def candidate(
        self,
        category: str,
        fact: str,
        span: Span,
        *,
        section: str | None = None,
        rule_id: str | None = None,
    ) -> None:
        self.candidates.append(
            Flag(
                id=f"F{len(self.candidates) + 1}",
                type="omission",
                category=category,
                verdict="absent",
                severity=SEVERITY[category],
                confidence="possible",
                origin="rules",
                rule_id=rule_id or f"{category}.missing",
                fact=fact,
                transcript_evidence=self.quote(span),
                expected_section=section or SECTION[category],
            )
        )

    def skip(
        self, category: str, item: str, reason: UncheckedReason, spans: Iterable[Span]
    ) -> None:
        self.unchecked.append(
            UncheckedItem(
                category=category,
                item=item,
                reasons=(reason,),
                transcript_evidence=self.quotes(spans),
            )
        )


def structured_omissions(
    transcript: Transcript,
    note: Note,
    *,
    brands: Mapping[str, Sequence[str]] | None = None,
) -> OmissionResult:
    """Candidate omissions of the patient's medicines, doses, frequencies, allergies and sides."""
    t_facts = transcript_facts(transcript, brands=brands)
    n_facts = note_facts(note, brands=brands)
    out = _Out(transcript)
    _medicines(out, transcript, note, t_facts, n_facts)
    _allergies(out, transcript, note, n_facts, brands)
    _sides(out, note, t_facts, n_facts)
    return OmissionResult(candidates=tuple(out.candidates), unchecked=tuple(out.unchecked))


# Medicines, doses and frequencies


def _name(key: frozenset[str]) -> str:
    return " / ".join(sorted(key))


def _drug_span(drug: DrugFact) -> Span:
    return Span(drug.turn, drug.match.start, drug.match.end)


def _allergy_spans(transcript: Transcript) -> set[tuple[int, int, int]]:
    """(turn, start, end) of every allergy statement, whose drugs are allergens, not use."""
    return {
        (turn.number, a.start, a.end)
        for turn in transcript.turns
        for a in find_allergies(turn.text)
    }


def _in_allergy(drug: DrugFact, spans: set[tuple[int, int, int]]) -> bool:
    return any(
        turn == drug.turn and start <= drug.match.start and drug.match.end <= end
        for turn, start, end in spans
    )


def _medicines(
    out: _Out, transcript: Transcript, note: Note, t_facts: Facts, n_facts: Facts
) -> None:
    allergies = _allergy_spans(transcript)

    def usable(d: DrugFact) -> bool:
        """The patient's own mention, current or planned, never past, stopped or asked about."""
        return not (
            d.attributes & _NOT_CURRENT
            or d.past
            or d.question
            or d.match.fuzzy
            or d.third_person
            or _in_allergy(d, allergies)
        )

    keys: list[frozenset[str]] = []
    for drug in t_facts.drugs:
        mine = not (drug.third_person or _in_allergy(drug, allergies))
        if drug.key not in keys and mine and not drug.attributes & {"historical", "negated"}:
            keys.append(drug.key)
    for key in keys:
        mentions = [d for d in t_facts.drugs if d.key == key and not _in_allergy(d, allergies)]
        current = [d for d in mentions if usable(d) and not d.planned]
        planned = [d for d in mentions if usable(d) and d.planned]
        in_note = [d for d in n_facts.drugs if d.key == key]
        spans = [_drug_span(d) for d in mentions]
        if in_note:
            if current:
                _values(out, key, current, note, n_facts, t_facts)
            continue
        if not current and not planned:
            reason = _medicine_reason(mentions)
            if reason is not None:
                out.skip("medicine", _name(key), reason, spans)
            continue
        if t_facts.orphan_stops:  # "I stopped it": maybe this drug
            out.skip("medicine", _name(key), "planned_or_past", spans)
        elif current:
            out.candidate("medicine", _name(key), _drug_span(current[0]))
        else:
            out.candidate(
                "medicine", f"{_name(key)} (planned)", _drug_span(planned[0]), section="Plan"
            )


def _medicine_reason(mentions: Sequence[DrugFact]) -> UncheckedReason | None:
    current = [d for d in mentions if not d.attributes & {"historical", "negated"}]
    current = [d for d in current if not d.third_person]
    if any(d.past or d.question for d in current):
        return "planned_or_past"
    if any(d.match.fuzzy for d in current):
        return "spelling_tolerance"
    if any("possible" in d.attributes for d in current):
        return "possible"
    return None


def _note_has_value(note: Note, n_facts: Facts, key: frozenset[str], kind: str) -> bool:
    """The note gives the drug any value of this kind, anywhere in its own lines.

    A drug's lines run from its name to the next drug name or blank line, so a colon,
    bracket, comma or a value on the line below ("Dose: 500 mg") does not hide the value.
    """
    text = note.text
    values: Sequence[DoseMatch] | Sequence[FrequencyMatch] = (
        find_doses(text) if kind == "doses" else find_frequencies(text)
    )
    starts = sorted(d.match.start for d in n_facts.drugs)
    for drug in n_facts.drugs:
        if drug.key != key:
            continue
        end = min([s for s in starts if s > drug.match.start] + [len(text)])
        blank = text.find("\n\n", drug.match.end)
        if blank != -1:
            end = min(end, blank)
        if any(drug.match.end <= v.start and v.end <= end for v in values):
            return True
        if kind == "frequencies" and _TIMING.search(text, drug.match.end, end):
            return True
    return False


def _values(
    out: _Out,
    key: frozenset[str],
    established: Sequence[DrugFact],
    note: Note,
    n_facts: Facts,
    t_facts: Facts,
) -> None:
    kinds: tuple[tuple[str, str, Callable[[object, object], Comparison]], ...] = (
        ("dose", "doses", _compare_doses),
        ("frequency", "frequencies", _compare_frequencies),
    )
    for category, kind, compare in kinds:
        values = [(d, v) for d in established for v in getattr(d, kind)]
        guarded = any(kind in d.ambiguous for d in established)
        if not values and not guarded:
            continue
        if _note_has_value(note, n_facts, key, kind):
            continue
        spans = [_drug_span(d) for d in established]
        reason = _value_reason(category, kind, established, values, compare, t_facts)
        if reason is not None:
            out.skip(category, _name(key), reason, spans)
            continue
        drug, value = values[0]
        start, end = min(drug.match.start, value.start), max(drug.match.end, value.end)
        out.candidate(category, f"{_name(key)} {_display(value)}", Span(drug.turn, start, end))


def _value_reason(
    category: str,
    kind: str,
    established: Sequence[DrugFact],
    values: Sequence[tuple[DrugFact, object]],
    compare: Callable[[object, object], Comparison],
    t_facts: Facts,
) -> UncheckedReason | None:
    if t_facts.orphan_values:
        return "unattributed_value"
    if any(kind in d.ambiguous for d in established):
        return "ambiguous_value"
    if any(d.range_or_max for d in established):
        return "range_or_maximum"
    if category == "dose" and any(d.has_count for d in established if d.doses):
        return "tablet_count"
    if not values:
        return "ambiguous_value"
    comparisons = {compare(a, b) for _, a in values for _, b in values[1:]}
    if "different" in comparisons:
        return "transcript_inconsistent"
    if "not_comparable" in comparisons:
        return "not_comparable"
    return None


_TIMING = re.compile(
    r"(?<![A-Za-z])(?:at\s+night|at\s+bedtime|in\s+the\s+(?:morning|evening|afternoon)|"
    r"nocte|mane|daily|weekly|morning|evening|bedtime|prn|as\s+needed|when\s+needed)(?![A-Za-z])",
    re.IGNORECASE,
)
"""Timing words a note line gives without a dose ("insulin glargine at night"); the frequency
finder needs a dose nearby, but for presence a timing word is enough."""
_PLURAL_UNITS = frozenset({Unit.UNIT, Unit.PUFF, Unit.DROP, Unit.TABLET, Unit.CAPSULE})


def _display(value: object) -> str:
    if isinstance(value, DoseMatch):
        number = Decimal(value.value).normalize()
        unit = value.unit.value
        if value.unit in _PLURAL_UNITS and number != 1:
            unit += "s"
        return f"{format(number, 'f')} {unit}"
    assert isinstance(value, FrequencyMatch)
    return value.text


def _compare_doses(a: object, b: object) -> Comparison:
    assert isinstance(a, DoseMatch) and isinstance(b, DoseMatch)
    return compare_doses(a, b)


def _compare_frequencies(a: object, b: object) -> Comparison:
    assert isinstance(a, FrequencyMatch) and isinstance(b, FrequencyMatch)
    return compare_frequencies(a, b)


# Allergies

_PERSON_WORDS = frozenset(
    {
        "husband", "wife", "partner", "mum", "mom", "mother", "dad", "father", "son",
        "daughter", "brother", "sister", "child", "children", "baby", "friend", "boy", "girl",
        "he", "she", "him", "her", "they", "them", "my", "his", "their",
    }
)  # fmt: skip
"""Words that make an "allergen" a person ("allergic to dust and my husband to cats")."""
_OUTGROWN = re.compile(
    r"(?<![A-Za-z])(?:used\s+to|outgrew|outgrown|grew\s+out|as\s+a\s+(?:child|kid|baby)|"
    r"no\s+longer|not\s+any\s*more|was\s+allergic|were\s+allergic)(?![A-Za-z])",
    re.IGNORECASE,
)
_NON_DRUG = frozenset(
    {
        "peanut", "nut", "shellfish", "fish", "egg", "milk", "dairy", "lactose", "wheat",
        "gluten", "soy", "soya", "sesame", "latex", "dust", "pollen", "grass", "hay", "cat",
        "dog", "animal", "bee", "wasp", "sting", "mould", "mold", "food", "strawberry",
    }
)  # fmt: skip
"""Food and environmental allergens; any other allergen is treated as a drug, because
class names ("penicillin", "sulfa") are often not in the drug lexicon."""
_DRUG_DENIAL = re.compile(r"(?<![A-Za-z])(?:drug|drugs|medicine|medication|nkda)(?![A-Za-z])", re.I)
_FOOD_DENIAL = re.compile(r"(?<![A-Za-z])(?:food|foods|nkfa)(?![A-Za-z])", re.I)


def _stem(word: str) -> str:
    """A crude singular: "peanuts" -> "peanut", "penicillins" -> "penicillin"."""
    if len(word) > 4 and word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _stems(text: str) -> frozenset[str]:
    return frozenset(_stem(w) for w in _WORD.findall(text.casefold()))


def _allergen_key(text: str, brands: Mapping[str, Sequence[str]] | None) -> frozenset[str]:
    """An allergen's ingredient set when the lexicon knows it, otherwise its word stems."""
    drugs = find_drugs(text, brands=brands)
    if drugs:
        return frozenset(g for d in drugs for g in d.generic)
    return _stems(text)


def _patient_allergy(turn: Turn, allergy: AllergyMatch) -> bool:
    if allergy.status != "allergic" or allergy.experiencer != "patient" or not allergy.allergens:
        return False
    if allergy.possible or allergy.hypothetical or allergy.historical:
        return False
    start, end = sentence_span(turn.text, allergy.start, allergy.end)
    sentence = turn.text[start:end]
    if is_question(turn.text, start, end) or _OUTGROWN.search(sentence):
        return False
    return not (turn.speaker == "patient" and _third_person_only(sentence))


def _third_person_only(sentence: str) -> bool:
    words = {w.lower() for w in re.findall(r"[A-Za-z'’]+", sentence)}
    third = words & {"he", "she", "he's", "she's", "his", "him", "her", "they", "their"}
    first = words & {"i", "i'm", "i've", "me", "my", "mine"}
    return bool(third) and not first


def _allergies(
    out: _Out,
    transcript: Transcript,
    note: Note,
    n_facts: Facts,
    brands: Mapping[str, Sequence[str]] | None,
) -> None:
    note_allergies = find_allergies(note.text)
    denials = [a for a in note_allergies if a.status == "none" and a.experiencer == "patient"]
    note_keys = {
        _allergen_key(x.text, brands)
        for a in note_allergies
        if a.status in ("allergic", "not_allergic")
        for x in a.allergens
    }
    note_keys |= {d.key for d in n_facts.drugs}
    note_stems = _stems(note.text)
    note_compact = re.sub(r"\W+", "", note.text.casefold())
    seen: list[frozenset[str]] = []
    no_allergies: tuple[Span, str] | None = None
    any_allergen = False
    for turn in transcript.turns:
        for allergy in find_allergies(turn.text):
            if no_allergies is None and _no_allergies(turn, allergy):
                drugs_only = _DRUG_DENIAL.search(allergy.text) is not None
                no_allergies = (
                    Span(turn.number, allergy.start, allergy.end),
                    "no known drug allergies" if drugs_only else "no known allergies",
                )
            # Any allergen of the patient's, however hedged, qualifies a "no allergies".
            any_allergen |= allergy.status == "allergic" and allergy.experiencer == "patient"
            if not _patient_allergy(turn, allergy):
                continue
            for allergen in allergy.allergens:
                words = _stems(allergen.text)
                if not words or words & _PERSON_WORDS:
                    continue
                key = _allergen_key(allergen.text, brands)
                if key in seen:
                    continue
                seen.append(key)
                is_drug = not words & _NON_DRUG
                if _denied(denials, is_drug):
                    continue  # "No allergies" or "NKDA": a contradiction for the model
                compact = re.sub(r"\W+", "", allergen.text.casefold())
                if (
                    key in note_keys
                    or words <= note_stems
                    or (len(compact) >= 5 and compact in note_compact)
                ):
                    continue
                span = Span(turn.number, allergy.start, allergy.end)
                out.candidate("allergy", f"allergy: {allergen.text.casefold()}", span)
    if no_allergies is not None and not any_allergen and not _records_allergy_status(note):
        span, fact = no_allergies
        out.candidate("allergy", fact, span, rule_id=STATUS_RULE)


def _no_allergies(turn: Turn, allergy: AllergyMatch) -> bool:
    """An explicit statement that the patient has no allergies, by anyone, not asked."""
    if allergy.status != "none" or allergy.experiencer != "patient":
        return False
    if allergy.possible or allergy.hypothetical or allergy.historical:
        return False
    start, end = sentence_span(turn.text, allergy.start, allergy.end)
    return not is_question(turn.text, start, end)


def _records_allergy_status(note: Note) -> bool:
    """The note says anything about allergies: an allergy word ("allergies", "allergic",
    "NKDA", "NKA"), an allergy statement, or an Allergies section with text."""
    return (
        _ALLERGY_WORD.search(note.text) is not None
        or bool(find_allergies(note.text))
        or any(s.body for s in note.sections_for(CanonicalSection.ALLERGIES))
    )


def is_allergy_status(text: str) -> bool:
    """`text` states that the patient has no allergies ("No known drug allergies", "NKDA",
    "NKA"), as opposed to naming an allergen."""
    if re.search(r"(?<![A-Za-z])(?:nkda|nka)(?![A-Za-z])", text, re.I):
        return True
    found = find_allergies(text)
    return bool(found) and all(a.status == "none" for a in found)


def _denied(denials: Sequence[AllergyMatch], is_drug: bool) -> bool:
    """A patient's "No allergies" covers every allergen; "NKDA" only drugs; "no food" only food."""
    for denial in denials:
        if _DRUG_DENIAL.search(denial.text) and not is_drug:
            continue
        if _FOOD_DENIAL.search(denial.text) and is_drug:
            continue
        return True
    return False


# Sides


def _site_pattern(site: str) -> re.Pattern[str]:
    forms = [form for form, part in {**_SINGULAR, **_PLURAL}.items() if part == site]
    alternation = "|".join(sorted(map(re.escape, forms), key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z])(?:{alternation})(?![A-Za-z])", re.IGNORECASE)


_SIDE_WORD = re.compile(r"(?<![A-Za-z])(?:left|right|bilateral|both|L|R|lt|rt)(?![A-Za-z])", re.I)


def _sides(out: _Out, note: Note, t_facts: Facts, n_facts: Facts) -> None:
    """A side candidate only when the note names the same body site without any side."""
    keys: list[tuple[str | None, str | None]] = []
    for side in t_facts.sides:
        if side.match.site not in (None, "side") and side.key not in keys:
            keys.append(side.key)
    lateral_sites = {s.match.site for s in n_facts.sides} | {p.site for p in n_facts.plural_sites}
    for key in keys:
        site, qualifier = key
        assert site is not None
        if site in lateral_sites or not _unlateral_site_in_note(note, site):
            continue
        mentions = [s for s in t_facts.sides if s.key == key]
        current = [
            s
            for s in mentions
            if s.role in PATIENT_SIDE
            and not s.question
            and not s.third_person
            and not s.attributes & _NOT_CURRENT
            and s.match.text.upper() not in {"OD", "OS", "OU"}
        ]
        if not current:
            continue
        spans = [Span(s.turn, s.match.start, s.match.end) for s in mentions]
        name = f"{site} {qualifier}" if qualifier else site
        if any(p.site == site for p in t_facts.plural_sites) or any(
            s.match.side == "bilateral" for s in current
        ):
            out.skip("side", name, "plural_site", spans)
            continue
        if len({s.match.side for s in current}) > 1:
            out.skip("side", name, "transcript_inconsistent", spans)
            continue
        first = current[0]
        span = Span(first.turn, first.match.start, first.match.end)
        out.candidate("side", f"{first.match.side} {qualifier or site}", span)


def _unlateral_site_in_note(note: Note, site: str) -> bool:
    """The note names the site, and no side word shares its line ("Knee pain, left")."""
    for found in _site_pattern(site).finditer(note.text):
        line_start = note.text.rfind("\n", 0, found.start()) + 1
        line_end = note.text.find("\n", found.end())
        line = note.text[line_start : line_end if line_end != -1 else len(note.text)]
        if not _SIDE_WORD.search(line):
            return True
    return False
