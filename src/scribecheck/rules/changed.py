"""Pass B, rules only: certain changed-fact flags, and unchecked items with reasons.

For each aligned item (a drug's dose, frequency and route; a body site's side; a
symptom's negation status) the transcript's current values are compared with the note's.
Precision comes first: a flag is emitted only at "certain" confidence, and any doubt
records the item as unchecked with the first reason that applies, so the recall each
guard costs can be measured.

Checked whenever both sides say something about the item:

- `planned_or_past`: a mention of the drug is in a question or speaks of a planned change
  or past use, on either side, or a sentence with no drug name does (a value given for
  "it" after "let's increase it").
- `unattributed_value`: the transcript has a dose or frequency in a sentence that names no
  drug, so a value may belong to this drug without saying so.
- `mixed_temporality`: the item also has a historical mention on either side.
- `ambiguous_value`: a value two drugs could equally claim.
- `range_or_maximum`: a range or maximum ("2-3 times a day", "up to 4 g").
- `tablet_count`: either side gives a tablet count and the counts are not identical, so
  count and strength are not compared like with like.
- `transcript_inconsistent` / `note_inconsistent`: more than one value on a side (a dose
  changed during the consultation, both knees discussed, a symptom present then absent).
- `not_comparable`: values that cannot be compared (500 mg against 5 ml, a frequency with
  no times value). Never treated as a change.

Checked when the values differ, before a flag is emitted:

- `eye_abbreviation`: a value came from OD, OS or OU.
- `spelling_tolerance`: a mention of the drug on either side matched with one edit.
- `unreviewed_data`: a mention relied on data marked "reviewed": false.
- `plural_site`: the site is also named in the plural ("knees", "both knees") or a side is
  bilateral: both sides, or unspecified.
- `status_change`: the transcript says a symptom has gone ("went away", "resolved").
- `negation_scope`: a negation used is not right next to its symptom ("no treatment tried
  for cough", "no fever, cough or headache"), or is the weak cue "without".
- `unconfirmed_presence`: a symptom's presence rests only on a patient fragment that is
  neither a first-person statement nor an answer to a plain question (reading back
  advice: "Chest pain, fainting, won't stop").
- `clinician_only` / `no_speaker_labels`: a symptom's presence rests only on clinician
  statements, or cannot be attributed to anyone.

When one side has values and the other none: `possible` (that side's mentions all carry a
"possible" cue) or `question_unanswered` (the transcript names the symptom only in
unbindable questions). Otherwise a value on one side only is Pass A's business. Drugs
found on one side only are `unaligned`; there is no drug-substituted flag here.

Severity follows the spec's rules-mode table: dose, frequency or side changed, and
negation reversed, are critical; route changed is supporting.

Certain flags are currently disabled (`CERTAIN_FLAGS_ENABLED`): every item is reported
as unchecked, and one that would have been a certain flag carries the reason
"certain_disabled_pending_allowlist". They return when certain flags are limited to
allow-listed sentence shapes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from scribecheck.flags import (
    ChangedFactResult,
    Flag,
    NoteEvidence,
    TranscriptEvidence,
    UncheckedItem,
    UncheckedReason,
    note_evidence,
    transcript_evidence,
)
from scribecheck.note import Note
from scribecheck.rules._match import Comparison
from scribecheck.rules.align import (
    DrugFact,
    Facts,
    SideFact,
    Span,
    SymptomFact,
    ValueKind,
    note_facts,
    transcript_facts,
)
from scribecheck.rules.frequency import FrequencyMatch, compare_frequencies
from scribecheck.rules.route import RouteMatch
from scribecheck.rules.units import DoseMatch, compare_doses
from scribecheck.transcript import Transcript

Category = Literal["dose", "frequency", "route", "side", "symptom"]

SEVERITY: Mapping[Category, Literal["critical", "supporting"]] = {
    "dose": "critical",
    "frequency": "critical",
    "route": "supporting",
    "side": "critical",
    "symptom": "critical",
}
RULE_ID: Mapping[Category, str] = {
    "dose": "dose.mismatch",
    "frequency": "frequency.mismatch",
    "route": "route.mismatch",
    "side": "side.mismatch",
    "symptom": "negation.reversed",
}
_EYE_TOKENS = frozenset({"OD", "OS", "OU"})

CERTAIN_FLAGS_ENABLED = False
"""Off until certain flags come only from allow-listed sentence shapes. Two adversarial
reviews showed that blocking cues (a deny-list) cannot rule out false certain flags in
open clinical language. While off, an item that would have been a certain flag is
unchecked with reason CERTAIN_DISABLED and the values it would have reported."""
CERTAIN_DISABLED: UncheckedReason = "certain_disabled_pending_allowlist"


def _with_disabled(reason: UncheckedReason) -> tuple[UncheckedReason, ...]:
    """An unchecked item's reasons: its own, plus the disabled marker while flags are off."""
    return (reason,) if CERTAIN_FLAGS_ENABLED else (reason, CERTAIN_DISABLED)


@dataclass(frozen=True)
class _Value:
    """One value of an item on one side, with the quote that shows it."""

    value: object
    display: str
    span: Span
    eye: bool = False


@dataclass(frozen=True)
class _Item:
    """One aligned item and attribute, ready to compare."""

    category: Category
    name: str
    transcript: tuple[_Value, ...]
    note: tuple[_Value, ...]
    compare: Callable[[object, object], Comparison]
    pre_checks: tuple[UncheckedReason, ...] = ()
    """Reasons reported whenever both sides say something about the item."""
    both_informed: bool = False
    """Both sides say something even where neither gives a comparable value (an
    ambiguous value, a tablet count)."""
    one_sided: UncheckedReason | None = None
    """What to report when one side has current values and the other has none."""
    blockers: tuple[UncheckedReason, ...] = ()
    """Reasons that stop a certain flag when the values differ."""
    mentions_t: tuple[Span, ...] = ()
    mentions_n: tuple[Span, ...] = ()


def check_changed_facts(
    transcript: Transcript,
    note: Note,
    *,
    brands: Mapping[str, Sequence[str]] | None = None,
    symptom_extra: Mapping[str, str] | None = None,
) -> ChangedFactResult:
    """Compare the patient's current facts in `transcript` and `note` (rules only)."""
    t_facts = transcript_facts(transcript, brands=brands, symptom_extra=symptom_extra)
    n_facts = note_facts(note, brands=brands, symptom_extra=symptom_extra)
    flags: list[Flag] = []
    unchecked: list[UncheckedItem] = []

    def t_quotes(spans: Iterable[Span]) -> tuple[TranscriptEvidence, ...]:
        return tuple(_t_evidence(transcript, s) for s in spans)

    def n_quotes(spans: Iterable[Span]) -> tuple[NoteEvidence, ...]:
        return tuple(note_evidence(note, s.start, s.end) for s in spans)

    for key in _unaligned(t_facts, n_facts):
        unchecked.append(
            UncheckedItem(
                category="drug",
                item=_drug_name(key),
                reasons=_with_disabled("unaligned"),
                transcript_evidence=t_quotes(_drug_span(d) for d in t_facts.drugs if d.key == key),
                note_evidence=n_quotes(_drug_span(d) for d in n_facts.drugs if d.key == key),
            )
        )

    for item in _items(t_facts, n_facts):
        outcome = _decide(item)
        if outcome is None:
            continue
        if outcome != "flag":
            unchecked.append(
                UncheckedItem(
                    category=item.category,
                    item=item.name,
                    reasons=_with_disabled(outcome),
                    transcript_evidence=t_quotes(item.mentions_t),
                    note_evidence=n_quotes(item.mentions_n),
                )
            )
            continue
        t_value, n_value = item.transcript[0], item.note[0]
        if not CERTAIN_FLAGS_ENABLED:
            unchecked.append(
                UncheckedItem(
                    category=item.category,
                    item=item.name,
                    reasons=(CERTAIN_DISABLED,),
                    transcript_value=t_value.display,
                    note_value=n_value.display,
                    transcript_evidence=t_quotes([t_value.span]),
                    note_evidence=n_quotes([n_value.span]),
                )
            )
            continue
        flags.append(
            Flag(
                id=f"F{len(flags) + 1}",
                type="changed_fact",
                category=item.category,
                verdict="changed",
                severity=SEVERITY[item.category],
                confidence="certain",
                origin="rules",
                rule_id=RULE_ID[item.category],
                transcript_value=t_value.display,
                note_value=n_value.display,
                transcript_evidence=_t_evidence(transcript, t_value.span),
                note_evidence=note_evidence(note, n_value.span.start, n_value.span.end),
            )
        )
    return ChangedFactResult(flags=tuple(flags), unchecked=tuple(unchecked))


def _decide(item: _Item) -> UncheckedReason | Literal["flag"] | None:
    """`flag`, the reason the item is unchecked, or None when there is nothing to report."""
    both = (bool(item.transcript) and bool(item.note)) or item.both_informed
    if not both:
        return item.one_sided if item.transcript or item.note else None
    if item.pre_checks:
        return item.pre_checks[0]
    if not item.transcript or not item.note:
        return None
    sides: tuple[tuple[tuple[_Value, ...], UncheckedReason], ...] = (
        (item.transcript, "transcript_inconsistent"),
        (item.note, "note_inconsistent"),
    )
    for values, reason in sides:
        comparisons = {item.compare(a.value, b.value) for a in values for b in values[1:]}
        if "different" in comparisons:
            return reason
        if "not_comparable" in comparisons:
            return "not_comparable"
    comparison = item.compare(item.transcript[0].value, item.note[0].value)
    if comparison == "equal":
        return None
    if comparison == "not_comparable":
        return "not_comparable"
    if any(v.eye for v in (*item.transcript, *item.note)):
        return "eye_abbreviation"
    if item.blockers:
        return item.blockers[0]
    return "flag"


# Building the items


def _items(t_facts: Facts, n_facts: Facts) -> list[_Item]:
    items: list[_Item] = []
    n_keys = {d.key for d in n_facts.drugs}
    seen: list[frozenset[str]] = []
    for drug in t_facts.drugs:
        if drug.key in n_keys and drug.key not in seen:
            seen.append(drug.key)
    for key in seen:
        t_drugs = [d for d in t_facts.drugs if d.key == key]
        n_drugs = [d for d in n_facts.drugs if d.key == key]
        items += _drug_items(key, t_drugs, n_drugs, t_facts, n_facts)

    sites: list[tuple[str | None, str | None]] = []
    for side in t_facts.sides:
        if side.match.site is not None and side.key not in sites:
            sites.append(side.key)
    for site in sites:
        n_sides = [s for s in n_facts.sides if s.key == site]
        if n_sides:
            t_sides = [s for s in t_facts.sides if s.key == site]
            plural = any(p.site == site[0] for p in (*t_facts.plural_sites, *n_facts.plural_sites))
            items.append(_side_item(site, t_sides, n_sides, plural))

    terms: list[str] = []
    for symptom in t_facts.symptoms:
        if symptom.match.term not in terms:
            terms.append(symptom.match.term)
    for term in terms:
        n_symptoms = [s for s in n_facts.symptoms if s.match.term == term]
        if n_symptoms:
            t_symptoms = [s for s in t_facts.symptoms if s.match.term == term]
            items.append(_symptom_item(term, t_symptoms, n_symptoms, bool(t_facts.resolutions)))
    return items


def _current(attributes: frozenset[str]) -> bool:
    """A mention that states a current fact: not historical, not merely possible."""
    return not attributes & {"historical", "possible"}


def _possible(attributes: frozenset[str]) -> bool:
    return "possible" in attributes and "historical" not in attributes


def _one_sided(
    t_values: tuple[_Value, ...],
    n_values: tuple[_Value, ...],
    t_possible: bool,
    n_possible: bool,
) -> UncheckedReason | None:
    """`possible` when one side's only mentions of the item carry a "possible" cue."""
    if (t_values and not n_values and n_possible) or (n_values and not t_values and t_possible):
        return "possible"
    return None


def _drug_items(
    key: frozenset[str],
    t_drugs: Sequence[DrugFact],
    n_drugs: Sequence[DrugFact],
    t_facts: Facts,
    n_facts: Facts,
) -> list[_Item]:
    every = (*t_drugs, *n_drugs)
    shared: list[UncheckedReason] = []
    if t_facts.orphan_changes or n_facts.orphan_changes or any(d.planned_or_past for d in every):
        shared.append("planned_or_past")
    if t_facts.orphan_values:
        shared.append("unattributed_value")
    if any("historical" in d.attributes for d in every):
        shared.append("mixed_temporality")
    blockers: list[UncheckedReason] = []
    if any(d.match.fuzzy for d in every):
        blockers.append("spelling_tolerance")
    if any(not d.match.reviewed for d in every):
        blockers.append("unreviewed_data")

    def usable(drug: DrugFact) -> bool:
        return _current(drug.attributes) and "negated" not in drug.attributes

    def values(drugs: Sequence[DrugFact], kind: ValueKind) -> tuple[_Value, ...]:
        return tuple(
            _drug_value(drug, value)
            for drug in drugs
            if usable(drug)
            for value in getattr(drug, kind)
        )

    def possible(drugs: Sequence[DrugFact], kind: ValueKind) -> bool:
        return any(
            _possible(d.attributes) and "negated" not in d.attributes and getattr(d, kind)
            for d in drugs
        )

    def informed(drugs: Sequence[DrugFact], kind: ValueKind) -> bool:
        return any(
            usable(d)
            and (getattr(d, kind) or kind in d.ambiguous or (kind == "doses" and d.has_count))
            for d in drugs
        )

    def counts(drugs: Sequence[DrugFact]) -> set[Decimal | None | Literal["none"]]:
        found: set[Decimal | None | Literal["none"]] = set()
        for d in drugs:
            if usable(d) and (d.doses or d.has_count):
                found.add(d.count if d.has_count else "none")
        return found

    kinds: tuple[tuple[Category, ValueKind, Callable[[object, object], Comparison]], ...] = (
        ("dose", "doses", _compare_doses),
        ("frequency", "frequencies", _compare_frequencies),
        ("route", "routes", _compare_routes),
    )
    items = []
    for category, kind, compare in kinds:
        t_values, n_values = values(t_drugs, kind), values(n_drugs, kind)
        pre = list(shared)
        if any(kind in d.ambiguous for d in every):
            pre.append("ambiguous_value")
        if any(d.range_or_max for d in every):
            pre.append("range_or_maximum")
        if kind == "doses":
            t_counts, n_counts = counts(t_drugs), counts(n_drugs)
            mixed = t_counts != n_counts or len(t_counts) > 1 or None in t_counts
            if (t_counts - {"none"} or n_counts - {"none"}) and mixed:
                pre.append("tablet_count")
        items.append(
            _Item(
                category=category,
                name=_drug_name(key),
                transcript=t_values,
                note=n_values,
                compare=compare,
                pre_checks=tuple(pre),
                both_informed=informed(t_drugs, kind) and informed(n_drugs, kind),
                one_sided=_one_sided(
                    t_values, n_values, possible(t_drugs, kind), possible(n_drugs, kind)
                ),
                blockers=tuple(blockers),
                mentions_t=tuple(_drug_span(d) for d in t_drugs),
                mentions_n=tuple(_drug_span(d) for d in n_drugs),
            )
        )
    return items


def _drug_value(drug: DrugFact, value: DoseMatch | FrequencyMatch | RouteMatch) -> _Value:
    """A dose, frequency or route, quoted from the drug name to the value (either order)."""
    start, end = min(drug.match.start, value.start), max(drug.match.end, value.end)
    span = Span(drug.turn, start, end)
    if isinstance(value, DoseMatch):
        return _Value(value, f"{_number(value.value)} {value.unit.value}", span)
    if isinstance(value, FrequencyMatch):
        return _Value(value, value.text, span, eye=value.text.upper() == "OD")
    return _Value(value.route, value.route.value, span)


def _side_item(
    key: tuple[str | None, str | None],
    t_sides: Sequence[SideFact],
    n_sides: Sequence[SideFact],
    plural: bool,
) -> _Item:
    site, qualifier = key

    def values(sides: Sequence[SideFact]) -> tuple[_Value, ...]:
        return tuple(
            _Value(
                s.match.side,
                f"{s.match.side} {qualifier or site}",
                Span(s.turn, s.match.start, s.match.end),
                eye=s.match.text.upper() in _EYE_TOKENS,
            )
            for s in sides
            if _current(s.attributes)
        )

    t_values, n_values = values(t_sides), values(n_sides)
    bilateral = any(v.value == "bilateral" for v in (*t_values, *n_values))
    return _Item(
        category="side",
        name=f"{site} {qualifier}" if qualifier else str(site),
        transcript=t_values,
        note=n_values,
        compare=_compare_equal,
        pre_checks=(
            ("mixed_temporality",)
            if any("historical" in s.attributes for s in (*t_sides, *n_sides))
            else ()
        ),
        one_sided=_one_sided(
            t_values,
            n_values,
            any(_possible(s.attributes) for s in t_sides),
            any(_possible(s.attributes) for s in n_sides),
        ),
        blockers=("plural_site",) if plural or bilateral else (),
        mentions_t=tuple(Span(s.turn, s.match.start, s.match.end) for s in t_sides),
        mentions_n=tuple(Span(s.turn, s.match.start, s.match.end) for s in n_sides),
    )


def _symptom_item(
    term: str,
    t_symptoms: Sequence[SymptomFact],
    n_symptoms: Sequence[SymptomFact],
    resolution: bool,
) -> _Item:
    def used(symptoms: Sequence[SymptomFact]) -> list[SymptomFact]:
        return [s for s in symptoms if s.status != "question" and _current(s.attributes)]

    def values(symptoms: Sequence[SymptomFact]) -> tuple[_Value, ...]:
        return tuple(
            _Value(s.status, "denied" if s.status == "negated" else "present", s.evidence)
            for s in used(symptoms)
        )

    t_values, n_values = values(t_symptoms), values(n_symptoms)
    one_sided = _one_sided(
        t_values,
        n_values,
        any(s.status != "question" and _possible(s.attributes) for s in t_symptoms),
        any(_possible(s.attributes) for s in n_symptoms),
    )
    if one_sided is None and n_values and not t_values:
        if any(s.status == "question" and _current(s.attributes) for s in t_symptoms):
            one_sided = "question_unanswered"

    blockers: list[UncheckedReason] = []
    if resolution:
        blockers.append("status_change")
    if any(
        s.status == "negated" and not s.negation_adjacent
        for s in (*used(t_symptoms), *used(n_symptoms))
    ):
        blockers.append("negation_scope")
    if any(not s.match.reviewed for s in (*t_symptoms, *n_symptoms)):
        blockers.append("unreviewed_data")
    presences = {
        s.presence for s in used(t_symptoms) if s.status == "present" and s.presence is not None
    }
    if presences and "patient" not in presences:
        if "fragment" in presences:
            blockers.append("unconfirmed_presence")
        elif "clinician" in presences:
            blockers.append("clinician_only")
        else:
            blockers.append("no_speaker_labels")
    return _Item(
        category="symptom",
        name=term,
        transcript=t_values,
        note=n_values,
        compare=_compare_equal,
        pre_checks=(
            ("mixed_temporality",)
            if any("historical" in s.attributes for s in (*t_symptoms, *n_symptoms))
            else ()
        ),
        one_sided=one_sided,
        blockers=tuple(blockers),
        mentions_t=tuple(Span(s.turn, s.match.start, s.match.end) for s in t_symptoms),
        mentions_n=tuple(Span(s.turn, s.match.start, s.match.end) for s in n_symptoms),
    )


# Comparisons and helpers


def _compare_doses(a: object, b: object) -> Comparison:
    assert isinstance(a, DoseMatch) and isinstance(b, DoseMatch)
    return compare_doses(a, b)


def _compare_frequencies(a: object, b: object) -> Comparison:
    assert isinstance(a, FrequencyMatch) and isinstance(b, FrequencyMatch)
    return compare_frequencies(a, b)


def _compare_routes(a: object, b: object) -> Comparison:
    return "equal" if a == b else "different"


def _compare_equal(a: object, b: object) -> Comparison:
    return "equal" if a == b else "different"


def _unaligned(t_facts: Facts, n_facts: Facts) -> list[frozenset[str]]:
    t_keys = [d.key for d in t_facts.drugs]
    n_keys = [d.key for d in n_facts.drugs]
    ordered = [*t_keys, *n_keys]
    one_side = {k for k in ordered if (k in t_keys) != (k in n_keys)}
    return [k for i, k in enumerate(ordered) if k in one_side and k not in ordered[:i]]


def _drug_span(drug: DrugFact) -> Span:
    return Span(drug.turn, drug.match.start, drug.match.end)


def _drug_name(key: frozenset[str]) -> str:
    return " / ".join(sorted(key))


def _number(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _t_evidence(transcript: Transcript, span: Span) -> TranscriptEvidence:
    assert span.turn is not None
    return transcript_evidence(transcript, span.turn, span.start, span.end)
