"""Allergy statements: allergen, reaction, or an explicit none.

Forms handled:

- "allergic to X", "allergy to X", "allergies to X", "allergic reaction to X", with
  lists ("X and Y", "X, Y");
- "penicillin allergy" (but not "drug allergy", "food allergy" or "allergy clinic");
- "X gives me a rash", "X caused me hives", "X brings me out in a rash", which need
  an allergic reaction from `ALLERGIC_REACTIONS` (so "Metformin gives me diarrhoea",
  a side effect, is not an allergy) and a subject that is not a common non-allergen
  trigger ("Stress gives me hives");
- a labelled line, "Allergies: X (rash)", "Drug allergies: X - rash; Y", "Allergic
  to: X", read as a list (see below);
- explicit none: NKDA, NKA, "no known (drug) allergies", "no drug allergies", and any
  negated allergy word without an allergen ("no allergies", "not allergic to anything").

`find_note_allergies` reads a parsed note: the body of every Allergies section is a
list, and the rest of the note goes through `find_allergies`.

Lists split on commas, semicolons, new lines and line-start bullets, never inside
parentheses. Each item is an allergen with an optional reaction, "penicillin (rash)",
"penicillin - rash" (a dash with spaces, so "co-trimoxazole" stays whole) or
"penicillin: rash"; NKDA, NKA, none, nil, "none known" and "no known allergies" are an
explicit none. A "?" or a leading "possible", "query" or "suspected" marks the item
possible. Items that defer ("see chart", "as above", "unknown") or do not read as an
allergen (verbs, reaction words, more than four words) give no result. Each list item
is its own result; its `cue` is the label or section heading before it.

Every result has a `kind`: "allergy" when the reaction includes an allergic reaction
(`ALLERGIC_REACTIONS`), "intolerance" when it only names a non-allergic one such as
nausea or headache (`INTOLERANCE_REACTIONS`), and "unknown" without a reaction.

Allergens are raw text; the drug lexicon normalises them later. A reaction is found
after the allergen in the same sentence (or label line) behind a connector such as
"gives me", "causes", "I get", "(" or a dash; a run like "a rash, hives or swelling"
is one reaction span.

Negation, family, certainty and time come from the ConText rules
(rules/context.py): "I'm not allergic to penicillin" is `not_allergic` for
penicillin, and "My mother is allergic to penicillin" has `experiencer="family"`,
so it is not the patient's.

Precision comes first, so these give no result and are left to the model:
questions ("Are you allergic to penicillin?"), a none with an exception ("No
allergies except penicillin"), and a family statement the patient shares ("My
mother is allergic to penicillin and so am I").
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, NamedTuple

from scribecheck.note import CanonicalSection, Note
from scribecheck.rules._context import sentence_span
from scribecheck.rules._match import Match, non_overlapping
from scribecheck.rules.context import classify
from scribecheck.rules.units import find_doses

AllergyStatus = Literal["allergic", "not_allergic", "none"]
""""none" is an explicit statement of no allergies."""
Experiencer = Literal["patient", "family"]

AllergyKind = Literal["allergy", "intolerance", "unknown"]

SKIN_REACTIONS: tuple[str, ...] = (
    "rash",
    "rashes",
    "hives",
    "urticaria",
    "swelling",
    "itching",
    "itch",
    "itchiness",
    "itchy",
    "anaphylaxis",
    "anaphylactic shock",
    "anaphylactic reaction",
    "angioedema",
    "blisters",
    "blistering",
)
ALLERGIC_REACTIONS: tuple[str, ...] = SKIN_REACTIONS + (
    "wheeze",
    "wheezing",
    "difficulty breathing",
    "trouble breathing",
    "breathing difficulty",
    "breathing difficulties",
    "breathing problems",
    "shortness of breath",
    "asthma",
    "bronchospasm",
)
""""X gives me ..." counts as an allergy only with one of these, and any of them makes the
kind "allergy"."""
INTOLERANCE_REACTIONS: tuple[str, ...] = (
    "nausea",
    "vomiting",
    "diarrhoea",
    "diarrhea",
    "headache",
    "headaches",
    "dizziness",
    "upset stomach",
    "stomach upset",
    "GI upset",
    "indigestion",
    "constipation",
    "drowsiness",
    "abdominal pain",
)
"""Reactions that are clearly not allergic; with only these the kind is "intolerance"."""
REACTIONS: tuple[str, ...] = ALLERGIC_REACTIONS + INTOLERANCE_REACTIONS


class AllergyMatch(Match):
    """One allergy statement or list item.

    The span covers the allergens, the reaction and, except for a list item, the cue.
    """

    status: AllergyStatus
    experiencer: Experiencer = "patient"
    cue: Match
    allergens: tuple[Match, ...] = ()
    """Raw allergen text; empty when status is "none"."""
    reaction: Match | None = None
    """None when no reaction is stated."""
    kind: AllergyKind = "unknown"
    possible: bool = False
    hypothetical: bool = False
    historical: bool = False


def _alternatives(phrases: tuple[str, ...]) -> str:
    ordered = sorted(phrases, key=len, reverse=True)
    return "|".join(r"[ \t]+".join(map(re.escape, phrase.split())) for phrase in ordered)


_B = r"(?<![A-Za-z])"
_E = r"(?![A-Za-z])"


def _run(reactions: tuple[str, ...]) -> re.Pattern[str]:
    """A reaction, or a coordinated run of them ("rash, hives or swelling")."""
    one = rf"{_B}(?:{_alternatives(reactions)}){_E}"
    return re.compile(
        rf"{one}(?:[ \t]*(?:,[ \t]*(?:(?:and|or)[ \t]+)?|(?:and|or)[ \t]+){one})*", re.IGNORECASE
    )


_REACTION_RUN = _run(REACTIONS)
_ALLERGIC_REACTION_RUN = _run(ALLERGIC_REACTIONS)
_ALLERGIC = re.compile(rf"{_B}(?:{_alternatives(ALLERGIC_REACTIONS)}){_E}", re.IGNORECASE)
_INTOLERANCE = re.compile(rf"{_B}(?:{_alternatives(INTOLERANCE_REACTIONS)}){_E}", re.IGNORECASE)
_CONNECTOR = re.compile(
    rf"{_B}(?:gives?|gave|causes?|caused|gets?|got|brings?|brought|reaction|with){_E}|[(\-–—:]",
    re.IGNORECASE,
)
_CLAUSE_BREAK = re.compile(rf"{_B}(?:but|however|although|though|whereas){_E}", re.IGNORECASE)

# Words after "allergy" that make it a service, not a statement ("allergy clinic").
_SERVICE = (
    r"(?![ \t]+(?:clinic|clinics|test|tests|testing|team|service|services|referral|nurse|"
    r"specialist|appointment|doctor)(?![A-Za-z]))"
)
_NONE = re.compile(
    rf"(?<!\?){_B}(?:nkda|nka"
    rf"|no[ \t]+known[ \t]+(?:(?:drug|medication|medicine)[ \t]+)?allerg(?:y|ies)"
    rf"|no[ \t]+(?:drug|medication|medicine)[ \t]+allerg(?:y|ies)){_E}{_SERVICE}",
    re.IGNORECASE,
)
_TO_FORM = re.compile(
    rf"{_B}allerg(?:ic|y|ies)(?P<reaction>[ \t]+reactions?)?[ \t]+to{_E}", re.IGNORECASE
)
_LABEL = re.compile(
    r"^[ \t]*(?:[#*\-•][ \t#*]*)?(?P<cue>(?:(?:drug|medication|medicine|food|known)[ \t]+)?"
    r"allerg(?:y|ies)|allergic[ \t]+to)[ \t]*\**[ \t]*:\**",
    re.IGNORECASE | re.MULTILINE,
)
_NOUN_FORM = re.compile(
    rf"{_B}allerg(?:y|ies)(?![A-Za-z])(?![ \t]*:)(?![ \t]+to{_E}){_SERVICE}", re.IGNORECASE
)
# Words before "allergy" that qualify it rather than name an allergen ("food allergy").
_NOUN_QUALIFIERS = frozenset(
    {
        "drug", "drugs", "food", "foods", "seasonal", "skin", "contact", "severe", "mild",
        "known", "true", "documented", "reported", "medication", "medicine", "multiple",
        "several", "other", "environmental", "possible", "suspected", "new", "old",
        "childhood", "adult", "family", "previous", "past", "recent", "presumed", "resolved",
    }
)  # fmt: skip
# "allergy" and "allergic" count bare only when no word follows ("no allergic reaction"
# is about one exposure, not allergies in general).
_BARE = re.compile(
    rf"{_B}(?:allergies(?![ \t]+to{_E})|allerg(?:ic|y)(?![ \t]*[A-Za-z]))(?![A-Za-z])(?![ \t]*:)",
    re.IGNORECASE,
)
_EXCEPTION = re.compile(
    rf"{_B}(?:except|apart[ \t]+from|other[ \t]+than|besides|aside[ \t]+from|but){_E}",
    re.IGNORECASE,
)
_PATIENT_TOO = re.compile(
    rf"{_B}(?:so[ \t]+(?:am|do|have)[ \t]+I|as[ \t]+am[ \t]+I|me[ \t]+too|I[ \t]+am[ \t]+too){_E}",
    re.IGNORECASE,
)
_SYMPTOM_VERB = re.compile(
    rf"{_B}(?P<cue>(?:gives?|gave|causes?|caused)[ \t]+me"
    r"|(?:brings?|brought)[ \t]+me[ \t]+out[ \t]+in"
    r"|(?:makes?|made)[ \t]+me[ \t]+come[ \t]+out[ \t]+in)"
    r"(?:[ \t]+(?:a|an))?(?:[ \t]+(?:bad|severe|nasty|terrible|awful|mild|really[ \t]+bad))?"
    rf"[ \t]+(?=(?:{_alternatives(ALLERGIC_REACTIONS)}){_E})",
    re.IGNORECASE,
)

_ITEM_WORD = re.compile(r"[ \t]*([A-Za-z0-9][\w/]*(?:-[\w/]+)*)")
_WORD = re.compile(r"[A-Za-z0-9][\w/]*(?:-[\w/]+)*")
_SEPARATOR = re.compile(r"[ \t]*(?:,(?:[ \t]*(?:and|or)(?![A-Za-z]))?|(?:and|or|&)(?![A-Za-z]))")
_PRONOUN_TAIL = re.compile(
    r"(?:,?[ \t]*(?<![A-Za-z])(?:it|that|this|they|these|those))?[ \t]*,?[ \t]*$", re.IGNORECASE
)

_DETERMINERS = frozenset({"the", "a", "an", "any", "some", "all", "my", "his", "her", "their"})
_STOP = frozenset(
    {
        "and", "or", "it", "its", "that", "this", "these", "those", "they", "which", "who",
        "since", "because", "due", "when", "but", "with", "causes", "caused", "gives", "gave",
        "makes", "made", "brings", "brought", "i", "i'm", "me", "as", "so", "is", "was", "are",
        "were", "in", "of", "for", "from", "too", "also", "think", "guess", "know", "believe",
        "say", "says", "said", "found", "get", "gets", "got", "she", "he", "we", "you", "not",
        "no", "to", "has", "have", "had", "does", "do", "did", "make", "tell", "see", "like",
        "unknown", "sure", "can", "will", "would", "should", "if", "then", "please", "other",
        "than", "except", "apart", "besides", "such", "including", "last", "ago", "summer",
        "winter", "spring", "autumn", "year", "years", "week", "weeks", "month", "months",
        "yesterday", "today", "at", "on", "during", "after", "while",
    }
)  # fmt: skip
# A later list item followed by one of these starts a new clause ("..., codeine makes me sick").
_CLAUSE_VERBS = frozenset(
    {
        "makes", "made", "make", "gives", "gave", "give", "causes", "caused", "cause", "has",
        "have", "had", "is", "was", "are", "were", "brings", "brought", "does", "did",
    }
)  # fmt: skip
# Common triggers that are not allergens ("Stress gives me hives").
_NON_ALLERGENS = frozenset(
    {
        "stress", "heat", "cold", "sun", "sunlight", "sunshine", "exercise", "walking",
        "running", "anxiety", "weather", "sweat", "sweating", "pressure", "work", "worry",
        "nerves", "water", "friction", "scratching",
    }
)  # fmt: skip
_CLASS_WORDS = frozenset(
    {
        "anything", "nothing", "everything", "medicine", "medicines", "medication",
        "medications", "drug", "drugs", "any",
    }
)  # fmt: skip
# Allergy no longer current: left to the model.
_NO_LONGER = re.compile(
    rf"{_B}(?:resolved|outgrown|outgrew|grew[ \t]+out[ \t]+of|no[ \t]+longer|used[ \t]+to){_E}",
    re.IGNORECASE,
)
_NOUN_WORD = re.compile(r"[A-Za-z0-9][\w/'’]*(?:-[\w/'’]+)*")
_MAX_ITEM_WORDS = 4
_MAX_SUBJECT_WORDS = 2
_MAX_REACTION_GAP_WORDS = 8


def find_allergies(text: str) -> tuple[AllergyMatch, ...]:
    """Find every allergy statement in `text`, left to right, without overlaps."""
    free = _free_text(text)
    return _merge(text, free, _label_regions(text, ()))


def find_note_allergies(note: Note) -> tuple[AllergyMatch, ...]:
    """Find allergies in a parsed note; offsets refer to `note.text`.

    The body of each Allergies section is read as a list, one result per item; an item
    that reads as prose ("Patient denies allergies") is read as free text instead. The
    rest of the note goes through the same finders as `find_allergies`.
    """
    text = note.text
    regions = [
        _Region(
            s.start, s.end, s.body_start, s.body_end, _span(text, s.heading_start, s.heading_end)
        )
        for s in note.sections_for(CanonicalSection.ALLERGIES)
    ]
    sections = [(region.start, region.end) for region in regions]
    regions.extend(_label_regions(text, sections))
    return _merge(text, _free_text(text), regions)


class _Region(NamedTuple):
    """A list: `start:end` covers its label or heading, `list_start:list_end` its items."""

    start: int
    end: int
    list_start: int
    list_end: int
    cue: Match


def _free_text(text: str) -> tuple[AllergyMatch, ...]:
    found: list[AllergyMatch | None] = []
    for match in _NONE.finditer(text):
        found.append(_build(text, _span(text, *match.span()), (), None, explicit_none=True))
    for match in _TO_FORM.finditer(text):
        limit = sentence_span(text, *match.span())[1]
        statement = _with_list(text, _span(text, *match.span()), match.end(), limit)
        # "no allergic reaction to the vaccine" is one exposure, not a non-allergy.
        if match.group("reaction") and statement and statement.status != "allergic":
            statement = None
        found.append(statement)
    nouns = [_noun_form(text, match) for match in _NOUN_FORM.finditer(text)]
    found.extend(nouns)
    found.extend(_symptom_form(text, match) for match in _SYMPTOM_VERB.finditer(text))
    noun_cues = {noun.cue.start for noun in nouns if noun is not None}
    for match in _BARE.finditer(text):
        if match.start() not in noun_cues:
            found.append(_build(text, _span(text, *match.span()), (), None))
    return non_overlapping(candidate for candidate in found if candidate is not None)


def _label_regions(text: str, skip: Sequence[tuple[int, int]]) -> list[_Region]:
    """Labelled lines ("Drug allergies: X"), plus the bullet lines under an empty label."""
    regions = []
    for match in _LABEL.finditer(text):
        if any(start <= match.start() < end for start, end in skip):
            continue
        line_end = text.find("\n", match.end())
        list_end = len(text) if line_end == -1 else line_end
        if not text[match.end() : list_end].strip():
            position = list_end + 1
            while position < len(text):
                next_end = text.find("\n", position)
                next_end = len(text) if next_end == -1 else next_end
                line = text[position:next_end]
                if not (_BULLET.match(line) or (line[:1] in " \t" and line.strip())):
                    break
                list_end, position = next_end, next_end + 1
        cue = _span(text, *match.span("cue"))
        regions.append(_Region(match.start(), list_end, match.end(), list_end, cue))
    return regions


def _merge(
    text: str, free: tuple[AllergyMatch, ...], regions: Sequence[_Region]
) -> tuple[AllergyMatch, ...]:
    """Free-text results outside the lists, plus each list read item by item."""
    results = [
        r
        for r in free
        if not any(r.start < region.end and region.start < r.end for region in regions)
    ]
    for region in regions:
        results.extend(_read_list(text, region, free))
    return tuple(sorted(results, key=lambda r: r.start))


def _span(text: str, start: int, end: int) -> Match:
    return Match(start=start, end=end, text=text[start:end])


def _with_list(text: str, cue: Match, position: int, limit: int) -> AllergyMatch | None:
    allergens, list_end = _parse_list(text, position, limit)
    reaction = _reaction_after(text, list_end, limit)
    return _build(text, cue, allergens, reaction, extent_end=list_end)


def _noun_form(text: str, match: re.Match[str]) -> AllergyMatch | None:
    """ "penicillin allergy": one or two allergen words straight before the noun."""
    sentence_start, sentence_end = sentence_span(text, *match.span())
    clause_start = sentence_start
    for index in range(match.start() - 1, sentence_start - 1, -1):
        if text[index] in ",;:()":
            clause_start = index + 1
            break
    for found in _CLAUSE_BREAK.finditer(text, clause_start, match.start()):
        clause_start = found.end()
    # After the noun, only "but" or "however" ends the clause, so "Penicillin allergy, no
    # longer relevant" is caught.
    clause_end = sentence_end
    following = _CLAUSE_BREAK.search(text, match.end(), clause_end)
    if following is not None:
        clause_end = following.start()
    if _NO_LONGER.search(text, clause_start, clause_end):
        return None  # "Her egg allergy has resolved": left to the model
    words = list(_NOUN_WORD.finditer(text, clause_start, match.start()))
    chosen: list[re.Match[str]] = []
    for word in reversed(words):
        lowered = word.group().lower()
        if (
            lowered in _STOP
            or lowered in _DETERMINERS
            or lowered in _NOT_ALLERGEN
            or len(chosen) == _MAX_SUBJECT_WORDS
        ):
            break
        chosen.insert(0, word)
    if not chosen or text[chosen[-1].end() : match.start()].strip():
        return None
    if any(word.group().lower() in _NOUN_QUALIFIERS | _CLASS_WORDS for word in chosen):
        return None
    allergen = _span(text, chosen[0].start(), chosen[-1].end())
    if _REACTION_RUN.search(allergen.text):
        return None
    reaction = _reaction_after(text, match.end(), sentence_end)
    return _build(text, _span(text, *match.span()), (allergen,), reaction)


def _symptom_form(text: str, match: re.Match[str]) -> AllergyMatch | None:
    sentence_start, sentence_end = sentence_span(text, *match.span())
    run = _ALLERGIC_REACTION_RUN.match(text, match.end(), sentence_end)
    if run is None:
        return None
    subject = _subject_before(text, sentence_start, match.start())
    if subject is None:
        return None
    reaction = _span(text, *run.span())
    return _build(text, _span(text, *match.span("cue")), (subject,), reaction)


def _subject_before(text: str, sentence_start: int, verb_start: int) -> Match | None:
    """The allergen before "gives me": up to two words, skipping a trailing ", it"."""
    tail = _PRONOUN_TAIL.search(text, sentence_start, verb_start)
    end = tail.start() if tail else verb_start
    clause_start = sentence_start
    for index in range(end - 1, sentence_start - 1, -1):
        if text[index] in ",;:()":
            clause_start = index + 1
            break
    words = list(_WORD.finditer(text, clause_start, end))
    chosen: list[re.Match[str]] = []
    for word in reversed(words):
        lowered = word.group().lower()
        if lowered in _STOP or lowered in _DETERMINERS or len(chosen) == _MAX_SUBJECT_WORDS:
            break
        chosen.insert(0, word)
    if not chosen or chosen[-1].end() != end:
        return None
    start, stop = chosen[0].start(), chosen[-1].end()
    lowered = " ".join(text[start:stop].lower().split())
    if (
        lowered in _CLASS_WORDS
        or lowered.split()[-1] in _NON_ALLERGENS
        or _REACTION_RUN.fullmatch(lowered)
    ):
        return None
    return _span(text, start, stop)


def _parse_list(text: str, position: int, limit: int) -> tuple[tuple[Match, ...], int]:
    """Allergens from text[position:limit]; returns them and where the list ends."""
    allergens: list[Match] = []
    end = position
    while True:
        item = _parse_item(text, position, limit)
        if item is None:
            break
        following = _ITEM_WORD.match(text, item.end, limit)
        if allergens and following and following.group(1).lower() in _CLAUSE_VERBS:
            break
        if " ".join(item.text.lower().split()) not in _CLASS_WORDS:
            allergens.append(item)
        end = item.end
        separator = _SEPARATOR.match(text, end, limit)
        if separator is None:
            break
        position = separator.end()
    return tuple(allergens), end


def _parse_item(text: str, position: int, limit: int) -> Match | None:
    words: list[tuple[int, int]] = []
    while len(words) < _MAX_ITEM_WORDS:
        match = _ITEM_WORD.match(text, position, limit)
        if match is None:
            break
        lowered = match.group(1).lower()
        if lowered in _STOP:
            break
        position = match.end()
        if not words and lowered in _DETERMINERS:
            continue
        words.append(match.span(1))
    if not words:
        return None
    return _span(text, words[0][0], words[-1][1])


def _reaction_after(text: str, position: int, limit: int) -> Match | None:
    run = _REACTION_RUN.search(text, position, limit)
    if run is None:
        return None
    between = text[position : run.start()]
    if not _CONNECTOR.search(between) or _CLAUSE_BREAK.search(between):
        return None
    if len(_WORD.findall(between)) > _MAX_REACTION_GAP_WORDS:
        return None
    return _span(text, *run.span())


def _build(
    text: str,
    cue: Match,
    allergens: tuple[Match, ...],
    reaction: Match | None,
    *,
    explicit_none: bool = False,
    extent_end: int = 0,
) -> AllergyMatch | None:
    parts = [cue, *allergens] + ([reaction] if reaction else [])
    start = min(part.start for part in parts)
    end = max(part.end for part in parts)
    sentence_start, sentence_end = sentence_span(text, start, end)
    if text[sentence_end : sentence_end + 1] == "?":
        return None
    (context,) = classify(text, [(start, cue.end)])
    rest = text[end:sentence_end]
    if context.family and _PATIENT_TOO.search(rest):
        return None
    status: AllergyStatus
    if explicit_none:
        status = "none"
    elif context.negated:
        status = "not_allergic" if allergens else "none"
    elif allergens:
        status = "allergic"
    else:
        return None
    if status == "none":
        if _EXCEPTION.search(rest):
            return None
        allergens, reaction = (), None
        end = max(cue.end, extent_end)
    # Widen the statement to the cues that shaped it ("My mother is allergic ...").
    for found in context.cues:
        start, end = min(start, found.start), max(end, found.end)
    if reaction is not None and text[end : end + 1] in ")]" and text[reaction.start - 1] in "([":
        end += 1
    return AllergyMatch(
        start=start,
        end=end,
        text=text[start:end],
        status=status,
        experiencer="family" if context.family else "patient",
        cue=cue,
        allergens=allergens,
        reaction=reaction,
        possible=context.possible,
        kind=_kind(reaction) if status != "none" else "unknown",
        hypothetical=context.hypothetical,
        historical=context.historical,
    )


def _kind(reaction: Match | None) -> AllergyKind:
    if reaction is None:
        return "unknown"
    if _ALLERGIC.search(reaction.text):
        return "allergy"
    if _INTOLERANCE.search(reaction.text):
        return "intolerance"
    return "unknown"


# Lists: Allergies section bodies and labelled lines

_PROSE = re.compile(
    rf"allerg|{_B}(?:is|was|are|were|has|have|had|gives?|gave|causes?|caused|reports?|"
    rf"reported|states?|stated|says?|said|gets?|got|patient|she|he|denies|denied){_E}",
    re.IGNORECASE,
)
_BULLET = re.compile(r"[ \t]*(?:[-*•–—]|\d{1,2}[.)])[ \t]+")
_ITEM_NONE = re.compile(
    r"(?:nkda|nka|(?:none|nil|nothing)(?:[ \t]+(?:known|reported|documented|recorded|stated))?"
    r"(?:[ \t]+(?:(?:drug|medication|medicine)[ \t]+)?allergies)?|"
    r"no[ \t]+(?:known[ \t]+)?(?:(?:drug|medication|medicine)[ \t]+)?allergies"
    r"(?:[ \t]+(?:known|reported|documented|recorded))?)",
    re.IGNORECASE,
)
_DEFERRAL = re.compile(
    r"(?:see|refer[ \t]+to)[ \t].*|as[ \t]+above|unknown|n/?a|not[ \t]+applicable|not[ \t]+"
    r"(?:known|recorded|asked|documented|checked|stated)|n/?k",
    re.IGNORECASE,
)
_POSSIBLE_BEFORE = re.compile(
    r"(?:\?[ \t]*|(?:possible|possibly|query|suspected)[ \t]+)", re.IGNORECASE
)
_POSSIBLE_AFTER = re.compile(r"[ \t]*\?+[ \t]*$")
_UNSURE = re.compile(rf"{_B}(?:unsure|not[ \t]+sure|uncertain|unclear|maybe|possibly){_E}", re.I)
_REACTION_DASH = re.compile(r"[ \t]+[-–—][ \t]+")
_REACTION_VERB = re.compile(
    r"[ \t]+(?:(?:which|that|it)[ \t]+)?(?:causes?|caused|gives?|gave)"
    r"(?:[ \t]+(?:me|her|him|them))?(?:[ \t]+an?)?[ \t]+",
    re.IGNORECASE,
)
_AND = re.compile(r"[ \t]+(?:and|&)[ \t]+", re.IGNORECASE)
_ALLERGY_NOUN = re.compile(r"[ \t]+allerg(?:y|ies)$", re.IGNORECASE)
_ALLERGEN_TEXT = re.compile(r"[A-Za-z0-9][\w/'’.-]*(?:[ \t]+[A-Za-z0-9][\w/'’.-]*)*")
# Words that show an item is a remark, not an allergen ("carries EpiPen", "reviewed today").
_NOT_ALLERGEN = frozenset(
    {
        "patient", "pt", "denies", "denied", "deny", "reports", "states", "says", "carries",
        "carry", "review", "change", "changes", "allergy", "allergies", "allergic", "reaction",
        "reactions", "history", "unclear", "unsure", "uncertain", "known", "nil", "none",
        "nothing", "nkda", "nka", "check", "ask", "above", "below", "chart", "notes", "record",
        "records", "epipen", "epi-pen", "maybe", "possibly", "query", "n/a", "na",
        "reviewed", "confirmed", "documented", "recorded", "reported", "noted", "updated",
        "checked", "discussed", "advised", "verified", "e.g.", "e.g", "eg", "i.e.", "i.e",
        "ie", "incl", "incl.",
    }
)  # fmt: skip
_STRENGTH = re.compile(
    r"\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)*(?:mg|mcg|micrograms?|g|ml|%|units?|iu)?", re.IGNORECASE
)
_REACTION_LABEL = re.compile(r"reactions?[ \t]*:[ \t]*", re.IGNORECASE)
_MAX_REACTION_WORDS = 6


def _read_list(text: str, region: _Region, free: tuple[AllergyMatch, ...]) -> list[AllergyMatch]:
    """One result per list item; prose items take the free-text results over them."""
    items: list[AllergyMatch] = []
    prose: list[tuple[int, int]] = []
    for start, end in _split_items(text, region.list_start, region.list_end):
        label = _REACTION_LABEL.match(text, start, end)
        if label is not None and items and items[-1].reaction is None and label.end() < end:
            reaction = _span(text, *_strip(text, label.end(), end))
            previous = items[-1]
            items[-1] = previous.model_copy(
                update={
                    "end": end,
                    "text": text[previous.start : end],
                    "reaction": reaction,
                    "kind": _kind(reaction),
                }
            )
            continue
        item = _list_item(text, start, end, region.cue)
        if item is not None:
            items.append(item)
        elif _PROSE.search(text, start, end):
            prose.append((start, end))
    statements = [
        r
        for r in free
        if region.list_start <= r.start
        and r.end <= region.list_end
        and any(r.start < end and start < r.end for start, end in prose)
        and not any(r.start < item.end and item.start < r.end for item in items)
    ]
    return items + statements


def _split_items(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Item spans: split on , ; new lines and full stops outside brackets, bullets removed.

    An indented line that is not a bullet continues the item above it. A piece that is
    only a reaction ("swelling" in "penicillin - rash, swelling") joins the item before.
    """
    lines: list[tuple[int, int]] = []
    previous_bulleted = False
    position = start
    while position < end:
        line_end = text.find("\n", position, end)
        line_end = end if line_end == -1 else line_end
        line = text[position:line_end]
        bullet = _BULLET.match(text, position, line_end)
        if previous_bulleted and line[:1] in " \t" and line.strip() and bullet is None:
            lines[-1] = (lines[-1][0], line_end)
        else:
            line_start = bullet.end() if bullet else position
            label = _LABEL.match(text, line_start, line_end)
            lines.append((label.end() if label else line_start, line_end))
            previous_bulleted = bullet is not None
        position = line_end + 1

    pieces: list[tuple[int, int]] = []
    for line_start, line_end in lines:
        depth = 0
        piece_start = line_start
        for index in range(line_start, line_end + 1):
            char = text[index] if index < line_end else ","
            if char in "([":
                depth += 1
            elif char in ")]":
                depth = max(0, depth - 1)
            elif depth == 0 and (char in ",;" or _is_full_stop(text, index, line_end)):
                pieces.append(_strip(text, piece_start, index))
                piece_start = index + 1

    items: list[tuple[int, int]] = []
    for piece_start, piece_end in pieces:
        if piece_start >= piece_end:
            continue
        previous = items[-1] if items else None
        if (
            previous is not None
            and _REACTION_RUN.fullmatch(text, piece_start, piece_end)
            and _has_reaction_separator(text, *previous)
        ):
            items[-1] = (previous[0], piece_end)
        else:
            items.append((piece_start, piece_end))
    return items


_LAST_WORD = re.compile(r"[A-Za-z]+$")
_ABBREVIATIONS = frozenset({"st", "dr", "mr", "mrs", "ms", "eg", "ie", "approx", "vs", "etc"})


def _is_full_stop(text: str, index: int, line_end: int) -> bool:
    if text[index] != ".":
        return False
    before = text[index - 1] if index > 0 else " "
    if before.isdigit() or not (index + 1 >= line_end or text[index + 1] in " \t"):
        return False
    word = _LAST_WORD.search(text, 0, index)
    return not (word and (len(word.group()) <= 2 or word.group().lower() in _ABBREVIATIONS))


def _has_reaction_separator(text: str, start: int, end: int) -> bool:
    if "(" in text[start:end] or "[" in text[start:end]:
        return False
    return bool(
        _REACTION_DASH.search(text, start, end)
        or _REACTION_VERB.search(text, start, end)
        or ":" in text[start:end]
    )


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in " \t\n":
        start += 1
    while end > start and text[end - 1] in " \t\n.":
        end -= 1
    return start, end


def _none_item(text: str, start: int, end: int, phrase: Match) -> AllergyMatch:
    return AllergyMatch(start=start, end=end, text=text[start:end], status="none", cue=phrase)


def _list_item(text: str, start: int, end: int, cue: Match) -> AllergyMatch | None:
    if _DEFERRAL.fullmatch(text, start, end):
        return None
    possible = False
    body_start, body_end = start, end
    before = _POSSIBLE_BEFORE.match(text, body_start, body_end)
    if before is not None:
        possible, body_start = True, before.end()
    after = _POSSIBLE_AFTER.search(text, body_start, body_end)
    if after is not None:
        possible, body_end = True, after.start()
    if _ITEM_NONE.fullmatch(text, body_start, body_end):
        if possible:
            return None  # "?NKDA" is neither an allergy nor a clear none
        return _none_item(text, start, end, _span(text, body_start, body_end))

    allergen_end, reaction_start, reaction_end = body_end, body_end, body_end
    needs_reaction_word = False
    bracket = min(
        (
            i
            for i in (text.find("(", body_start, body_end), text.find("[", body_start, body_end))
            if i != -1
        ),
        default=-1,
    )
    dash = _REACTION_DASH.search(text, body_start, body_end)
    verb = _REACTION_VERB.search(text, body_start, body_end)
    colon = text.find(":", body_start, body_end)
    if bracket != -1 and text[body_end - 1] in ")]":
        allergen_end, reaction_start, reaction_end = bracket, bracket + 1, body_end - 1
    elif dash is not None:
        allergen_end, reaction_start = dash.start(), dash.end()
    elif verb is not None:
        allergen_end, reaction_start = verb.start(), verb.end()
        needs_reaction_word = True
    elif colon != -1:
        allergen_end, reaction_start = colon, colon + 1

    allergen_start, allergen_end = _strip(text, body_start, allergen_end)
    if _ITEM_NONE.fullmatch(text, allergen_start, allergen_end):
        if possible:
            return None
        return _none_item(text, start, end, _span(text, allergen_start, allergen_end))
    allergens = _allergens(text, allergen_start, allergen_end)
    if not allergens:
        return None

    reaction: Match | None = None
    reaction_start, reaction_end = _strip(text, reaction_start, reaction_end)
    if reaction_start < reaction_end:
        before = _POSSIBLE_BEFORE.match(text, reaction_start, reaction_end)
        if before is not None:
            possible, reaction_start = True, before.end()
        after = _POSSIBLE_AFTER.search(text, reaction_start, reaction_end)
        if after is not None:
            possible, reaction_end = True, after.start()
        second_dash = _REACTION_DASH.search(text, reaction_start, reaction_end)
        if second_dash is not None:
            reaction_end = second_dash.start()  # "anaphylaxis — epipen"
        reaction_start, reaction_end = _strip(text, reaction_start, reaction_end)
        named = _REACTION_RUN.search(text, reaction_start, reaction_end)
        if _UNSURE.search(text, reaction_start, reaction_end):
            possible = True
        words = text[reaction_start:reaction_end].lower().split()
        if named is None and (
            needs_reaction_word or any(word in _NOT_ALLERGEN or word in _STOP for word in words)
        ):
            reaction = None  # a remark such as "patient unsure", not a reaction
        elif not 0 < len(_WORD.findall(text, reaction_start, reaction_end)) <= _MAX_REACTION_WORDS:
            return None
        else:
            reaction = _span(text, reaction_start, reaction_end)
    elif needs_reaction_word:
        return None

    return AllergyMatch(
        start=start,
        end=end,
        text=text[start:end],
        status="allergic",
        cue=cue,
        allergens=allergens,
        reaction=reaction,
        kind=_kind(reaction),
        possible=possible,
    )


def _allergens(text: str, start: int, end: int) -> tuple[Match, ...]:
    """Allergens in text[start:end] ("nuts and penicillin"), or () if any part is not one."""
    noun = _ALLERGY_NOUN.search(text, start, end)
    if noun is not None:
        end = noun.start()
    for dose in find_doses(text[start:end]):
        end = _strip(text, start, start + dose.start)[1]
        break
    # Cut strengths such as "30/500" after the name, keeping "Vitamin B12".
    for word in list(_WORD.finditer(text, start, end))[1:]:
        if _STRENGTH.fullmatch(word.group()):
            end = _strip(text, start, word.start())[1]
            break
    parts = []
    part_start = start
    for joiner in _AND.finditer(text, start, end):
        parts.append((part_start, joiner.start()))
        part_start = joiner.end()
    parts.append((part_start, end))
    # Split on "and" only between single words, so "Salt and vinegar crisps" stays whole.
    joined = len(parts) > 1 and any(len(text[s:e].split()) > 1 for s, e in parts)
    if joined:
        parts = [(start, end)]
    allergens = []
    for part_start, part_end in parts:
        part = text[part_start:part_end]
        words = [w for w in part.lower().split() if not (joined and w in ("and", "&"))]
        if (
            not _ALLERGEN_TEXT.fullmatch(part.replace("&", "and"))
            or len(words) > _MAX_ITEM_WORDS
            or any(word in _STOP or word in _NOT_ALLERGEN for word in words)
            or all(word in _CLASS_WORDS or word in _DETERMINERS for word in words)
            or _REACTION_RUN.search(part)
        ):
            return ()
        allergens.append(_span(text, part_start, part_end))
    return tuple(allergens)
