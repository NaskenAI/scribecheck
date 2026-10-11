"""Allergy statements: allergen, reaction, or an explicit none.

Forms handled:

- "allergic to X", "allergy to X", "allergies to X", with lists ("X and Y", "X, Y");
- "X gives me a rash", "X caused me hives", "X brings me out in a rash", which need
  a skin or anaphylactic reaction from `SKIN_REACTIONS` (so "Metformin gives me
  diarrhoea", a side effect, is not an allergy) and a subject that is not a common
  non-allergen trigger ("Stress gives me hives");
- a label line, "Allergies: X (rash)", "Allergy: none";
- explicit none: NKDA, NKA, "no known (drug) allergies", "no drug allergies", and any
  negated allergy word without an allergen ("no allergies", "not allergic to anything").

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
from typing import Literal

from scribecheck.rules._context import sentence_span
from scribecheck.rules._match import Match, non_overlapping
from scribecheck.rules.context import classify

AllergyStatus = Literal["allergic", "not_allergic", "none"]
""""none" is an explicit statement of no allergies."""
Experiencer = Literal["patient", "family"]

REACTIONS: tuple[str, ...] = (
    "rash",
    "rashes",
    "hives",
    "urticaria",
    "swelling",
    "itching",
    "itch",
    "itchiness",
    "anaphylaxis",
    "anaphylactic shock",
    "anaphylactic reaction",
    "angioedema",
    "wheeze",
    "wheezing",
    "difficulty breathing",
    "trouble breathing",
    "breathing problems",
    "shortness of breath",
    "vomiting",
    "nausea",
    "diarrhoea",
    "diarrhea",
    "blisters",
    "blistering",
)
SKIN_REACTIONS: tuple[str, ...] = (
    "rash",
    "rashes",
    "hives",
    "urticaria",
    "swelling",
    "itching",
    "itch",
    "itchiness",
    "anaphylaxis",
    "anaphylactic shock",
    "anaphylactic reaction",
    "angioedema",
    "blisters",
    "blistering",
)
""""X gives me ..." counts as an allergy only with one of these reactions."""


class AllergyMatch(Match):
    """One allergy statement; the span covers the cue, allergens and reaction."""

    status: AllergyStatus
    experiencer: Experiencer = "patient"
    cue: Match
    allergens: tuple[Match, ...] = ()
    """Raw allergen text; empty when status is "none"."""
    reaction: Match | None = None
    """None when no reaction is stated."""
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
_SKIN_REACTION_RUN = _run(SKIN_REACTIONS)
_CONNECTOR = re.compile(
    rf"{_B}(?:gives?|gave|causes?|caused|gets?|got|brings?|brought|reaction|with){_E}|[(\-–—:]",
    re.IGNORECASE,
)
_CLAUSE_BREAK = re.compile(rf"{_B}(?:but|however|although|though|whereas){_E}", re.IGNORECASE)

_NONE = re.compile(
    rf"{_B}(?:nkda|nka|no[ \t]+known[ \t]+(?:(?:drug|medication|medicine)[ \t]+)?allerg(?:y|ies)"
    rf"|no[ \t]+(?:drug|medication|medicine)[ \t]+allerg(?:y|ies)){_E}",
    re.IGNORECASE,
)
_TO_FORM = re.compile(rf"{_B}allerg(?:ic|y|ies)[ \t]+to{_E}", re.IGNORECASE)
_LABEL = re.compile(
    r"^[ \t]*(?:[#*\-][ \t#*]*)?(?P<cue>allerg(?:y|ies))[ \t]*\**[ \t]*:\**",
    re.IGNORECASE | re.MULTILINE,
)
_LABEL_NONE = re.compile(rf"[ \t]*(?P<cue>none(?:[ \t]+known)?|nil(?:[ \t]+known)?){_E}", re.I)
_LABEL_DEFERS = re.compile(rf"[ \t]*(?:nkda|nka|no){_E}", re.IGNORECASE)
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
    rf"[ \t]+(?=(?:{_alternatives(SKIN_REACTIONS)}){_E})",
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
        "than", "except", "apart", "besides", "such", "including",
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
_MAX_ITEM_WORDS = 4
_MAX_SUBJECT_WORDS = 2
_MAX_REACTION_GAP_WORDS = 8


def find_allergies(text: str) -> tuple[AllergyMatch, ...]:
    """Find every allergy statement in `text`, left to right, without overlaps."""
    found: list[AllergyMatch | None] = []
    for match in _NONE.finditer(text):
        found.append(_build(text, _span(text, *match.span()), (), None, explicit_none=True))
    for match in _TO_FORM.finditer(text):
        limit = sentence_span(text, *match.span())[1]
        found.append(_with_list(text, _span(text, *match.span()), match.end(), limit))
    found.extend(_label(text, match) for match in _LABEL.finditer(text))
    found.extend(_symptom_form(text, match) for match in _SYMPTOM_VERB.finditer(text))
    for match in _BARE.finditer(text):
        found.append(_build(text, _span(text, *match.span()), (), None))
    return non_overlapping(candidate for candidate in found if candidate is not None)


def _span(text: str, start: int, end: int) -> Match:
    return Match(start=start, end=end, text=text[start:end])


def _with_list(text: str, cue: Match, position: int, limit: int) -> AllergyMatch | None:
    allergens, list_end = _parse_list(text, position, limit)
    reaction = _reaction_after(text, list_end, limit)
    return _build(text, cue, allergens, reaction, extent_end=list_end)


def _label(text: str, match: re.Match[str]) -> AllergyMatch | None:
    line_end = text.find("\n", match.end())
    limit = len(text) if line_end == -1 else line_end
    none = _LABEL_NONE.match(text, match.end(), limit)
    if none is not None:
        cue = _span(text, *none.span("cue"))
        return _build(text, cue, (), None, explicit_none=True)
    if _LABEL_DEFERS.match(text, match.end(), limit):
        return None  # NKDA and "no allergies" are found by their own patterns
    return _with_list(text, _span(text, *match.span("cue")), match.end(), limit)


def _symptom_form(text: str, match: re.Match[str]) -> AllergyMatch | None:
    sentence_start, sentence_end = sentence_span(text, *match.span())
    run = _SKIN_REACTION_RUN.match(text, match.end(), sentence_end)
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
        hypothetical=context.hypothetical,
        historical=context.historical,
    )
