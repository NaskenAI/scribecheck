"""Drug and symptom lexicons: find mentions in text, with positions.

Data ships in `scribecheck/data/`:

- drugs.json: RxNorm Current Prescribable Content (NLM), ingredient names (IN, PIN, MIN)
  and brand names (BN) with their ingredients, built by scripts/build_drug_lexicon.py;
- inn_usan.json: international (INN/BAN) names mapped to the US names RxNorm uses
  ("paracetamol" to acetaminophen); a draft, marked "reviewed": false;
- symptoms.json: common symptoms from the Symptom Ontology (SYMP, CC0) with exact
  synonyms and lay phrasings, built by scripts/build_symptom_lexicon.py.

Matching is case-insensitive and whole-word; at each position the longest name wins.
Words are runs of letters and digits in any script, so "émetformin" and "metformin_x"
are not metformin. A hyphen and a space are interchangeable between words, so
"metformin-induced" matches metformin and "short-of-breath" matches short of breath;
a possessive "'s" may follow a name ("Augmentin's"). A name joined by a hyphen to
"free", "deficiency", "negative", "naive" or "resistant", or after "anti" or "non", is
not a mention ("pain-free", "anti-nausea", "iron-deficiency"). Multi-word names match
across spaces and hyphens only, and a multiple-ingredient name ("A / B") only across a
slash. Every drug match resolves to US ingredient (generic) names.

Spelling tolerance is conservative: one edit (insertion, deletion, substitution or swap
of adjacent letters), only for single-word ingredient and international names of 7 or
more letters (brand names must be spelt exactly), and never for a name within one edit
of another lexicon name, nor for an English word or inflection one edit from a name
("aspiring", "creating").

Some names never match on their own: brand names that are everyday words or first
names ("Align", "Julie") and ingredients that are everyday words or lab analytes
("water", "sodium"). Both lists are in drugs.json. A user brand map entry can still
add such a brand.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any, Generic, Literal, TypeVar

from scribecheck.rules._match import Match

DrugVia = Literal["ingredient", "brand", "international", "user_brand"]
SymptomVia = Literal["label", "synonym", "lay", "user"]

FUZZY_MIN_LENGTH = 7


class DrugMatch(Match):
    """A drug mention; `generic` holds its US ingredient name(s)."""

    name: str
    """The lexicon name matched, as stored ("Parlodel", "paracetamol")."""
    generic: tuple[str, ...]
    rxcui: str | None = None
    """RxNorm concept ID of the matched name; None for international and user names."""
    via: DrugVia
    fuzzy: bool = False
    """True when matched with one spelling edit."""


class SymptomMatch(Match):
    """A symptom mention, normalised to its lexicon term."""

    term: str
    symp_id: str | None
    """The SYMP ID, or None for a user term outside the lexicon."""
    via: SymptomVia


def find_drugs(
    text: str, *, brands: Mapping[str, Sequence[str]] | None = None
) -> tuple[DrugMatch, ...]:
    """Find drug mentions in `text`, left to right.

    `brands` is a user brand map: brand name -> generic names, each an RxNorm ingredient
    or an international name from inn_usan.json. Raises ValueError for a generic name
    the lexicon does not know.
    """
    index = _drug_index()
    if brands:
        index = index.with_user_brands(brands)
    results = []
    for start, end, entry, fuzzy in index.scan(text):
        results.append(
            DrugMatch(
                start=start,
                end=end,
                text=text[start:end],
                name=entry.name,
                generic=entry.generic,
                rxcui=entry.rxcui,
                via=entry.via,
                fuzzy=fuzzy,
            )
        )
    return tuple(results)


def find_symptoms(text: str, *, extra: Mapping[str, str] | None = None) -> tuple[SymptomMatch, ...]:
    """Find symptom mentions in `text`, left to right.

    `extra` adds user phrasings: phrase -> term. A term that is a lexicon label keeps
    its SYMP ID; any other term is new and has no ID.
    """
    index = _symptom_index()
    if extra:
        labels = {entry.term.casefold(): entry for entry in index.entries.values()}
        additions = {}
        for phrase, term in extra.items():
            known = labels.get(term.casefold())
            additions[phrase] = _Symptom(
                known.term if known else term, known.symp_id if known else None, "user"
            )
        index = index.extended(additions.items())
    return tuple(
        SymptomMatch(
            start=start,
            end=end,
            text=text[start:end],
            term=entry.term,
            symp_id=entry.symp_id,
            via=entry.via,
        )
        for start, end, entry, _ in index.scan(text)
    )


def within_one_edit(a: str, b: str) -> bool:
    """True when a and b differ by one insertion, deletion, substitution or adjacent swap."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        return (
            len(diffs) == 2
            and diffs[1] == diffs[0] + 1
            and a[diffs[0]] == b[diffs[1]]
            and a[diffs[1]] == b[diffs[0]]
        )
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1 :]


# Tokens and keys

_TOKEN = re.compile(r"\w+(?:['’]\w+)*")
_POSSESSIVE = re.compile(r"['’]s$", re.IGNORECASE)
_SLASH_GAP = re.compile(r"[ \t]*/[ \t]*")
_SPACE_GAP = re.compile(r"[ \t]+")
_HYPHEN_GAP = re.compile(r"[-\u2010\u2011]")

Key = tuple[str, ...]
"""Normalised tokens alternating with gap kinds: ("sulfamethoxazole", "/", "trimethoprim")."""


def _norm(token: str) -> str:
    return token.casefold().replace("’", "'")


# A mention joined by a hyphen to one of these words is not a mention.
_HYPHEN_SUFFIXES = frozenset({"free", "deficiency", "deficient", "negative", "naive", "resistant"})
_HYPHEN_PREFIXES = frozenset({"anti", "non"})


def _tokens(text: str) -> list[tuple[int, int, str]]:
    """(start, end, normalised) for each word."""
    return [(m.start(), m.end(), _norm(m.group())) for m in _TOKEN.finditer(text)]


def _without_possessive(token: str) -> str | None:
    return token[:-2] if len(token) > 2 and _POSSESSIVE.search(token) else None


def _gap(text: str) -> str | None:
    if _SPACE_GAP.fullmatch(text) or _HYPHEN_GAP.fullmatch(text):
        return " "
    if _SLASH_GAP.fullmatch(text):
        return "/"
    return None


def _key(name: str) -> Key | None:
    """The match key for a lexicon name, or None if it has punctuation between words."""
    tokens = _tokens(name)
    if not tokens or name[: tokens[0][0]].strip() or name[tokens[-1][1] :].strip():
        return None
    key: list[str] = [tokens[0][2]]
    for previous, token in zip(tokens, tokens[1:], strict=False):
        gap = _gap(name[previous[1] : token[0]])
        if gap is None:
            return None
        key += [gap, token[2]]
    return tuple(key)


def _deletes(word: str) -> set[str]:
    return {word} | {word[:i] + word[i + 1 :] for i in range(len(word))}


E = TypeVar("E")


class _Index(Generic[E]):
    """Lexicon keys to entries, with optional one-edit matching of single words."""

    def __init__(
        self,
        entries: dict[Key, E],
        *,
        fuzzy: Callable[[E], bool] | None = None,
        fuzzy_blocked: frozenset[str] = frozenset(),
    ) -> None:
        self.entries = entries
        self.fuzzy = fuzzy
        """Which entries may match with one spelling edit; None turns tolerance off."""
        self.fuzzy_blocked = fuzzy_blocked
        self.longest: dict[str, int] = {}
        for key in entries:
            self.longest[key[0]] = max(self.longest.get(key[0], 1), (len(key) + 1) // 2)
        self._neighbours: dict[str, set[str]] | None = None
        self._ambiguous: set[str] = set()
        self._longest: int | None = None

    def extended(self, additions: Iterable[tuple[str, E]]) -> _Index[E]:
        entries = dict(self.entries)
        for name, entry in additions:
            key = _key(name)
            if key is None:
                raise ValueError(f"cannot match {name!r}: use words separated by spaces")
            entries[key] = entry
        return _Index(entries, fuzzy=self.fuzzy, fuzzy_blocked=self.fuzzy_blocked)

    def scan(self, text: str) -> list[tuple[int, int, E, bool]]:
        tokens = _tokens(text)
        between = [text[a[1] : b[0]] for a, b in zip(tokens, tokens[1:], strict=False)]
        gaps = [_gap(gap) for gap in between]
        hyphens = [bool(_HYPHEN_GAP.fullmatch(gap)) for gap in between]

        def compound(first: int, last: int) -> bool:
            """Whether tokens first..last are joined by a hyphen to a negating word."""
            before = first > 0 and hyphens[first - 1] and tokens[first - 1][2] in _HYPHEN_PREFIXES
            after = (
                last + 1 < len(tokens) and hyphens[last] and tokens[last + 1][2] in _HYPHEN_SUFFIXES
            )
            return before or after

        found: list[tuple[int, int, E, bool]] = []
        i = 0
        while i < len(tokens):
            matched = self._match_at(tokens, gaps, i)
            if matched is not None:
                length, key, possessive = matched
                last = i + length - 1
                if not compound(i, last):
                    end = tokens[last][1] - (2 if possessive else 0)
                    found.append((tokens[i][0], end, self.entries[key], False))
                i += length
                continue
            entry = self._fuzzy(tokens[i][2]) if self.fuzzy is not None else None
            if entry is not None and not compound(i, i):
                found.append((tokens[i][0], tokens[i][1], entry, True))
            i += 1
        return found

    def _key_at(
        self, tokens: list[tuple[int, int, str]], gaps: list[str | None], i: int, length: int
    ) -> Key | None:
        key = [tokens[i][2]]
        for j in range(i, i + length - 1):
            if gaps[j] is None:
                return None
            key += [gaps[j], tokens[j + 1][2]]  # type: ignore[list-item]
        return tuple(key)

    def _match_at(
        self, tokens: list[tuple[int, int, str]], gaps: list[str | None], i: int
    ) -> tuple[int, Key, bool] | None:
        """(length, key, whether a possessive "'s" follows) of the longest name at i."""
        first = tokens[i][2]
        stripped_first = _without_possessive(first)
        longest = max(self.longest.get(first, 0), self.longest.get(stripped_first or "", 0))
        for length in range(min(longest, len(tokens) - i), 0, -1):
            key = self._key_at(tokens, gaps, i, length)
            if key is None:
                continue
            if key in self.entries:
                return length, key, False
            stripped = _without_possessive(key[-1])
            if stripped is not None and key[:-1] + (stripped,) in self.entries:
                return length, key[:-1] + (stripped,), True
        return None

    def _fuzzy(self, word: str) -> E | None:
        # A name of 7+ letters is one edit from words of 6 letters up to its length + 1.
        if not FUZZY_MIN_LENGTH - 1 <= len(word) <= self._longest_name + 1:
            return None
        if word in self.fuzzy_blocked or not (word.isascii() and word.isalpha()):
            return None
        neighbours = self._fuzzy_neighbours()
        candidates = set().union(*(neighbours.get(d, set()) for d in _deletes(word)))
        close = [name for name in candidates if within_one_edit(word, name)]
        if len(close) != 1 or close[0] in self._ambiguous:
            return None
        return self.entries[(close[0],)]

    @property
    def _longest_name(self) -> int:
        if self._longest is None:
            self._longest = max((len(key[0]) for key in self.entries if len(key) == 1), default=0)
        return self._longest

    def _fuzzy_neighbours(self) -> dict[str, set[str]]:
        """Deletion index of single-word names of 7+ letters; finds ambiguous names once."""
        if self._neighbours is None:
            singles = [key[0] for key in self.entries if len(key) == 1]
            every: dict[str, set[str]] = {}
            for name in singles:
                for d in _deletes(name):
                    every.setdefault(d, set()).add(name)
            for name in singles:
                close = set().union(*(every[d] for d in _deletes(name)))
                if any(within_one_edit(name, other) for other in close):
                    self._ambiguous.add(name)
            assert self.fuzzy is not None
            tolerant = {
                name
                for name in singles
                if len(name) >= FUZZY_MIN_LENGTH and self.fuzzy(self.entries[(name,)])
            }
            self._neighbours = {d: names & tolerant for d, names in every.items()}
        return self._neighbours


# Drug data


@dataclass(frozen=True)
class _Drug:
    name: str
    generic: tuple[str, ...]
    rxcui: str | None
    via: DrugVia


def _spelling_tolerant(entry: _Drug) -> bool:
    return entry.via in ("ingredient", "international")


class _DrugIndex(_Index[_Drug]):
    def __init__(
        self,
        entries: dict[Key, _Drug],
        generics: dict[str, tuple[str, ...]],
        fuzzy_blocked: frozenset[str],
    ) -> None:
        super().__init__(entries, fuzzy=_spelling_tolerant, fuzzy_blocked=fuzzy_blocked)
        self.generics = generics
        """Ingredient and international names (casefolded) -> US generic names."""

    def with_user_brands(self, brands: Mapping[str, Sequence[str]]) -> _DrugIndex:
        entries = dict(self.entries)
        for brand, names in brands.items():
            if not names:
                raise ValueError(f"brand {brand!r}: no generic names given")
            existing = entries.get(_key(brand) or ())
            if existing is not None and existing.via in ("ingredient", "international"):
                raise ValueError(f"brand {brand!r} is already a generic name")
            generic: list[str] = []
            for name in names:
                resolved = self.generics.get(name.casefold())
                if resolved is None:
                    raise ValueError(f"brand {brand!r}: unknown generic name {name!r}")
                generic.extend(resolved)
            key = _key(brand)
            if key is None:
                raise ValueError(f"cannot match brand {brand!r}: use words separated by spaces")
            entries[key] = _Drug(brand, tuple(sorted(set(generic))), None, "user_brand")
        return _DrugIndex(entries, self.generics, self.fuzzy_blocked)


def _read(name: str) -> Any:
    path = resources.files("scribecheck").joinpath("data").joinpath(name)
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def _drug_index() -> _DrugIndex:
    drugs = _read("drugs.json")
    names = {row[0]: row[1] for row in drugs["ingredients"]}
    blocked_ingredients = {name.casefold() for name in drugs["blocked_ingredients"]}
    blocked_brands = {name.casefold() for name in drugs["blocked_brands"]}

    entries: dict[Key, _Drug] = {}
    generics: dict[str, tuple[str, ...]] = {}
    for rxcui, name, tty, base in drugs["ingredients"]:
        if tty == "IN":
            generic: tuple[str, ...] = (name,)
        elif tty == "PIN":
            generic = (names[base],) if base else (name,)
        else:
            generic = tuple(sorted(names[part] for part in base))
        if tty != "MIN":
            generics.setdefault(name.casefold(), generic)
        key = _key(name)
        if key is not None and name.casefold() not in blocked_ingredients:
            entries.setdefault(key, _Drug(name, generic, rxcui, "ingredient"))

    for pair in _read("inn_usan.json")["pairs"]:
        generic = tuple(sorted({g for us in pair["us"] for g in generics[us.casefold()]}))
        international = pair["international"]
        generics.setdefault(international.casefold(), generic)
        key = _key(international)
        if key is not None:
            entries.setdefault(key, _Drug(international, generic, None, "international"))

    for rxcui, name, ingredients in drugs["brands"]:
        key = _key(name)
        if key is None or name.casefold() in blocked_brands:
            continue
        generic = tuple(sorted(names[i] for i in ingredients))
        entries.setdefault(key, _Drug(name, generic, rxcui, "brand"))

    # Blocked names never match, not even as a near spelling of another name.
    never = blocked_ingredients | blocked_brands
    return _DrugIndex(entries, generics, frozenset(drugs["fuzzy_blocked"]) | never)


# Symptom data


@dataclass(frozen=True)
class _Symptom:
    term: str
    symp_id: str | None
    via: SymptomVia


@cache
def _symptom_index() -> _Index[_Symptom]:
    entries: dict[Key, _Symptom] = {}
    for term in _read("symptoms.json")["terms"]:
        forms: list[tuple[str, SymptomVia]] = []
        if term["match_label"]:
            forms.append((term["label"], "label"))
        forms += [(synonym, "synonym") for synonym in term["synonyms"]]
        for merged in term["merged"]:
            if merged["match_label"]:
                forms.append((merged["label"], "synonym"))
            forms += [(synonym, "synonym") for synonym in merged["synonyms"]]
        forms += [(lay["text"], "lay") for lay in term["lay"]]
        for form, via in forms:
            key = _key(form)
            if key is not None:
                entries.setdefault(key, _Symptom(term["label"], term["id"], via))
    return _Index(entries)
