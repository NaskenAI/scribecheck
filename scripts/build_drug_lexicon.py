"""Build src/scribecheck/data/drugs.json from an RxNorm Current Prescribable Content zip.

Usage:
    python scripts/build_drug_lexicon.py RxNorm_full_prescribe_MMDDYYYY.zip \
        [--english-words /usr/share/dict/words] [--output src/scribecheck/data/drugs.json]

Reads rrf/RXNCONSO.RRF and rrf/RXNREL.RRF straight from the zip (nothing is extracted)
and keeps SAB=RXNORM ingredient names (TTY IN, PIN, MIN) and brand names (TTY BN), with
these links:

- brand to ingredient (RELA has_tradename: the IN "has_tradename" the BN);
- precise ingredient to ingredient (RELA form_of: the PIN is a "form_of" the IN);
- multiple ingredient to its parts (RELA has_part: the MIN "has_part" the IN or PIN).

Three reviewable lists in the output keep matching precise:

- blocked_brands: never matched. A curated list (below) of single-word brand names
  that are everyday words or first names ("Align", "Julie"), plus every multi-word
  brand made only of everyday words ("My Choice", "Morning After") unless a word is a
  single letter or hyphenated ("Plan B", "Gas-X"). Brand names that only collide with
  rare dictionary words ("Flector") are kept; the script prints them for review.
- blocked_ingredients: a curated list (below) of single-word ingredient names that are
  everyday words or lab analytes ("water", "sodium 135", "amylase level"), never
  matched on their own.
- fuzzy_blocked: English words, including their inflected forms ("appeared",
  "creating"), that are one edit from a name the matcher spells tolerantly (ingredient,
  international and extra ingredient names of 7+ letters), never matched by spelling
  tolerance. The international and extra names are read from inn_usan.json and
  ingredients_extra.json.

The output is deterministic: the same zip and wordlist give the same bytes. Run it with
the package installed (`pip install -e .`); it reuses the matcher's edit check.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from scribecheck.rules.lexicon import FUZZY_MIN_LENGTH, within_one_edit

DATA = Path(__file__).resolve().parents[1] / "src" / "scribecheck" / "data"
DEFAULT_OUTPUT = DATA / "drugs.json"
INN_USAN = DATA / "inn_usan.json"
EXTRA_INGREDIENTS = DATA / "ingredients_extra.json"
KEPT_TTYS = frozenset({"IN", "PIN", "MIN", "BN"})

# Brand names that are everyday words, first names, or hyphenated phrases that read as
# ordinary words once a hyphen counts as a space ("bedside care").
BLOCKED_BRANDS = (
    "acne-clear", "align", "alli", "bedside-care", "dragon", "ella", "halls", "julie",
    "kirsty", "muse", "perform", "pronto", "react", "rid", "senna-time", "sojourn", "stye",
    "tabloid", "tandem", "thrive", "tobi", "tucks", "vanquish",
)  # fmt: skip

# Single-word ingredient names that are everyday words or lab analytes. Matching them
# alone would read "drink water" or "sodium 135" as a medicine. Multi-word names that
# contain them ("potassium chloride", "sodium bicarbonate") still match.
BLOCKED_INGREDIENTS = (
    "air", "amber", "ammonia", "amylase", "beeswax", "bran", "caffeine", "calcium",
    "cholesterol", "cocaine", "collagen", "copper", "creatine", "creatinine", "ethanol",
    "fibrinogen", "glucose", "hemp", "honey", "indigo", "lactate", "lactose", "licorice",
    "lipase", "magnesium", "mica", "molasses", "nitrogen", "perlite", "potassium",
    "prothrombin", "pumice", "rosin", "silver", "sodium", "starch", "talc", "urea", "water",
)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("zip", type=Path, help="RxNorm Current Prescribable Content zip")
    parser.add_argument("--english-words", type=Path, default=Path("/usr/share/dict/words"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    with zipfile.ZipFile(args.zip) as archive:
        release = _release_date(archive)
        concepts = dict(_concepts(archive))
        links = _links(archive, concepts)
    words, common = _english_words(args.english_words)
    international = [p["international"] for p in json.loads(INN_USAN.read_text())["pairs"]]
    international += [e["name"] for e in json.loads(EXTRA_INGREDIENTS.read_text())["ingredients"]]
    lexicon = _lexicon(concepts, links, words, common, international)
    lexicon["source"] = {
        "name": "RxNorm Current Prescribable Content",
        "publisher": "U.S. National Library of Medicine (NLM)",
        "file": args.zip.name,
        "release_date": release,
        "licence": "Public domain; no UMLS licence needed for the prescribable subset.",
        "built_by": "scripts/build_drug_lexicon.py",
        "english_words": {
            "file": str(args.english_words),
            "sha256": hashlib.sha256(args.english_words.read_bytes()).hexdigest(),
            "entries": len(words),
        },
    }
    ordered = {key: lexicon[key] for key in _KEY_ORDER}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_dump(ordered), encoding="utf-8")
    counts = {key: len(value) for key, value in ordered.items() if isinstance(value, list)}
    print(f"wrote {args.output} from {args.zip.name} ({release}): {counts}", file=sys.stderr)
    return 0


_KEY_ORDER = (
    "source",
    "ingredients",
    "brands",
    "blocked_brands",
    "blocked_ingredients",
    "fuzzy_blocked",
)


def _release_date(archive: zipfile.ZipFile) -> str:
    readme = next(n for n in archive.namelist() if n.lower().startswith("readme"))
    first = archive.read(readme).decode("utf-8").strip().splitlines()[0].strip()
    return datetime.strptime(first, "%B %d, %Y").date().isoformat()


def _rows(archive: zipfile.ZipFile, name: str) -> Iterator[list[str]]:
    with archive.open(name) as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            yield line.rstrip("\n").split("|")


def _concepts(archive: zipfile.ZipFile) -> Iterator[tuple[str, tuple[str, str]]]:
    """(rxcui, (tty, name)) for the kept term types."""
    for row in _rows(archive, "rrf/RXNCONSO.RRF"):
        if row[11] == "RXNORM" and row[12] in KEPT_TTYS and row[16] == "N":
            yield row[0], (row[12], row[14])


def _links(
    archive: zipfile.ZipFile, concepts: dict[str, tuple[str, str]]
) -> dict[str, dict[str, set[str]]]:
    """Links by kind: brand -> ingredients, pin -> base, min -> parts."""
    links: dict[str, dict[str, set[str]]] = {
        "brand": defaultdict(set),
        "base": defaultdict(set),
        "parts": defaultdict(set),
    }
    for row in _rows(archive, "rrf/RXNREL.RRF"):
        if row[10] != "RXNORM":
            continue
        first, second, rela = row[0], row[4], row[7]
        if first not in concepts or second not in concepts:
            continue
        first_tty, second_tty = concepts[first][0], concepts[second][0]
        # RXNREL reads "RXCUI2 <RELA> RXCUI1".
        if rela == "has_tradename" and first_tty == "BN" and second_tty == "IN":
            links["brand"][first].add(second)
        elif rela == "form_of" and first_tty == "IN" and second_tty == "PIN":
            links["base"][second].add(first)
        elif rela == "has_part" and first_tty in ("IN", "PIN") and second_tty == "MIN":
            links["parts"][second].add(first)
    return links


def _english_words(path: Path) -> tuple[frozenset[str], frozenset[str]]:
    """All words (casefolded), and the common ones: entries written in lowercase."""
    entries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    words = frozenset(entry.casefold() for entry in entries if entry)
    common = frozenset(entry for entry in entries if entry and entry == entry.lower())
    return words, common


def _lexicon(
    concepts: dict[str, tuple[str, str]],
    links: dict[str, dict[str, set[str]]],
    words: frozenset[str],
    common: frozenset[str],
    international: list[str],
) -> dict[str, object]:
    def base_of(rxcui: str) -> str | None:
        bases = links["base"].get(rxcui, set())
        return min(bases, key=int) if len(bases) == 1 else None

    ingredients = []
    for rxcui, (tty, name) in sorted(concepts.items(), key=lambda item: int(item[0])):
        if tty == "IN":
            ingredients.append([rxcui, name, tty, None])
        elif tty == "PIN":
            ingredients.append([rxcui, name, tty, base_of(rxcui)])
        elif tty == "MIN":
            parts = sorted(
                {base_of(p) or p if concepts[p][0] == "PIN" else p for p in links["parts"][rxcui]},
                key=int,
            )
            ingredients.append([rxcui, name, tty, parts])

    brands = []
    blocked_brands = []
    for rxcui, (tty, name) in sorted(concepts.items(), key=lambda item: int(item[0])):
        if tty != "BN":
            continue
        brands.append([rxcui, name, sorted(links["brand"][rxcui], key=int)])
        if name.casefold() in BLOCKED_BRANDS or _everyday_phrase(name, common):
            blocked_brands.append(name)
        elif name.casefold() in common:
            print(f"wordlist collision kept: {name}", file=sys.stderr)

    names = {name.casefold() for _, (_, name) in concepts.items()}
    blocked_ingredients = sorted(n for n in BLOCKED_INGREDIENTS if n in names)
    tolerant = {name.casefold() for _, (tty, name) in concepts.items() if tty in ("IN", "PIN")} | {
        name.casefold() for name in international
    }
    return {
        "ingredients": ingredients,
        "brands": brands,
        "blocked_brands": sorted(blocked_brands, key=str.casefold),
        "blocked_ingredients": blocked_ingredients,
        "fuzzy_blocked": _fuzzy_blocked(tolerant, names, words),
    }


def _everyday_phrase(name: str, common: frozenset[str]) -> bool:
    """A name of two or more everyday words, none a single letter or hyphenated.

    "My Choice" qualifies; "Plan B One-Step" does not. Hyphenated coinages ("Nitro-Bid",
    "Slow-Fe") are kept; the few that read as phrases are in BLOCKED_BRANDS.
    """
    words = name.split()
    return len(words) > 1 and all(
        len(word) > 1 and "-" not in word and word.casefold() in common for word in words
    )


_SINGLE_WORD = re.compile(r"[a-z]+")


def _fuzzy_blocked(tolerant: set[str], names: set[str], words: frozenset[str]) -> list[str]:
    """English words and their inflections one edit from a spelling-tolerant name."""
    targets = {n for n in tolerant if len(n) >= FUZZY_MIN_LENGTH and _SINGLE_WORD.fullmatch(n)}
    index: dict[str, set[str]] = defaultdict(set)
    for name in targets:
        for key in _deletes(name):
            index[key].add(name)
    blocked = set()
    for form in {form for word in words for form in _inflections(word)}:
        if len(form) < FUZZY_MIN_LENGTH - 1 or form in names or not _SINGLE_WORD.fullmatch(form):
            continue
        candidates = set().union(*(index.get(key, set()) for key in _deletes(form)))
        if any(within_one_edit(form, name) for name in candidates):
            blocked.add(form)
    return sorted(blocked)


def _inflections(word: str) -> set[str]:
    """The word with common English endings, since the wordlist lists base forms only."""
    forms = {word, word + "s", word + "es", word + "ed", word + "d", word + "ing", word + "er"}
    forms |= {word + "ers", word + "ly", word + "est"}
    if word.endswith("e"):
        forms |= {word[:-1] + "ing", word[:-1] + "ed"}
    if word.endswith("y"):
        forms |= {word[:-1] + "ies", word[:-1] + "ied"}
    if len(word) > 2 and word[-1] not in "aeiouwxy" and word[-2] in "aeiou":
        forms |= {word + word[-1] + "ing", word + word[-1] + "ed"}
    return forms


def _deletes(word: str) -> set[str]:
    return {word} | {word[:i] + word[i + 1 :] for i in range(len(word))}


def _dump(data: dict[str, object]) -> str:
    """JSON with one list item per line, so diffs between releases stay readable."""
    lines = ["{"]
    keys = list(data)
    for position, key in enumerate(keys):
        value = data[key]
        comma = "," if position < len(keys) - 1 else ""
        if isinstance(value, list):
            lines.append(f"  {json.dumps(key)}: [")
            for index, item in enumerate(value):
                tail = "," if index < len(value) - 1 else ""
                lines.append(f"    {json.dumps(item, ensure_ascii=False)}{tail}")
            lines.append(f"  ]{comma}")
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
