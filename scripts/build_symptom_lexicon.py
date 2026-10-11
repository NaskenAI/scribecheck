"""Build src/scribecheck/data/symptoms.json from the Symptom Ontology (SYMP, CC0).

Usage:
    python scripts/build_symptom_lexicon.py symp.obo \
        [--input scripts/symptom_lexicon_input.json] [--output src/scribecheck/data/symptoms.json]

The curation input lists the SYMP labels to include, SYMP surface forms to exclude
(synonyms that would mislead matching, such as "fecal incontinence" under diarrhea),
terms to merge into another that means the same for matching (convulsion into
seizure), lay phrasings to add, and phrasings withheld from matching with the reason
(recorded in symptoms.json so they still reach clinical review). IDs, labels and exact
synonyms come only from the OBO file; every lay phrasing, merge and withheld phrasing
is written with "reviewed": false. Obsolete terms are never used.

The output is deterministic: the same inputs give the same bytes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "scripts" / "symptom_lexicon_input.json"
DEFAULT_OUTPUT = ROOT / "src" / "scribecheck" / "data" / "symptoms.json"

_SYNONYM = re.compile(r'"((?:[^"\\]|\\.)*)" (EXACT|RELATED|BROAD|NARROW)')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("obo", type=Path, help="SYMP release in OBO format")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    header, terms = parse_obo(args.obo.read_text(encoding="utf-8"))
    curation = json.loads(args.input.read_text(encoding="utf-8"))
    by_label = {term["name"]: term for term in terms if not term["obsolete"]}

    problems = [label for label in curation["terms"] if label not in by_label]
    problems += [f"exclude: {label}" for label in curation["exclude"] if label not in by_label]
    problems += [f"lay: {label}" for label in curation["lay"] if label not in curation["terms"]]
    withheld = curation.get("withheld", {})
    reasons = curation.get("exclude_reasons", {})
    problems += [
        f"exclude_reasons: {form}"
        for forms in curation["exclude"].values()
        for form in forms
        if form not in reasons
    ]
    problems += [f"withheld: {label}" for label in withheld if label not in curation["terms"]]
    merge = curation.get("merge", {})
    problems += [
        f"merge: {source} -> {target}"
        for source, target in merge.items()
        if source not in curation["terms"] or target not in curation["terms"] or target in merge
    ]
    if problems:
        print("labels not found or not selected: " + ", ".join(problems), file=sys.stderr)
        return 1

    def kept(label: str) -> tuple[bool, list[str]]:
        """Whether the label matches, and the synonyms left after exclusions."""
        excluded = {form.casefold() for form in curation["exclude"].get(label, [])}
        synonyms = [s for s in by_label[label]["synonyms"] if s.casefold() not in excluded]
        return label.casefold() not in excluded, synonyms

    entries = []
    seen: dict[str, str] = {}
    targets = [label for label in curation["terms"] if label not in merge]
    for label in sorted(targets, key=lambda name: by_label[name]["id"]):
        term = by_label[label]
        match_label, synonyms = kept(label)
        merged = []
        for source in sorted(s for s, t in merge.items() if t == label):
            source_label, source_synonyms = kept(source)
            merged.append(
                {
                    "id": by_label[source]["id"],
                    "label": source,
                    "match_label": source_label,
                    "synonyms": source_synonyms,
                    "reviewed": False,
                }
            )
        forms = ([label] if match_label else []) + synonyms
        for item in merged:
            forms += ([item["label"]] if item["match_label"] else []) + item["synonyms"]
        lay = []
        for phrase in curation["lay"].get(label, []):
            if phrase.casefold() in {form.casefold() for form in forms}:
                print(f"lay phrase already in SYMP, skipped: {phrase} ({label})", file=sys.stderr)
                continue
            lay.append({"text": phrase, "reviewed": False})
        for form in forms + [item["text"] for item in lay]:
            key = form.casefold()
            if key in seen and seen[key] != label:
                print(f"'{form}' is under both {seen[key]} and {label}", file=sys.stderr)
                return 1
            seen[key] = label
        entries.append(
            {
                "id": term["id"],
                "label": label,
                "match_label": match_label,
                "synonyms": synonyms,
                "merged": merged,
                "lay": lay,
            }
        )

    output = {
        "source": {
            "name": "Symptom Ontology (SYMP)",
            "file": args.obo.name,
            "version": header.get("data-version", ""),
            "licence": "CC0 1.0",
            "url": "http://purl.obolibrary.org/obo/symp.obo",
            "built_by": "scripts/build_symptom_lexicon.py",
        },
        "reviewed_note": (
            "Labels, IDs and exact synonyms come from SYMP. The term selection, exclusions "
            "and merges are in scripts/symptom_lexicon_input.json. Lay phrasings and merges "
            'marked "reviewed": false are in-house additions awaiting clinical review.'
        ),
        "excluded_forms": {k: curation["exclude"][k] for k in sorted(curation["exclude"])},
        "excluded_reasons": {k: reasons[k] for k in sorted(reasons)},
        "merges": {k: merge[k] for k in sorted(merge)},
        "withheld": [
            {
                "text": item["text"],
                "term": label,
                "id": by_label[label]["id"],
                "note": item["note"],
                "reviewed": False,
            }
            for label in sorted(withheld)
            for item in withheld[label]
        ],
        "terms": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lay_count = sum(len(entry["lay"]) for entry in entries)
    print(f"wrote {args.output}: {len(entries)} terms, {lay_count} lay phrasings", file=sys.stderr)
    return 0


def parse_obo(text: str) -> tuple[dict[str, str], list[dict]]:
    """The header tags and every [Term] stanza (id, name, exact synonyms, obsolete flag)."""
    header: dict[str, str] = {}
    terms: list[dict] = []
    current: dict | None = None
    in_header = True
    for line in text.splitlines():
        if line.startswith("["):
            in_header = False
            current = None
            if line.strip() == "[Term]":
                current = {"id": "", "name": "", "synonyms": [], "obsolete": False}
                terms.append(current)
            continue
        tag, _, value = line.partition(": ")
        if in_header:
            if tag and value:
                header.setdefault(tag, value.strip())
        elif current is not None:
            if tag == "id":
                current["id"] = value.strip()
            elif tag == "name":
                current["name"] = value.strip()
            elif tag == "is_obsolete":
                current["obsolete"] = value.strip() == "true"
            elif tag == "synonym":
                found = _SYNONYM.match(value)
                if found and found.group(2) == "EXACT":
                    current["synonyms"].append(found.group(1).replace('\\"', '"'))
    return header, terms


if __name__ == "__main__":
    raise SystemExit(main())
