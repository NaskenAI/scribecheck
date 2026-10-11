"""The packaged lexicon data: provenance, integrity and review flags."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

import pytest


def _load(name: str) -> Any:
    path = resources.files("scribecheck").joinpath("data").joinpath(name)
    assert path.is_file(), f"{name} is not packaged"
    return json.loads(path.read_text(encoding="utf-8"))


DRUGS = _load("drugs.json")
INN = _load("inn_usan.json")
SYMPTOMS = _load("symptoms.json")
TEMPLATE = _load("brands_user_template.json")
INGREDIENT_NAMES = {row[1].casefold() for row in DRUGS["ingredients"] if row[2] in ("IN", "PIN")}


def test_drug_source_is_recorded() -> None:
    source = DRUGS["source"]
    assert source["file"] == "RxNorm_full_prescribe_10052026.zip"
    assert source["release_date"] == "2026-10-05"
    assert "National Library of Medicine" in source["publisher"]


def test_every_brand_and_combination_links_to_known_ingredients() -> None:
    ids = {row[0] for row in DRUGS["ingredients"]}
    for _, name, ingredients in DRUGS["brands"]:
        assert ingredients, name
        assert set(ingredients) <= ids, name
    for _rxcui, name, tty, base in DRUGS["ingredients"]:
        if tty == "MIN":
            assert base and set(base) <= ids, name
        elif tty == "PIN" and base is not None:
            assert base in ids, name


def test_blocked_lists_name_real_entries() -> None:
    brands = {row[1] for row in DRUGS["brands"]}
    assert set(DRUGS["blocked_brands"]) <= brands
    assert set(DRUGS["blocked_ingredients"]) <= INGREDIENT_NAMES


def test_inn_map_is_marked_unreviewed_and_targets_exist() -> None:
    assert INN["reviewed"] is False
    for pair in INN["pairs"]:
        assert pair["international"].casefold() not in INGREDIENT_NAMES, pair
        for us in pair["us"]:
            assert us.casefold() in INGREDIENT_NAMES, pair


def test_brand_template_has_only_invented_examples() -> None:
    real = {row[1].casefold() for row in DRUGS["brands"]}
    assert set(TEMPLATE["brands"]) == set(TEMPLATE["_examples"])
    for brand in TEMPLATE["brands"]:
        assert brand.casefold() not in real


def test_symptom_source_and_ids() -> None:
    assert SYMPTOMS["source"]["licence"] == "CC0 1.0"
    assert SYMPTOMS["source"]["version"]
    terms = SYMPTOMS["terms"]
    assert 150 <= len(terms) <= 250
    assert len({t["id"] for t in terms}) == len(terms)
    assert all(t["id"].startswith("SYMP:") for t in terms)


@pytest.mark.parametrize("term", SYMPTOMS["terms"], ids=lambda t: t["label"])
def test_every_lay_phrase_and_merge_is_marked_unreviewed(term: dict[str, Any]) -> None:
    for addition in term["lay"] + term["merged"]:
        assert addition["reviewed"] is False


def test_no_surface_form_is_shared_between_terms() -> None:
    owner: dict[str, str] = {}
    for term in SYMPTOMS["terms"]:
        forms = term["synonyms"] + [lay["text"] for lay in term["lay"]]
        for merged in term["merged"]:
            forms += [merged["label"], *merged["synonyms"]]
        if term["match_label"]:
            forms.append(term["label"])
        for form in forms:
            key = form.casefold()
            assert owner.setdefault(key, term["label"]) == term["label"], form
