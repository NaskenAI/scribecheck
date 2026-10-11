from __future__ import annotations

import pytest
from _spans import assert_round_trip

from scribecheck.rules.lexicon import DrugMatch, SymptomMatch, find_drugs, find_symptoms

# Drugs


def _drugs(text: str, **kwargs: object) -> tuple[DrugMatch, ...]:
    found = find_drugs(text, **kwargs)  # type: ignore[arg-type]
    assert_round_trip(text, found)
    return found


def _one_drug(text: str, **kwargs: object) -> DrugMatch:
    (found,) = _drugs(text, **kwargs)
    return found


@pytest.mark.parametrize(
    ("text", "matched", "generic", "via"),
    [
        ("Started metformin 500 mg BD", "metformin", ("metformin",), "ingredient"),
        ("METFORMIN 1 g", "METFORMIN", ("metformin",), "ingredient"),
        ("metformin hydrochloride 500 mg", "metformin hydrochloride", ("metformin",), "ingredient"),
        (
            "on sulfamethoxazole/trimethoprim",
            "sulfamethoxazole/trimethoprim",
            ("sulfamethoxazole", "trimethoprim"),
            "ingredient",
        ),
        ("potassium chloride 20 mmol", "potassium chloride", ("potassium chloride",), "ingredient"),
        ("Parlodel 2.5 mg", "Parlodel", ("bromocriptine",), "brand"),
        ("takes Augmentin", "Augmentin", ("amoxicillin", "clavulanate"), "brand"),
        ("paracetamol 1 g QDS", "paracetamol", ("acetaminophen",), "international"),
        ("salbutamol inhaler", "salbutamol", ("albuterol",), "international"),
        ("glyceryl trinitrate spray", "glyceryl trinitrate", ("nitroglycerin",), "international"),
        ("co-amoxiclav 625", "co-amoxiclav", ("amoxicillin", "clavulanate"), "international"),
    ],
)
def test_drug_lookup(text: str, matched: str, generic: tuple[str, ...], via: str) -> None:
    found = _one_drug(text)
    assert found.text == matched
    assert found.generic == generic
    assert found.via == via
    assert not found.fuzzy


def test_brand_carries_its_rxnorm_id() -> None:
    assert _one_drug("Parlodel").rxcui == "38"
    assert _one_drug("paracetamol").rxcui is None


def test_several_drugs_left_to_right() -> None:
    found = _drugs("Stop paracetamol, start codeine and Parlodel.")
    assert [f.generic for f in found] == [("acetaminophen",), ("codeine",), ("bromocriptine",)]


@pytest.mark.parametrize(
    ("text", "matched", "generic"),
    [
        ("metfromin 500 mg", "metfromin", ("metformin",)),
        ("warfrin 3 mg", "warfrin", ("warfarin",)),
        ("paracetmol", "paracetmol", ("acetaminophen",)),
        ("codine", "codine", ("codeine",)),  # the name has 7 letters
    ],
)
def test_one_edit_spelling_for_long_names(
    text: str, matched: str, generic: tuple[str, ...]
) -> None:
    found = _one_drug(text)
    assert (found.text, found.generic, found.fuzzy) == (matched, generic, True)


@pytest.mark.parametrize(
    "text",
    [
        "aspiring to run",  # an English word one edit from aspirin
        "naicin",  # one edit from niacin, a name of only 6 letters
        "dimethicon",  # dimethicone and simethicone are within one edit of each other
        "metfrmn",  # two edits
        "Tablets taken",  # whole-word only
        "metforminhydro",  # not a word boundary
        "We'll align the plan",  # brand name that is an everyday word
        "React to the news",
        "Julie came with her",  # brand name that is a first name
        "drink more water",  # ingredient that is an everyday word
        "sodium 135, potassium 4.1",  # lab analytes
        "glucose 7.2",
    ],
)
def test_near_misses_do_not_match(text: str) -> None:
    assert _drugs(text) == ()


def test_user_brand_map() -> None:
    found = _one_drug("Takes Examplol daily", brands={"Examplol": ["paracetamol"]})
    assert (found.generic, found.via, found.rxcui) == (("acetaminophen",), "user_brand", None)


def test_user_brand_can_unblock_a_blocked_brand() -> None:
    found = _one_drug("Align capsules", brands={"Align": ["lactase"]})
    assert found.via == "user_brand"


def test_user_brand_with_unknown_generic_raises() -> None:
    with pytest.raises(ValueError, match="notadrug"):
        find_drugs("x", brands={"Examplol": ["notadrug"]})


def test_user_brand_makes_close_names_ambiguous() -> None:
    # "Warfarim" is one edit from warfarin, so a typo near both no longer matches.
    assert _drugs("warfarix", brands={"Warfarim": ["warfarin"]}) == ()


# Symptoms


def _symptoms(text: str, **kwargs: object) -> tuple[SymptomMatch, ...]:
    found = find_symptoms(text, **kwargs)  # type: ignore[arg-type]
    assert_round_trip(text, found)
    return found


def _one_symptom(text: str, **kwargs: object) -> SymptomMatch:
    (found,) = _symptoms(text, **kwargs)
    return found


@pytest.mark.parametrize(
    ("text", "matched", "term", "via"),
    [
        ("Three days of loose motions", "loose motions", "diarrhea", "lay"),
        ("Some giddiness on standing", "giddiness", "dizziness", "lay"),
        ("Gets short of breath on stairs", "short of breath", "dyspnea", "lay"),
        ("shortness of breath", "shortness of breath", "dyspnea", "synonym"),
        ("Diarrhoea since Monday", "Diarrhoea", "diarrhea", "synonym"),
        ("Chest Pain at rest", "Chest Pain", "chest pain", "label"),
        ("burning micturition", "burning micturition", "dysuria", "lay"),
        ("runny nose", "runny nose", "rhinorrhea", "synonym"),
        ("pins and needles in the hand", "pins and needles", "paresthesia", "lay"),
    ],
)
def test_symptom_lookup(text: str, matched: str, term: str, via: str) -> None:
    found = _one_symptom(text)
    assert (found.text, found.term, found.via) == (matched, term, via)
    assert found.symp_id is not None and found.symp_id.startswith("SYMP:")


def test_longest_phrase_wins() -> None:
    assert [s.term for s in _symptoms("chest pain and pain in the knee")] == ["chest pain", "pain"]


@pytest.mark.parametrize(
    ("text", "terms"),
    [
        ("fecal incontinence", []),  # a SYMP synonym of diarrhea, excluded as misleading
        ("gas and air in labour", []),  # "gas" is excluded from flatulence
        ("a faint murmur", []),  # "faint" is excluded from syncope
        ("painful", []),  # whole words only
        ("chest congestion", []),  # bare "congestion" is excluded
        ("blocked nose", ["congestion"]),
    ],
)
def test_symptom_near_misses(text: str, terms: list[str]) -> None:
    assert [s.term for s in _symptoms(text)] == terms


def test_extra_symptoms() -> None:
    known = _one_symptom("wobbly legs", extra={"wobbly legs": "leg weakness"})
    assert (known.term, known.via) == ("leg weakness", "user")
    assert known.symp_id is not None
    new = _one_symptom("brain fog all week", extra={"brain fog": "brain fog"})
    assert (new.term, new.symp_id) == ("brain fog", None)


# Review findings


@pytest.mark.parametrize(
    "text",
    [
        "The rash appeared on Monday and entered the wound.",
        "It enables her to walk; creating a plan.",
        "morphing, augmenting, concerts and intestines",
        "It's your choice, and my choice is to wait.",
        "She came in the morning after the fall.",
        "We will take action today.",
        "A new day, my way.",
        "a stye on her eyelid",
        "Alli is my daughter; Tobi came along.",
        "the amylase level was 5",
        "creatine kinase raised",
        "lactose intolerant",
        "caffeine intake is high",
        "émetformin and metformin_x",
        "hyoscine 300 micrograms",  # ambiguous: hydrobromide or butylbromide
    ],
)
def test_review_near_misses(text: str) -> None:
    assert _drugs(text) == ()


@pytest.mark.parametrize(
    ("text", "matched", "generic"),
    [
        ("metformin-induced lactic acidosis", "metformin", ("metformin",)),
        ("Augmentin's side effects", "Augmentin", ("amoxicillin", "clavulanate")),
        ("aspirin\u2010sensitive asthma", "aspirin", ("aspirin",)),
        ("aspirin-sensitive asthma", "aspirin", ("aspirin",)),
        ("hyoscine butylbromide 20 mg", "hyoscine butylbromide", ("butylscopolamine",)),
        ("hyoscine hydrobromide patch", "hyoscine hydrobromide", ("scopolamine",)),
        ("took Plan B One-Step", "Plan B One-Step", ("levonorgestrel",)),
    ],
)
def test_review_matches(text: str, matched: str, generic: tuple[str, ...]) -> None:
    found = _one_drug(text)
    assert (found.text, found.generic) == (matched, generic)


def test_brands_are_not_spelling_tolerant() -> None:
    assert _drugs("Augmentan") == ()


def test_very_long_word_is_quick() -> None:
    assert _drugs("a" * 50_000) == ()


@pytest.mark.parametrize(
    "brands",
    [{"Examplol": []}, {"aspirin": ["ibuprofen"]}],
)
def test_user_brand_validation(brands: dict[str, list[str]]) -> None:
    with pytest.raises(ValueError):
        find_drugs("x", brands=brands)


@pytest.mark.parametrize(
    "text",
    [
        "it fits well",
        "she started to sob",
        "boil the water",
        "he shakes his head",
        "flushing the cannula",
        "skin erythema",
        "a power blackout",
    ],
)
def test_review_symptom_near_misses(text: str) -> None:
    assert _symptoms(text) == ()


@pytest.mark.parametrize(
    ("text", "term"),
    [
        ("low blood pressure", "hypotension"),
        ("a convulsion", "seizure"),
        ("tiredness", "fatigue"),
        ("weariness", "fatigue"),
        ("hoarse voice", "hoarseness"),
        ("slurred speech", "dysarthria"),
        ("nasal congestion", "congestion"),
        ("hot flushes", "flushing"),
        ("a furuncle on the neck", "boil"),
    ],
)
def test_merged_and_lay_terms(text: str, term: str) -> None:
    (found,) = _symptoms(text)
    assert found.term == term


# Second review round


@pytest.mark.parametrize(
    "text",
    [
        "Pain-free since Monday.",
        "Seizure-free for a year",
        "chest pain-free, headache-free and fever-free",
        "anti-nausea tablets",
        "non-drowsy antihistamine",
    ],
)
def test_hyphen_compounds_are_not_symptoms(text: str) -> None:
    assert _symptoms(text) == ()


@pytest.mark.parametrize(
    "text",
    ["iron-deficiency anaemia", "penicillin-resistant organism", "a bedside care plan"],
)
def test_hyphen_compounds_are_not_drugs(text: str) -> None:
    assert _drugs(text) == ()


@pytest.mark.parametrize(
    ("text", "matched"),
    [
        ("Luden's drops", "Luden's drops"),
        ("one Luden's", "Luden's"),
        ("Buckley's Cough Suppressant", "Buckley's Cough Suppressant"),
        ("Augmentin's", "Augmentin"),
        ("zinc-oxide cream", "zinc-oxide"),
        ("beta alanine powder", "beta alanine"),
        ("Ex-Lax Chocolated", "Ex-Lax Chocolated"),
    ],
)
def test_names_with_possessives_and_hyphens(text: str, matched: str) -> None:
    assert _one_drug(text).text == matched


def test_hyphen_and_space_are_interchangeable_for_symptoms() -> None:
    assert _one_symptom("short-of-breath on exertion").term == "dyspnea"


# Extra (non-US) ingredients and withheld phrases


@pytest.mark.parametrize(
    ("text", "matched", "fuzzy"),
    [
        ("domperidone 10 mg TDS", "domperidone", False),
        ("Fusidic acid cream", "Fusidic acid", False),
        ("domperidon 10 mg", "domperidon", True),
    ],
)
def test_extra_ingredients(text: str, matched: str, fuzzy: bool) -> None:
    found = _one_drug(text)
    assert (found.text, found.via, found.rxcui, found.fuzzy) == (
        matched,
        "extra_ingredient",
        None,
        fuzzy,
    )
    assert found.generic == (found.name,)


def test_user_brand_can_point_to_an_extra_ingredient() -> None:
    found = _one_drug("Examplol", brands={"Examplol": ["domperidone"]})
    assert found.generic == ("domperidone",)


def test_withheld_phrase_does_not_match() -> None:
    assert _symptoms("white discharge for a week") == ()
    assert _one_symptom("vaginal discharge").term == "vaginal discharge"


def test_extra_ingredient_loader_rejects_existing_names() -> None:
    from scribecheck.rules.lexicon import _add_extra_ingredients, _drug_index

    index = _drug_index()
    entries, generics = dict(index.entries), dict(index.generics)
    with pytest.raises(ValueError, match="metformin"):
        _add_extra_ingredients(entries, generics, ["metformin"])
    with pytest.raises(ValueError, match="domperidone"):
        _add_extra_ingredients(entries, generics, ["domperidone"])
    _add_extra_ingredients(entries, generics, ["examplazole"])
    assert generics["examplazole"] == ("examplazole",)
