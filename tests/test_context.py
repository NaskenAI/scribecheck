from __future__ import annotations

import pytest

from scribecheck.rules.context import TargetContext, classify


def _span(text: str, target: str) -> tuple[int, int]:
    start = text.index(target)
    return start, start + len(target)


def _classify_one(text: str, target: str) -> TargetContext:
    (result,) = classify(text, [_span(text, target)])
    assert text[result.start : result.end] == result.text == target
    for cue in result.cues:
        assert text[cue.start : cue.end] == cue.text
    return result


@pytest.mark.parametrize(
    ("text", "target", "attribute", "cue"),
    [
        ("no chest pain", "chest pain", "negated", "no"),
        ("She denies fever.", "fever", "negated", "denies"),
        ("Denied any vomiting", "vomiting", "negated", "Denied"),
        ("He has not had a cough", "cough", "negated", "not"),
        ("without nausea", "nausea", "negated", "without"),
        ("Negative for blood in the urine", "blood", "negated", "Negative for"),
        ("no sign of a rash", "rash", "negated", "no sign of"),
        ("I don't have any headache", "headache", "negated", "don't have"),
        ("Never had chest pain", "chest pain", "negated", "Never had"),
        ("DVT was ruled out", "DVT", "negated", "was ruled out"),
        ("possible fracture", "fracture", "possible", "possible"),
        ("It could be a sprain", "sprain", "possible", "could be"),
        ("probably a virus", "virus", "possible", "probably"),
        ("suspected appendicitis", "appendicitis", "possible", "suspected"),
        ("?fracture of the wrist", "fracture", "possible", "?"),
        ("Not sure whether it is asthma", "asthma", "possible", "Not sure whether"),
        ("Fracture cannot be ruled out", "Fracture", "possible", "cannot be ruled out"),
        ("history of asthma", "asthma", "historical", "history of"),
        ("He had a stroke in 2019", "stroke", "historical", "had"),
        ("Previous DVT", "DVT", "historical", "Previous"),
        ("Eczema as a child", "Eczema", "historical", "as a child"),
        ("her mother has diabetes", "diabetes", "family", "mother"),
        ("My dad had a heart attack", "heart attack", "family", "dad"),
        ("Family history of bowel cancer", "bowel cancer", "family", "Family history of"),
        ("Diabetes runs in the family", "Diabetes", "family", "runs in the family"),
        ("if the pain gets worse, come back", "pain", "hypothetical", "if"),
        ("Come back if a fever develops", "fever", "hypothetical", "Come back if"),
        ("Watch out for a rash", "rash", "hypothetical", "Watch out for"),
    ],
)
def test_attribute_and_cue(text: str, target: str, attribute: str, cue: str) -> None:
    result = _classify_one(text, target)
    assert attribute in result.attributes
    assert any(c.attribute == attribute and c.text == cue for c in result.cues)


def test_properties_match_attributes() -> None:
    result = _classify_one("her mother had no history of asthma", "asthma")
    assert result.family and result.negated and result.historical
    assert not result.possible and not result.hypothetical
    assert result.attributes == frozenset({"family", "negated", "historical"})


@pytest.mark.parametrize(
    ("text", "target"),
    [
        ("chest pain", "chest pain"),
        ("no fever but has a cough", "cough"),
        ("no fever however a cough", "cough"),
        ("No fever. Cough for a week.", "Cough"),
        ("No fever; cough for a week", "cough"),
        ("no change in the headache", "headache"),
        ("no increase in pain", "pain"),
        ("not only a cough", "cough"),
        ("no problems at work and it has been a long and tiring week with a cough", "cough"),
        ("Any chest pain?", "chest pain"),
    ],
)
def test_not_negated(text: str, target: str) -> None:
    assert not _classify_one(text, target).negated


def test_negation_spans_a_list_of_targets() -> None:
    text = "no fever, cough or headache"
    spans = [_span(text, word) for word in ("fever", "cough", "headache")]
    assert all(result.negated for result in classify(text, spans))


@pytest.mark.parametrize(
    ("text", "target"),
    [
        # A comma ends the scope unless only other targets and separators lie between.
        ("no fever, cough or headache", "cough"),
        ("No relief, cough persists", "cough"),
        ("Not sleeping, chest pain at night", "chest pain"),
        ("No, I've got chest pain", "chest pain"),
        ("Nil by mouth, has pain", "pain"),
        ("Not keen on tablets, takes ibuprofen", "ibuprofen"),
        # "no" that is not about the target.
        ("Patient has no idea about the rash", "rash"),
        ("No improvement in cough", "cough"),
        ("no better with the cough", "cough"),
        ("no response to the inhaler for wheeze", "wheeze"),
    ],
)
def test_comma_and_pseudo_cues_stop_negation(text: str, target: str) -> None:
    assert not _classify_one(text, target).negated


@pytest.mark.parametrize(
    ("text", "target"),
    [
        ("son brought him in with chest pain", "chest pain"),
        ("mum says he has a fever", "fever"),
        ("Her father thinks she has asthma", "asthma"),
        ("My mother noticed I had a rash", "rash"),
    ],
)
def test_family_member_reporting_the_patients_symptom(text: str, target: str) -> None:
    assert not _classify_one(text, target).family


@pytest.mark.parametrize(
    ("text", "target"),
    [
        ("previous GP visit, now headache", "headache"),
        ("Previously well, presents with fever", "fever"),
        ("history of present illness: cough", "cough"),
        ("prior to the fever", "fever"),
    ],
)
def test_historical_cue_not_about_the_target(text: str, target: str) -> None:
    assert not _classify_one(text, target).historical


def test_terminator_stops_every_attribute() -> None:
    result = _classify_one("her mother had asthma but he has eczema", "eczema")
    assert result.attributes == frozenset()


def test_present_perfect_had_is_not_historical() -> None:
    assert not _classify_one("I've had a cough for three days", "cough").historical
    assert not _classify_one("She has had a rash since Monday", "rash").historical
    assert not _classify_one("Have you had a rash?", "rash").historical


def test_hypothetical_if_stops_at_comma() -> None:
    text = "if the pain gets worse, take paracetamol"
    assert not _classify_one(text, "paracetamol").hypothetical


def test_word_window_is_configurable() -> None:
    text = "no trouble getting to sleep at night and a cough"
    (default,) = classify(text, [_span(text, "cough")])
    (wide,) = classify(text, [_span(text, "cough")], max_words=10)
    assert not default.negated
    assert wide.negated


def test_decimal_point_is_not_a_sentence_break() -> None:
    assert _classify_one("no 0.5 mm lesion", "lesion").negated


def test_results_follow_target_order() -> None:
    text = "no fever but has a cough"
    spans = [_span(text, "cough"), _span(text, "fever")]
    results = classify(text, spans)
    assert [r.text for r in results] == ["cough", "fever"]
    assert [r.negated for r in results] == [False, True]


def test_empty_targets() -> None:
    assert classify("no fever", []) == ()


@pytest.mark.parametrize("span", [(-1, 2), (3, 3), (5, 2), (0, 99)])
def test_invalid_span_raises(span: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        classify("no fever", [span])


@pytest.mark.parametrize(
    ("text", "target", "attribute"),
    [
        ("Mother: diabetes", "diabetes", "family"),
        ("FH: diabetes", "diabetes", "family"),
        ("Denies: fever", "fever", "negated"),
    ],
)
def test_label_colon_after_cue(text: str, target: str, attribute: str) -> None:
    assert attribute in _classify_one(text, target).attributes


@pytest.mark.parametrize(
    ("text", "targets", "attribute"),
    [
        (
            "She denies any chest pain, shortness of breath, or palpitations",
            ("chest pain", "shortness of breath", "palpitations"),
            "negated",
        ),
        (
            "if the pain, swelling or redness gets worse",
            ("pain", "swelling", "redness"),
            "hypothetical",
        ),
    ],
)
def test_list_of_targets_shares_the_cue(
    text: str, targets: tuple[str, ...], attribute: str
) -> None:
    results = classify(text, [_span(text, target) for target in targets])
    assert all(attribute in result.attributes for result in results)
