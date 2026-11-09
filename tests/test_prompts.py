"""Prompt files: route one unmodified from the harness, ours present, both shipped."""

from __future__ import annotations

import hashlib
from importlib import resources

import pytest

from scribecheck import prompts

# The SHA-256 values recorded in the OmissionBench harness's w2_prompts/PROMPTS.sha256.
HARNESS = {
    "pipeline_audit": "7ed7339a3db9bb8a65a3b613259875f367ee018b14d0d28f2f367f55390b734f",
    "pipeline_check_binary": "b5338cc78e684e96db6fcd8f03d430d9116e3945713e685cf1fc709e8a538c2d",
    "pipeline_check_tri": "47406081e8b612fab3c320b15d0ec001c741ce3ebd55fe6b02f73ce0d62bcd41",
    "pipeline_extract": "24c071b86aa5d4bf398f9d04585146273e7eaed4e57fa345bb4bc4e65106b23c",
    "pipeline_refute": "0d28769d43e8a49eee1dcdf192150694f78578143d31c519df781820ab838417",
}


@pytest.mark.parametrize("name", sorted(HARNESS))
def test_route_one_prompts_are_byte_identical_to_the_harness(name: str) -> None:
    data = (resources.files("scribecheck.prompts") / "route_one" / f"{name}.txt").read_bytes()
    assert hashlib.sha256(data).hexdigest() == HARNESS[name]


def test_route_one_attribution_and_licence_ship() -> None:
    folder = resources.files("scribecheck.prompts") / "route_one"
    licence = (folder / "LICENSE").read_text(encoding="utf-8")
    assert licence.startswith("MIT License") and "Composo Limited" in licence
    readme = (folder / "README.md").read_text(encoding="utf-8")
    assert "Fox" in readme and "arXiv:2608.31016" in readme and "MIT" in readme


@pytest.mark.parametrize(
    ("name", "slots"),
    [
        ("system", {}),
        ("extract", {"chunk_note": ""}),
        ("audit", {"facts_block": "f1: x"}),
        ("check", {"facts_block": "f1: x"}),
        ("second_look", {"facts_block": "f1: x"}),
        ("candidate", {"facts_block": "c1: x"}),
        ("quote_recheck", {"source": "NOTE", "unit": "passage", "facts_block": "q1: x"}),
    ],
)
def test_scribecheck_prompts_render_every_slot(name: str, slots: dict[str, str]) -> None:
    text = prompts.render(prompts.load("scribecheck_v3", name), **slots)
    assert "{" not in text.replace('{"', "").replace("{}", "")


def test_render_is_literal_and_strict() -> None:
    template = prompts.load("route_one", "pipeline_check_tri")
    out = prompts.render(template, note="N", facts_block="f1: x")
    assert '{"verdicts": {"f1": "full"' in out  # literal JSON in the prompt survives
    with pytest.raises(KeyError):
        prompts.render("no slots here", note="N")


def test_audit_prompt_attributes_the_rubric() -> None:
    assert "Fox et al., OmissionBench harness, MIT" in prompts.load("scribecheck_v3", "audit")


def test_v2_changes_only_the_candidate_prompt() -> None:
    folder = resources.files(prompts)
    v1 = {p.name for p in (folder / "scribecheck_v1").iterdir() if p.name.endswith(".txt")}
    v2 = {p.name for p in (folder / "scribecheck_v2").iterdir() if p.name.endswith(".txt")}
    assert v1 == v2
    changed = {
        name
        for name in v1
        if prompts.load("scribecheck_v1", name[:-4]) != prompts.load("scribecheck_v2", name[:-4])
    }
    assert changed == {"candidate.txt"}
    assert "recounts a past episode" in prompts.load("scribecheck_v2", "candidate")


def _changed(old: prompts.PromptSet, new: prompts.PromptSet) -> set[str]:
    folder = resources.files(prompts)
    names = {p.name for p in (folder / old).iterdir() if p.name.endswith(".txt")}
    assert names == {p.name for p in (folder / new).iterdir() if p.name.endswith(".txt")}
    return {n for n in names if prompts.load(old, n[:-4]) != prompts.load(new, n[:-4])}


def test_v3_makes_facts_atomic() -> None:
    assert _changed("scribecheck_v2", "scribecheck_v3") == {
        "extract.txt",
        "audit.txt",
        "check.txt",
        "second_look.txt",
    }
    extract = " ".join(prompts.load("scribecheck_v3", "extract").split())
    for rule in (
        "one clinical statement",
        "each medicine",
        "each denied symptom",
        "decisions not to treat",
        "each safety-net instruction",
        "each numeric threshold",
        "no known allergies",
    ):
        assert rule in extract, rule
    check = " ".join(prompts.load("scribecheck_v3", "check").split())
    assert "does not carry this one" in check
    look = " ".join(prompts.load("scribecheck_v3", "second_look").split())
    assert "in part" not in look and "only gestures at" in look
