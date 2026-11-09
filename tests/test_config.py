from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from scribecheck.config import STARTER_CONFIG, Config, config_from_mapping, load_config
from scribecheck.errors import InputError
from scribecheck.note import CanonicalSection


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "scribecheck.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults() -> None:
    config = Config()
    assert config.mode == "rules"
    assert config.standard == "clinical"
    assert config.fail_on == ()
    assert config.fail_confidence == ("certain", "confirmed")


def test_starter_config_loads_as_the_defaults(tmp_path: Path) -> None:
    assert load_config(_write(tmp_path, STARTER_CONFIG)) == Config()


def test_full_file(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
mode = "rules"
standard = "strict"

[decision]
fail_on = ["critical"]
fail_confidence = ["confirmed"]

[speakers]
"GP" = "Clinician"

[headings]
"Current meds" = "medications"

[brands]
"Examplol" = ["paracetamol"]

[symptoms]
"wheezy" = "wheezing"

[model]
base_url = "http://localhost:11434/v1"
name = "some-model"
""",
    )
    config = load_config(path)
    assert config.standard == "strict"
    assert config.fail_on == ("critical",)
    assert config.fail_confidence == ("confirmed",)
    assert config.brands == {"Examplol": ["paracetamol"]}
    assert config.symptoms == {"wheezy": "wheezing"}
    assert (config.base_url, config.model) == ("http://localhost:11434/v1", "some-model")
    speakers = config.speaker_map()
    assert speakers["gp"] == "clinician"
    assert speakers["doctor"] == "clinician"  # the built-in labels are kept
    headings = config.heading_map()
    assert headings["current meds"] is CanonicalSection.MEDICATIONS
    assert headings["plan"] is CanonicalSection.PLAN  # the built-in headings are kept


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("mood = 'rules'", "mood"),
        ("mode = 'fast'", "mode"),
        ("standard = 'loose'", "standard"),
        ("[decision]\nfail_on = ['major']", "decision.fail_on"),
        ("[decision]\nfail_confidence = 'confirmed'", "decision.fail_confidence"),
        ("[extras]\nx = 1", "extras"),
        ("[model]\nflavour = 'x'", "model.flavour"),
        ("[model]\nprovider = 'gemini'", "model.provider"),
        ("[model]\neffort = 'extreme'", "model.effort"),
        ("[model.prices]\ninput = 1.0", "model.prices.output"),
        ("[model.prices]\ninput = -1.0\noutput = 1.0", "model.prices.input"),
        ("[budget]\ncost_cap_usd = -5", "budget.cost_cap_usd"),
        ("[headings]\n'Meds' = 'Pharmacy'", "Pharmacy"),
        ("[brands]\n'Examplol' = ['notadrugatall']", "notadrugatall"),
        ("[brands]\n'Examplol' = 'paracetamol'", "brands.Examplol"),
        ("mode = ", "invalid TOML"),
        ("[speakers]\n'GP' = 'doctorr'", "doctorr"),
    ],
)
def test_invalid_files_are_input_errors(tmp_path: Path, text: str, message: str) -> None:
    path = _write(tmp_path, text)
    with pytest.raises(InputError) as exc:
        load_config(path)
    assert str(path) in str(exc.value)
    assert message in str(exc.value)


def test_missing_file_is_an_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="cannot read config"):
        load_config(tmp_path / "absent.toml")


def test_mapping_source_is_named() -> None:
    with pytest.raises(InputError, match="^settings: "):
        config_from_mapping({"mode": 3}, source="settings")


def test_starter_config_is_valid_toml_with_every_section() -> None:
    data = tomllib.loads(STARTER_CONFIG)
    assert set(data) == {
        "mode",
        "standard",
        "decision",
        "speakers",
        "headings",
        "brands",
        "symptoms",
    }
    assert all(data[k] == {} for k in ("speakers", "headings", "brands", "symptoms"))
    assert "# [model]" in STARTER_CONFIG


def test_report_options() -> None:
    assert (Config().partial_as_flags, Config().pass_a_trace) == (False, False)
    config = config_from_mapping(
        {"report": {"partial_as_flags": True, "pass_a_trace": True}}, source="test"
    )
    assert (config.partial_as_flags, config.pass_a_trace) == (True, True)
    with pytest.raises(InputError):
        config_from_mapping({"report": {"partial_as_flags": "yes"}}, source="test")
