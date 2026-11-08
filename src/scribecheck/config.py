"""Settings, and the TOML config file that sets them.

`Config` is what `check` takes. `load_config` reads a TOML file with the standard
library's tomllib; any unknown key, wrong type or unknown value is an `InputError`
naming the file, so a typo never silently falls back to a default. The user's speaker
and heading maps extend the built-in ones rather than replacing them.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from scribecheck.errors import InputError
from scribecheck.flags import Confidence, Severity
from scribecheck.note import CanonicalSection, build_heading_map
from scribecheck.rules.lexicon import find_drugs
from scribecheck.transcript import DEFAULT_SPEAKER_MAP

Mode = Literal["rules", "full"]
Standard = Literal["clinical", "strict"]

CONFIG_FILENAME = "scribecheck.toml"
ROLES = frozenset({"clinician", "patient", "relative", "carer", "parent", "nurse"})
"""The roles a speaker label can map to."""


class Config(BaseModel):
    """Checker settings. Defaults: rules mode, clinical standard, nothing fails.

    Invalid settings raise pydantic's `ValidationError` (a `ValueError`); `load_config`
    turns it into an `InputError` naming the file.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Mode = "rules"
    standard: Standard = "clinical"
    fail_on: tuple[Severity, ...] = ()
    """Severities that fail a note; full mode only, and opt-in."""
    fail_confidence: tuple[Confidence, ...] = ("certain", "confirmed")
    """Confidence levels at which a flag of a `fail_on` severity fails the note."""
    speakers: dict[str, str] = Field(default_factory=dict)
    """Speaker label -> role, added to the built-in speaker map."""
    headings: dict[str, str] = Field(default_factory=dict)
    """Note heading -> section name, added to the built-in heading map."""
    brands: dict[str, list[str]] = Field(default_factory=dict)
    """Brand name -> generic names."""
    symptoms: dict[str, str] = Field(default_factory=dict)
    """Extra symptom phrasing -> symptom term."""
    base_url: str | None = None
    """Model endpoint, for full mode (M3)."""
    model: str | None = None
    """Model name, for full mode (M3)."""

    @field_validator("speakers")
    @classmethod
    def _known_roles(cls, value: dict[str, str]) -> dict[str, str]:
        for label, role in value.items():
            if role.casefold() not in ROLES:
                allowed = ", ".join(sorted(ROLES))
                raise ValueError(
                    f"speaker {label!r} maps to unknown role {role!r}; expected one of: {allowed}"
                )
        return value

    @field_validator("headings")
    @classmethod
    def _known_sections(cls, value: dict[str, str]) -> dict[str, str]:
        build_heading_map(value)
        return value

    @field_validator("brands")
    @classmethod
    def _known_generics(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        if value:
            find_drugs("", brands=value)
        return value

    def speaker_map(self) -> dict[str, str]:
        """The built-in speaker map with the user's labels added (theirs win)."""
        merged = dict(DEFAULT_SPEAKER_MAP)
        merged.update({label.casefold(): role.casefold() for label, role in self.speakers.items()})
        return merged

    def heading_map(self) -> dict[str, CanonicalSection]:
        """The built-in heading map with the user's headings added (theirs win)."""
        return build_heading_map(self.headings)


class _Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fail_on: list[Severity] = []
    fail_confidence: list[Confidence] = ["certain", "confirmed"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str | None = None
    name: str | None = None


class _File(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Mode = "rules"
    standard: Standard = "clinical"
    decision: _Decision = _Decision()
    speakers: dict[str, str] = {}
    headings: dict[str, str] = {}
    brands: dict[str, list[str]] = {}
    symptoms: dict[str, str] = {}
    model: _Model = _Model()


def load_config(path: str | Path) -> Config:
    """Read a TOML config file. Raises `InputError` for an unreadable or invalid file."""
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(f"{path}: cannot read config ({exc.strerror or exc})") from exc
    except UnicodeDecodeError as exc:
        raise InputError(f"{path}: config is not valid UTF-8") from exc
    except tomllib.TOMLDecodeError as exc:
        raise InputError(f"{path}: invalid TOML: {exc}") from exc
    return config_from_mapping(data, source=str(path))


def config_from_mapping(data: Mapping[str, object], *, source: str = "config") -> Config:
    """Build a `Config` from parsed TOML. Raises `InputError` naming `source` and the key."""
    try:
        parsed = _File.model_validate(data)
        return Config(
            mode=parsed.mode,
            standard=parsed.standard,
            fail_on=tuple(parsed.decision.fail_on),
            fail_confidence=tuple(parsed.decision.fail_confidence),
            speakers=parsed.speakers,
            headings=parsed.headings,
            brands=parsed.brands,
            symptoms=parsed.symptoms,
            base_url=parsed.model.base_url,
            model=parsed.model.name,
        )
    except ValidationError as exc:
        raise InputError(f"{source}: {_describe(exc)}") from exc
    except (InputError, ValueError) as exc:
        raise InputError(f"{source}: {exc}") from exc


def _describe(error: ValidationError) -> str:
    problems = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"])
        problems.append(f"{location or 'config'}: {item['msg']}")
    return "; ".join(problems)


STARTER_CONFIG = """\
# scribecheck configuration. Every setting is optional; delete what you don't need.
# The checker reads ./scribecheck.toml, or the file given with --config.
# Command-line options override this file.

# "rules" needs no model and makes no network calls. "full" arrives in a later release.
mode = "rules"

# "clinical" (default): critical and supporting facts must appear in the note.
# "strict": every fact must appear, including peripheral ones; for benchmarking only.
standard = "clinical"

[decision]
# Nothing fails by default, and in rules mode nothing ever fails: findings are warnings.
# In full mode, failing is opt-in: run the stability command on your own test set first.
fail_on = []                              # e.g. ["critical"]
fail_confidence = ["certain", "confirmed"]

[speakers]
# Speaker label in your transcripts -> role (clinician, patient, relative, carer,
# parent, nurse). Added to the built-in labels such as "Doctor" and "Patient".
# "GP" = "clinician"

[headings]
# Note heading -> section (History, Medications, Allergies, Examination, Results,
# Assessment, Plan, ...). Added to the built-in headings.
# "Current meds" = "Medications"

[brands]
# Brand name -> generic (ingredient) names. These two are invented examples.
# "Examplol" = ["paracetamol"]
# "Fictivir" = ["acyclovir"]

[symptoms]
# Extra phrasing -> symptom term.
# "wheezy" = "wheezing"

# [model]
# For full mode, in a later release:
# base_url = "http://localhost:11434/v1"
# name = "<model name>"
"""
"""Written by `scribecheck init`."""
