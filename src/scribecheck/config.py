"""Settings, and the TOML config file that sets them.

`Config` is what `check` takes. `load_config` reads a TOML file with the standard
library's tomllib; any unknown key, wrong type or unknown value is an `InputError`
naming the file, so a typo never silently falls back to a default. The user's speaker
and heading maps extend the built-in ones rather than replacing them.

API keys never go in the config: the client reads them from environment variables
(`api_key_env` names a different variable), and an `api_key` key is rejected.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    ValidationError,
    field_validator,
)

from scribecheck._url import check_base_url, check_env_name, check_query
from scribecheck.errors import InputError
from scribecheck.flags import Confidence, Severity
from scribecheck.note import CanonicalSection, build_heading_map
from scribecheck.rules.lexicon import find_drugs
from scribecheck.transcript import DEFAULT_SPEAKER_MAP

Mode = Literal["rules", "full"]
Standard = Literal["clinical", "strict"]
Provider = Literal["anthropic", "openai"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]
CacheTTL = Literal["5m", "1h"]
JsonMode = Literal["json_schema", "json_object", "none"]

CONFIG_FILENAME = "scribecheck.toml"
ROLES = frozenset({"clinician", "patient", "relative", "carer", "parent", "nurse"})
"""The roles a speaker label can map to."""


class PriceTable(BaseModel):
    """Model prices in USD per million tokens, from config; nothing is built in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_write_5m: float | None = Field(default=None, ge=0)
    cache_write_1h: float | None = Field(default=None, ge=0)
    cache_read: float | None = Field(default=None, ge=0)
    batch_multiplier: float | None = Field(default=None, gt=0)
    """Batch cost relative to live calls."""


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
    provider: Provider | None = None
    """Full mode's model API: "anthropic" or "openai" (any OpenAI-compatible endpoint)."""
    base_url: str | None = None
    """Model endpoint; the Anthropic API by default for "anthropic"."""
    model: str | None = None
    """Model name, as the provider names it."""
    api_key_env: str | None = None
    """Environment variable holding the key, if not the provider's usual one."""
    effort: Effort | None = None
    """Anthropic effort level, fixed for the run; None leaves the model's default."""
    max_tokens: int = Field(default=16000, gt=0)
    cache_ttl: CacheTTL = "5m"
    timeout_s: float = Field(default=120.0, gt=0)
    temperature: float | None = 0.0
    """OpenAI-compatible only; None omits it (the Anthropic API accepts none)."""
    seed: int | None = 0
    """OpenAI-compatible only; None omits it."""
    json_mode: JsonMode = "json_schema"
    auth_header: str = "authorization"
    """"authorization" sends a Bearer token; Azure uses "api-key"."""
    query: dict[str, str] = Field(default_factory=dict)
    """Extra URL query parameters, such as Azure's api-version."""
    prices: PriceTable | None = None
    cost_cap_usd: float | None = Field(default=None, ge=0)
    """Stop making model calls once the estimated spend reaches this."""

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

    @field_validator("base_url")
    @classmethod
    def _clean_url(cls, value: str | None) -> str | None:
        return None if value is None else check_base_url(value)

    @field_validator("api_key_env")
    @classmethod
    def _env_name(cls, value: str | None) -> str | None:
        return None if value is None else check_env_name(value)

    @field_validator("query")
    @classmethod
    def _no_key_in_query(cls, value: dict[str, str]) -> dict[str, str]:
        return check_query(value)

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

    provider: Provider | None = None
    base_url: str | None = None
    name: str | None = None
    api_key_env: str | None = None
    effort: Effort | None = None
    max_tokens: StrictInt = Field(default=16000, gt=0)
    cache_ttl: CacheTTL = "5m"
    timeout_s: StrictFloat | StrictInt = Field(default=120.0, gt=0)
    temperature: StrictFloat | StrictInt | Literal["none"] = 0.0
    seed: StrictInt | Literal["none"] = 0
    json_mode: JsonMode = "json_schema"
    auth_header: str = "authorization"
    query: dict[str, str] = {}
    prices: PriceTable | None = None


class _Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cost_cap_usd: float | None = Field(default=None, ge=0)


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
    budget: _Budget = _Budget()


_KEY_NAMES = frozenset({"api_key", "key", "apikey", "token", "secret", "password"})


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
    model_table = data.get("model")
    if isinstance(model_table, Mapping) and _KEY_NAMES.intersection(
        str(k).casefold() for k in model_table
    ):
        raise InputError(
            f"{source}: never put an API key in the config file; set ANTHROPIC_API_KEY, "
            "OPENAI_API_KEY or OPENROUTER_API_KEY, or name your own variable with "
            "[model] api_key_env"
        )
    try:
        parsed = _File.model_validate(data)
        m = parsed.model
        return Config(
            mode=parsed.mode,
            standard=parsed.standard,
            fail_on=tuple(parsed.decision.fail_on),
            fail_confidence=tuple(parsed.decision.fail_confidence),
            speakers=parsed.speakers,
            headings=parsed.headings,
            brands=parsed.brands,
            symptoms=parsed.symptoms,
            provider=m.provider,
            base_url=m.base_url,
            model=m.name,
            api_key_env=m.api_key_env,
            effort=m.effort,
            max_tokens=m.max_tokens,
            cache_ttl=m.cache_ttl,
            timeout_s=m.timeout_s,
            temperature=None if m.temperature == "none" else m.temperature,
            seed=None if m.seed == "none" else m.seed,
            json_mode=m.json_mode,
            auth_header=m.auth_header,
            query=m.query,
            prices=m.prices,
            cost_cap_usd=parsed.budget.cost_cap_usd,
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
# For full mode, in a later release. API keys are read from environment variables
# (ANTHROPIC_API_KEY, OPENAI_API_KEY or OPENROUTER_API_KEY), never from this file.
# provider = "openai"                     # or "anthropic"
# base_url = "http://localhost:11434/v1"  # any OpenAI-compatible endpoint
# name = "<model name>"
# api_key_env = "MY_KEY_VARIABLE"         # if the key is in another variable
# max_tokens = 16000
# cache_ttl = "5m"                        # Anthropic prompt caching: "5m" or "1h"
# effort = "low"                          # Anthropic only; fixed for the whole run
# temperature = 0.0                       # OpenAI-compatible only; "none" omits it
# seed = 0                                # OpenAI-compatible only; "none" omits it
# json_mode = "json_schema"               # or "json_object", or "none"
# auth_header = "api-key"                 # Azure; the default sends a Bearer token
# query = { "api-version" = "<version>" } # Azure

# [model.prices]
# USD per million tokens, from your provider's price list; nothing is built in.
# input = 0.0
# output = 0.0
# cache_write_5m = 0.0
# cache_write_1h = 0.0
# cache_read = 0.0
# batch_multiplier = 0.5                  # batch cost relative to live calls

# [budget]
# cost_cap_usd = 10.0                     # needs [model.prices]
"""
"""Written by `scribecheck init`."""
