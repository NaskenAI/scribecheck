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
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from scribecheck._url import check_base_url, check_env_name, check_extra_body, check_query
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
Pipeline = Literal["scribecheck", "route_one"]
ROUTE_ONE_STAGES = frozenset({"extract", "audit", "check", "refute"})
SCRIBECHECK_STAGES = frozenset(
    {
        "extract",
        "audit",
        "check",
        "second_look",
        "candidate",
        "quote_recheck",
        "changes",
        "changes_second",
        "split",
        "statements",
        "statements_second",
    }
)
PassName = Literal["a", "b", "c"]
ExemptKind = Literal["vital", "result", "record_history", "demographic", "examination"]
DEFAULT_EXEMPT: tuple[ExemptKind, ...] = (
    "vital",
    "result",
    "record_history",
    "demographic",
    "examination",
)

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


class StageSettings(BaseModel):
    """One Pass A stage's model, effort and token limit; unset values use [model]."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str | None = None
    effort: Effort | None = None
    max_tokens: StrictInt | None = Field(default=None, gt=0)


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
    extra_body: dict[str, Any] = Field(default_factory=dict)
    """Extra fields for OpenAI-compatible request bodies, such as an OpenRouter provider pin."""
    prices: PriceTable | None = None
    cost_cap_usd: float | None = Field(default=None, ge=0)
    """Stop making model calls once the estimated spend reaches this."""
    pipeline: Pipeline = "scribecheck"
    """Pass A's configuration: scribecheck's own, or Fox et al.'s route one."""
    check_facts_per_call: int = Field(default=10, gt=0)
    chunk_chars: int = Field(default=60000, gt=1000)
    """Transcripts longer than this are extracted in overlapping chunks."""
    chunk_overlap_turns: int = Field(default=4, ge=0)
    stages: dict[str, StageSettings] = Field(default_factory=dict)
    """Per-stage settings, keyed "route_one.<stage>" or "scribecheck.<stage>"."""
    passes: tuple[PassName, ...] = ("a", "b", "c")
    """The passes to run and report: a (omissions), b (changed facts), c (unsupported)."""
    exempt: tuple[ExemptKind, ...] = DEFAULT_EXEMPT
    """Pass C statement kinds reported as not verifiable instead of checked."""
    show_rejected_as_possible: bool = False
    """Report Pass B and C verdicts the second check rejected as possible flags, instead of
    as unchecked items ("second_check_rejected")."""
    partial_as_flags: bool = False
    """Report partial verdicts as flags that warn, instead of as information."""
    pass_a_trace: bool = False
    """Add each fact's path through Pass A (extraction, audit, check, second look) to a
    full-mode report, for analysing misses on development data."""

    @field_validator("passes")
    @classmethod
    def _some_pass(cls, value: tuple[PassName, ...]) -> tuple[PassName, ...]:
        if not value:
            raise ValueError("choose at least one pass: a, b or c")
        return tuple(sorted(set(value)))

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

    @field_validator("extra_body")
    @classmethod
    def _safe_extra_body(cls, value: dict[str, Any]) -> dict[str, Any]:
        return check_extra_body(value)

    @model_validator(mode="after")
    def _extra_body_needs_openai(self) -> Config:
        if self.extra_body and self.provider == "anthropic":
            raise ValueError(
                "[model] extra_body applies to OpenAI-compatible endpoints only, not anthropic"
            )
        return self

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
    extra_body: dict[str, Any] = {}
    prices: PriceTable | None = None


class _PassA(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline: Pipeline = "scribecheck"
    check_facts_per_call: StrictInt = Field(default=10, gt=0)
    chunk_chars: StrictInt = Field(default=60000, gt=1000)
    chunk_overlap_turns: StrictInt = Field(default=4, ge=0)
    route_one: dict[str, StageSettings] = {}
    scribecheck: dict[str, StageSettings] = {}

    @field_validator("route_one")
    @classmethod
    def _route_one_stages(cls, value: dict[str, StageSettings]) -> dict[str, StageSettings]:
        return _known_stages(value, ROUTE_ONE_STAGES)

    @field_validator("scribecheck")
    @classmethod
    def _scribecheck_stages(cls, value: dict[str, StageSettings]) -> dict[str, StageSettings]:
        return _known_stages(value, SCRIBECHECK_STAGES)


def _known_stages(
    value: dict[str, StageSettings], known: frozenset[str]
) -> dict[str, StageSettings]:
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(
            f"unknown stage {unknown[0]!r}; expected one of: {', '.join(sorted(known))}"
        )
    return value


class _PassC(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exempt: list[ExemptKind] = list(DEFAULT_EXEMPT)


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid")

    show_rejected_as_possible: StrictBool = False
    partial_as_flags: StrictBool = False
    pass_a_trace: StrictBool = False


class _Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cost_cap_usd: float | None = Field(default=None, ge=0)


class _File(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Mode = "rules"
    standard: Standard = "clinical"
    passes: list[PassName] = ["a", "b", "c"]
    decision: _Decision = _Decision()
    speakers: dict[str, str] = {}
    headings: dict[str, str] = {}
    brands: dict[str, list[str]] = {}
    symptoms: dict[str, str] = {}
    model: _Model = _Model()
    budget: _Budget = _Budget()
    pass_a: _PassA = _PassA()
    pass_c: _PassC = _PassC()
    report: _Report = _Report()


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
            extra_body=m.extra_body,
            prices=m.prices,
            cost_cap_usd=parsed.budget.cost_cap_usd,
            pipeline=parsed.pass_a.pipeline,
            check_facts_per_call=parsed.pass_a.check_facts_per_call,
            chunk_chars=parsed.pass_a.chunk_chars,
            chunk_overlap_turns=parsed.pass_a.chunk_overlap_turns,
            passes=tuple(parsed.passes),
            exempt=tuple(parsed.pass_c.exempt),
            show_rejected_as_possible=parsed.report.show_rejected_as_possible,
            partial_as_flags=parsed.report.partial_as_flags,
            pass_a_trace=parsed.report.pass_a_trace,
            stages={
                **{f"route_one.{k}": v for k, v in parsed.pass_a.route_one.items()},
                **{f"scribecheck.{k}": v for k, v in parsed.pass_a.scribecheck.items()},
            },
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

# The passes to run and report: a (omissions), b (changed facts), c (unsupported
# statements). Benchmark (OmissionBench) runs use ["a"].
# passes = ["a", "b", "c"]

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
# extra_body = { provider = { order = ["<Provider>"], allow_fallbacks = false } }  # OpenRouter pin

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

# [pass_a]
# Full mode's omission check: "scribecheck" (default) or "route_one", Fox et al.'s
# published per-fact pipeline.
# pipeline = "scribecheck"
# check_facts_per_call = 10
# chunk_chars = 60000                     # longer transcripts are extracted in chunks
# chunk_overlap_turns = 4

# [pass_a.route_one.audit]               # per-stage model, effort and token limit
# name = "<model name>"                  # stages: extract, audit, check, refute
# effort = "high"
# max_tokens = 24000

# [pass_a.scribecheck.check]             # stages: extract, audit, check, second_look,
# effort = "low"                         # candidate, quote_recheck, changes,
#                                        # changes_second, split, statements,
#                                        # statements_second

# [pass_c]
# Note statements reported as not verifiable instead of checked against the transcript.
# record_history applies only where the note marks a statement as from the record.
# exempt = ["vital", "result", "record_history", "demographic", "examination"]

# [report]
# Show Pass B and C verdicts their second check rejected as possible flags, instead of as
# unchecked items.
# show_rejected_as_possible = false
# Report partial verdicts (the note carries part of a fact) as flags that warn, instead of
# as information that never changes the decision.
# partial_as_flags = false
# Add each fact's path through Pass A to full-mode reports (development analysis only).
# pass_a_trace = false
"""
"""Written by `scribecheck init`."""
