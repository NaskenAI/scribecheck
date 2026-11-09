"""Provider-agnostic types for model calls.

A request is a system prompt, messages made of text blocks, and a pydantic output model.
A block marked `cache=True` ends a prefix the provider may cache: put the transcript
(then the note) first and mark them, so every call about one consultation reuses them.
Adapters turn a request into one HTTP call and return a `RawReply`; `structured.complete`
parses and validates it, retries once on a parse failure and records the usage.

Nothing here holds or logs prompt or response text beyond what the caller passes in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel

from scribecheck.errors import InputError
from scribecheck.model.endpoint import Endpoint

T = TypeVar("T", bound=BaseModel)

JsonMode = Literal["schema", "prompt"]
"""How the reply's JSON is obtained: "schema" asks the provider to constrain the output to
the output model's schema; "prompt" relies on the prompt alone (as route one ran) and takes
the first JSON object in the reply."""
FailureReason = Literal["model_parse_failure", "cost_cap_reached", "model_refusal", "model_error"]
StopReason = Literal["end", "max_tokens", "refusal", "other"]


@dataclass(frozen=True)
class Block:
    """A piece of message text; `cache` marks the end of a cacheable prefix."""

    text: str
    cache: bool = False


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    blocks: tuple[Block, ...]


@dataclass(frozen=True)
class ModelRequest(Generic[T]):
    """One structured call: instructions, messages, and the model the reply must match."""

    system: str
    """Empty for none."""
    messages: tuple[Message, ...]
    output: type[T]
    max_tokens: int | None = None
    """Overrides the client's default."""
    custom_id: str = "request"
    """Names the request in a batch: 1-64 letters, digits, hyphens or underscores."""
    json_mode: JsonMode = "schema"
    model: str | None = None
    """Overrides the client's model, for pipelines that use a different model per stage."""
    effort: str | None = None
    """Overrides the client's effort level: Anthropic `output_config.effort`; sent as
    `reasoning_effort` to an OpenAI-compatible endpoint."""
    temperature: float | None = None
    """Overrides an OpenAI-compatible client's temperature (the Anthropic API takes none)."""


@dataclass(frozen=True)
class Usage:
    """Token counts as the provider reports them."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0
    cache_read_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_write_5m_tokens + other.cache_write_5m_tokens,
            self.cache_write_1h_tokens + other.cache_write_1h_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
        )


@dataclass(frozen=True)
class RawReply:
    """One provider response: the reply text, why it stopped, usage and the model id."""

    text: str
    stop: StopReason
    usage: Usage
    model_id: str | None


@dataclass(frozen=True)
class Completion(Generic[T]):
    """A validated reply."""

    parsed: T
    usage: Usage
    """All attempts together."""
    model_id: str | None
    """The model the provider says answered."""
    attempts: int


@dataclass(frozen=True)
class ModelFailure:
    """A request that produced no validated reply; its item is reported as unchecked."""

    reason: FailureReason
    detail: str
    """Why, without any prompt or response text."""
    usage: Usage = field(default_factory=Usage)
    attempts: int = 0
    """Calls made before failing (0 when stopped before any call)."""


class ModelConfigError(InputError):
    """The provider rejected the setup (bad request shape, key, model or endpoint).

    The run stops (exit 2) instead of marking every item unchecked.
    """


class ModelTransportError(RuntimeError):
    """The provider could not be reached or kept failing after the retries."""


class ModelClient(Protocol):
    """What `structured.complete` needs from an adapter."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def endpoint(self) -> Endpoint: ...

    def send(self, request: ModelRequest[T], *, correction: str | None = None) -> RawReply:
        """One call. `correction`, on a retry, is appended as the last text block."""
        ...
