"""Structured calls: validate the reply against the output model, retry once, count cost.

`complete` makes one validated call; `complete_batch` does the same through the Message
Batches API. In both, a reply that is not valid JSON for the output model, or that was cut
off by the token limit, is retried once with the validation error appended to the request
(the bad reply itself is not replayed, so the cached prefix stays intact). A second
failure is "model_parse_failure", a refusal "model_refusal", a call that kept failing
"model_error", and a call the cost cap stops is never made: "cost_cap_reached".
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, NamedTuple, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from scribecheck.model.base import (
    Completion,
    ModelClient,
    ModelFailure,
    ModelRequest,
    ModelTransportError,
    RawReply,
    Usage,
)
from scribecheck.model.budget import Budget

T = TypeVar("T", bound=BaseModel)

_UNSUPPORTED = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
    }
)
_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)
"""String formats structured outputs accept; others are dropped (pydantic still checks)."""
_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(?P<body>.*?)\n?\s*```\s*$", re.DOTALL)


def strict_schema(output: type[BaseModel]) -> dict[str, Any]:
    """The output model's JSON schema in the subset structured outputs accept.

    Every object gets `additionalProperties: false`; constraints the providers don't
    enforce (numeric bounds, string lengths, most array bounds, unlisted string formats)
    are removed, `oneOf` becomes `anyOf`, and pydantic still checks everything when the
    reply is validated. Raises `ValueError` for shapes that cannot be expressed: a free-form
    mapping (a `dict` field) or a recursive model.
    """
    schema = output.model_json_schema()
    _reject_recursion(schema)
    return _strict(schema)  # type: ignore[no-any-return]


def _strict(node: Any) -> Any:
    if isinstance(node, list):
        return [_strict(item) for item in node]
    if not isinstance(node, dict):
        return node
    extra = node.get("additionalProperties")
    if extra is True or isinstance(extra, dict):
        raise ValueError("output models cannot have free-form mappings (dict fields)")
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _UNSUPPORTED or key == "discriminator":
            continue
        if key == "minItems" and isinstance(value, int) and value > 1:
            continue
        if key == "format" and value not in _FORMATS:
            continue
        if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
            out[key] = {name: _strict(sub) for name, sub in value.items()}
        elif key == "oneOf":
            out["anyOf"] = _strict(value)
        else:
            out[key] = _strict(value)
    if out.get("type") == "object" or "properties" in out:
        out["additionalProperties"] = False
    return out


def _refs(node: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        if isinstance(node.get("$ref"), str):
            found.add(node["$ref"].rsplit("/", 1)[-1])
        for value in node.values():
            found |= _refs(value)
    elif isinstance(node, list):
        for value in node:
            found |= _refs(value)
    return found


def _reject_recursion(schema: dict[str, Any]) -> None:
    graph = {name: _refs(body) for name, body in (schema.get("$defs") or {}).items()}
    for start in graph:
        seen: set[str] = set()
        stack = list(graph[start])
        while stack:
            name = stack.pop()
            if name == start:
                raise ValueError("output models cannot be recursive")
            if name not in seen:
                seen.add(name)
                stack.extend(graph.get(name, ()))


class Invalid(NamedTuple):
    """Why a reply failed: `feedback` for the model (it may quote the reply) and `detail`
    for reports and logs (error types only, never reply text)."""

    feedback: str
    detail: str


def parse(text: str, output: type[T]) -> T | Invalid:
    """The validated reply, or what was wrong with it."""
    match = _FENCE.match(text)
    body = match.group("body") if match else text.strip()
    if not body:
        return Invalid("the reply was empty", "empty reply")
    try:
        return output.model_validate_json(body)
    except ValidationError as exc:
        errors = exc.errors()
        problems = []
        for error in errors[:5]:
            where = ".".join(str(p) for p in error["loc"]) or "reply"
            problems.append(f"{where}: {error['msg']}")
        kinds = sorted({str(error["type"]) for error in errors})
        return Invalid("; ".join(problems), "validation failed: " + ", ".join(kinds))


def correction(problem: str) -> str:
    return (
        f"Your previous reply did not match the required JSON schema ({problem}). "
        "Reply again with only a JSON object that matches the schema."
    )


def _problem(reply: RawReply, output: type[T]) -> T | Invalid:
    if reply.stop == "max_tokens":
        return Invalid("the reply was cut off at the token limit", "cut off at the token limit")
    return parse(reply.text, output)


def complete(
    client: ModelClient, request: ModelRequest[T], budget: Budget
) -> Completion[T] | ModelFailure:
    """One validated call, with at most one retry after a parse failure."""
    usage = Usage()
    note: str | None = None
    detail = ""
    for attempt in (1, 2):
        if not budget.can_spend():
            return ModelFailure("cost_cap_reached", "the run's cost cap was reached", usage)
        try:
            reply = client.send(request, correction=note)
        except ModelTransportError as exc:
            return ModelFailure("model_error", str(exc), usage)
        budget.record(reply.usage)
        usage = usage + reply.usage
        if reply.stop == "refusal":
            return ModelFailure("model_refusal", "the model declined the request", usage)
        result = _problem(reply, request.output)
        if not isinstance(result, Invalid):
            return Completion(result, usage, reply.model_id, attempt)
        note, detail = correction(result.feedback), result.detail
    return ModelFailure("model_parse_failure", f"no valid reply after 2 attempts ({detail})", usage)


# Batches


class BatchClient(ModelClient, Protocol):
    """An adapter that can also run Message Batches."""

    def run_batch(
        self, requests: Sequence[tuple[ModelRequest[Any], str | None]]
    ) -> dict[str, RawReply | str]:
        """Submit, poll and fetch; each custom_id maps to its reply or an error kind."""
        ...


def complete_batch(
    client: BatchClient, requests: Sequence[ModelRequest[T]], budget: Budget
) -> dict[str, Completion[T] | ModelFailure]:
    """Validated replies for many requests through the batch API, keyed by custom_id.

    Parse failures are retried once, together, in a second batch. Before each batch the
    cap is checked against a worst-case estimate; requests that don't fit are not sent.
    """
    ids = [r.custom_id for r in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("batch requests need distinct custom_ids")
    results: dict[str, Completion[T] | ModelFailure] = {}
    usage: dict[str, Usage] = {r.custom_id: Usage() for r in requests}
    pending: list[tuple[ModelRequest[T], str | None]] = [(r, None) for r in requests]
    for attempt in (1, 2):
        sendable: list[tuple[ModelRequest[T], str | None]] = []
        reserved = 0.0
        for request, note in pending:
            estimate = budget.estimate(
                _chars(request, note), _max_tokens(client, request), batch=True
            )
            if budget.can_spend(reserved + estimate):
                sendable.append((request, note))
                reserved += estimate
            else:
                results[request.custom_id] = ModelFailure(
                    "cost_cap_reached", "the run's cost cap was reached", usage[request.custom_id]
                )
        if not sendable:
            break
        try:
            replies = client.run_batch(sendable)
        except ModelTransportError as exc:
            for request, _ in sendable:
                results[request.custom_id] = ModelFailure(
                    "model_error", str(exc), usage[request.custom_id]
                )
            break
        pending = []
        for request, _ in sendable:
            cid = request.custom_id
            reply = replies.get(cid, "missing")
            if isinstance(reply, str):
                results[cid] = ModelFailure("model_error", f"batch result {reply}", usage[cid])
                continue
            budget.record(reply.usage, batch=True)
            usage[cid] = usage[cid] + reply.usage
            if reply.stop == "refusal":
                results[cid] = ModelFailure(
                    "model_refusal", "the model declined the request", usage[cid]
                )
                continue
            result = _problem(reply, request.output)
            if not isinstance(result, Invalid):
                results[cid] = Completion(result, usage[cid], reply.model_id, attempt)
            elif attempt == 1:
                pending.append((request, correction(result.feedback)))
            else:
                results[cid] = ModelFailure(
                    "model_parse_failure",
                    f"no valid reply after 2 attempts ({result.detail})",
                    usage[cid],
                )
    return {r.custom_id: results[r.custom_id] for r in requests}


def _chars(request: ModelRequest[Any], note: str | None) -> int:
    text = len(request.system) + sum(len(b.text) for m in request.messages for b in m.blocks)
    schema = len(json.dumps(strict_schema(request.output)))
    return text + schema + (len(note) if note else 0)


def _max_tokens(client: ModelClient, request: ModelRequest[Any]) -> int:
    if request.max_tokens is not None:
        return request.max_tokens
    default = getattr(client, "max_tokens", None)
    return default if isinstance(default, int) else 16000
