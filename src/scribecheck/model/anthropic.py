"""The Anthropic Messages API and Message Batches API, over httpx.

Requests use API version 2026-10-21, on which `temperature`, `top_p` and `top_k` are
rejected on every model, so none is sent; nor are `thinking` (adaptive by default) or
`tool_choice` (forced tool use is rejected on current models). The reply's JSON is
requested with `output_config.format` and its effort with `output_config.effort`, one
fixed level per run, since changing it invalidates the prompt cache. Blocks marked for
caching get `cache_control`, at most four per request.

Batches take the same request bodies; results come back as JSONL in any order and are
matched by custom_id.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from typing import Any, Literal

import httpx

from scribecheck.model._http import Sleep, log, request_json, request_text
from scribecheck.model.base import (
    ModelRequest,
    ModelTransportError,
    RawReply,
    StopReason,
    T,
    Usage,
)
from scribecheck.model.endpoint import Endpoint, Secret
from scribecheck.model.structured import strict_schema

API_VERSION = "2026-10-21"
DEFAULT_BASE_URL = "https://api.anthropic.com"
MAX_CACHE_BREAKPOINTS = 4
Effort = Literal["low", "medium", "high", "xhigh", "max"]
CacheTTL = Literal["5m", "1h"]

_CUSTOM_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_BATCH_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_STOPS: dict[str, StopReason] = {
    "end_turn": "end",
    "stop_sequence": "end",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}


class AnthropicClient:
    """A Claude model through the Anthropic API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: Secret,
        base_url: str = DEFAULT_BASE_URL,
        effort: Effort | None = None,
        max_tokens: int = 16000,
        cache_ttl: CacheTTL = "5m",
        batch_cache_ttl: CacheTTL = "1h",
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Sleep = time.sleep,
    ) -> None:
        self._model = model
        self._key = api_key
        self._endpoint = Endpoint.from_url(base_url)
        self.effort = effort
        self.max_tokens = max_tokens
        self.cache_ttl = cache_ttl
        self.batch_cache_ttl = batch_cache_ttl
        self._sleep = sleep
        self._http = httpx.Client(timeout=timeout, transport=transport)

    def __repr__(self) -> str:
        return f"AnthropicClient(model={self._model!r}, host={self._endpoint.host!r})"

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    @property
    def endpoint(self) -> Endpoint:
        return self._endpoint

    def close(self) -> None:
        self._http.close()

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._key.reveal(),
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

    def body(
        self, request: ModelRequest[Any], *, correction: str | None = None, batch: bool = False
    ) -> dict[str, Any]:
        """The Messages API request body (also a batch request's `params`)."""
        ttl = self.batch_cache_ttl if batch else self.cache_ttl
        marker: dict[str, str] = {"type": "ephemeral"}
        if ttl == "1h":
            marker["ttl"] = "1h"
        messages: list[dict[str, Any]] = []
        breakpoints = 0
        for message in request.messages:
            content = []
            for block in message.blocks:
                part: dict[str, Any] = {"type": "text", "text": block.text}
                if block.cache:
                    breakpoints += 1
                    part["cache_control"] = dict(marker)
                content.append(part)
            messages.append({"role": message.role, "content": content})
        if breakpoints > MAX_CACHE_BREAKPOINTS:
            raise ValueError(f"at most {MAX_CACHE_BREAKPOINTS} cached blocks per request")
        if correction is not None:
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"].append({"type": "text", "text": correction})
            else:
                messages.append({"role": "user", "content": [{"type": "text", "text": correction}]})
        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": strict_schema(request.output)}
        }
        if self.effort is not None:
            output_config["effort"] = self.effort
        return {
            "model": self._model,
            "max_tokens": request.max_tokens or self.max_tokens,
            "system": request.system,
            "messages": messages,
            "output_config": output_config,
        }

    def send(self, request: ModelRequest[T], *, correction: str | None = None) -> RawReply:
        data = request_json(
            self._http,
            "POST",
            f"{self._endpoint.base_url}/v1/messages",
            headers=self._headers(),
            body=self.body(request, correction=correction),
            sleep=self._sleep,
            provider=self.provider,
        )
        return _reply(data)

    # Batches

    def submit_batch(self, requests: Sequence[tuple[ModelRequest[Any], str | None]]) -> str:
        """Submit requests (each with an optional correction); returns the batch id."""
        items = []
        for request, correction in requests:
            if not _CUSTOM_ID.fullmatch(request.custom_id):
                raise ValueError(
                    f"custom_id {request.custom_id!r} must be 1-64 letters, digits, - or _"
                )
            params = self.body(request, correction=correction, batch=True)
            items.append({"custom_id": request.custom_id, "params": params})
        data = request_json(
            self._http,
            "POST",
            f"{self._endpoint.base_url}/v1/messages/batches",
            headers=self._headers(),
            body={"requests": items},
            sleep=self._sleep,
            provider=self.provider,
        )
        return _checked_id(data.get("id"))

    def poll_batch(
        self, batch_id: str, *, interval: float = 30.0, timeout: float = 25 * 3600
    ) -> dict[str, Any]:
        """Wait until the batch has ended; returns the batch object."""
        batch_id = _checked_id(batch_id)
        waited = 0.0
        while True:
            batch = request_json(
                self._http,
                "GET",
                f"{self._endpoint.base_url}/v1/messages/batches/{batch_id}",
                headers=self._headers(),
                sleep=self._sleep,
                provider=self.provider,
            )
            if batch.get("processing_status") == "ended":
                return dict(batch)
            if waited >= timeout:
                raise ModelTransportError(f"anthropic: batch {batch_id} did not end in time")
            self._sleep(interval)
            waited += interval

    def fetch_batch(self, batch: dict[str, Any]) -> dict[str, RawReply | str]:
        """Each custom_id's reply, or its result type when it did not succeed.

        Results are always fetched from the configured endpoint, never from a host the
        response names, so the key goes nowhere else. A malformed line is skipped (its
        request is reported as missing) rather than losing the batch's other results.
        """
        batch_id = _checked_id(batch.get("id"))
        url = f"{self._endpoint.base_url}/v1/messages/batches/{batch_id}/results"
        text = request_text(
            self._http, url, headers=self._headers(), sleep=self._sleep, provider=self.provider
        )
        results: dict[str, RawReply | str] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                cid = entry["custom_id"]
                result = entry.get("result") or {}
                kind = result.get("type")
                value: RawReply | str = (
                    _reply(result["message"]) if kind == "succeeded" else str(kind or "unknown")
                )
            except (ValueError, KeyError, TypeError, AttributeError):
                log.warning("anthropic: skipped a malformed batch result line")
                continue
            if isinstance(cid, str):
                results[cid] = value
        return results

    def run_batch(
        self, requests: Sequence[tuple[ModelRequest[Any], str | None]]
    ) -> dict[str, RawReply | str]:
        return self.fetch_batch(self.poll_batch(self.submit_batch(requests)))


def _checked_id(value: object) -> str:
    """A batch id safe to put in a URL path."""
    if not isinstance(value, str) or not _BATCH_ID.fullmatch(value):
        raise ModelTransportError("anthropic: the API returned an invalid batch id")
    return value


def _reply(data: dict[str, Any]) -> RawReply:
    text = "".join(
        block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
    )
    usage = data.get("usage") or {}
    creation = usage.get("cache_creation") or {}
    write_5m = creation.get("ephemeral_5m_input_tokens")
    write_1h = creation.get("ephemeral_1h_input_tokens", 0)
    if write_5m is None:  # no breakdown: count every write as 5-minute
        write_5m = usage.get("cache_creation_input_tokens", 0) - (write_1h or 0)
    return RawReply(
        text=text,
        stop=_STOPS.get(str(data.get("stop_reason")), "other"),
        usage=Usage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_write_5m_tokens=int(write_5m or 0),
            cache_write_1h_tokens=int(write_1h or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
        ),
        model_id=data.get("model"),
    )
