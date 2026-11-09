"""OpenAI-compatible chat completions over httpx: Ollama, vLLM, LM Studio, llama.cpp server,
OpenRouter, Azure OpenAI and OpenAI.

Calls use temperature 0 and a fixed seed unless configured otherwise (a null temperature
omits the field, for models that reject it). JSON is requested with `response_format`:
a JSON schema (default), plain JSON mode, or nothing for servers that reject the field, in
which case the schema is given in the system prompt. Cache markers are ignored: these
providers cache prompt prefixes automatically, where they cache at all.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from typing import Any, Literal

import httpx

from scribecheck.model._http import Sleep, request_json
from scribecheck.model.base import ModelRequest, RawReply, StopReason, T, Usage
from scribecheck.model.endpoint import Endpoint, Secret
from scribecheck.model.structured import strict_schema

JsonMode = Literal["json_schema", "json_object", "none"]

_STOPS: dict[str, StopReason] = {
    "stop": "end",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class OpenAICompatClient:
    """A model behind an OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: Secret | None = None,
        temperature: float | None = 0.0,
        seed: int | None = 0,
        json_mode: JsonMode = "json_schema",
        max_tokens: int = 16000,
        auth_header: str = "authorization",
        query: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Sleep = time.sleep,
    ) -> None:
        self._model = model
        self._key = api_key
        self._endpoint = Endpoint.from_url(base_url)
        self.temperature = temperature
        self.seed = seed
        self.json_mode = json_mode
        self.max_tokens = max_tokens
        self._auth_header = auth_header.lower()
        self._query = dict(query or {})
        self._extra = dict(extra_body or {})
        self._sleep = sleep
        self._http = httpx.Client(timeout=timeout, transport=transport)

    def __repr__(self) -> str:
        return f"OpenAICompatClient(model={self._model!r}, host={self._endpoint.host!r})"

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    @property
    def endpoint(self) -> Endpoint:
        return self._endpoint

    def close(self) -> None:
        self._http.close()

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._key is not None:
            if self._auth_header == "authorization":
                headers["authorization"] = f"Bearer {self._key.reveal()}"
            else:
                headers[self._auth_header] = self._key.reveal()
        return headers

    def body(self, request: ModelRequest[Any], *, correction: str | None = None) -> dict[str, Any]:
        prompt_only = request.json_mode == "prompt"
        schema = None if prompt_only else strict_schema(request.output)
        system = request.system
        if schema is not None and self.json_mode != "json_schema":
            system += "\n\nReply with only a JSON object matching this JSON schema:\n" + json.dumps(
                schema, sort_keys=True
            )
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        for message in request.messages:
            messages.append(
                {"role": message.role, "content": "\n\n".join(b.text for b in message.blocks)}
            )
        if correction is not None:
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\n" + correction
            else:
                messages.append({"role": "user", "content": correction})
        body: dict[str, Any] = {
            "model": request.model or self._model,
            "messages": messages,
            "max_tokens": request.max_tokens or self.max_tokens,
        }
        for name, value in self._extra.items():  # e.g. OpenRouter's provider pin
            body.setdefault(name, value)
        temperature = request.temperature if request.temperature is not None else self.temperature
        if temperature is not None:
            body["temperature"] = temperature
        if request.effort is not None:
            body["reasoning_effort"] = request.effort
        if self.seed is not None:
            body["seed"] = self.seed
        if schema is not None and self.json_mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _schema_name(request.output),
                    "schema": schema,
                    "strict": False,
                },
            }
        elif schema is not None and self.json_mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        return body

    def send(self, request: ModelRequest[T], *, correction: str | None = None) -> RawReply:
        data = request_json(
            self._http,
            "POST",
            f"{self._endpoint.base_url}/chat/completions",
            headers=self._headers(),
            body=self.body(request, correction=correction),
            params=self._query or None,
            sleep=self._sleep,
            provider=self.provider,
        )
        return _reply(data)


def _schema_name(output: type[Any]) -> str:
    """The output model's name in the form OpenAI requires: 1-64 of [A-Za-z0-9_-]."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", output.__name__).strip("_")[:64]
    return name or "output"


def _reply(data: dict[str, Any]) -> RawReply:
    choices = data.get("choices") or [{}]
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):  # some servers return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    stop = _STOPS.get(str(choice.get("finish_reason")), "other")
    if message.get("refusal"):
        stop = "refusal"
    return RawReply(
        text=content or "",
        stop=stop,
        usage=Usage(
            input_tokens=max(int(usage.get("prompt_tokens") or 0) - cached, 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cache_read_tokens=cached,
        ),
        model_id=data.get("model"),
    )
