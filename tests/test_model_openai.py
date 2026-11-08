"""The OpenAI-compatible adapter: request shape, auth, JSON modes and usage (mocked HTTP)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from _mock_http import Recorder, json_response, openai_completion
from pydantic import BaseModel

from scribecheck.model.base import Block, Completion, Message, ModelRequest
from scribecheck.model.budget import Budget
from scribecheck.model.endpoint import Secret
from scribecheck.model.openai_compat import OpenAICompatClient
from scribecheck.model.structured import complete


class Presence(BaseModel):
    present: bool


REQUEST = ModelRequest(
    system="You check clinical notes.",
    messages=(Message("user", (Block("transcript", cache=True), Block("question"))),),
    output=Presence,
)


def _client(recorder: Recorder, **kwargs: Any) -> OpenAICompatClient:
    defaults: dict[str, Any] = {"base_url": "http://localhost:11434/v1"}
    defaults.update(kwargs)
    return OpenAICompatClient(
        "local-model", transport=recorder.transport, sleep=lambda _: None, **defaults
    )


def _ok(**kwargs: Any) -> Any:
    return json_response(openai_completion('{"present": true}', **kwargs))


def test_default_body_uses_temperature_zero_seed_and_json_schema() -> None:
    recorder = Recorder(_ok())
    _client(recorder).send(REQUEST)
    request = recorder.requests[0]
    assert str(request.url) == "http://localhost:11434/v1/chat/completions"
    body = recorder.body()
    assert body["temperature"] == 0.0 and body["seed"] == 0
    assert body["messages"] == [
        {"role": "system", "content": "You check clinical notes."},
        {"role": "user", "content": "transcript\n\nquestion"},
    ]
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "Presence"
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False
    assert "authorization" not in request.headers  # local server, no key


def test_temperature_and_seed_can_be_omitted() -> None:
    recorder = Recorder(_ok())
    _client(recorder, temperature=None, seed=None).send(REQUEST)
    body = recorder.body()
    assert "temperature" not in body and "seed" not in body


@pytest.mark.parametrize("mode", ["json_object", "none"])
def test_other_json_modes_put_the_schema_in_the_system_prompt(mode: str) -> None:
    recorder = Recorder(_ok())
    _client(recorder, json_mode=mode).send(REQUEST)
    body = recorder.body()
    system = body["messages"][0]["content"]
    assert "JSON schema" in system and '"present"' in system
    if mode == "json_object":
        assert body["response_format"] == {"type": "json_object"}
    else:
        assert "response_format" not in body


def test_bearer_auth() -> None:
    recorder = Recorder(_ok())
    _client(recorder, base_url="https://openrouter.ai/api/v1", api_key=Secret("sk-or-1")).send(
        REQUEST
    )
    assert recorder.requests[0].headers["authorization"] == "Bearer sk-or-1"


def test_azure_api_key_header_and_version_query() -> None:
    recorder = Recorder(_ok())
    _client(
        recorder,
        base_url="https://example-resource.openai.azure.com/openai/deployments/dep1",
        api_key=Secret("azure-key"),
        auth_header="api-key",
        query={"api-version": "2026-01-01"},
    ).send(REQUEST)
    request = recorder.requests[0]
    assert request.headers["api-key"] == "azure-key"
    assert "authorization" not in request.headers
    assert request.url.params["api-version"] == "2026-01-01"
    assert request.url.path == "/openai/deployments/dep1/chat/completions"


def test_usage_and_cached_tokens_and_model_id() -> None:
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 30,
        "prompt_tokens_details": {"cached_tokens": 800},
    }
    recorder = Recorder(_ok(usage=usage, model="qwen-test-7b"))
    reply = _client(recorder).send(REQUEST)
    assert (reply.usage.input_tokens, reply.usage.cache_read_tokens) == (200, 800)
    assert reply.usage.output_tokens == 30
    assert reply.model_id == "qwen-test-7b"


def test_length_finish_is_retried_and_correction_appended() -> None:
    recorder = Recorder(_ok(finish="length"), _ok())
    result = complete(_client(recorder), REQUEST, Budget())
    assert isinstance(result, Completion) and result.attempts == 2
    last = recorder.body(1)["messages"][-1]["content"]
    assert last.endswith("Reply again with only a JSON object that matches the schema.")


def test_content_parts_and_refusal() -> None:
    parts = openai_completion("")
    parts["choices"][0]["message"]["content"] = [{"type": "text", "text": '{"present": false}'}]
    reply = _client(Recorder(json_response(parts))).send(REQUEST)
    assert json.loads(reply.text) == {"present": False}
    refused = openai_completion("")
    refused["choices"][0]["message"]["refusal"] = "I can't help with that."
    assert _client(Recorder(json_response(refused))).send(REQUEST).stop == "refusal"
