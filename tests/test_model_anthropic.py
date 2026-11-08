"""The Anthropic adapter's request shape, caching markers and reply parsing (mocked HTTP)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from _mock_http import Recorder, anthropic_message, json_response
from pydantic import BaseModel, Field

from scribecheck.model.anthropic import API_VERSION, AnthropicClient
from scribecheck.model.base import Block, Message, ModelRequest
from scribecheck.model.endpoint import Secret
from scribecheck.model.structured import strict_schema

KEY = "sk-ant-test-0000"


class Verdict(BaseModel):
    verdict: str = Field(min_length=1, max_length=20)
    confidence: float = Field(ge=0, le=1)
    quotes: list[str] = Field(default_factory=list, min_length=0, max_length=5)


class Nested(BaseModel):
    items: list[Verdict] = Field(min_length=2)


def _client(recorder: Recorder, **kwargs: Any) -> AnthropicClient:
    return AnthropicClient(
        "claude-sonnet-5-5",
        api_key=Secret(KEY),
        transport=recorder.transport,
        sleep=lambda _: None,
        **kwargs,
    )


def _request(output: type[BaseModel] = Verdict, **kwargs: Any) -> ModelRequest[Any]:
    return ModelRequest(
        system="You check clinical notes.",
        messages=(
            Message(
                "user",
                (
                    Block("<transcript>synthetic transcript</transcript>", cache=True),
                    Block("<note>synthetic note</note>", cache=True),
                    Block("Is the fact present?"),
                ),
            ),
        ),
        output=output,
        **kwargs,
    )


def _reply(**kwargs: Any) -> Any:
    return json_response(anthropic_message('{"verdict": "present", "confidence": 0.9}', **kwargs))


def test_headers_and_endpoint() -> None:
    recorder = Recorder(_reply())
    _client(recorder).send(_request())
    request = recorder.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == KEY
    assert request.headers["anthropic-version"] == API_VERSION == "2026-10-21"
    assert "authorization" not in request.headers


def test_body_sends_no_sampling_thinking_or_tool_choice() -> None:
    recorder = Recorder(_reply())
    _client(recorder, effort="low", max_tokens=2048).send(_request())
    body = recorder.body()
    for field in ("temperature", "top_p", "top_k", "thinking", "tool_choice", "tools", "stream"):
        assert field not in body
    assert body["model"] == "claude-sonnet-5-5"
    assert body["max_tokens"] == 2048
    assert body["system"] == "You check clinical notes."
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"


def test_effort_is_left_out_when_unset() -> None:
    recorder = Recorder(_reply())
    _client(recorder).send(_request())
    assert "effort" not in recorder.body()["output_config"]


def _walk(node: Any) -> list[dict[str, Any]]:
    found = []
    if isinstance(node, dict):
        found.append(node)
        for value in node.values():
            found += _walk(value)
    elif isinstance(node, list):
        for value in node:
            found += _walk(value)
    return found


@pytest.mark.parametrize("output", [Verdict, Nested])
def test_schema_is_in_the_supported_subset(output: type[BaseModel]) -> None:
    schema = strict_schema(output)
    nodes = _walk(schema)
    objects = [n for n in nodes if n.get("type") == "object"]
    assert objects and all(n["additionalProperties"] is False for n in objects)
    banned = {"minimum", "maximum", "minLength", "maxLength", "maxItems"}
    assert not [n for n in nodes if banned & n.keys()]
    assert not [n for n in nodes if isinstance(n.get("minItems"), int) and n["minItems"] > 1]


def test_cache_markers_only_on_flagged_blocks_with_5m_default() -> None:
    recorder = Recorder(_reply())
    _client(recorder).send(_request())
    content = recorder.body()["messages"][0]["content"]
    assert [part.get("cache_control") for part in content] == [
        {"type": "ephemeral"},
        {"type": "ephemeral"},
        None,
    ]
    assert content[0]["text"].startswith("<transcript>")  # the transcript comes first


def test_cache_ttl_one_hour() -> None:
    recorder = Recorder(_reply())
    _client(recorder, cache_ttl="1h").send(_request())
    content = recorder.body()["messages"][0]["content"]
    assert content[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_more_than_four_cache_markers_is_an_error() -> None:
    blocks = tuple(Block(f"part {i}", cache=True) for i in range(5))
    request = ModelRequest(system="s", messages=(Message("user", blocks),), output=Verdict)
    with pytest.raises(ValueError, match="at most 4"):
        _client(Recorder()).body(request)


def test_correction_is_appended_after_the_cached_prefix() -> None:
    recorder = Recorder(_reply())
    _client(recorder).send(_request(), correction="Fix the JSON.")
    content = recorder.body()["messages"][0]["content"]
    assert content[-1] == {"type": "text", "text": "Fix the JSON."}
    assert content[0]["cache_control"] == {"type": "ephemeral"}


def test_reply_skips_thinking_and_maps_usage_and_model() -> None:
    usage = {
        "input_tokens": 50,
        "output_tokens": 20,
        "cache_creation_input_tokens": 700,
        "cache_read_input_tokens": 300,
        "cache_creation": {"ephemeral_5m_input_tokens": 400, "ephemeral_1h_input_tokens": 300},
    }
    recorder = Recorder(_reply(usage=usage, thinking=True, model="claude-sonnet-5-5-20261101"))
    reply = _client(recorder).send(_request())
    assert json.loads(reply.text) == {"verdict": "present", "confidence": 0.9}
    assert reply.stop == "end"
    assert reply.model_id == "claude-sonnet-5-5-20261101"
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (50, 20)
    assert (reply.usage.cache_write_5m_tokens, reply.usage.cache_write_1h_tokens) == (400, 300)
    assert reply.usage.cache_read_tokens == 300


def test_usage_without_a_breakdown_counts_writes_as_5m() -> None:
    usage = {"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 90}
    reply = _client(Recorder(_reply(usage=usage))).send(_request())
    assert (reply.usage.cache_write_5m_tokens, reply.usage.cache_write_1h_tokens) == (90, 0)


@pytest.mark.parametrize(
    ("stop", "expected"),
    [("end_turn", "end"), ("max_tokens", "max_tokens"), ("refusal", "refusal"), ("pause", "other")],
)
def test_stop_reasons(stop: str, expected: str) -> None:
    reply = _client(Recorder(_reply(stop=stop))).send(_request())
    assert reply.stop == expected


def test_custom_base_url() -> None:
    recorder = Recorder(_reply())
    AnthropicClient(
        "claude-sonnet-5-5",
        api_key=Secret(KEY),
        base_url="https://gateway.example/anthropic/",
        transport=recorder.transport,
    ).send(_request())
    assert str(recorder.requests[0].url) == "https://gateway.example/anthropic/v1/messages"
