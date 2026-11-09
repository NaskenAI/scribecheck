"""Regression tests for the model client's review findings (mocked HTTP only)."""

from __future__ import annotations

import json
import logging
from typing import Any, Generic, Literal, TypeVar

import httpx
import pytest
from _mock_http import Recorder, anthropic_message, json_response, openai_completion
from pydantic import BaseModel, ConfigDict

from scribecheck.config import Config, config_from_mapping
from scribecheck.errors import InputError
from scribecheck.model._http import _backoff
from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.base import Block, Completion, Message, ModelFailure, ModelRequest
from scribecheck.model.budget import Budget, Prices
from scribecheck.model.endpoint import Endpoint, Secret, api_key
from scribecheck.model.openai_compat import OpenAICompatClient
from scribecheck.model.structured import complete, complete_batch, parse, strict_schema

KEY = "sk-test-HARDEN-91b2"
BASE = "https://api.anthropic.com/v1/messages/batches"


class Presence(BaseModel):
    present: bool


def _request(cid: str = "r1") -> ModelRequest[Presence]:
    return ModelRequest(
        system="s",
        messages=(Message("user", (Block("transcript", cache=True), Block("q"))),),
        output=Presence,
        custom_id=cid,
    )


def _anthropic(recorder: Recorder) -> AnthropicClient:
    return AnthropicClient(
        "claude-sonnet-5-5",
        api_key=Secret(KEY),
        transport=recorder.transport,
        sleep=lambda _: None,
        max_tokens=100,
    )


def test_batch_results_never_go_to_another_host() -> None:
    recorder = Recorder(
        json_response({"id": "msgbatch_1", "processing_status": "in_progress"}),
        json_response(
            {
                "id": "msgbatch_1",
                "processing_status": "ended",
                "results_url": "https://elsewhere.example.net/steal",
            }
        ),
        httpx.Response(
            200,
            text=json.dumps(
                {
                    "custom_id": "r1",
                    "result": {
                        "type": "succeeded",
                        "message": anthropic_message('{"present": true}'),
                    },
                }
            ),
        ),
    )
    out = complete_batch(_anthropic(recorder), [_request()], Budget())
    assert isinstance(out["r1"], Completion)
    assert {r.url.host for r in recorder.requests} == {"api.anthropic.com"}
    assert str(recorder.requests[-1].url) == f"{BASE}/msgbatch_1/results"


def test_invalid_batch_id_is_refused() -> None:
    recorder = Recorder(json_response({"id": "../../v1/other", "processing_status": "x"}))
    out = complete_batch(_anthropic(recorder), [_request()], Budget())
    failure = out["r1"]
    assert isinstance(failure, ModelFailure) and failure.reason == "model_error"
    assert len(recorder.requests) == 1


def test_malformed_result_lines_keep_the_good_ones_and_their_cost() -> None:
    good = {
        "custom_id": "a",
        "result": {"type": "succeeded", "message": anthropic_message('{"present": true}')},
    }
    recorder = Recorder(
        json_response({"id": "msgbatch_2", "processing_status": "in_progress"}),
        json_response({"id": "msgbatch_2", "processing_status": "ended"}),
        httpx.Response(200, text=json.dumps(good) + '\n{"custom_id": "b", "resu\n{"no": 1}\n'),
    )
    budget = Budget()
    out = complete_batch(_anthropic(recorder), [_request("a"), _request("b")], budget)
    assert isinstance(out["a"], Completion)
    failure = out["b"]
    assert isinstance(failure, ModelFailure) and failure.reason == "model_error"
    assert budget.usage.input_tokens == 10  # the paid result is still counted


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "soon", "Wed, 21 Oct 2026 07:28:00 GMT"])
def test_retry_after_values_that_are_not_finite_seconds(value: str) -> None:
    assert 0 < _backoff(1, value) <= 60


def test_408_is_retried() -> None:
    recorder = Recorder(
        json_response({}, 408), json_response(anthropic_message('{"present": true}'))
    )
    assert isinstance(complete(_anthropic(recorder), _request(), Budget()), Completion)


def test_a_post_that_may_have_reached_the_server_is_not_repeated() -> None:
    def read_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    recorder = Recorder(read_timeout)
    result = complete(_anthropic(recorder), _request(), Budget())
    assert isinstance(result, ModelFailure) and result.reason == "model_error"
    assert len(recorder.requests) == 1


def test_a_connection_that_never_opened_is_retried() -> None:
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    recorder = Recorder(refused, json_response(anthropic_message('{"present": true}')))
    assert isinstance(complete(_anthropic(recorder), _request(), Budget()), Completion)


def test_provider_error_messages_are_not_repeated(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    error = {"error": {"type": "server_error", "message": "failed on: 'Patient SECRET pain'"}}
    recorder = Recorder(*(json_response(error, 500) for _ in range(3)))
    client = OpenAICompatClient(
        "m", base_url="http://localhost:8000/v1", transport=recorder.transport, sleep=lambda _: None
    )
    result = complete(client, _request(), Budget())
    assert isinstance(result, ModelFailure) and result.reason == "model_error"
    assert "server_error" in result.detail
    assert "SECRET" not in result.detail and "SECRET" not in caplog.text


def test_failure_detail_never_quotes_the_reply() -> None:
    class Strict(BaseModel):
        model_config = ConfigDict(extra="forbid")
        kind: Literal["present", "absent"]

    bad = '{"kind": "patient SECRET has pain", "SECRET_name_Smith": 1}'
    invalid = parse(bad, Strict)
    assert not isinstance(invalid, BaseModel)
    assert "SECRET" in invalid.feedback  # the model may see its own reply
    assert "SECRET" not in invalid.detail  # reports and logs may not
    recorder = Recorder(*(json_response(anthropic_message(bad)) for _ in range(2)))
    request = ModelRequest(system="s", messages=(Message("user", (Block("q"),)),), output=Strict)
    failure = complete(_anthropic(recorder), request, Budget())
    assert isinstance(failure, ModelFailure) and failure.reason == "model_parse_failure"
    assert "SECRET" not in failure.detail


def test_batch_estimate_assumes_the_dearest_input_rate() -> None:
    budget = Budget(
        prices=Prices(input=3.0, output=15.0, cache_write_1h=6.0, batch_multiplier=0.5),
        cap_usd=100.0,
    )
    # 300,000 characters: 150,001 tokens at $6/MTok, plus 0 output, halved.
    assert budget.estimate(300_000, 0, batch=True) == pytest.approx(150_001 * 6 / 1e6 / 2)


def test_dict_fields_and_recursive_models_are_refused() -> None:
    class Counts(BaseModel):
        counts: dict[str, int]

    class Node(BaseModel):
        children: list[Node] = []

    with pytest.raises(ValueError, match="free-form"):
        strict_schema(Counts)
    with pytest.raises(ValueError, match="recursive"):
        strict_schema(Node)


def test_unsupported_formats_and_oneof() -> None:
    from typing import Annotated
    from uuid import UUID

    from pydantic import Field
    from pydantic.types import UUID4

    class A(BaseModel):
        kind: Literal["a"]

    class B(BaseModel):
        kind: Literal["b"]

    class Mixed(BaseModel):
        ident: UUID
        ident4: UUID4
        item: Annotated[A | B, Field(discriminator="kind")]

    schema = json.dumps(strict_schema(Mixed))
    assert '"uuid4"' not in schema and '"discriminator"' not in schema and '"oneOf"' not in schema
    assert '"anyOf"' in schema and '"format": "uuid"' in schema


T = TypeVar("T")


class Wrapper(BaseModel, Generic[T]):
    value: T


def test_generic_output_names_are_cleaned_for_openai() -> None:
    recorder = Recorder(json_response(openai_completion('{"value": 1}')))
    client = OpenAICompatClient("m", base_url="http://localhost:1/v1", transport=recorder.transport)
    request: ModelRequest[Any] = ModelRequest(
        system="s", messages=(Message("user", (Block("q"),)),), output=Wrapper[int]
    )
    client.send(request)
    assert recorder.body()["response_format"]["json_schema"]["name"] == "Wrapper_int"


@pytest.mark.parametrize(
    "url",
    [
        "https://user:hunter2@api.example.com/v1",
        "https://api.example.com/v1?api_key=abc",
        "https://api.example.com/v1#frag",
        "ftp://user:hunter2@x",
        "https://api.example.com:notaport/v1",
    ],
)
def test_urls_with_credentials_queries_or_bad_ports_are_refused(url: str) -> None:
    with pytest.raises(InputError) as raised:
        Endpoint.from_url(url)
    assert "hunter2" not in str(raised.value) and "abc" not in str(raised.value)
    with pytest.raises(InputError) as raised_config:
        config_from_mapping({"model": {"base_url": url}})
    assert "hunter2" not in str(raised_config.value)


def test_a_key_pasted_as_the_variable_name_is_not_echoed() -> None:
    endpoint = Endpoint.from_url("https://api.example.com/v1")
    with pytest.raises(InputError) as raised:
        api_key("openai", endpoint, "sk-LIVEKEY-123")
    assert "sk-LIVEKEY-123" not in str(raised.value)
    with pytest.raises(InputError) as raised_config:
        config_from_mapping({"model": {"api_key_env": "sk-LIVEKEY-123"}})
    assert "sk-LIVEKEY-123" not in str(raised_config.value)
    assert Config(api_key_env="MY_KEY_1").api_key_env == "MY_KEY_1"


@pytest.mark.parametrize("name", ["api_key", "key", "token", "sig"])
def test_a_key_in_the_query_table_is_refused(name: str) -> None:
    with pytest.raises(InputError, match="never put an API key"):
        config_from_mapping({"model": {"query": {name: "abc"}}})
    assert config_from_mapping({"model": {"query": {"api-version": "2026-01-01"}}}).query == {
        "api-version": "2026-01-01"
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("seed", True),
        ("seed", 1.0),
        ("temperature", "0.7"),
        ("temperature", True),
        ("max_tokens", 1.5),
        ("max_tokens", "100"),
    ],
)
def test_toml_types_are_strict(key: str, value: object) -> None:
    with pytest.raises(InputError, match=f"model.{key}"):
        config_from_mapping({"model": {key: value}})


def test_integer_temperature_and_none_values_are_accepted() -> None:
    config = config_from_mapping({"model": {"temperature": 0, "seed": "none", "timeout_s": 30}})
    assert (config.temperature, config.seed, config.timeout_s) == (0, None, 30)


def test_httpx_request_logging_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="scribecheck")
    recorder = Recorder(json_response(openai_completion('{"present": true}')))
    client = OpenAICompatClient(
        "m",
        base_url="https://api.example.com/v1",
        api_key=Secret(KEY),
        query={"api-version": "1"},
        transport=recorder.transport,
    )
    complete(client, _request(), Budget())
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
    assert "HTTP Request" not in caplog.text and KEY not in caplog.text


def test_extra_body_is_merged_for_openai_compatible_requests() -> None:
    recorder = Recorder(json_response(openai_completion('{"present": true}')))
    pin = {"provider": {"order": ["GoodFP8"], "allow_fallbacks": False}}
    client = OpenAICompatClient(
        "m",
        base_url="https://openrouter.ai/api/v1",
        api_key=Secret(KEY),
        extra_body=pin,
        transport=recorder.transport,
    )
    complete(client, _request(), Budget())
    body = recorder.body()
    assert body["provider"] == pin["provider"] and body["model"] == "m"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"model": "other"}, "may not set 'model'"),
        ({"temperature": 1}, "may not set 'temperature'"),
        ({"provider": {"api_key": "abc"}}, "never put an API key"),
        ({"headers": [{"token": "abc"}]}, "never put an API key"),
    ],
)
def test_extra_body_rejects_overrides_and_keys(extra: dict[str, Any], message: str) -> None:
    with pytest.raises(InputError, match=message) as raised:
        config_from_mapping({"model": {"extra_body": extra}})
    assert "abc" not in str(raised.value)


def test_extra_body_is_refused_for_anthropic() -> None:
    with pytest.raises(InputError, match="OpenAI-compatible endpoints only"):
        config_from_mapping(
            {"model": {"provider": "anthropic", "extra_body": {"provider": {"order": ["x"]}}}}
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"stream": True},
        {"models": ["a", "b"]},
        {"route": "fallback"},
        {"max_completion_tokens": 5},
        {"provider": {"model": "other"}},
        {"headers": {"Authorization": "x"}},
        {"provider": {"api_token": "x"}},
        {"note": "sk-or-v1-abc"},
    ],
)
def test_extra_body_rejects_bypasses(extra: dict[str, Any]) -> None:
    with pytest.raises(InputError) as raised:
        config_from_mapping({"model": {"extra_body": extra}})
    assert "sk-or-v1-abc" not in str(raised.value)
