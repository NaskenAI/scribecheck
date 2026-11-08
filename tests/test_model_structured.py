"""Validation, the one retry, refusals, transport retries and setup errors (mocked HTTP)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from _mock_http import Recorder, anthropic_message, json_response
from pydantic import BaseModel

from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.base import (
    Block,
    Completion,
    Message,
    ModelConfigError,
    ModelFailure,
    ModelRequest,
)
from scribecheck.model.budget import Budget
from scribecheck.model.endpoint import Secret
from scribecheck.model.structured import complete, parse

GOOD = '{"present": true, "quote": "metformin"}'


class Presence(BaseModel):
    present: bool
    quote: str | None = None


def _client(recorder: Recorder, sleeps: list[float] | None = None) -> AnthropicClient:
    record = sleeps if sleeps is not None else []
    return AnthropicClient(
        "claude-sonnet-5-5",
        api_key=Secret("sk-test"),
        transport=recorder.transport,
        sleep=record.append,
    )


REQUEST = ModelRequest(
    system="s",
    messages=(Message("user", (Block("transcript", cache=True), Block("question"))),),
    output=Presence,
)


def _msg(text: str, **kwargs: Any) -> httpx.Response:
    return json_response(anthropic_message(text, **kwargs))


def test_valid_reply_first_time() -> None:
    recorder = Recorder(_msg(GOOD))
    result = complete(_client(recorder), REQUEST, Budget())
    assert isinstance(result, Completion)
    assert result.parsed == Presence(present=True, quote="metformin")
    assert (result.attempts, result.model_id) == (1, "claude-sonnet-5-5")


def test_invalid_then_valid_retries_once_with_the_error() -> None:
    recorder = Recorder(_msg('{"present": "maybe"}'), _msg(GOOD))
    budget = Budget()
    result = complete(_client(recorder), REQUEST, budget)
    assert isinstance(result, Completion) and result.attempts == 2
    assert len(recorder.requests) == 2
    retry = recorder.body(1)["messages"][0]["content"]
    assert "did not match the required JSON schema" in retry[-1]["text"]
    assert "present" in retry[-1]["text"]  # names the failing field
    assert retry[0]["cache_control"] == {"type": "ephemeral"}  # the prefix is unchanged
    assert result.usage.input_tokens == 20 and budget.calls == 2


def test_two_failures_give_model_parse_failure() -> None:
    recorder = Recorder(_msg("not json"), _msg("still not json"))
    result = complete(_client(recorder), REQUEST, Budget())
    assert isinstance(result, ModelFailure)
    assert result.reason == "model_parse_failure"
    assert len(recorder.requests) == 2
    assert result.usage.output_tokens == 10


def test_truncated_reply_is_retried() -> None:
    recorder = Recorder(_msg('{"present": tr', stop="max_tokens"), _msg(GOOD))
    result = complete(_client(recorder), REQUEST, Budget())
    assert isinstance(result, Completion)
    assert "cut off at the token limit" in recorder.body(1)["messages"][0]["content"][-1]["text"]


def test_refusal_is_not_retried() -> None:
    recorder = Recorder(_msg("", stop="refusal"))
    result = complete(_client(recorder), REQUEST, Budget())
    assert isinstance(result, ModelFailure) and result.reason == "model_refusal"
    assert len(recorder.requests) == 1


def test_fenced_json_is_accepted() -> None:
    assert parse(f"```json\n{GOOD}\n```", Presence) == Presence(present=True, quote="metformin")
    assert parse("", Presence) == ("the reply was empty", "empty reply")


@pytest.mark.parametrize("status", [429, 500, 529])
def test_transient_errors_are_retried_honouring_retry_after(status: int) -> None:
    sleeps: list[float] = []
    recorder = Recorder(
        json_response({"error": {"type": "overloaded_error"}}, status, {"retry-after": "7"}),
        _msg(GOOD),
    )
    result = complete(_client(recorder, sleeps), REQUEST, Budget())
    assert isinstance(result, Completion)
    assert sleeps == [7.0]


def test_exhausted_retries_give_model_error() -> None:
    recorder = Recorder(*(json_response({}, 529) for _ in range(3)))
    result = complete(_client(recorder), REQUEST, Budget())
    assert isinstance(result, ModelFailure) and result.reason == "model_error"
    assert "HTTP 529" in result.detail
    assert len(recorder.requests) == 3


def test_timeouts_are_retried_then_give_model_error() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    recorder = Recorder(timeout, timeout, timeout)
    result = complete(_client(recorder), REQUEST, Budget())
    assert isinstance(result, ModelFailure) and result.reason == "model_error"
    assert "ReadTimeout" in result.detail


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_setup_errors_stop_the_run(status: int) -> None:
    error = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad field"}}
    recorder = Recorder(json_response(error, status))
    with pytest.raises(ModelConfigError, match=f"HTTP {status} \\(invalid_request_error\\)$"):
        complete(_client(recorder), REQUEST, Budget())
    assert len(recorder.requests) == 1
