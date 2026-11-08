"""Message Batches: submit, poll, fetch, the correction batch and the cost cap (mocked HTTP)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from _mock_http import Recorder, anthropic_message, json_response
from pydantic import BaseModel

from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.base import Block, Completion, Message, ModelFailure, ModelRequest
from scribecheck.model.budget import Budget, Prices
from scribecheck.model.endpoint import Secret
from scribecheck.model.structured import complete_batch

BASE = "https://api.anthropic.com/v1/messages/batches"


class Presence(BaseModel):
    present: bool


def _request(cid: str, text: str = "question") -> ModelRequest[Presence]:
    return ModelRequest(
        system="s",
        messages=(Message("user", (Block("transcript", cache=True), Block(text))),),
        output=Presence,
        custom_id=cid,
    )


def _client(recorder: Recorder, sleeps: list[float] | None = None) -> AnthropicClient:
    return AnthropicClient(
        "claude-sonnet-5-5",
        api_key=Secret("sk-test"),
        transport=recorder.transport,
        sleep=(sleeps if sleeps is not None else []).append,
        max_tokens=1000,
    )


def _batch(batch_id: str, status: str) -> httpx.Response:
    return json_response(
        {
            "id": batch_id,
            "type": "message_batch",
            "processing_status": status,
            "request_counts": {
                "processing": 0,
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
            "results_url": f"{BASE}/{batch_id}/results" if status == "ended" else None,
        }
    )


def _results(*lines: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, text="\n".join(json.dumps(line) for line in lines) + "\n")


def _ok(cid: str, text: str) -> dict[str, Any]:
    return {"custom_id": cid, "result": {"type": "succeeded", "message": anthropic_message(text)}}


def test_submit_poll_fetch_matches_results_by_custom_id() -> None:
    sleeps: list[float] = []
    recorder = Recorder(
        _batch("b1", "in_progress"),
        _batch("b1", "in_progress"),
        _batch("b1", "ended"),
        _results(_ok("r-2", '{"present": false}'), _ok("r-1", '{"present": true}')),
    )
    budget = Budget()
    out = complete_batch(_client(recorder, sleeps), [_request("r-1"), _request("r-2")], budget)
    assert list(out) == ["r-1", "r-2"]
    first, second = out["r-1"], out["r-2"]
    assert isinstance(first, Completion) and first.parsed.present is True
    assert isinstance(second, Completion) and second.parsed.present is False
    submit = recorder.requests[0]
    assert (submit.method, str(submit.url)) == ("POST", BASE)
    body = json.loads(submit.content)
    assert [r["custom_id"] for r in body["requests"]] == ["r-1", "r-2"]
    params = body["requests"][0]["params"]
    assert "temperature" not in params and "stream" not in params
    assert params["output_config"]["format"]["type"] == "json_schema"
    content = params["messages"][0]["content"]
    assert content[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}  # batch TTL
    # The first reply answers the submission; then two polls and the results.
    assert [str(r.url) for r in recorder.requests[1:]] == [
        f"{BASE}/b1",
        f"{BASE}/b1",
        f"{BASE}/b1/results",
    ]
    assert len(sleeps) == 1
    assert budget.calls == 2


def test_batch_params_match_a_live_call() -> None:
    client = _client(Recorder())
    live, batch = client.body(_request("r-1")), client.body(_request("r-1"), batch=True)
    live_content = live["messages"][0]["content"]
    for part in batch["messages"][0]["content"]:
        part.pop("cache_control", None)
    for part in live_content:
        part.pop("cache_control", None)
    assert live == batch


@pytest.mark.parametrize("cid", ["", "has space", "x" * 65, "semi;colon"])
def test_invalid_custom_id_is_rejected(cid: str) -> None:
    with pytest.raises(ValueError, match="custom_id"):
        _client(Recorder()).submit_batch([(_request(cid), None)])


def test_duplicate_custom_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="distinct"):
        complete_batch(_client(Recorder()), [_request("a"), _request("a")], Budget())


def test_parse_failures_go_into_one_correction_batch() -> None:
    recorder = Recorder(
        json_response({"id": "b1", "processing_status": "in_progress"}),
        _batch("b1", "ended"),
        _results(_ok("good", '{"present": true}'), _ok("bad", "nope"), _ok("worse", "{}")),
        json_response({"id": "b2", "processing_status": "in_progress"}),
        _batch("b2", "ended"),
        _results(_ok("bad", '{"present": false}'), _ok("worse", "still wrong")),
    )
    requests = [_request("good"), _request("bad"), _request("worse")]
    out = complete_batch(_client(recorder), requests, Budget())
    good, bad, worse = out["good"], out["bad"], out["worse"]
    assert isinstance(good, Completion) and good.attempts == 1
    assert isinstance(bad, Completion) and bad.attempts == 2
    assert isinstance(worse, ModelFailure) and worse.reason == "model_parse_failure"
    second = json.loads(recorder.requests[3].content)
    assert [r["custom_id"] for r in second["requests"]] == ["bad", "worse"]
    last = second["requests"][0]["params"]["messages"][0]["content"][-1]["text"]
    assert "did not match the required JSON schema" in last


def test_errored_and_expired_results_are_model_errors() -> None:
    recorder = Recorder(
        json_response({"id": "b1", "processing_status": "in_progress"}),
        _batch("b1", "ended"),
        _results(
            {
                "custom_id": "e",
                "result": {
                    "type": "errored",
                    "error": {"type": "error", "error": {"type": "api_error"}},
                },
            },
            {"custom_id": "x", "result": {"type": "expired"}},
            {
                "custom_id": "r",
                "result": {"type": "succeeded", "message": anthropic_message("", stop="refusal")},
            },
        ),
    )
    out = complete_batch(
        _client(recorder), [_request("e"), _request("x"), _request("r"), _request("lost")], Budget()
    )
    assert [(k, v.reason) for k, v in out.items() if isinstance(v, ModelFailure)] == [
        ("e", "model_error"),
        ("x", "model_error"),
        ("r", "model_refusal"),
        ("lost", "model_error"),
    ]


def test_cost_cap_leaves_requests_unsubmitted() -> None:
    prices = Prices(input=1000.0, output=1000.0, batch_multiplier=0.5)
    # Each request is estimated at about (chars/3 + 1000 tokens) at $1000/MTok, halved.
    budget = Budget(prices=prices, cap_usd=1.2)
    recorder = Recorder(
        json_response({"id": "b1", "processing_status": "in_progress"}),
        _batch("b1", "ended"),
        _results(_ok("a", '{"present": true}'), _ok("b", '{"present": true}')),
    )
    out = complete_batch(_client(recorder), [_request("a"), _request("b"), _request("c")], budget)
    submitted = json.loads(recorder.requests[0].content)["requests"]
    assert len(submitted) < 3
    unsent = [
        k for k, v in out.items() if isinstance(v, ModelFailure) and v.reason == "cost_cap_reached"
    ]
    assert unsent and not {r["custom_id"] for r in submitted} & set(unsent)


def test_batch_prices_need_a_multiplier() -> None:
    from scribecheck.errors import InputError

    budget = Budget(prices=Prices(input=1.0, output=1.0))
    recorder = Recorder(
        json_response({"id": "b1", "processing_status": "in_progress"}),
        _batch("b1", "ended"),
        _results(_ok("a", '{"present": true}')),
    )
    with pytest.raises(InputError, match="batch_multiplier"):
        complete_batch(_client(recorder), [_request("a")], budget)
