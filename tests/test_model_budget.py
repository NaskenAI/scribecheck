"""Usage, cost from configured prices, and the cost cap (mocked HTTP)."""

from __future__ import annotations

import pytest
from _mock_http import Recorder, anthropic_message, json_response
from pydantic import BaseModel

from scribecheck.config import Config, PriceTable
from scribecheck.errors import InputError
from scribecheck.model import create_budget
from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.base import Block, Completion, Message, ModelFailure, ModelRequest, Usage
from scribecheck.model.budget import Budget, Prices
from scribecheck.model.endpoint import Secret
from scribecheck.model.structured import complete


class Presence(BaseModel):
    present: bool


REQUEST = ModelRequest(system="s", messages=(Message("user", (Block("q"),)),), output=Presence)
USAGE = Usage(
    input_tokens=1_000_000,
    output_tokens=100_000,
    cache_write_5m_tokens=200_000,
    cache_write_1h_tokens=100_000,
    cache_read_tokens=500_000,
)


def test_cost_uses_every_configured_rate() -> None:
    prices = Prices(
        input=2.0,
        output=10.0,
        cache_write_5m=2.5,
        cache_write_1h=4.0,
        cache_read=0.2,
        batch_multiplier=0.5,
    )
    expected = 2.0 + 1.0 + 0.5 + 0.4 + 0.1
    assert prices.cost(USAGE) == pytest.approx(expected)
    assert prices.cost(USAGE, batch=True) == pytest.approx(expected / 2)


def test_cache_rates_default_to_the_input_price() -> None:
    assert Prices(input=1.0, output=0.0).cost(USAGE) == pytest.approx(1.0 + 0.2 + 0.1 + 0.5)


def test_without_prices_usage_is_recorded_but_cost_unknown() -> None:
    budget = Budget()
    budget.record(USAGE)
    assert budget.usage == USAGE and budget.calls == 1
    assert budget.spent_usd is None
    assert budget.can_spend()


def test_a_cap_needs_prices() -> None:
    with pytest.raises(InputError, match="needs prices"):
        Budget(cap_usd=5.0)


def test_create_budget_from_config() -> None:
    config = Config(prices=PriceTable(input=2.0, output=10.0), cost_cap_usd=3.0)
    budget = create_budget(config)
    assert budget.cap_usd == 3.0 and budget.prices == Prices(input=2.0, output=10.0)
    with pytest.raises(InputError, match="needs prices"):
        create_budget(Config(cost_cap_usd=3.0))


def test_cap_stops_new_calls_and_marks_the_rest() -> None:
    # Each reply costs 1M input tokens at $1/MTok = $1; the cap is $2.
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    recorder = Recorder(
        *(json_response(anthropic_message('{"present": true}', usage=usage)) for _ in range(2))
    )
    client = AnthropicClient(
        "claude-sonnet-5-5", api_key=Secret("sk-test"), transport=recorder.transport
    )
    budget = Budget(prices=Prices(input=1.0, output=1.0), cap_usd=2.0)
    results = [complete(client, REQUEST, budget) for _ in range(5)]
    assert len(recorder.requests) == 2  # no HTTP call once the cap is reached
    assert [type(r).__name__ for r in results] == ["Completion", "Completion"] + [
        "ModelFailure"
    ] * 3
    assert all(isinstance(r, ModelFailure) and r.reason == "cost_cap_reached" for r in results[2:])
    assert budget.spent_usd == pytest.approx(2.0)


def test_a_retry_is_not_made_past_the_cap() -> None:
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    recorder = Recorder(json_response(anthropic_message("not json", usage=usage)))
    client = AnthropicClient(
        "claude-sonnet-5-5", api_key=Secret("sk-test"), transport=recorder.transport
    )
    budget = Budget(prices=Prices(input=1.0, output=1.0), cap_usd=1.0)
    result = complete(client, REQUEST, budget)
    assert isinstance(result, ModelFailure) and result.reason == "cost_cap_reached"
    assert len(recorder.requests) == 1
    assert result.usage.input_tokens == 1_000_000  # the spent attempt is still counted


def test_completion_usage_sums_attempts() -> None:
    recorder = Recorder(
        json_response(anthropic_message("bad")),
        json_response(anthropic_message('{"present": false}')),
    )
    client = AnthropicClient(
        "claude-sonnet-5-5", api_key=Secret("sk-test"), transport=recorder.transport
    )
    result = complete(client, REQUEST, Budget())
    assert isinstance(result, Completion)
    assert (result.usage.input_tokens, result.usage.output_tokens) == (20, 10)
