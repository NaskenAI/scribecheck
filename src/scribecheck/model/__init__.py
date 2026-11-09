"""The model client, for full mode. Rules mode never imports this package.

`create_client(config)` builds the configured adapter, reading its API key from the
environment, and logs the spec's warning when the endpoint is not local.
`create_budget(config)` builds the run's usage tracker and cost cap.
"""

from __future__ import annotations

import logging

from scribecheck.config import Config
from scribecheck.errors import InputError
from scribecheck.model.anthropic import DEFAULT_BASE_URL, AnthropicClient
from scribecheck.model.base import (
    Block,
    Completion,
    Message,
    ModelClient,
    ModelConfigError,
    ModelFailure,
    ModelRequest,
    Usage,
)
from scribecheck.model.budget import Budget, Prices
from scribecheck.model.endpoint import Endpoint, api_key, privacy_warning
from scribecheck.model.openai_compat import OpenAICompatClient
from scribecheck.model.structured import complete, complete_batch

__all__ = [
    "AnthropicClient",
    "Block",
    "Budget",
    "Completion",
    "Message",
    "ModelClient",
    "ModelConfigError",
    "ModelFailure",
    "ModelRequest",
    "OpenAICompatClient",
    "Prices",
    "Usage",
    "complete",
    "complete_batch",
    "create_budget",
    "create_client",
]

log = logging.getLogger("scribecheck.model")


def create_client(config: Config) -> AnthropicClient | OpenAICompatClient:
    """The adapter `config` names. Raises `InputError` for an incomplete setup or no key."""
    if config.provider is None or not config.model:
        raise InputError("full mode needs [model] provider and name in the config")
    if config.provider == "anthropic":
        endpoint = Endpoint.from_url(config.base_url or DEFAULT_BASE_URL)
        key = api_key("anthropic", endpoint, config.api_key_env)
        if key is None:
            raise InputError("the Anthropic API needs a key")
        client: AnthropicClient | OpenAICompatClient = AnthropicClient(
            config.model,
            api_key=key,
            base_url=endpoint.base_url,
            effort=config.effort,
            max_tokens=config.max_tokens,
            cache_ttl=config.cache_ttl,
            timeout=config.timeout_s,
        )
    else:
        if not config.base_url:
            raise InputError("an OpenAI-compatible model needs [model] base_url")
        endpoint = Endpoint.from_url(config.base_url)
        client = OpenAICompatClient(
            config.model,
            base_url=endpoint.base_url,
            api_key=api_key("openai", endpoint, config.api_key_env),
            temperature=config.temperature,
            seed=config.seed,
            json_mode=config.json_mode,
            max_tokens=config.max_tokens,
            auth_header=config.auth_header,
            query=config.query,
            extra_body=config.extra_body,
            timeout=config.timeout_s,
        )
    warning = privacy_warning(client.endpoint)
    if warning:
        log.warning("%s", warning)
    return client


def create_budget(config: Config) -> Budget:
    """The run's usage tracker, with prices and the cost cap from `config`."""
    table = config.prices
    prices = (
        None
        if table is None
        else Prices(
            input=table.input,
            output=table.output,
            cache_write_5m=table.cache_write_5m,
            cache_write_1h=table.cache_write_1h,
            cache_read=table.cache_read,
            batch_multiplier=table.batch_multiplier,
        )
    )
    return Budget(prices=prices, cap_usd=config.cost_cap_usd)
