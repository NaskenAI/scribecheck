"""Smoke test for the model adapters: one tiny call each, with your own key.

Run it yourself; CI never does. It sends only the synthetic text below and prints the
adapter, the endpoint host, the model id the provider reports, the token usage and
whether the JSON reply parsed. It never prints a key, a request or a reply body.

    # Anthropic (reads ANTHROPIC_API_KEY)
    python scripts/smoke_model.py --anthropic

    # Any OpenAI-compatible endpoint (reads OPENAI_API_KEY, or OPENROUTER_API_KEY for
    # openrouter.ai; a local server needs no key)
    python scripts/smoke_model.py --base-url http://localhost:11434/v1 --model <name>

Options: --check-cache makes a second Anthropic call with a ~700-token synthetic prefix
and prints its cache reads; --batch also runs a one-request Anthropic batch (minutes).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import httpx
from pydantic import BaseModel

from scribecheck.errors import InputError
from scribecheck.model.anthropic import DEFAULT_BASE_URL, AnthropicClient
from scribecheck.model.base import Block, Completion, Message, ModelClient, ModelRequest
from scribecheck.model.budget import Budget
from scribecheck.model.endpoint import Endpoint, api_key, privacy_warning
from scribecheck.model.openai_compat import OpenAICompatClient
from scribecheck.model.structured import complete, complete_batch

ANTHROPIC_MODEL = "claude-sonnet-5-5"
SYSTEM = "You answer questions about a short synthetic text. Reply with JSON only."
QUESTION = "Synthetic text: 'The cat sat on the mat.' Does the text mention a cat?"
# A synthetic, repetitive prefix long enough (about 700 tokens) to be cached.
PREFIX = "Synthetic context for a cache test. " + " ".join(
    f"Line {i}: the quick brown fox jumps over the lazy dog." for i in range(60)
)


class Answer(BaseModel):
    answer: bool


def _request(prefix: str | None = None, custom_id: str = "smoke-1") -> ModelRequest[Answer]:
    blocks = (Block(prefix, cache=True), Block(QUESTION)) if prefix else (Block(QUESTION),)
    return ModelRequest(
        system=SYSTEM,
        messages=(Message("user", blocks),),
        output=Answer,
        max_tokens=1024,
        custom_id=custom_id,
    )


def _show(label: str, client: ModelClient, result: object) -> None:
    print(f"[{label}] endpoint host: {client.endpoint.host}")
    warning = privacy_warning(client.endpoint)
    if warning:
        print(f"[{label}] warning: {warning}")
    if isinstance(result, Completion):
        u = result.usage
        print(f"[{label}] model id: {result.model_id}")
        print(
            f"[{label}] usage: input {u.input_tokens}, output {u.output_tokens}, "
            f"cache write {u.cache_write_5m_tokens + u.cache_write_1h_tokens}, "
            f"cache read {u.cache_read_tokens}"
        )
        print(
            f"[{label}] parsed: yes ({result.parsed.model_dump_json()}, attempts {result.attempts})"
        )
    else:
        reason = getattr(result, "reason", "unknown")
        detail = getattr(result, "detail", "")
        print(f"[{label}] parsed: no ({reason}: {detail})")


def main(argv: Sequence[str] | None = None, *, transport: httpx.BaseTransport | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--anthropic", action="store_true", help="call the Anthropic API")
    parser.add_argument("--anthropic-model", default=ANTHROPIC_MODEL)
    parser.add_argument("--base-url", help="an OpenAI-compatible endpoint")
    parser.add_argument("--model", help="the model name at --base-url")
    parser.add_argument("--check-cache", action="store_true")
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args(argv)
    if not args.anthropic and not args.base_url:
        parser.error("choose --anthropic and/or --base-url with --model")
    if args.base_url and not args.model:
        parser.error("--base-url needs --model")

    status = 0
    try:
        if args.anthropic:
            endpoint = Endpoint.from_url(DEFAULT_BASE_URL)
            key = api_key("anthropic", endpoint)
            assert key is not None
            client = AnthropicClient(
                args.anthropic_model,
                api_key=key,
                effort="low",
                max_tokens=1024,
                transport=transport,
            )
            budget = Budget()
            _show("anthropic", client, complete(client, _request(), budget))
            if args.check_cache:
                complete(client, _request(PREFIX), budget)
                second = complete(client, _request(PREFIX), budget)
                reads = second.usage.cache_read_tokens if isinstance(second, Completion) else 0
                print(f"[anthropic] cache check: second call read {reads} cached tokens")
            if args.batch:
                out = complete_batch(client, [_request(custom_id="smoke-batch-1")], budget)
                _show("anthropic batch", client, out["smoke-batch-1"])
        if args.base_url:
            endpoint = Endpoint.from_url(args.base_url)
            openai = OpenAICompatClient(
                args.model,
                base_url=args.base_url,
                api_key=api_key("openai", endpoint),
                max_tokens=1024,
                transport=transport,
            )
            _show("openai-compatible", openai, complete(openai, _request(), Budget()))
    except InputError as exc:
        print(f"smoke test stopped: {exc}", file=sys.stderr)
        status = 2
    return status


if __name__ == "__main__":
    raise SystemExit(main())
