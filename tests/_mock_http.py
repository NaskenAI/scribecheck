"""A recording `httpx.MockTransport` for model-client tests: no network, ever."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

Handler = Callable[[httpx.Request], httpx.Response]


class Recorder:
    """Replies in turn from `replies` (responses, or callables taking the request)."""

    def __init__(self, *replies: httpx.Response | Handler) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.replies:
            raise AssertionError(f"unexpected request {request.method} {request.url}")
        reply = self.replies.pop(0)
        return reply(request) if callable(reply) else reply

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def body(self, index: int = -1) -> Any:
        return json.loads(self.requests[index].content)


def json_response(
    data: Any, status: int = 200, headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(status, json=data, headers=headers)


def anthropic_message(
    text: str,
    *,
    stop: str = "end_turn",
    model: str = "claude-sonnet-5-5",
    usage: dict[str, Any] | None = None,
    thinking: bool = False,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if thinking:
        content.append({"type": "thinking", "thinking": "", "signature": "sig"})
    content.append({"type": "text", "text": text})
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop,
        "usage": usage or {"input_tokens": 10, "output_tokens": 5},
    }


def openai_completion(
    text: str,
    *,
    finish: str = "stop",
    model: str = "local-model",
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
        ],
        "usage": usage or {"prompt_tokens": 12, "completion_tokens": 4},
    }
