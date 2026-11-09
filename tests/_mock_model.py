"""A scripted mock of the Anthropic Messages and Message Batches APIs for Pass A tests.

Each request is routed to a reply by the stage marker in its prompt; a test supplies one
function per stage that turns the request's facts into a JSON reply. Every request body is
recorded with its stage. httpx.MockTransport only: nothing reaches a network.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import httpx

MARKERS = (
    ("TASK: EXTRACT FACTS", "extract"),
    ("TASK: AUDIT FACTS", "audit"),
    ("TASK: CHECK FACTS AGAINST THE NOTE", "check"),
    ("TASK: SECOND LOOK", "second_look"),
    ("TASK: THE SPECIFIC ELEMENT", "partial_look"),
    ("TASK: CONFIRM RULE-BASED FINDINGS", "candidate"),
    ("TASK: GIVE THE EXACT QUOTE", "quote_recheck"),
    ("TASK: FIND CHANGED FACTS", "changes"),
    ("TASK: SECOND CHECK OF CHANGED FACTS", "changes_second"),
    ("TASK: SPLIT THE NOTE INTO STATEMENTS", "split"),
    ("TASK: CHECK NOTE STATEMENTS AGAINST THE TRANSCRIPT", "statements"),
    ("TASK: SECOND CHECK OF UNSUPPORTED STATEMENTS", "statements_second"),
    ("You are a UK GP and a meticulous clinical auditor", "r1_extract"),
    ("CANDIDATE FACTS:", "r1_audit"),
    ("CLINICAL NOTE UNDER REVIEW:", "r1_check"),
    ("You are a skeptical reviewer.", "r1_refute"),
)
_LINE = re.compile(r"^([a-z]\d+): (.*)$", re.MULTILINE)

Script = Callable[[list[tuple[str, str]], dict[str, Any]], Any]
"""(facts as (id, text) pairs from the prompt, request body) -> the reply's JSON value."""


def text_of(body: dict[str, Any]) -> str:
    parts = []
    for message in body.get("messages", []):
        content = message["content"]
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(block.get("text", "") for block in content)
    return "\n".join(parts)


def stage_of(body: dict[str, Any]) -> str:
    text = text_of(body)
    for marker, stage in MARKERS:
        if marker in text:
            return stage
    raise AssertionError("request with no known stage marker")


def listed(body: dict[str, Any]) -> list[tuple[str, str]]:
    """The (id, text) lines of the last block that lists facts."""
    text = text_of(body)
    for heading in (
        "FACTS:\n",
        "FINDINGS:\n",
        "ITEMS:\n",
        "CLAIMED-MISSING FACTS:\n",
        "ITEMS TO JUDGE:\n",
        "CLAIMS:\n",
        "STATEMENTS:\n",
    ):
        if heading in text:
            text = text.rsplit(heading, 1)[1]
            break
    else:
        if "Each fact has an ID:\n" in text:
            text = text.rsplit("Each fact has an ID:\n", 1)[1]
    return [(m.group(1), m.group(2)) for m in _LINE.finditer(text)]


def _no_element(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
    """The partial look's default: no fact has a specific element the note leaves out."""
    return {"facts": [{"id": fid, "element": "", "missing": False} for fid, _ in listed]}


DEFAULTS: dict[str, Script] = {"partial_look": _no_element}
"""Stages a test need not script: their default reply changes nothing."""


class MockClaude:
    """Routes requests to `scripts[stage]`, live or through the batch endpoints."""

    def __init__(self, scripts: dict[str, Script], usage: dict[str, int] | None = None) -> None:
        self.scripts = scripts
        self.usage = usage or {"input_tokens": 100, "output_tokens": 20}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.paths: list[str] = []
        self._batches: dict[str, list[dict[str, Any]]] = {}

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def stages(self) -> list[str]:
        return [stage for stage, _ in self.calls]

    def reply(self, body: dict[str, Any]) -> dict[str, Any]:
        stage = stage_of(body)
        self.calls.append((stage, body))
        script = self.scripts.get(stage) or DEFAULTS[stage]
        value = script(listed(body), body)
        text = value if isinstance(value, str) else json.dumps(value)
        return {
            "id": "msg_mock",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5-mock",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": dict(self.usage),
        }

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(f"{request.method} {path}")
        if request.method == "POST" and path == "/v1/messages":
            return httpx.Response(200, json=self.reply(json.loads(request.content)))
        if request.method == "POST" and path == "/v1/messages/batches":
            batch_id = f"msgbatch_{len(self._batches) + 1}"
            self._batches[batch_id] = json.loads(request.content)["requests"]
            return httpx.Response(200, json={"id": batch_id, "processing_status": "in_progress"})
        match = re.fullmatch(r"/v1/messages/batches/(msgbatch_\d+)(/results)?", path)
        if request.method == "GET" and match:
            batch_id = match.group(1)
            if not match.group(2):
                return httpx.Response(200, json={"id": batch_id, "processing_status": "ended"})
            lines = [
                json.dumps(
                    {
                        "custom_id": item["custom_id"],
                        "result": {"type": "succeeded", "message": self.reply(item["params"])},
                    }
                )
                for item in reversed(self._batches[batch_id])  # any order
            ]
            return httpx.Response(200, text="\n".join(lines) + "\n")
        raise AssertionError(f"unexpected request {request.method} {path}")
