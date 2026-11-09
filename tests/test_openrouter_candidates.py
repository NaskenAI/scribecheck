"""openrouter_candidates.py on a mocked listing (no network)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "openrouter_candidates.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("openrouter_candidates", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["openrouter_candidates"] = module
    spec.loader.exec_module(module)
    return module


FULL = ["structured_outputs", "seed", "temperature", "max_tokens"]
MODELS = [
    {
        "id": "moonshot/kimi-k2",
        "name": "Kimi K2 (1T total, 32B active)",
        "context_length": 131072,
        "hugging_face_id": "moonshot/Kimi-K2",
        "created": 1_780_000_000,
    },
    {
        "id": "microsoft/phi-4",
        "name": "Phi 4",
        "context_length": 32768,
        "hugging_face_id": "microsoft/phi-4",
        "created": 1_700_000_000,
    },
    {
        "id": "qwen/qwen3-32b",
        "name": "Qwen3 32B",
        "context_length": 131072,
        "hugging_face_id": "Qwen/Qwen3-32B",
        "created": 1_760_000_000,
    },
    {
        "id": "mistralai/mistral-small-3.2-24b-instruct",
        "name": "Mistral Small 3.2 24B",
        "context_length": 131072,
        "hugging_face_id": "mistralai/Mistral-Small-3.2",
        "created": 1_750_000_000,
    },
    {
        "id": "acme/huge-405b",
        "name": "Huge 405B",
        "context_length": 131072,
        "hugging_face_id": "acme/huge",
        "created": 1_770_000_000,
    },
    {
        "id": "acme/closed-model",
        "name": "Closed",
        "context_length": 200000,
        "hugging_face_id": "",
        "created": 1_770_000_000,
    },
    {
        "id": "acme/tiny-ctx-8b",
        "name": "Tiny 8B",
        "context_length": 8192,
        "hugging_face_id": "acme/tiny",
        "created": 1_770_000_000,
    },
]
ENDPOINTS: dict[str, list[dict[str, Any]]] = {
    "qwen/qwen3-32b": [
        {
            "provider_name": "Cheap4bit",
            "quantization": "int4",
            "context_length": 131072,
            "supported_parameters": FULL,
            "pricing": {"prompt": "0.00000005", "completion": "0.0000001"},
        },
        {
            "provider_name": "GoodFP8",
            "quantization": "fp8",
            "context_length": 131072,
            "supported_parameters": FULL,
            "pricing": {"prompt": "0.0000001", "completion": "0.0000003"},
        },
        {
            "provider_name": "NoSeed",
            "quantization": "fp8",
            "context_length": 131072,
            "supported_parameters": ["response_format", "temperature", "max_tokens"],
            "pricing": {"prompt": "0.00000001", "completion": "0.00000001"},
        },
        {
            "provider_name": "NoPrice",
            "quantization": "bf16",
            "context_length": 131072,
            "supported_parameters": FULL,
            "pricing": {"prompt": "n/a", "completion": "0"},
        },
        {
            "provider_name": "NoJSON",
            "quantization": "bf16",
            "context_length": 131072,
            "supported_parameters": ["temperature"],
            "pricing": {"prompt": "0.00000001", "completion": "0.00000001"},
        },
    ],
    "mistralai/mistral-small-3.2-24b-instruct": [
        {
            "provider_name": "ShortCtx",
            "quantization": "fp8",
            "context_length": 16000,
            "supported_parameters": FULL,
            "pricing": {"prompt": "0", "completion": "0"},
        },
    ],
}


def _transport(seen: list[str]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        assert request.method == "GET" and "authorization" not in request.headers
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json={"data": MODELS})
        model = request.url.path.removeprefix("/api/v1/models/").removesuffix("/endpoints")
        return httpx.Response(200, json={"data": {"endpoints": ENDPOINTS.get(model, [])}})

    return httpx.MockTransport(handle)


def test_filters_and_pins(capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[str] = []
    assert _load().main([], transport=_transport(seen)) == 0
    out = capsys.readouterr().out
    assert "2 open-weight model(s)" in out  # not the 405B, the closed or the 8k-context one
    assert "qwen/qwen3-32b  (32B" in out and "usually permissive" in out
    assert "GoodFP8" in out and "Cheap4bit" in out
    assert "NoJSON" not in out and "NoSeed" not in out and "NoPrice" not in out
    pinned = [line for line in out.splitlines() if "<- pin" in line]
    assert len(pinned) == 1 and "GoodFP8" in pinned[0]  # cheapest at FP8 or better
    assert "mistral-small" not in out  # its only provider has too little context
    assert "1 candidate(s) with a provider that supports the run's parameters" in out
    assert "kimi" not in out.split("check by hand")[0]  # 1T total is not 32B
    assert "check by hand): microsoft/phi-4" in out
    assert all(not path.endswith(("/completions", "/chat/completions")) for path in seen)


def test_params_from_ids() -> None:
    module = _load()
    assert module.params_billion({"id": "qwen/qwen3-30b-a3b"}) == 30
    assert module.params_billion({"id": "openai/gpt-oss-20b"}) == 20
    assert module.params_billion({"id": "acme/no-size"}) is None
    assert module.params_billion({"id": "x/kimi", "name": "Kimi (1T total, 32B active)"}) == 1000
    assert module.params_billion({"id": "mistralai/mixtral-8x7b"}) == 56
    assert module.params_billion({"id": "x/model-3.5b"}) == 3.5
    assert module.pin([{"quantization": "fp4", "input": 0.0, "output": 0.0}]) is None
    assert module.licence_note("google/gemma-3-27b-it") == "review licence"
