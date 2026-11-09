"""List open-weight models on OpenRouter that meet the spec's criteria (M3 session 4).

    python scripts/openrouter_candidates.py [--max-params 40] [--min-context 32768]

Reads only OpenRouter's public model and endpoint listings (no API key, no completion
calls). A candidate must have:

- open weights: a Hugging Face id in the listing (check the licence on that page; the
  licence column only marks families whose usual licence is permissive or needs review);
- a total parameter count, read from its id or name ("32b", "1T", "8x7b"; the largest
  figure, so "32B active" never stands in for the total), of at most --max-params billion,
  which fits one 24-48 GB GPU at FP8 or below (models whose size is unclear are listed
  separately for a manual check);
- a context window of at least --min-context tokens;
- at least one provider that supports everything the runbook's config sends with
  `require_parameters`: `structured_outputs`, `seed`, `temperature` and `max_tokens`.

For each candidate it prints the release date (prefer the newest family that qualifies),
and each qualifying provider's quantization and prices per million tokens, marking the
cheapest one at FP8 or better as the provider to pin (none is marked if no provider runs
at FP8 or better).
"""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import httpx

BASE_URL = "https://openrouter.ai/api/v1"
_SIZE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([bt])(?![a-z])", re.IGNORECASE)
_MIXTURE = re.compile(r"(?<![\w.])(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*b(?![a-z])", re.IGNORECASE)
REQUIRED = ("structured_outputs", "seed", "temperature", "max_tokens")
"""Parameters the runbook's OpenRouter config sends, with require_parameters = true."""
PERMISSIVE = ("qwen", "mistral", "devstral", "gpt-oss", "phi", "olmo", "granite", "deepseek")
REVIEW = ("gemma", "llama")
GOOD_QUANT = ("fp8", "bf16", "fp16", "fp32")


def params_billion(model: dict[str, Any]) -> float | None:
    """Total parameters in billions, from the id and name ("qwen3-32b", "30B-A3B", "1T",
    "8x7b"): the largest figure found in either, so an active count never wins."""
    sizes: list[float] = []
    for text in (str(model.get("id", "")), str(model.get("name", ""))):
        for count, each in _MIXTURE.findall(text):
            sizes.append(float(count) * float(each))
        for value, unit in _SIZE.findall(_MIXTURE.sub(" ", text)):
            sizes.append(float(value) * (1000 if unit.lower() == "t" else 1))
    return max(sizes) if sizes else None


def licence_note(model_id: str) -> str:
    lowered = model_id.lower()
    if any(f in lowered for f in REVIEW):
        return "review licence"
    if any(f in lowered for f in PERMISSIVE):
        return "usually permissive"
    return "check licence"


def supports_run(params: Sequence[str]) -> bool:
    return all(p in params for p in REQUIRED)


def per_million(price: Any) -> float | None:
    try:
        value = float(price) * 1_000_000
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def candidates(
    models: Sequence[dict[str, Any]], *, max_params: float, min_context: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(candidates, open-weight models with enough context whose size is unclear)."""
    out, unclear = [], []
    for model in models:
        if not model.get("hugging_face_id"):
            continue
        if int(model.get("context_length") or 0) < min_context:
            continue
        size = params_billion(model)
        if size is None:
            unclear.append(model)
        elif size <= max_params:
            out.append(model)
    newest = sorted(out, key=lambda m: -int(m.get("created") or 0))
    return newest, sorted(unclear, key=lambda m: str(m.get("id")))


def providers(endpoints: Sequence[dict[str, Any]], min_context: int) -> list[dict[str, Any]]:
    rows = []
    for endpoint in endpoints:
        if not supports_run(endpoint.get("supported_parameters") or []):
            continue
        if int(endpoint.get("context_length") or 0) < min_context:
            continue
        pricing = endpoint.get("pricing") or {}
        price_in = per_million(pricing.get("prompt"))
        price_out = per_million(pricing.get("completion"))
        if price_in is None or price_out is None:
            continue  # no usable price: a cost cap could not be enforced
        rows.append(
            {
                "provider": endpoint.get("provider_name") or endpoint.get("name") or "?",
                "quantization": str(endpoint.get("quantization") or "unknown").lower(),
                "input": price_in,
                "output": price_out,
            }
        )
    return rows


def pin(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The cheapest provider at FP8 or better, or None if there is none."""
    good = [r for r in rows if r["quantization"] in GOOD_QUANT]
    return min(good, key=lambda r: r["input"] + r["output"]) if good else None


def main(argv: Sequence[str] | None = None, *, transport: httpx.BaseTransport | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-params", type=float, default=40.0)
    parser.add_argument("--min-context", type=int, default=32768)
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args(argv)
    with httpx.Client(timeout=60, transport=transport) as http:
        listing = http.get(f"{args.base_url}/models").raise_for_status().json()
        models, unclear = candidates(
            listing.get("data", []), max_params=args.max_params, min_context=args.min_context
        )
        print(
            f"{len(models)} open-weight model(s) of at most {args.max_params:g}B with "
            f">= {args.min_context} context (newest first)"
        )
        shown = 0
        for model in models:
            url = f"{args.base_url}/models/{model['id']}/endpoints"
            data = http.get(url).raise_for_status().json().get("data") or {}
            rows = providers(data.get("endpoints") or [], args.min_context)
            if not rows:
                continue
            shown += 1
            created = datetime.fromtimestamp(int(model.get("created") or 0), UTC).date()
            print(
                f"\n{model['id']}  ({params_billion(model):g}B, context "
                f"{model.get('context_length')}, released {created}, "
                f"{licence_note(model['id'])}; HF: {model.get('hugging_face_id')})"
            )
            chosen = pin(rows)
            if chosen is None:
                print("   (no provider runs it at FP8 or better: none marked to pin)")
            for row in sorted(rows, key=lambda r: r["input"] + r["output"]):
                mark = "  <- pin" if row is chosen else ""
                print(
                    f"   {row['provider']:<20} {row['quantization']:<8} "
                    f"${row['input']:.3f} in / ${row['output']:.3f} out per MTok{mark}"
                )
        print(f"\n{shown} candidate(s) with a provider that supports the run's parameters")
        if unclear:
            print(
                "open-weight models whose size could not be read (check by hand): "
                + ", ".join(str(m.get("id")) for m in unclear)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
