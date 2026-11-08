"""HTTP calls with retries, shared by the adapters.

Transient failures (429, 500, 502, 503, 504, 529, timeouts and connection errors) are
retried with backoff, honouring `retry-after`. A rejected setup (400, 401, 403, 404, 422)
raises `ModelConfigError` at once. Logs carry the host, status and attempt only: never
headers (which hold the key) or bodies (which hold patient text).
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from scribecheck.model.base import ModelConfigError, ModelTransportError

log = logging.getLogger("scribecheck.model")
# httpx logs every request URL at INFO; keep it and httpcore quiet unless a user has
# chosen a level for them.
for _name in ("httpx", "httpcore"):
    if logging.getLogger(_name).level == logging.NOTSET:
        logging.getLogger(_name).setLevel(logging.WARNING)

RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})
CONFIG_STATUSES = frozenset({400, 401, 403, 404, 422})
MAX_ATTEMPTS = 3
_MAX_WAIT = 60.0

Sleep = Callable[[float], None]


def request_json(
    http: httpx.Client,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    body: Any = None,
    params: Mapping[str, str] | None = None,
    sleep: Sleep = time.sleep,
    provider: str,
) -> Any:
    """Send a request and return its decoded JSON body, retrying transient failures."""
    response = _send(
        http, method, url, headers=headers, body=body, params=params, sleep=sleep, provider=provider
    )
    try:
        return response.json()
    except ValueError as exc:
        raise ModelTransportError(f"{provider}: the response was not JSON") from exc


def request_text(
    http: httpx.Client,
    url: str,
    *,
    headers: Mapping[str, str],
    sleep: Sleep = time.sleep,
    provider: str,
) -> str:
    """GET a text body (batch results), retrying transient failures."""
    return _send(http, "GET", url, headers=headers, sleep=sleep, provider=provider).text


def _send(
    http: httpx.Client,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    body: Any = None,
    params: Mapping[str, str] | None = None,
    sleep: Sleep,
    provider: str,
) -> httpx.Response:
    host = httpx.URL(url).host
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = http.request(method, url, headers=dict(headers), json=body, params=params)
        except httpx.TransportError as exc:
            log.warning("%s %s: %s (attempt %d)", provider, host, type(exc).__name__, attempt)
            # A POST that may have reached the server (it timed out while sending or
            # reading) is not repeated: it may already have been billed.
            unsent = isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))
            if attempt == MAX_ATTEMPTS or (method == "POST" and not unsent):
                raise ModelTransportError(
                    f"{provider}: {host} unreachable after {attempt} attempts "
                    f"({type(exc).__name__})"
                ) from None
            sleep(_backoff(attempt, None))
            continue
        log.debug(
            "%s %s %s -> %d (attempt %d)", provider, method, host, response.status_code, attempt
        )
        status = response.status_code
        if status < 400:
            return response
        if status in CONFIG_STATUSES:
            raise ModelConfigError(f"{provider}: {host} rejected the request: {_error(response)}")
        if status in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
            sleep(_backoff(attempt, response.headers.get("retry-after")))
            continue
        raise ModelTransportError(f"{provider}: {host} failed: {_error(response)}")
    raise AssertionError("unreachable")


def _backoff(attempt: int, retry_after: str | None) -> float:
    if retry_after is not None:
        try:
            seconds = float(retry_after)
        except ValueError:
            seconds = math.nan
        if math.isfinite(seconds):
            return min(max(seconds, 0.0), _MAX_WAIT)
    return float(min(2**attempt, _MAX_WAIT))


def _error(response: httpx.Response) -> str:
    """HTTP status and the provider's error type only.

    The provider's message is left out on purpose: some servers quote the input in it,
    and these strings reach exceptions, failure details and logs.
    """
    kind = ""
    try:
        data = response.json()
    except (ValueError, json.JSONDecodeError):
        data = None
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict) and isinstance(error.get("type"), str):
            kind = error["type"]
        elif isinstance(data.get("type"), str) and data.get("type") != "error":
            kind = data["type"]
    kind = re.sub(r"[^A-Za-z0-9_.-]", "", kind)[:60]
    return f"HTTP {response.status_code}" + (f" ({kind})" if kind else "")
