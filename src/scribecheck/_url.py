"""Checks on user-supplied endpoint settings that keep credentials out of URLs.

Shared by config.py and the model client without importing the client (and httpx), so
rules mode stays free of both. Error messages never repeat the URL or value they reject.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from scribecheck.errors import InputError

ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
KEY_LIKE = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "api_token",
        "key",
        "token",
        "access_token",
        "secret",
        "password",
        "sig",
        "authorization",
        "headers",
    }
)
_KEY_VALUE = re.compile(r"^(?:sk-|sk_|Bearer\s)", re.IGNORECASE)


def check_base_url(base_url: str) -> str:
    """Return `base_url` without a trailing slash; raise `InputError` for an unusable one.

    It must be http(s) with a host, and may not carry a user name or password, a query or
    a fragment: credentials go in environment variables, and query parameters (such as
    Azure's api-version) in `[model] query`.
    """
    try:
        parts = urlsplit(base_url)
        parts.port  # noqa: B018 - raises ValueError for a malformed port
    except ValueError:
        raise InputError("model base_url is not a valid URL") from None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise InputError("model base_url must be an http:// or https:// URL with a host")
    if parts.username is not None or parts.password is not None:
        raise InputError(
            "model base_url must not contain a user name or password; "
            "put the key in an environment variable"
        )
    if parts.query or parts.fragment:
        raise InputError(
            "model base_url must not contain a query or fragment; use [model] query for "
            "parameters such as api-version"
        )
    return base_url.rstrip("/")


def check_env_name(name: str) -> str:
    """An environment variable name; the message never repeats a rejected value (a key)."""
    if not ENV_NAME.fullmatch(name):
        raise InputError(
            "api_key_env must name an environment variable (letters, digits and "
            "underscores), not hold the key itself"
        )
    return name


RESERVED_BODY = frozenset(
    {
        "model",
        "models",
        "route",
        "messages",
        "response_format",
        "temperature",
        "seed",
        "max_tokens",
        "max_completion_tokens",
        "stream",
        "n",
    }
)
"""Fields the client sets, or that would change which model answers or how replies come
back (OpenRouter's `models` and `route` fallbacks would defeat a provider pin)."""


def check_extra_body(body: dict[str, Any]) -> dict[str, Any]:
    """Extra request-body fields (an OpenRouter provider pin): no credentials anywhere in
    them, and none of the fields the client sets itself."""
    clash = sorted(RESERVED_BODY & set(body))
    if clash:
        raise InputError(f"[model] extra_body may not set {clash[0]!r}; the client sets it")

    def walk(node: Any) -> None:
        if isinstance(node, str) and _KEY_VALUE.match(node):
            raise InputError(
                "never put an API key in [model] extra_body; set it in an environment variable"
            )
        if isinstance(node, dict):
            for name, value in node.items():
                label = str(name).casefold()
                if label in KEY_LIKE:
                    raise InputError(
                        "never put an API key in [model] extra_body; set it in an "
                        "environment variable"
                    )
                if label in ("model", "models"):
                    raise InputError(f"[model] extra_body may not set {label!r} at any depth")
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(body)
    return body


def check_query(query: dict[str, str]) -> dict[str, str]:
    """URL query parameters, refusing any that look like credentials."""
    for name in query:
        if name.casefold() in KEY_LIKE:
            raise InputError(
                "never put an API key in [model] query; set it in an environment variable"
            )
    return query
