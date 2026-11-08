"""Checks on user-supplied endpoint settings that keep credentials out of URLs.

Shared by config.py and the model client without importing the client (and httpx), so
rules mode stays free of both. Error messages never repeat the URL or value they reject.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from scribecheck.errors import InputError

ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
KEY_LIKE = frozenset(
    {"api_key", "apikey", "api-key", "key", "token", "access_token", "secret", "password", "sig"}
)


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


def check_query(query: dict[str, str]) -> dict[str, str]:
    """URL query parameters, refusing any that look like credentials."""
    for name in query:
        if name.casefold() in KEY_LIKE:
            raise InputError(
                "never put an API key in [model] query; set it in an environment variable"
            )
    return query
