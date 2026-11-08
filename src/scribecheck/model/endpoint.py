"""Where calls go, whether that is local, and the API key for it.

Keys are read from environment variables only, never from config, and are held in a
`Secret` that prints as "***". Locality is judged from the URL alone, with no DNS lookup:
a host name other than localhost or a ".local" name counts as not local.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from urllib.parse import urlsplit

from scribecheck._url import check_base_url, check_env_name
from scribecheck.errors import InputError

ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
OPENAI_KEY_ENV = "OPENAI_API_KEY"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

_SHARED_ADDRESS_SPACE = ipaddress.ip_network("100.64.0.0/10")


class Secret:
    """An API key that never shows itself in repr, str, logs or tracebacks."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        """The key itself, for building a request header only."""
        return self._value

    def __repr__(self) -> str:
        return "Secret('***')"

    __str__ = __repr__


@dataclass(frozen=True)
class Endpoint:
    """A model endpoint's base URL, host and locality."""

    base_url: str
    host: str
    is_local: bool

    @classmethod
    def from_url(cls, base_url: str) -> Endpoint:
        """Raises `InputError` for a URL that is not http(s), or that carries credentials,
        a query or a fragment (see `scribecheck._url.check_base_url`)."""
        clean = check_base_url(base_url)
        host = (urlsplit(clean).hostname or "").lower()
        return cls(clean, host, _is_local(host))


def _is_local(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or (address.version == 4 and address in _SHARED_ADDRESS_SPACE)
    )


def privacy_warning(endpoint: Endpoint) -> str | None:
    """The spec's warning for an endpoint off the local machine and network, else None."""
    if endpoint.is_local:
        return None
    return (
        f"the model endpoint {endpoint.host} is not on this machine or the local network: "
        "transcript and note text will be sent there"
    )


def api_key(provider: str, endpoint: Endpoint, key_env: str | None = None) -> Secret | None:
    """The key from the environment: `key_env` if given, else the provider's usual variable.

    Raises `InputError` when a non-local endpoint has no key.
    """
    if key_env is not None:
        check_env_name(key_env)
    else:
        if provider == "anthropic":
            key_env = ANTHROPIC_KEY_ENV
        elif endpoint.host == "openrouter.ai" or endpoint.host.endswith(".openrouter.ai"):
            key_env = OPENROUTER_KEY_ENV
        else:
            key_env = OPENAI_KEY_ENV
    value = os.environ.get(key_env, "").strip()
    if value:
        return Secret(value)
    if endpoint.is_local:
        return None
    raise InputError(f"no API key: set the environment variable {key_env}")
