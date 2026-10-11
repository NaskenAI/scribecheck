"""Test-wide network guard.

Every test runs with outbound network connections blocked, so nothing can quietly
depend on the network. Loopback (127.0.0.1, ::1, localhost) and Unix sockets stay
open because asyncio and socket.socketpair() use them, especially on Windows.
Rules mode's independence from the model client gets its own test in M2.

Tests that genuinely need the network must be marked @pytest.mark.network.
Do not weaken or remove this guard (see CLAUDE.md).
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest

BLOCKED_MESSAGE = (
    "Network access is blocked in tests. "
    "Mark the test with @pytest.mark.network if it truly needs the network."
)

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", ""}


def _is_local_host(host: Any) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode(errors="ignore")
    host = str(host)
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


def _is_local_address(address: Any) -> bool:
    # Unix-domain sockets use a path (str or bytes) as the address.
    if isinstance(address, (str, bytes)):
        return True
    if isinstance(address, tuple) and address:
        return _is_local_host(address[0])
    return False


@pytest.fixture(autouse=True)
def _block_network(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    if request.node.get_closest_marker("network") is not None:
        yield
        return

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if not _is_local_address(address):
            raise RuntimeError(BLOCKED_MESSAGE)
        original_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if not _is_local_address(address):
            raise RuntimeError(BLOCKED_MESSAGE)
        return original_connect_ex(self, address)

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_local_host(host):
            raise RuntimeError(BLOCKED_MESSAGE)
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    yield
