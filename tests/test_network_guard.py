"""The network guard in conftest.py must block real connections and allow loopback."""

from __future__ import annotations

import socket

import pytest


def test_dns_lookup_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="Network access is blocked"):
        socket.getaddrinfo("example.com", 443)


def test_connection_by_hostname_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="Network access is blocked"):
        socket.create_connection(("example.com", 443), timeout=1)


def test_connection_by_ip_is_blocked() -> None:
    # 192.0.2.0/24 is reserved for documentation (RFC 5737), so this never reaches anyone.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="Network access is blocked"):
            sock.connect(("192.0.2.1", 443))
        with pytest.raises(RuntimeError, match="Network access is blocked"):
            sock.connect_ex(("192.0.2.1", 443))
    finally:
        sock.close()


def test_loopback_socketpair_still_works() -> None:
    left, right = socket.socketpair()
    try:
        left.sendall(b"ping")
        assert right.recv(4) == b"ping"
    finally:
        left.close()
        right.close()
