"""Block every socket that leaves this machine. Loopback and unix sockets stay open."""

from __future__ import annotations

import ipaddress
import socket
from typing import Any

_LOCAL_NAMES = frozenset({"", "localhost", "localhost.localdomain", "ip6-localhost"})
_MARK = "_recall_netguard"


class NetworkBlocked(RuntimeError):
    """A test tried to reach a host outside this machine."""


def _is_local_host(host: object) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    text = str(host).strip().lower()
    if text in _LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(text.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _is_local_address(address: object) -> bool:
    if isinstance(address, (str, bytes)):
        return True  # unix socket path
    if isinstance(address, tuple) and address:
        return _is_local_host(address[0])
    return False


def _guarded(original: Any, name: str, address_index: int) -> Any:
    def wrapper(self: socket.socket, *args: Any, **kwargs: Any) -> Any:
        address = args[address_index] if len(args) > address_index else None
        if not _is_local_address(address):
            raise NetworkBlocked(f"socket.{name} to {address!r} is blocked in archive tests")
        return original(self, *args, **kwargs)

    setattr(wrapper, _MARK, True)
    return wrapper


def _guarded_lookup(original: Any, name: str) -> Any:
    def wrapper(host: object = None, *args: Any, **kwargs: Any) -> Any:
        if not _is_local_host(host):
            raise NetworkBlocked(f"socket.{name}({host!r}) is blocked in archive tests")
        return original(host, *args, **kwargs)

    setattr(wrapper, _MARK, True)
    return wrapper


def install() -> None:
    """Patch the socket module once. Safe to call again."""
    if is_installed():
        return
    socket.socket.connect = _guarded(socket.socket.connect, "connect", 0)  # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded(socket.socket.connect_ex, "connect_ex", 0)  # type: ignore[method-assign]
    socket.socket.sendto = _guarded(socket.socket.sendto, "sendto", -1)  # type: ignore[method-assign]
    socket.getaddrinfo = _guarded_lookup(socket.getaddrinfo, "getaddrinfo")
    socket.gethostbyname = _guarded_lookup(socket.gethostbyname, "gethostbyname")


def is_installed() -> bool:
    return bool(getattr(socket.socket.connect, _MARK, False)) and bool(getattr(socket.getaddrinfo, _MARK, False))
