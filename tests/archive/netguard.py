"""Block every socket that leaves this machine. Loopback and unix sockets stay open."""

from __future__ import annotations

import ipaddress
import socket
from typing import Any

_LOCAL_NAMES = frozenset({"", "localhost", "localhost.localdomain", "ip6-localhost"})
_MARK = "_recall_netguard"
# TEST-NET-1 (RFC 5737): reserved for documentation, never routed
PROBE_ADDRESS = ("192.0.2.1", 9)


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
        address = args[address_index] if args else None
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
    """Patch the socket module. Safe to call again: a function that is already guarded is left alone."""
    for name, index in (("connect", 0), ("connect_ex", 0), ("sendto", -1)):
        current = getattr(socket.socket, name)
        if not getattr(current, _MARK, False):
            setattr(socket.socket, name, _guarded(current, name, index))
    for name in ("getaddrinfo", "gethostbyname"):
        current = getattr(socket, name)
        if not getattr(current, _MARK, False):
            setattr(socket, name, _guarded_lookup(current, name))


def blocks_outbound() -> bool:
    """True when a connection to another machine is refused before any packet leaves.

    The probe is a UDP connect: it only sets a destination and sends nothing,
    also when no guard is active. It works with any guard, not only this one.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(PROBE_ADDRESS)
    except Exception:  # noqa: BLE001 - any refusal counts
        return True
    finally:
        probe.close()
    return False


def ensure() -> None:
    """Make sure outbound sockets are blocked now, by this guard or by another one."""
    if not blocks_outbound():
        install()
