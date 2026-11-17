"""Test setup for report_engine.

Order matters. The dummy key is set before any project import, so a
`load_dotenv()` anywhere in the tree can not put a real key in the process.
Outbound sockets are blocked at import time, so no test and no import can
reach a model provider. Loopback stays open for local test servers.
"""

import os

os.environ["OPENAI_API_KEY"] = "sk-test-dummy-not-a-real-key"

import ipaddress  # noqa: E402
import socket  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402


class OutboundNetworkBlocked(RuntimeError):
    """Raised when a test tries to open a socket to a non-loopback host."""


BLOCKED_ATTEMPTS: list[str] = []

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_getaddrinfo = socket.getaddrinfo


def _is_loopback_host(host: Any) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    if host in {"localhost", ""}:
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _check(sock: socket.socket, address: Any) -> None:
    if sock.family == socket.AF_UNIX:
        return
    host = address[0] if isinstance(address, tuple) and address else address
    if _is_loopback_host(host):
        return
    BLOCKED_ATTEMPTS.append(str(host))
    raise OutboundNetworkBlocked(f"outbound socket blocked in tests: {host!r}")


def _guarded_connect(self: socket.socket, address: Any) -> Any:
    _check(self, address)
    return _real_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> Any:
    _check(self, address)
    return _real_connect_ex(self, address)


def _guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    if host is not None and not _is_loopback_host(host):
        BLOCKED_ATTEMPTS.append(str(host))
        raise OutboundNetworkBlocked(f"name lookup blocked in tests: {host!r}")
    return _real_getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]
socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]


@pytest.fixture
def blocked_attempts() -> list[str]:
    """Hosts that code under test tried to reach. Stays empty in a clean run."""
    return BLOCKED_ATTEMPTS


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "period-cache.sqlite"


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    return tmp_path / "reports"
