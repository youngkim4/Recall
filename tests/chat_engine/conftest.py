"""Guards for the chat_engine tests: a dummy key, no network, a fixed clock."""

import os

# before any project import: load_dotenv() must never find a real key
os.environ["OPENAI_API_KEY"] = "test-key-not-real"

import socket  # noqa: E402
from datetime import datetime  # noqa: E402

import pytest  # noqa: E402

FIXED_NOW = datetime(2026, 11, 16, 9, 30, 0)


class NetworkBlocked(RuntimeError):
    """Raised when a test tries to open an outbound connection."""


def _blocked(*_args: object, **_kwargs: object) -> None:
    raise NetworkBlocked("chat_engine tests must not use the network")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")


@pytest.fixture
def fixed_now() -> datetime:
    return FIXED_NOW
