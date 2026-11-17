"""Guards for the chat_engine tests: a dummy key, no network, a fixed clock.

The shared guards are set before any project import: see tests/guards.py.
These tests open no socket at all, so each one also has loopback blocked.
"""

from tests import guards

guards.install()

from datetime import datetime  # noqa: E402

import pytest  # noqa: E402

FIXED_NOW = datetime(2026, 11, 16, 9, 30, 0)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    guards.block_loopback_too(monkeypatch)


@pytest.fixture
def fixed_now() -> datetime:
    return FIXED_NOW
