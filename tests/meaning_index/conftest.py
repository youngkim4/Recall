"""Test setup for meaning_index.

The guards are set before any project import: see tests/guards.py. These tests
open no socket at all, so each one also has loopback blocked. Every text, id
and vector in these tests is invented.
"""

from tests import guards

guards.install()

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    guards.block_loopback_too(monkeypatch)


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """Where a test keeps its index. The store makes the folder."""
    return tmp_path / "meaning"
