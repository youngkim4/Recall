"""Test setup for the archive package.

The lines before the first project import do three things:
- set a dummy OPENAI_API_KEY (load_dotenv can find a real .env above a worktree)
- point RECALL_DATA_DIR at a temp folder, so no default path reaches real data
- block every socket that leaves this machine (loopback stays open)
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

os.environ["OPENAI_API_KEY"] = "test-key-not-real"
_DATA_DIR = tempfile.mkdtemp(prefix="recall-archive-tests-")
atexit.register(shutil.rmtree, _DATA_DIR, True)
os.environ.setdefault("RECALL_DATA_DIR", _DATA_DIR)

from tests.archive import netguard  # noqa: E402

netguard.install()

import pytest  # noqa: E402

from tests.archive.builders import ChatDbBuilder  # noqa: E402

TEST_TIME_ZONE = "America/Los_Angeles"


@pytest.fixture(autouse=True)
def _guards() -> Iterator[None]:
    """Fixed time zone and a live socket block for every archive test."""
    assert netguard.is_installed(), "the socket block was removed"
    previous = os.environ.get("TZ")
    os.environ["TZ"] = TEST_TIME_ZONE
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


@pytest.fixture
def archive_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "archive.sqlite3"


@pytest.fixture
def source_path(tmp_path: Path) -> Path:
    folder = tmp_path / "source"
    folder.mkdir()
    return folder / "chat.db"


@pytest.fixture
def builder(source_path: Path) -> Iterator[ChatDbBuilder]:
    """An empty synthetic chat.db with Apple's table layout."""
    made = ChatDbBuilder(source_path)
    try:
        yield made
    finally:
        made.close()
