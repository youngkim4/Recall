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
os.environ["RECALL_DATA_DIR"] = _DATA_DIR  # also when the shell of the developer names the real folder

from tests.archive import netguard

netguard.install()

from dataclasses import dataclass
from typing import Any

import pytest

from archive import Archive, import_chat_db
from tests.archive import worlds
from tests.archive.builders import ChatDbBuilder

TEST_TIME_ZONE = "America/Los_Angeles"
DATA_FILES = {
    "SAVES_DIR": "saves",
    "OUT_DIR": "out",
    "DEFAULT_MESSAGES_CSV": "messages.csv",
    "SNAPSHOT_DB": "chat.db",
    "DOTENV_PATH": ".env",
}


def _pin_data_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point recall_paths at the temp folder.

    recall_paths reads RECALL_DATA_DIR one time, at its first import. In a run of the whole
    test suite another package can import it first, with the real data folder.
    """
    import recall_paths

    monkeypatch.setenv("RECALL_DATA_DIR", _DATA_DIR)
    monkeypatch.setattr(recall_paths, "DATA_DIR", Path(_DATA_DIR))
    for name, leaf in DATA_FILES.items():
        monkeypatch.setattr(recall_paths, name, Path(_DATA_DIR) / leaf)


@pytest.fixture(autouse=True)
def _guards(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fixed time zone, a temp data folder and a live socket block for every archive test."""
    netguard.ensure()
    assert netguard.blocks_outbound(), "sockets that leave this machine are not blocked"
    _pin_data_folder(monkeypatch)
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


@dataclass(frozen=True)
class Harbor:
    """The imported "Harbor" history (tests/archive/worlds.py) and an open Archive on it."""

    archive: Archive
    path: Path
    guids: dict[str, str]

    def id(self, label: str) -> str:
        return self.guids[label]

    def labels(self, messages: list[dict[str, Any]]) -> list[str]:
        names = {guid: label for label, guid in self.guids.items()}
        return [names[message["id"]] for message in messages]


@pytest.fixture(scope="session")
def harbor_files(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, str]]:
    """Import the Harbor history once. Tests only read it."""
    folder = tmp_path_factory.mktemp("harbor")
    source = folder / "source" / "chat.db"
    source.parent.mkdir()
    made = ChatDbBuilder(source)
    world = worlds.build_harbor(made)
    made.close()
    archive_path = folder / "data" / "archive.sqlite3"
    import_chat_db(source, archive_path, contacts=[worlds.ROWAN_CONTACT, worlds.MIRA_CONTACT])
    return archive_path, dict(world.guids)


@pytest.fixture
def harbor(harbor_files: tuple[Path, dict[str, str]]) -> Iterator[Harbor]:
    path, guids = harbor_files
    with Archive.open(path) as archive:
        yield Harbor(archive, path, guids)
