"""Test setup for the archive package.

The guards are set before the first project import: a dummy key, an empty temp
home, a temp data folder, and no socket that leaves this machine. See
tests/guards.py. Each archive test also has a fixed time zone.
"""

from __future__ import annotations

from tests import guards

guards.install()

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from archive import Archive, import_chat_db
from tests.archive import worlds
from tests.archive.builders import ChatDbBuilder


@pytest.fixture(autouse=True)
def _guards(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A fixed time zone and the temp data folder for every archive test."""
    guards.pin_data_folder(monkeypatch)
    with guards.time_zone():
        yield


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
