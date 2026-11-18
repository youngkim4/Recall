"""Test setup for the service layer.

The guards are set before the first project import: a dummy key, an empty
temp home, a temp data folder, no dotenv file, and no socket that leaves this
machine. See tests/guards.py.
"""

from __future__ import annotations

from tests import guards

guards.install()

from collections.abc import Iterator  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import pytest  # noqa: E402

from archive import Archive  # noqa: E402
from recall_services.names import Names  # noqa: E402
from tests.recall_services import world  # noqa: E402

if TYPE_CHECKING:
    from tests.recall_services import memory_kit, report_world


@pytest.fixture(autouse=True)
def _guards(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A fixed time zone and the temp data folder for every test."""
    guards.pin_data_folder(monkeypatch)
    with guards.time_zone():
        yield


@pytest.fixture(scope="session")
def orchard(tmp_path_factory: pytest.TempPathFactory) -> world.Orchard:
    """The Orchard history, pulled into a real archive once. Tests only read it."""
    return world.build_orchard(tmp_path_factory.mktemp("orchard"))


@pytest.fixture
def archive(orchard: world.Orchard) -> Iterator[Archive]:
    with Archive.open(orchard.archive_path) as opened:
        yield opened


@pytest.fixture
def saved() -> Names:
    return Names.from_records(world.RECORDS)


@pytest.fixture
def desk(tmp_path: Path) -> Iterator["report_world.Desk"]:
    """An archive with a long and a short conversation, and places for the files of a report."""
    from tests.recall_services import report_world

    made = report_world.build_desk(tmp_path)
    try:
        yield made
    finally:
        report_world.close_desk(made)


@pytest.fixture
def lake(tmp_path: Path) -> Iterator["memory_kit.Lake"]:
    """An archive with a few short exchanges of two people, and a folder for the index of meanings."""
    from tests.recall_services import memory_kit

    made = memory_kit.build_lake(tmp_path)
    try:
        yield made
    finally:
        made.close()
