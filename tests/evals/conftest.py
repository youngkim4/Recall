"""Test setup for the eval harness.

The guards are set before the first project import: a dummy key, an empty
temp home, a temp data folder and no socket that leaves this machine. See
tests/guards.py. The world is built once and only read.
"""

from __future__ import annotations

from tests import guards

guards.install()

from collections.abc import Iterator  # noqa: E402

import pytest  # noqa: E402

from archive import Archive  # noqa: E402
from evals.world import Built, build_world, zone  # noqa: E402


@pytest.fixture(autouse=True)
def _guards(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The temp data folder and the time zone of the world for every test."""
    guards.pin_data_folder(monkeypatch)
    with zone():
        yield


@pytest.fixture(scope="session")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    return build_world(tmp_path_factory.mktemp("eval-world"))


@pytest.fixture
def archive(built: Built) -> Iterator[Archive]:
    with Archive.open(built.archive_path) as opened:
        yield opened
