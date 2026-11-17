"""Test setup for the tests of the joined branch.

The guards are set before the first project import: a dummy key, a temp data
folder, and no socket that leaves this machine. See tests/guards.py.
"""

from __future__ import annotations

from tests import guards

guards.install()

from collections.abc import Iterator  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _guards(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A fixed time zone and the temp data folder for every test."""
    guards.pin_data_folder(monkeypatch)
    with guards.time_zone():
        yield
