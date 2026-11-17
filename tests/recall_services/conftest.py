"""Test setup for the service layer.

The guards are set before the first project import: a dummy key, an empty
temp home, a temp data folder, no dotenv file, and no socket that leaves this
machine. See tests/integration/guards.py.
"""

from __future__ import annotations

from tests.integration import guards

guards.install()

from collections.abc import Iterator  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _guards(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A fixed time zone, a temp data folder and a live socket block for every test."""
    guards.assert_no_way_out()
    guards.pin_data_folder(monkeypatch)
    with guards.time_zone():
        yield
