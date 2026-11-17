"""Test setup for report_engine.

The guards are set before any project import: see tests/guards.py. The dummy
key is there first, so a `load_dotenv()` anywhere in the tree can not put a
real key in the process. Outbound sockets are blocked at import time, so no
test and no import can reach a model provider. Loopback stays open for local
test servers.
"""

from tests import guards

guards.install()

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture
def blocked_attempts() -> list[str]:
    """Hosts that code under test tried to reach. Stays empty in a clean run."""
    return guards.BLOCKED_ATTEMPTS


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "period-cache.sqlite"


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    return tmp_path / "reports"
