"""Guards shared by the tests of the joined branch.

`install()` runs before any project import. It does five things:
- sets a dummy OPENAI_API_KEY
- points HOME at an empty temp folder: contact_names.py and setup_status.py take
  the Contacts and Messages folders from it when they load
- makes load_dotenv do nothing: openai_client.py calls it with no path when the
  data folder has no .env, and it then walks up to a real .env above a worktree
- points RECALL_DATA_DIR at a temp folder, so no default path reaches real data
- blocks every socket that leaves this machine (loopback stays open), and the
  lookups and the `sendmsg` that the socket guard of the archive tests leaves open

A test must not start another process: no guard of this module holds there.

This module imports nothing of the project but the socket guard of the
archive tests, which uses the standard library only.
"""

from __future__ import annotations

import atexit
import os
import shutil
import socket
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import dotenv
import pytest

from tests.archive import netguard

DUMMY_KEY = "test-key-not-real"
TEST_TIME_ZONE = "America/Los_Angeles"
HOME_LEAF = "home"
DATA_FILES = {
    "SAVES_DIR": "saves",
    "OUT_DIR": "out",
    "DEFAULT_MESSAGES_CSV": "messages.csv",
    "SNAPSHOT_DB": "chat.db",
    "DOTENV_PATH": ".env",
}
# nothing in the environment may point a client at another address or at a proxy
_CLIENT_SETTINGS = (
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

_data_dir: str | None = None


def _read_no_dotenv(*_args: object, **_kwargs: object) -> bool:
    """Stands in for dotenv.load_dotenv: no file is looked for and none is read."""
    return False


_GUARDED = "_recall_joined_guard"
_THIS_MACHINE = frozenset({"", "localhost", "127.0.0.1", "::1"})
_LOOKUPS = ("gethostbyname_ex", "gethostbyaddr", "getnameinfo")


def _refuse(name: str) -> OSError:
    return OSError(f"network access is blocked in tests: {name}")


def _host_of(target: object) -> str:
    return str(target[0] if isinstance(target, tuple) and target else target or "")


def _guard_lookups() -> None:
    """Refuse the lookups that the socket guard of the archive tests does not cover."""
    for name in _LOOKUPS:
        real = getattr(socket, name)
        if getattr(real, _GUARDED, False):
            continue

        def guarded(target: object, *more: object, _real: object = real, _name: str = name) -> object:
            if _host_of(target) not in _THIS_MACHINE:
                raise _refuse(_name)
            return _real(target, *more)  # type: ignore[operator]

        setattr(guarded, _GUARDED, True)
        setattr(socket, name, guarded)


def _guard_sendmsg() -> None:
    real = socket.socket.sendmsg
    if getattr(real, _GUARDED, False):
        return

    def guarded(self: socket.socket, buffers: object, *more: object) -> int:
        address = more[2] if len(more) > 2 else None
        if address is not None and _host_of(address) not in _THIS_MACHINE:
            raise _refuse("sendmsg")
        return real(self, buffers, *more)  # type: ignore[arg-type]

    setattr(guarded, _GUARDED, True)
    socket.socket.sendmsg = guarded  # type: ignore[method-assign,assignment]


def install() -> Path:
    """Set the guards and return the temp data folder. Safe to call again."""
    global _data_dir
    os.environ["OPENAI_API_KEY"] = DUMMY_KEY
    for name in _CLIENT_SETTINGS:
        os.environ.pop(name, None)
    dotenv.load_dotenv = _read_no_dotenv  # type: ignore[assignment]
    if _data_dir is None:
        _data_dir = tempfile.mkdtemp(prefix="recall-joined-tests-")
        atexit.register(shutil.rmtree, _data_dir, True)
        (Path(_data_dir) / HOME_LEAF).mkdir()
    os.environ["RECALL_DATA_DIR"] = _data_dir
    os.environ["HOME"] = str(Path(_data_dir) / HOME_LEAF)
    netguard.install()
    _guard_lookups()
    _guard_sendmsg()
    return Path(_data_dir)


def pin_data_folder(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point recall_paths at the temp folder.

    recall_paths reads RECALL_DATA_DIR one time, at its first import. In a run
    of the whole test suite another package can import it first.
    """
    import recall_paths

    folder = install()
    monkeypatch.setenv("RECALL_DATA_DIR", str(folder))
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
    monkeypatch.setattr(recall_paths, "DATA_DIR", folder)
    for name, leaf in DATA_FILES.items():
        monkeypatch.setattr(recall_paths, name, folder / leaf)
    return folder


@contextmanager
def time_zone(name: str = TEST_TIME_ZONE) -> Iterator[None]:
    """Run a block in one fixed local time zone."""
    previous = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


def assert_no_way_out() -> None:
    netguard.ensure()
    assert netguard.blocks_outbound(), "sockets that leave this machine are not blocked"
