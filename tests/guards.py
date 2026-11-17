"""The one guard of every test suite.

`install()` runs before the first project import: tests/conftest.py calls it. It:
- sets one dummy OPENAI_API_KEY
- removes every setting that points a client at another address or at a proxy
- points HOME at an empty temp folder: contact_names.py and setup_status.py take
  the Contacts and Messages folders from it when they load
- points RECALL_DATA_DIR at a temp folder, so no default path reaches real data
- makes load_dotenv do nothing: openai_client.py calls it with no path when the
  data folder has no .env, and it then walks up to a real .env above a worktree
- blocks every socket that leaves this machine. Loopback and unix sockets stay open.

`hold()` runs at the start of each test: the guards are in force again, whatever
an earlier test left behind.

A child process has no socket guard: a test that starts one must open nothing there.

This module uses the standard library only and imports nothing of the project
when it loads. `dotenv` is silenced when it is installed; it is not needed.
"""

from __future__ import annotations

import atexit
import ipaddress
import os
import shutil
import socket
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

DUMMY_KEY = "test-key-not-real"
TEST_TIME_ZONE = "America/Los_Angeles"
TEMP_PREFIX = "recall-tests-"
HOME_LEAF = "home"
DATA_FILES = {
    "SAVES_DIR": "saves",
    "OUT_DIR": "out",
    "DEFAULT_MESSAGES_CSV": "messages.csv",
    "SNAPSHOT_DB": "chat.db",
    "DOTENV_PATH": ".env",
}
# nothing in the environment may point a client at another address or at a proxy
CLIENT_SETTINGS = (
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
SOCKET_METHODS = ("connect", "connect_ex", "sendto", "sendmsg")
SOCKET_FUNCTIONS = (
    "create_connection",
    "getaddrinfo",
    "gethostbyname",
    "gethostbyname_ex",
    "gethostbyaddr",
    "getnameinfo",
)
# TEST-NET-1 (RFC 5737): reserved for documentation, never routed
PROBE_ADDRESS = ("192.0.2.1", 9)
# each host that a guard refused, in order. A test reads the part it added.
BLOCKED_ATTEMPTS: list[str] = []

_LOCAL_NAMES = frozenset({"", "localhost", "localhost.localdomain", "ip6-localhost"})
_MARK = "_recall_guard"
_Guards = dict[tuple[Any, str], Callable[..., Any]]

_data_dir: str | None = None


class NetworkBlocked(OSError):
    """A test tried to reach a host outside this machine."""


# --- which address stays on this machine ---------------------------------------


def _host_of(target: object) -> object:
    return target[0] if isinstance(target, tuple) and target else target


def _is_local_host(host: object) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    text = host.strip().lower()
    if text in _LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(text.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _stays_here(sock: socket.socket, address: object) -> bool:
    # no address: the socket is connected already, and `connect` was guarded
    if address is None or sock.family == socket.AF_UNIX:
        return True
    return _is_local_host(_host_of(address))


def _address_of(name: str, args: tuple[Any, ...]) -> object:
    """The destination among the arguments of a socket method, or None."""
    if name == "sendmsg":
        return args[3] if len(args) > 3 else None
    if name == "sendto":
        return args[-1] if len(args) > 1 else None
    return args[0] if args else None


def _refusal(way: str, host: object) -> NetworkBlocked:
    BLOCKED_ATTEMPTS.append(str(host))
    return NetworkBlocked(f"network access is blocked in tests: socket.{way} to {host!r}")


# --- the guards ----------------------------------------------------------------


def _guard_method(name: str, loopback: bool) -> Callable[..., Any]:
    real = getattr(socket.socket, name)

    def guarded(self: socket.socket, *args: Any) -> Any:
        address = _address_of(name, args)
        if not (loopback and _stays_here(self, address)):
            raise _refusal(name, _host_of(address))
        return real(self, *args)

    setattr(guarded, _MARK, True)
    return guarded


def _guard_function(name: str, loopback: bool) -> Callable[..., Any]:
    real = getattr(socket, name)

    def guarded(*args: Any, **kwargs: Any) -> Any:
        host = _host_of(args[0] if args else kwargs.get("host", kwargs.get("address")))
        if not (loopback and _is_local_host(host)):
            raise _refusal(name, host)
        return real(*args, **kwargs)

    setattr(guarded, _MARK, True)
    return guarded


def _build(loopback: bool) -> _Guards:
    methods: _Guards = {(socket.socket, name): _guard_method(name, loopback) for name in SOCKET_METHODS}
    functions: _Guards = {(socket, name): _guard_function(name, loopback) for name in SOCKET_FUNCTIONS}
    return {**methods, **functions}


# both are made here, before a guard is set: each one calls the real function
_LOOPBACK_OPEN = _build(loopback=True)
_NOTHING_OPEN = _build(loopback=False)


def _guard_sockets() -> None:
    """Guard each way out that has no guard now. A stricter guard of this module is left alone."""
    for (owner, name), guard in _LOOPBACK_OPEN.items():
        if not getattr(getattr(owner, name), _MARK, False):
            setattr(owner, name, guard)


def _every_way_is_guarded() -> bool:
    return all(getattr(getattr(owner, name), _MARK, False) for owner, name in _LOOPBACK_OPEN)


def _probe_is_refused() -> bool:
    """A UDP connect only sets a destination: it sends nothing, also when no guard is active."""
    seen = len(BLOCKED_ATTEMPTS)
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(PROBE_ADDRESS)
    except Exception:  # noqa: BLE001 - any refusal counts
        return True
    finally:
        probe.close()
        del BLOCKED_ATTEMPTS[seen:]  # the probe is not an attempt of a test
    return False


def blocks_outbound() -> bool:
    """True when each way out has a guard, and a connection to another machine is refused."""
    return _every_way_is_guarded() and _probe_is_refused()


def ensure() -> None:
    """Make sure that outbound sockets are blocked now."""
    if not blocks_outbound():
        _guard_sockets()


def block_loopback_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Until the test ends no socket connects, sends or looks up a name: not on this machine either."""
    for (owner, name), guard in _NOTHING_OPEN.items():
        monkeypatch.setattr(owner, name, guard)


# --- the environment -----------------------------------------------------------


def _read_no_dotenv(*_args: object, **_kwargs: object) -> bool:
    """Stands in for dotenv.load_dotenv: no file is looked for and none is read."""
    return False


def _silence_dotenv() -> None:
    try:
        import dotenv
    except ImportError:  # no code can read a .env file then
        return
    dotenv.load_dotenv = _read_no_dotenv  # type: ignore[assignment]


def data_folder() -> Path:
    """The temp folder that stands in for the data folder. The empty home is in it."""
    global _data_dir
    if _data_dir is None:
        # the real path, as recall_paths resolves the folder that it reads
        _data_dir = os.path.realpath(tempfile.mkdtemp(prefix=TEMP_PREFIX))
        atexit.register(shutil.rmtree, _data_dir, True)
        (Path(_data_dir) / HOME_LEAF).mkdir()
    return Path(_data_dir)


def _environment() -> dict[str, str]:
    folder = data_folder()
    return {"OPENAI_API_KEY": DUMMY_KEY, "RECALL_DATA_DIR": str(folder), "HOME": str(folder / HOME_LEAF)}


def install() -> Path:
    """Set the guards and return the temp data folder. Safe to call again."""
    os.environ.update(_environment())
    for name in CLIENT_SETTINGS:
        os.environ.pop(name, None)
    _silence_dotenv()
    _guard_sockets()
    return data_folder()


def hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """For the start of each test: the guards are in force, whatever an earlier test left."""
    ensure()
    assert blocks_outbound(), "sockets that leave this machine are not blocked"
    for name, value in _environment().items():
        monkeypatch.setenv(name, value)
    for name in CLIENT_SETTINGS:
        monkeypatch.delenv(name, raising=False)


def pin_data_folder(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point recall_paths at the temp folder.

    recall_paths reads RECALL_DATA_DIR one time, at its first import. A test can
    change its paths; the next test must find the temp folder again.
    """
    import recall_paths

    folder = data_folder()
    monkeypatch.setenv("RECALL_DATA_DIR", str(folder))
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
