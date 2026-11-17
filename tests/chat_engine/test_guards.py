"""The guards themselves: no network, no real key, no import of the old server."""

import importlib
import os
import pkgutil
import socket
import subprocess
import sys

import pytest

import chat_engine
from tests.chat_engine.conftest import NetworkBlocked


def _module_names() -> list[str]:
    names = [chat_engine.__name__]
    for info in pkgutil.walk_packages(chat_engine.__path__, prefix="chat_engine."):
        names.append(info.name)
    return names


def test_outbound_connections_are_blocked() -> None:
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("example.com", 443), timeout=1)
    with pytest.raises(NetworkBlocked):
        socket.socket().connect(("127.0.0.1", 9))


def test_api_key_is_a_dummy() -> None:
    assert os.environ["OPENAI_API_KEY"] == "test-key-not-real"


def test_every_module_imports_without_network() -> None:
    for name in _module_names():
        module = importlib.import_module(name)
        importlib.reload(module)


def test_package_never_imports_the_old_server() -> None:
    names = [name for name in _module_names() if not name.endswith("__main__")]
    code = (
        "import importlib, sys\n"
        f"for name in {names!r}:\n"
        "    importlib.import_module(name)\n"
        "banned = {'ui_server', 'openai_client', 'retrieval_planner', 'conversation', 'tiktoken'}\n"
        "found = sorted(banned & set(sys.modules))\n"
        "assert not found, found\n"
    )
    env = {**os.environ, "OPENAI_API_KEY": "test-key-not-real"}
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )
    assert done.returncode == 0, done.stderr
