"""The guards themselves: no network, no real key, no import of the old server."""

import os
import pkgutil
import socket
import subprocess
import sys

import pytest

import chat_engine
from tests.chat_engine.conftest import NetworkBlocked

_BANNED = ("ui_server", "openai_client", "retrieval_planner", "conversation", "tiktoken", "dotenv")


def _module_names() -> list[str]:
    names = [chat_engine.__name__]
    for info in pkgutil.walk_packages(chat_engine.__path__, prefix="chat_engine."):
        if not info.name.endswith("__main__"):
            names.append(info.name)
    return names


def test_outbound_connections_are_blocked() -> None:
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("example.com", 443), timeout=1)
    with pytest.raises(NetworkBlocked):
        socket.socket().connect(("127.0.0.1", 9))


def test_api_key_is_a_dummy() -> None:
    assert os.environ["OPENAI_API_KEY"] == "test-key-not-real"


def test_every_module_imports_in_a_clean_process_without_network_or_old_code() -> None:
    """A fresh interpreter with sockets blocked imports the whole package."""
    code = (
        "import importlib, socket, sys\n"
        "def blocked(*a, **k):\n"
        "    raise RuntimeError('network used at import time')\n"
        "socket.socket.connect = blocked\n"
        "socket.socket.connect_ex = blocked\n"
        "socket.create_connection = blocked\n"
        "socket.getaddrinfo = blocked\n"
        f"for name in {_module_names()!r}:\n"
        "    importlib.import_module(name)\n"
        f"found = sorted(set({_BANNED!r}) & set(sys.modules))\n"
        "assert not found, found\n"
    )
    env = {**os.environ, "OPENAI_API_KEY": "test-key-not-real"}
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120, check=False
    )
    assert done.returncode == 0, done.stderr
