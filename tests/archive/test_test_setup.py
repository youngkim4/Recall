"""The safety setup of the archive tests: no real key, no real data folder, no network."""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path

import pytest

from archive import default_archive_path
from tests import guards
from tests.archive.legacy import REPO_ROOT


def test_api_key_is_a_dummy() -> None:
    assert os.environ["OPENAI_API_KEY"] == "test-key-not-real"


def test_default_data_folder_is_a_temp_folder() -> None:
    data_dir = Path(os.environ["RECALL_DATA_DIR"]).resolve()
    assert REPO_ROOT not in data_dir.parents and data_dir != REPO_ROOT
    assert data_dir.is_relative_to(Path(tempfile.gettempdir()).resolve())
    assert default_archive_path().resolve().is_relative_to(data_dir)


def test_a_data_folder_that_was_read_earlier_is_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another test package can import recall_paths first: the guard must still win."""
    import recall_paths

    monkeypatch.setattr(recall_paths, "SAVES_DIR", REPO_ROOT / "saves")
    guards.pin_data_folder(monkeypatch)

    assert default_archive_path().resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())


@pytest.mark.parametrize("address", [("192.0.2.1", 443), ("2001:db8::1", 443), ("api.example.invalid", 443)])
def test_connect_to_another_machine_is_blocked(address: tuple[str, int]) -> None:
    family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock, pytest.raises(guards.NetworkBlocked):
        sock.connect(address)
    with socket.socket(family, socket.SOCK_DGRAM) as sock, pytest.raises(guards.NetworkBlocked):
        sock.sendto(b"x", address)


def test_name_lookup_of_another_machine_is_blocked() -> None:
    with pytest.raises(guards.NetworkBlocked):
        socket.getaddrinfo("api.openai.com", 443)
    with pytest.raises(guards.NetworkBlocked):
        socket.create_connection(("api.openai.com", 443), timeout=1)
    with pytest.raises(guards.NetworkBlocked):
        socket.gethostbyname("example.com")


def test_loopback_stays_open() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=5) as client:
            accepted, _ = server.accept()
            client.sendall(b"ping")
            assert accepted.recv(4) == b"ping"
            accepted.close()
    assert socket.getaddrinfo("localhost", 80)


def test_guard_comes_back_when_another_patch_removes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    import _socket

    monkeypatch.setattr(socket.socket, "connect", _socket.socket.connect)  # the C method, with no guard
    assert not guards.blocks_outbound()
    guards.ensure()
    assert guards.blocks_outbound()
