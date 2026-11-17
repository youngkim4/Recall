"""The guards of every suite (tests/guards.py). The tests of the packages check them too:
these are the cases that no package checks."""

import os
import socket
import tempfile
from pathlib import Path

import pytest

from tests import guards

# TEST-NET-1 (RFC 5737) and .invalid (RFC 2606): these reach nobody, with a guard or without
OUTSIDE = ("192.0.2.1", 9)
NO_HOST = "name.invalid"
WAYS_BY_NAME = {
    "create_connection": lambda: socket.create_connection((NO_HOST, 443), timeout=1),
    "getaddrinfo": lambda: socket.getaddrinfo(NO_HOST, 443),
    "gethostbyname": lambda: socket.gethostbyname(NO_HOST),
    "gethostbyname_ex": lambda: socket.gethostbyname_ex(NO_HOST),
    "gethostbyaddr": lambda: socket.gethostbyaddr(NO_HOST),
    "getnameinfo": lambda: socket.getnameinfo((NO_HOST, 443), 0),
}
WAYS_BY_SOCKET = {
    "connect": lambda sock: sock.connect(OUTSIDE),
    "connect_ex": lambda sock: sock.connect_ex(OUTSIDE),
    "sendto": lambda sock: sock.sendto(b"x", OUTSIDE),
    "sendmsg": lambda sock: sock.sendmsg([b"x"], [], 0, OUTSIDE),
}


def test_the_cases_name_every_way_out() -> None:
    assert sorted(WAYS_BY_NAME) == sorted(guards.SOCKET_FUNCTIONS)
    assert sorted(WAYS_BY_SOCKET) == sorted(guards.SOCKET_METHODS)


@pytest.mark.parametrize("way", sorted(WAYS_BY_NAME))
def test_no_name_outside_this_machine_is_asked_or_reached(way: str) -> None:
    before = len(guards.BLOCKED_ATTEMPTS)
    with pytest.raises(guards.NetworkBlocked, match="blocked in tests"):
        WAYS_BY_NAME[way]()
    assert guards.BLOCKED_ATTEMPTS[before:] == [NO_HOST]


@pytest.mark.parametrize("way", sorted(WAYS_BY_SOCKET))
def test_no_socket_sends_to_another_machine(way: str) -> None:
    before = len(guards.BLOCKED_ATTEMPTS)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(guards.NetworkBlocked, match="blocked in tests"):
            WAYS_BY_SOCKET[way](sock)
    assert guards.BLOCKED_ATTEMPTS[before:] == [OUTSIDE[0]]


def test_a_blocked_way_out_looks_like_a_network_that_is_down() -> None:
    assert issubclass(guards.NetworkBlocked, OSError)


def test_the_probe_is_not_counted_as_an_attempt() -> None:
    before = list(guards.BLOCKED_ATTEMPTS)
    assert guards.blocks_outbound()
    assert guards.BLOCKED_ATTEMPTS == before


def test_a_socket_on_loopback_sends_with_every_call() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server, socket.socket(
        socket.AF_INET, socket.SOCK_DGRAM
    ) as client:
        server.bind(("127.0.0.1", 0))
        here = server.getsockname()
        client.sendto(b"a", here)
        client.sendmsg([b"b"], [], 0, here)
        assert client.connect_ex(here) == 0
        client.sendmsg([b"c"])
        assert [server.recv(1) for _ in range(3)] == [b"a", b"b", b"c"]
    assert socket.getaddrinfo("::1", 80) and socket.gethostbyname("localhost")


def test_a_socket_file_on_this_machine_stays_open() -> None:
    # not tmp_path: the path of a socket file must be short
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "s")
        with socket.socket(socket.AF_UNIX) as server, socket.socket(socket.AF_UNIX) as client:
            server.bind(path)
            server.listen(1)
            client.connect(path)
            accepted, _ = server.accept()
            accepted.close()


def test_the_stricter_block_shuts_loopback_and_ends_with_its_test() -> None:
    with pytest.MonkeyPatch.context() as patch:
        guards.block_loopback_too(patch)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, pytest.raises(guards.NetworkBlocked):
            sock.connect(("127.0.0.1", 9))
        with pytest.raises(guards.NetworkBlocked):
            socket.getaddrinfo("localhost", 80)
        assert guards.blocks_outbound()
    assert socket.getaddrinfo("localhost", 80)
    assert guards.blocks_outbound()


def test_the_start_of_a_test_puts_back_what_an_earlier_test_left() -> None:
    import _socket

    with pytest.MonkeyPatch.context() as left, pytest.MonkeyPatch.context() as patch:
        left.setenv("OPENAI_API_KEY", "left by a test")
        left.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
        left.setenv("HOME", "/nowhere")
        left.setattr(socket.socket, "connect", _socket.socket.connect)  # the C method, with no guard
        assert not guards.blocks_outbound()
        guards.hold(patch)
        assert guards.blocks_outbound()
        assert os.environ["OPENAI_API_KEY"] == guards.DUMMY_KEY
        assert "HTTPS_PROXY" not in os.environ
        assert Path(os.environ["HOME"]) == guards.data_folder() / guards.HOME_LEAF


def test_the_guards_can_be_set_again() -> None:
    folder = guards.data_folder()
    assert guards.install() == folder
    assert Path(os.environ["RECALL_DATA_DIR"]) == folder
    assert guards.blocks_outbound()


def test_the_data_folder_and_the_home_are_temp_folders_from_the_first_import() -> None:
    import recall_paths

    folder = guards.data_folder()
    assert folder.name.startswith(guards.TEMP_PREFIX)
    assert folder.is_relative_to(Path(tempfile.gettempdir()).resolve())
    assert recall_paths.DATA_DIR == folder
    assert list((folder / guards.HOME_LEAF).iterdir()) == []
