"""The guards of this folder hold: a dummy key, a temp data folder, no way out, one time zone."""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path

import pytest

from tests import guards


def test_the_key_is_a_dummy() -> None:
    assert os.environ["OPENAI_API_KEY"] == guards.DUMMY_KEY


def test_no_dotenv_file_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dotenv

    planted = tmp_path / ".env"
    planted.write_text("RECALL_PLANTED_VALUE=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert dotenv.load_dotenv() is False
    assert dotenv.load_dotenv(planted) is False
    assert "RECALL_PLANTED_VALUE" not in os.environ


def test_importing_the_model_client_reads_no_key() -> None:
    import openai_client  # noqa: F401  calls load_dotenv when it loads

    assert os.environ["OPENAI_API_KEY"] == guards.DUMMY_KEY


def test_no_client_setting_points_somewhere_else() -> None:
    for name in ("OPENAI_BASE_URL", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        assert name not in os.environ


def test_default_paths_are_in_a_temp_folder() -> None:
    import recall_paths
    from archive import default_archive_path

    folder = Path(os.environ["RECALL_DATA_DIR"])
    assert folder.name.startswith(guards.TEMP_PREFIX)
    assert recall_paths.DATA_DIR == folder
    assert recall_paths.SAVES_DIR == folder / "saves"
    assert recall_paths.DEFAULT_MESSAGES_CSV == folder / "messages.csv"
    assert recall_paths.DOTENV_PATH == folder / ".env"
    assert default_archive_path() == folder / "saves" / "archive.sqlite3"


def test_home_is_an_empty_temp_folder() -> None:
    home = Path(os.environ["HOME"])
    assert home == Path(os.environ["RECALL_DATA_DIR"]) / guards.HOME_LEAF
    assert Path("~/Library/Messages").expanduser() == home / "Library" / "Messages"
    assert list(home.iterdir()) == []


def test_modules_that_read_the_home_folder_point_into_the_temp_folder() -> None:
    import contact_names
    import setup_status

    home = Path(os.environ["HOME"])
    for folder in (contact_names.ADDRESS_BOOK_DIR, setup_status.LIVE_CHAT_DB):
        assert home in folder.parents
    assert setup_status.probe_database().status == "missing"


def test_no_socket_leaves_this_machine() -> None:
    assert guards.blocks_outbound()
    with pytest.raises(guards.NetworkBlocked):
        socket.getaddrinfo("api.openai.com", 443)
    with pytest.raises(guards.NetworkBlocked):
        socket.create_connection(("192.0.2.1", 9), timeout=1)


@pytest.mark.parametrize(
    "lookup",
    [
        lambda: socket.gethostbyname_ex("example.com"),
        lambda: socket.gethostbyaddr("192.0.2.1"),
        lambda: socket.getnameinfo(("192.0.2.1", 443), 0),
    ],
)
def test_no_name_is_asked_of_another_machine(lookup: object) -> None:
    with pytest.raises(guards.NetworkBlocked, match="blocked in tests"):
        lookup()  # type: ignore[operator]


def test_no_packet_is_sent_with_sendmsg() -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(guards.NetworkBlocked, match="blocked in tests"):
            probe.sendmsg([b"x"], [], 0, ("192.0.2.1", 9))
    finally:
        probe.close()


def test_the_names_of_this_machine_can_be_asked() -> None:
    assert socket.gethostbyname_ex("localhost")[0]


def test_loopback_stays_open() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        client = socket.create_connection(server.getsockname(), timeout=2)
        client.close()
    finally:
        server.close()


def test_the_time_zone_is_fixed() -> None:
    assert os.environ["TZ"] == guards.TEST_TIME_ZONE
    # a summer and a winter moment have two offsets: the zone has clock changes
    summer = time.localtime(1_720_000_000).tm_gmtoff
    winter = time.localtime(1_705_000_000).tm_gmtoff
    assert (summer, winter) == (-7 * 3600, -8 * 3600)


def test_time_zone_is_put_back() -> None:
    with guards.time_zone("UTC"):
        assert time.localtime(1_720_000_000).tm_gmtoff == 0
    assert os.environ["TZ"] == guards.TEST_TIME_ZONE
