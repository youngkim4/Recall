"""Whether macOS lets Recall read Messages: /api/permissions/fulldisk.

The answer comes from the same check that the guide uses, so a database that
macOS hides is "denied" and not "missing", and a busy one is not "denied".
"""
import json

import pytest

import ui_server
from setup_status import DbProbe
from tests.invented_db import messages_db
from tests.loopback import fetch


def access(server):
    status, _, body = fetch(server, "/api/permissions/fulldisk")
    return status, json.loads(body)


@pytest.mark.parametrize("found, said", [
    ("readable", "granted"),
    ("empty", "granted"),
    ("locked", "granted"),
    ("invalid", "granted"),
    ("error", "granted"),
    ("fda_blocked", "denied"),
    ("missing", "missing"),
])
def test_the_answer_follows_the_check_of_the_database(server, messages_home, monkeypatch, found, said):
    monkeypatch.setattr(ui_server, "probe_database", lambda path=None, deep=False: DbProbe(status=found, kind="live", path=str(path)))

    status, answer = access(server)

    assert (status, answer) == (200, {"status": said, "dbPath": str(messages_home / "chat.db")})


def test_a_database_that_is_there_is_read_through_the_check(server, messages_home):
    messages_db(messages_home / "chat.db", ["one"])

    assert access(server) == (200, {"status": "granted", "dbPath": str(messages_home / "chat.db")})


def test_no_database_is_missing(server, messages_home):
    assert access(server)[1]["status"] == "missing"
