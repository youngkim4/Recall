"""The list of conversations: /api/contacts.

With an archive the list comes from the archive. With none it comes from
Messages itself. The client names no database. A database that is busy, or
that fails, is answered with a plain sentence.
"""
import json
import sqlite3
from urllib.parse import quote

import pytest

import ui_server
from setup_status import DbProbe
from tests.archive_kit import messages_csv
from tests.invented_db import GROUP, PERSON, messages_db
from tests.loopback import fetch

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache", "no_saved_names")


def conversations(server, export, more=""):
    status, _, body = fetch(server, f"/api/contacts?messagesPath={quote(str(export))}{more}")
    return status, json.loads(body)


def raising(error):
    def reader(*_args, **_more):
        raise error

    return reader


def test_with_an_archive_the_list_comes_from_the_archive(server, messages_home, archive):
    messages_db(messages_home / "chat.db", ["in messages", "and not in the archive"])

    status, answer = conversations(server, messages_csv(archive))

    assert status == 200
    assert (answer["source"], answer["messagesPath"]) == ("messages", str(messages_csv(archive)))
    assert [(row["chat_id"], row["message_count"]) for row in answer["contacts"]] == [(PERSON, 1)]
    assert answer["archive"]["messages"] == 1
    assert "dbPath" not in answer


def test_with_no_archive_the_list_comes_from_messages_itself(server, messages_home, tmp_path):
    messages_db(messages_home / "chat.db", ["one", "two"])

    status, answer = conversations(server, tmp_path / "none.csv")

    assert status == 200
    assert answer["source"] == "database"
    assert [(row["chat_id"], row["message_count"]) for row in answer["contacts"]] == [(PERSON, 3)]
    assert "dbPath" not in answer and "archive" not in answer


def test_a_database_that_the_client_names_is_not_read(server, messages_home, tmp_path):
    messages_db(messages_home / "chat.db", ["one"])
    copy = messages_db(tmp_path / "copy.db", ["one", "two", "three"], group_name="Named In A Copy")

    status, answer = conversations(server, tmp_path / "none.csv", f"&dbPath={quote(str(copy))}")

    assert status == 200
    assert [row["message_count"] for row in answer["contacts"]] == [2]


def test_with_no_archive_and_no_database_the_answer_says_so(server, messages_home, tmp_path):
    status, answer = conversations(server, tmp_path / "none.csv")

    assert status == 400
    assert answer == {"error": f"There is no archive at {tmp_path / 'none.csv'}, and Messages cannot be read."}


@pytest.mark.parametrize("status", ["fda_blocked", "empty", "invalid", "error"])
def test_a_database_that_cannot_be_read_is_not_tried(server, messages_home, tmp_path, monkeypatch, status):
    monkeypatch.setattr(ui_server, "probe_database", lambda path=None, deep=False: DbProbe(status=status, kind="live", path=str(path)))
    monkeypatch.setattr(ui_server, "list_contacts", raising(AssertionError("the database was read")))

    answered, answer = conversations(server, tmp_path / "none.csv")

    assert answered == 400
    assert "Messages cannot be read" in answer["error"]


def test_a_database_that_the_check_finds_busy_is_tried(server, messages_home, tmp_path, monkeypatch):
    messages_db(messages_home / "chat.db", ["one"])
    monkeypatch.setattr(ui_server, "probe_database", lambda path=None, deep=False: DbProbe(status="locked", kind="live", path=str(path)))

    status, answer = conversations(server, tmp_path / "none.csv")

    assert (status, answer["source"]) == (200, "database")


def test_a_busy_database_is_answered_with_a_plain_sentence(server, messages_home, tmp_path, monkeypatch):
    messages_db(messages_home / "chat.db", ["one"])
    monkeypatch.setattr(ui_server, "list_contacts", raising(sqlite3.OperationalError("database is locked")))

    status, answer = conversations(server, tmp_path / "none.csv")

    assert status == 503
    assert answer == {"error": "The Messages database on this Mac is busy. Quit Messages or wait a moment, then try again."}


def test_a_database_that_fails_is_answered_with_a_plain_sentence(server, messages_home, tmp_path):
    # the check passes: the messages are there. The table of conversations is not.
    live = messages_db(messages_home / "chat.db", ["one"])
    conn = sqlite3.connect(str(live))
    conn.execute("DROP TABLE chat")
    conn.commit()
    conn.close()

    status, answer = conversations(server, tmp_path / "none.csv")

    assert status == 500
    assert answer == {"error": "The file that Messages keeps on this Mac is not a Messages database."}
    assert "SELECT" not in json.dumps(answer)


def test_a_group_with_no_name_in_the_archive_is_named_from_messages(server, messages_home, tmp_path):
    messages_db(messages_home / "chat.db", ["anything"], group_name="Project Crew")
    export = tmp_path / "group.csv"
    export.write_text(f"timestamp,chat_id,text,is_from_me\n2025-01-01,{GROUP},who is driving,1\n", encoding="utf-8")

    status, answer = conversations(server, export)

    assert status == 200
    assert {row["chat_id"]: row["display_name"] for row in answer["contacts"]} == {GROUP: "Project Crew"}
