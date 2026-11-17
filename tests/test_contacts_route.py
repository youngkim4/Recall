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
from recall_services import places
from setup_status import DbProbe
from tests.archive_kit import (
    PERSON,
    bridged,
    databases_opened,
    listed,
    no_names,  # noqa: F401
    not_an_archive,
    own_archives,  # noqa: F401
    pulled,
    write_database,
)
from tests.invented_db import PERSON as PERSON_IN_MESSAGES
from tests.invented_db import messages_db
from tests.loopback import fetch

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache", "no_names", "own_archives", "no_jobs")

NO_ARCHIVE = {"error": "There is no archive yet, and Messages cannot be read.", "code": "no_archive"}


def conversations(server, messages_path, more=""):
    status, _, body = fetch(server, f"/api/contacts?messagesPath={quote(str(messages_path))}{more}")
    return status, json.loads(body)


def raising(error):
    def reader(*_args, **_more):
        raise error

    return reader


def save_a_name(cache, handle):
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"contacts": [{"name": "Avery Stone", "phones": [handle], "emails": []}]}), encoding="utf-8")


# ---- with an archive ----


def test_with_an_archive_the_list_comes_from_the_archive(server, messages_home, tmp_path, monkeypatch):
    archive_file = pulled(tmp_path / "saves" / "archive.sqlite3", ["in the archive"])
    live = messages_db(messages_home / "chat.db", ["in messages", "and not in the archive"])
    opened = databases_opened(monkeypatch)

    status, answer = conversations(server, archive_file)

    assert status == 200
    assert (answer["source"], answer["messagesPath"]) == ("messages", str(archive_file))
    assert [(row["chat_id"], row["message_count"]) for row in answer["contacts"]] == [(PERSON, 1)]
    assert answer["archive"] == {"messages": 1, "conversations": 1, "firstMessage": "2024-07-01T09:00:00", "lastMessage": "2024-07-01T09:00:00"}
    assert "dbPath" not in answer
    # Messages itself was not opened for it
    assert not any(str(live) in address for address in opened)


def test_a_row_of_the_list_has_the_figures_of_its_conversation(server, tmp_path):
    archive_file = pulled(tmp_path / "saves" / "archive.sqlite3", ["one", "two", "three"])

    _, answer = conversations(server, archive_file)

    assert answer["contacts"] == [{
        "chat_id": PERSON,
        "display_name": "",
        "is_group": 0,
        "first_msg": "2024-07-01T09:00:00",
        "last_msg": "2024-07-01T09:02:00",
        "message_count": 3,
        "sent_count": 0,
        "received_count": 3,
    }]
    assert answer["contactNameCount"] == 0


def test_the_answer_names_the_archive_file_when_the_app_named_its_export(server, tmp_path):
    export = bridged(tmp_path / "earlier" / "messages.csv", ["one", "two"])

    status, answer = conversations(server, export)

    assert status == 200
    assert answer["messagesPath"] == str(export.with_suffix(".sqlite3"))
    assert [(row["chat_id"], row["message_count"]) for row in answer["contacts"]] == [(PERSON, 2)]


def test_a_database_that_the_client_names_is_not_read_beside_an_archive(server, messages_home, tmp_path, monkeypatch):
    archive_file = pulled(tmp_path / "saves" / "archive.sqlite3", ["in the archive"])
    copy = write_database(tmp_path / "copy.db", ["one", "two", "three"])
    opened = databases_opened(monkeypatch)

    status, answer = conversations(server, archive_file, f"&dbPath={quote(str(copy))}")

    assert status == 200
    assert [row["message_count"] for row in answer["contacts"]] == [1]
    assert not any(str(copy) in address for address in opened)


def test_a_file_that_is_no_archive_is_told_so_and_messages_is_not_read_in_its_place(server, messages_home, tmp_path, monkeypatch):
    wrong = not_an_archive(tmp_path / "saves" / "archive.sqlite3")
    messages_db(messages_home / "chat.db", ["one"])
    monkeypatch.setattr(ui_server, "list_contacts", raising(AssertionError("the database was read")))

    status, answer = conversations(server, wrong)

    assert status == 500
    assert answer == {"error": "The archive file could not be opened. Open the setup guide and update the archive.", "code": "archive_failed"}
    assert listed() == []


# ---- with no archive ----


def test_with_no_archive_the_list_comes_from_messages_itself(server, messages_home, tmp_path):
    messages_db(messages_home / "chat.db", ["one", "two"])

    status, answer = conversations(server, tmp_path / "saves" / "archive.sqlite3")

    assert status == 200
    assert answer["source"] == "database"
    assert [(row["chat_id"], row["message_count"]) for row in answer["contacts"]] == [(PERSON_IN_MESSAGES, 3)]
    assert "dbPath" not in answer and "archive" not in answer
    # the list made no archive, and no job
    assert not (tmp_path / "saves").exists()
    assert listed() == []


def test_a_database_that_the_client_names_is_not_read(server, messages_home, tmp_path):
    messages_db(messages_home / "chat.db", ["one"])
    copy = messages_db(tmp_path / "copy.db", ["one", "two", "three"], group_name="Named In A Copy")

    status, answer = conversations(server, tmp_path / "none.csv", f"&dbPath={quote(str(copy))}")

    assert status == 200
    assert [row["message_count"] for row in answer["contacts"]] == [2]


def test_with_no_archive_and_no_database_the_answer_says_so(server, messages_home, tmp_path):
    status, answer = conversations(server, tmp_path / "none.csv")

    assert (status, answer) == (400, NO_ARCHIVE)


@pytest.mark.parametrize("status", ["fda_blocked", "empty", "invalid", "error"])
def test_a_database_that_cannot_be_read_is_not_tried(server, messages_home, tmp_path, monkeypatch, status):
    monkeypatch.setattr(ui_server, "probe_database", lambda path=None, deep=False: DbProbe(status=status, kind="live", path=str(path)))
    monkeypatch.setattr(ui_server, "list_contacts", raising(AssertionError("the database was read")))

    answered, answer = conversations(server, tmp_path / "none.csv")

    assert (answered, answer) == (400, NO_ARCHIVE)


def test_a_database_that_the_check_finds_busy_is_tried(server, messages_home, tmp_path, monkeypatch):
    messages_db(messages_home / "chat.db", ["one"])
    monkeypatch.setattr(ui_server, "probe_database", lambda path=None, deep=False: DbProbe(status="locked", kind="live", path=str(path)))

    status, answer = conversations(server, tmp_path / "none.csv")

    assert (status, answer["source"]) == (200, "database")


@pytest.mark.parametrize("raised", [sqlite3.OperationalError("database is locked"), sqlite3.OperationalError("database is busy")], ids=str)
def test_a_busy_database_is_answered_with_a_plain_sentence(server, messages_home, tmp_path, monkeypatch, raised):
    messages_db(messages_home / "chat.db", ["one"])
    monkeypatch.setattr(ui_server, "list_contacts", raising(raised))

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


@pytest.mark.parametrize("raised, sentence", [
    (sqlite3.OperationalError("unable to open database file"), "Recall cannot read Messages. Give it Full Disk Access"),
    (sqlite3.OperationalError("disk I/O error in SELECT chat_id FROM message"), "The Messages database on this Mac could not be read."),
    (sqlite3.DatabaseError("database disk image is malformed"), "The Messages database on this Mac could not be read."),
], ids=lambda value: str(value) if isinstance(value, Exception) else None)
def test_what_the_reader_raised_is_not_in_the_answer(server, messages_home, tmp_path, monkeypatch, raised, sentence):
    messages_db(messages_home / "chat.db", ["one"])
    monkeypatch.setattr(ui_server, "list_contacts", raising(raised))

    status, answer = conversations(server, tmp_path / "none.csv")

    assert status == 500
    assert list(answer) == ["error"] and answer["error"].startswith(sentence)
    assert str(raised) not in answer["error"] and "SELECT" not in answer["error"]
    # there is no archive, and nothing is said of one
    assert "archive" not in answer["error"]


# ---- the limit, and the names ----


@pytest.mark.parametrize("source", ["archive", "database"])
def test_a_limit_that_is_no_number_is_refused(server, messages_home, tmp_path, source):
    archive_file = tmp_path / "saves" / "archive.sqlite3"
    if source == "archive":
        pulled(archive_file, ["one"])
    else:
        messages_db(messages_home / "chat.db", ["one"])

    status, answer = conversations(server, archive_file, "&limit=many")

    assert status == 400
    assert answer == {"error": "limit must be a whole number, got 'many'"}


def test_the_list_is_cut_at_the_limit_and_the_totals_are_of_the_whole_archive(server, messages_home, tmp_path):
    from tests.archive.builders import ChatDbBuilder
    from tests.recall_services.state_data import OTHER, START

    builder = ChatDbBuilder(tmp_path / "chat.db")
    for handle, count in ((PERSON, 2), (OTHER, 1)):
        person = builder.handle(handle)
        thread = builder.chat(handle, members=[person])
        for place in range(count):
            builder.message(thread, when=START, text=f"to {handle} {place}", handle=person)
    builder.close()
    archive_file = tmp_path / "saves" / "archive.sqlite3"
    assert places.state_for(archive_file).pull_from_messages(tmp_path / "chat.db", None).status == "done"

    status, answer = conversations(server, archive_file, "&limit=1")

    assert status == 200
    assert [(row["chat_id"], row["message_count"]) for row in answer["contacts"]] == [(PERSON, 2)]
    assert (answer["archive"]["messages"], answer["archive"]["conversations"]) == (3, 2)


@pytest.mark.parametrize("source", ["archive", "database"])
def test_a_saved_name_is_on_the_row(server, messages_home, tmp_path, no_names, source):  # noqa: F811
    archive_file = tmp_path / "saves" / "archive.sqlite3"
    if source == "archive":
        pulled(archive_file, ["one"])
        save_a_name(no_names, PERSON)
    else:
        messages_db(messages_home / "chat.db", ["one"])
        save_a_name(no_names, PERSON_IN_MESSAGES)

    status, answer = conversations(server, archive_file)

    assert status == 200
    assert [row["display_name"] for row in answer["contacts"]] == ["Avery Stone"]
    assert answer["contactNameCount"] == 1
