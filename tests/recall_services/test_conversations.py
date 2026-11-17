"""conversations.py: the rows of /api/contacts and the totals of the archive."""

from __future__ import annotations

import json

import pytest

from archive import Archive
from recall_services import conversations
from recall_services.names import Names
from tests.recall_services import world

# Contact in app/src/types.ts at 89ab504; the server sends display_name, never displayName
ROW_KEYS = {
    "chat_id",
    "display_name",
    "is_group",
    "first_msg",
    "last_msg",
    "message_count",
    "sent_count",
    "received_count",
}
# ArchiveSummary in app/src/types.ts
SUMMARY_KEYS = {"messages", "conversations", "firstMessage", "lastMessage"}
# ContactsResponse in app/src/types.ts
RESPONSE_KEYS = {"contacts", "source", "messagesPath", "contactNameCount", "archive"}
INES_MESSAGES = 4 + world.LONG_DAY_MESSAGES + 5
ALL_MESSAGES = INES_MESSAGES + 2 + 3 + 5 + 10


def test_a_row_has_the_keys_the_app_reads(archive: Archive, saved: Names) -> None:
    rows = conversations.rows(archive, saved)
    assert all(set(row) == ROW_KEYS for row in rows)
    json.dumps(rows)


def test_the_rows_are_the_conversations_with_the_most_messages_first(archive: Archive, saved: Names) -> None:
    rows = conversations.rows(archive, saved)
    assert [(row["chat_id"], row["message_count"]) for row in rows] == [
        (world.INES_PHONE, INES_MESSAGES),
        (world.UNTITLED, 10),
        (world.GARDEN, 5),
        (world.UNSAVED, 3),
        (world.INES_EMAIL, 2),
    ]


def test_one_row(archive: Archive, saved: Names) -> None:
    by_id = {row["chat_id"]: row for row in conversations.rows(archive, saved)}
    assert by_id[world.GARDEN] == {
        "chat_id": world.GARDEN,
        "display_name": world.GARDEN_TITLE,
        "is_group": 1,
        # written as the server writes a time in this list: with a T
        "first_msg": "2024-04-01T10:00:00",
        "last_msg": "2024-04-02T10:00:00",
        "message_count": 5,
        "sent_count": 1,
        "received_count": 4,
    }
    assert by_id[world.INES_PHONE]["is_group"] == 0


def test_the_names_of_the_rows(archive: Archive, saved: Names) -> None:
    names = {row["chat_id"]: row["display_name"] for row in conversations.rows(archive, saved)}
    assert names == {
        world.INES_PHONE: "Ines Calder",
        world.INES_EMAIL: "Ines Calder",
        world.GARDEN: world.GARDEN_TITLE,
        world.UNTITLED: world.UNTITLED_NAME,
        # the interface labels a conversation that has no name
        world.UNSAVED: "",
    }


def test_sent_and_received_are_counted_each_by_itself(archive: Archive, saved: Names) -> None:
    for row in conversations.rows(archive, saved):
        assert row["sent_count"] + row["received_count"] == row["message_count"]


@pytest.mark.parametrize(("limit", "held"), [(2, 2), ("3", 3), (0, 0), (-4, 0), (None, 5), ("", 5), (400, 5)])
def test_the_list_is_cut_at_the_limit(archive: Archive, saved: Names, limit: object, held: int) -> None:
    assert len(conversations.rows(archive, saved, limit)) == held  # type: ignore[arg-type]


def test_a_limit_that_is_no_number_is_refused(archive: Archive, saved: Names) -> None:
    with pytest.raises(ValueError):
        conversations.rows(archive, saved, "many")  # type: ignore[arg-type]


def test_the_totals_never_come_from_adding_up_the_list(archive: Archive) -> None:
    found = conversations.summary(archive)
    assert set(found) == SUMMARY_KEYS
    assert found == {
        "messages": ALL_MESSAGES,
        "conversations": 5,
        "firstMessage": "2024-03-01T09:00:00",
        "lastMessage": "2024-05-01T11:05:00",
    }


def test_a_message_with_no_conversation_is_in_no_total(archive: Archive) -> None:
    assert archive.summary()["messages_without_conversation"] == 1
    assert conversations.summary(archive)["messages"] == ALL_MESSAGES


def test_how_many_rows_have_a_name(archive: Archive, saved: Names) -> None:
    assert conversations.name_count(conversations.rows(archive, saved)) == 4
    assert conversations.name_count(conversations.rows(archive, Names())) == 1  # only the group with a title
    assert conversations.name_count([]) == 0


def test_the_whole_answer_of_the_route(archive: Archive, saved: Names) -> None:
    found = conversations.payload(archive, saved, limit=2, messages_path="/data/messages.csv")
    assert set(found) == RESPONSE_KEYS
    assert found["source"] == "messages"
    assert found["messagesPath"] == "/data/messages.csv"
    assert len(found["contacts"]) == 2
    assert found["contactNameCount"] == 2
    assert found["archive"] == conversations.summary(archive)
    json.dumps(found)


def test_an_archive_with_no_message(tmp_path_factory: pytest.TempPathFactory) -> None:
    path = tmp_path_factory.mktemp("empty") / "archive.sqlite3"
    Archive.open(path, readonly=False).close()
    with Archive.open(path) as empty:
        assert conversations.rows(empty, Names()) == []
        assert conversations.summary(empty) == {
            "messages": 0,
            "conversations": 0,
            "firstMessage": None,
            "lastMessage": None,
        }
