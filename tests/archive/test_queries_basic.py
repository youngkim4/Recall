"""Archive.open, summary, conversations, people, reactions and thread safety."""

from __future__ import annotations

import gc
import json
import threading
from pathlib import Path

import pytest

from archive import Archive, ArchiveError, ArchiveNotFoundError, import_chat_db
from tests.archive import worlds
from tests.archive.builders import ChatDbBuilder, utc
from tests.archive.conftest import Harbor

A, B, G = worlds.ROWAN_PHONE, worlds.ROWAN_EMAIL, worlds.HARBOR_GROUP_ID
CONVERSATION_KEYS = [
    "id", "is_group", "name", "message_count", "sent_count", "received_count",
    "first_timestamp", "last_timestamp", "members", "person_id",
]  # fmt: skip


class TestOpen:
    def test_missing_archive_is_an_error_for_readers(self, archive_path: Path) -> None:
        with pytest.raises(ArchiveNotFoundError):
            Archive.open(archive_path)
        with pytest.raises(FileNotFoundError):
            Archive.open(archive_path, readonly=True)
        assert not archive_path.exists()

    def test_writable_open_creates_an_empty_archive(self, archive_path: Path) -> None:
        with Archive.open(archive_path, readonly=False) as archive:
            assert archive.summary() == {
                "messages": 0,
                "conversations": 0,
                "first_timestamp": None,
                "last_timestamp": None,
                "messages_without_conversation": 0,
            }
            assert archive.conversations() == []
            assert archive.people() == []
        with Archive.open(archive_path) as archive:
            assert archive.path == archive_path

    def test_default_path_is_under_the_data_folder(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import recall_paths
        from archive import default_archive_path

        assert default_archive_path() == recall_paths.SAVES_DIR / "archive.sqlite3"
        monkeypatch.setattr(recall_paths, "SAVES_DIR", tmp_path / "saves")
        with Archive.open(None, readonly=False) as archive:
            assert archive.path == tmp_path / "saves" / "archive.sqlite3"

    def test_closed_archive_refuses_queries(self, harbor_files: tuple[Path, dict[str, str]]) -> None:
        archive = Archive.open(harbor_files[0])
        archive.summary()
        archive.close()
        archive.close()
        with pytest.raises(ArchiveError, match="closed"):
            archive.summary()

    def test_archive_cannot_be_written_through_its_connections(self, harbor: Harbor) -> None:
        import sqlite3

        with pytest.raises(sqlite3.OperationalError):
            harbor.archive._slot().conn.execute("DELETE FROM messages")
        assert harbor.archive.summary()["messages"] == 19


class TestSummary:
    def test_figures(self, harbor: Harbor) -> None:
        assert harbor.archive.summary() == {
            "messages": 19,
            "conversations": 3,
            "first_timestamp": "2023-12-31 23:59:59",
            "last_timestamp": "2024-03-25 08:00:00",
            "messages_without_conversation": 0,
        }


class TestConversations:
    def test_order_by_count_is_the_default(self, harbor: Harbor) -> None:
        rows = harbor.archive.conversations()
        assert [row["id"] for row in rows] == [A, G, B]
        assert [row["message_count"] for row in rows] == [10, 7, 2]

    def test_order_by_recent(self, harbor: Harbor) -> None:
        assert [row["id"] for row in harbor.archive.conversations(order="recent")] == [A, B, G]

    def test_shape_of_a_one_to_one_conversation(self, harbor: Harbor) -> None:
        row = harbor.archive.conversation(A)
        assert row is not None and list(row) == CONVERSATION_KEYS
        person = row.pop("person_id")
        assert row == {
            "id": A,
            "is_group": False,
            "name": "",
            "message_count": 10,
            "sent_count": 5,
            "received_count": 5,
            "first_timestamp": "2024-03-08 09:00:00",
            "last_timestamp": "2024-03-25 08:00:00",
            "members": [A],
        }
        assert isinstance(person, int)

    def test_shape_of_a_group(self, harbor: Harbor) -> None:
        row = harbor.archive.conversation(G)
        assert row == {
            "id": G,
            "is_group": True,
            "name": worlds.HARBOR_GROUP_TITLE,
            "message_count": 7,
            "sent_count": 2,
            "received_count": 5,
            "first_timestamp": "2023-12-31 23:59:59",
            "last_timestamp": "2024-03-10 03:00:00",
            "members": sorted([worlds.ROWAN_PHONE, worlds.MIRA_PHONE, worlds.TESSA_EMAIL]),
            "person_id": None,
        }

    def test_phone_and_email_threads_share_one_person(self, harbor: Harbor) -> None:
        phone, email = harbor.archive.conversation(A), harbor.archive.conversation(B)
        assert phone is not None and email is not None
        assert phone["person_id"] == email["person_id"] is not None

    def test_limit_and_offset(self, harbor: Harbor) -> None:
        assert [row["id"] for row in harbor.archive.conversations(limit=2)] == [A, G]
        assert [row["id"] for row in harbor.archive.conversations(limit=2, offset=2)] == [B]
        assert harbor.archive.conversations(limit=0) == []
        assert harbor.archive.conversations(offset=10) == []

    def test_conversation_without_messages_is_listed_only_on_request(self, harbor: Harbor) -> None:
        listed = [row["id"] for row in harbor.archive.conversations(include_empty=True)]
        assert listed == [A, G, B, worlds.HARBOR_EMPTY_ID]
        empty = harbor.archive.conversation(worlds.HARBOR_EMPTY_ID)
        assert empty is not None
        assert (empty["message_count"], empty["first_timestamp"], empty["last_timestamp"]) == (0, None, None)

    def test_unknown_conversation_is_none(self, harbor: Harbor) -> None:
        assert harbor.archive.conversation("+15550109999") is None

    @pytest.mark.parametrize("bad", [{"order": "name"}, {"limit": -1}, {"offset": -1}, {"limit": "5"}])
    def test_bad_arguments_raise_value_error(self, harbor: Harbor, bad: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            harbor.archive.conversations(**bad)  # type: ignore[arg-type]

    def test_everything_is_json(self, harbor: Harbor) -> None:
        json.dumps(harbor.archive.conversations(include_empty=True))
        json.dumps(harbor.archive.people())
        json.dumps(harbor.archive.summary())


class TestPeople:
    def test_people_are_named_people_first_then_by_id(self, harbor: Harbor) -> None:
        people = harbor.archive.people()
        assert [person["name"] for person in people] == ["Mira Okafor", "Rowan Vale", "", ""]
        assert all(list(person) == ["id", "name", "handles", "conversations"] for person in people)

    def test_one_person_has_both_handles_and_both_threads(self, harbor: Harbor) -> None:
        rowan = next(person for person in harbor.archive.people() if person["name"] == "Rowan Vale")
        assert rowan["handles"] == sorted([worlds.ROWAN_PHONE, worlds.ROWAN_EMAIL])
        assert rowan["conversations"] == [A, B]
        assert harbor.archive.person(rowan["id"]) == rowan

    def test_group_only_member_has_no_conversation_of_their_own(self, harbor: Harbor) -> None:
        mira = next(person for person in harbor.archive.people() if person["name"] == "Mira Okafor")
        assert mira["handles"] == [worlds.MIRA_PHONE]
        assert mira["conversations"] == []

    def test_unknown_person_is_none(self, harbor: Harbor) -> None:
        assert harbor.archive.person(987_654) is None


class TestReactions:
    def test_current_reactions_of_a_message(self, harbor: Harbor) -> None:
        found = harbor.archive.reactions([harbor.id("g15")])
        assert found == {
            harbor.id("g15"): [
                {"kind": "laugh", "is_from_me": False, "sender": worlds.MIRA_PHONE},
                {"kind": "emoji", "is_from_me": False, "sender": worlds.TESSA_EMAIL, "emoji": "🔥"},
                {"kind": "love", "is_from_me": True, "sender": ""},
            ]
        }

    def test_removed_tapback_is_not_listed(self, harbor: Harbor) -> None:
        found = harbor.archive.reactions([harbor.id("a1"), harbor.id("a2")])
        assert found[harbor.id("a1")] == []
        assert found[harbor.id("a2")] == [{"kind": "love", "is_from_me": False, "sender": worlds.ROWAN_PHONE}]

    def test_every_requested_id_is_a_key(self, harbor: Harbor) -> None:
        found = harbor.archive.reactions([harbor.id("a3"), "no-such-guid", harbor.id("a3")])
        assert found == {harbor.id("a3"): [], "no-such-guid": []}
        assert harbor.archive.reactions([]) == {}

    def test_one_id_as_a_string(self, harbor: Harbor) -> None:
        assert list(harbor.archive.reactions(harbor.id("a2"))) == [harbor.id("a2")]

    def test_many_ids(self, harbor: Harbor) -> None:
        ids = [f"missing-{index}" for index in range(1500)] + [harbor.id("a2")]
        found = harbor.archive.reactions(ids)
        assert len(found) == 1501 and len(found[harbor.id("a2")]) == 1


class TestThreads:
    def test_queries_from_many_threads_agree(self, harbor: Harbor) -> None:
        expected = (
            harbor.archive.summary(),
            harbor.archive.search("happy")["total"],
            harbor.archive.count(group_by="day"),
        )
        results: list[object] = []
        errors: list[BaseException] = []

        def work() -> None:
            try:
                for _ in range(20):
                    results.append(
                        (
                            harbor.archive.summary(),
                            harbor.archive.search("happy")["total"],
                            harbor.archive.count(group_by="day"),
                        )
                    )
                    list(harbor.archive.iter_messages(A))
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        assert len(results) == 160 and all(result == expected for result in results)

    def test_each_thread_gets_its_own_connection_and_gives_it_back(self, harbor: Harbor) -> None:
        harbor.archive.summary()
        seen: list[int] = []

        def work() -> None:
            harbor.archive.summary()
            seen.append(id(harbor.archive._slot().conn))

        for _ in range(12):
            thread = threading.Thread(target=work)
            thread.start()
            thread.join()
        gc.collect()
        assert id(harbor.archive._slot().conn) not in seen
        assert harbor.archive.open_connections() == 1

    def test_close_from_another_thread(self, harbor_files: tuple[Path, dict[str, str]]) -> None:
        archive = Archive.open(harbor_files[0])
        archive.summary()
        thread = threading.Thread(target=archive.close)
        thread.start()
        thread.join()
        assert archive.open_connections() == 0

    def test_reader_sees_a_new_import(self, builder: ChatDbBuilder, source_path: Path, archive_path: Path) -> None:
        rowan = builder.handle(worlds.ROWAN_PHONE)
        chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
        builder.message(chat, when=utc(2024, 1, 1, 12), text="first", handle=rowan)
        builder.commit()
        import_chat_db(source_path, archive_path)
        with Archive.open(archive_path) as archive:
            assert archive.summary()["messages"] == 1
            builder.message(chat, when=utc(2024, 1, 2, 12), text="second", handle=rowan)
            builder.commit()
            import_chat_db(source_path, archive_path)
            assert archive.summary()["messages"] == 2
            assert [message["text"] for message in archive.iter_messages(worlds.ROWAN_PHONE)] == ["first", "second"]
