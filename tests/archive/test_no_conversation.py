"""A message that the source links to no chat is stored, and it belongs to no conversation."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from archive import Archive, import_chat_db, import_legacy_csv
from archive.compat import messages_frame
from tests.archive.builders import ChatDbBuilder, utc
from tests.archive.inspect_archive import fts_is_consistent, totals_match_messages

START = utc(2024, 9, 1, 16, 0)
DAY = "2024-09-01"
OTIS = "+15550100171"


@pytest.fixture
def world(builder: ChatDbBuilder) -> dict[str, int]:
    otis = builder.handle(OTIS)
    chat = builder.chat(OTIS, members=[otis])
    for index in range(3):
        builder.message(chat, when=START + timedelta(minutes=index), text=f"filed {index}", handle=otis)
    lost = builder.message(None, when=START + timedelta(minutes=10), text="orphan from Otis", handle=otis)
    mine = builder.message(None, when=START + timedelta(minutes=11), text="orphan from me", from_me=True)
    builder.commit()
    return {"chat": chat, "lost": lost, "mine": mine}


@pytest.fixture
def archive(world: dict[str, int], source_path: Path, archive_path: Path) -> Iterator[Archive]:
    import_chat_db(source_path, archive_path)
    with Archive.open(archive_path) as opened:
        yield opened


class TestStored:
    def test_the_rows_are_stored_and_counted_apart(
        self, world: dict[str, int], source_path: Path, archive_path: Path
    ) -> None:
        result = import_chat_db(source_path, archive_path)

        assert result.skipped == {}
        assert (result.messages_added, result.counts["without_conversation"]) == (5, 2)
        with Archive.open(archive_path) as archive:
            assert archive.summary()["messages"] == 3
            assert archive.summary()["messages_without_conversation"] == 2
        assert totals_match_messages(archive_path) and fts_is_consistent(archive_path)

    def test_a_second_pull_changes_nothing(self, world: dict[str, int], source_path: Path, archive_path: Path) -> None:
        import_chat_db(source_path, archive_path)
        again = import_chat_db(source_path, archive_path)
        assert (again.messages_added, again.messages_updated, again.messages_removed) == (0, 0, 0)

    def test_a_message_that_gets_a_chat_later_moves_into_the_conversation(
        self, world: dict[str, int], builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        import_chat_db(source_path, archive_path)
        builder.link(world["chat"], world["lost"])
        builder.commit()

        result = import_chat_db(source_path, archive_path)

        assert result.messages_updated == 1
        with Archive.open(archive_path) as archive:
            assert archive.summary()["messages"] == 4
            assert archive.summary()["messages_without_conversation"] == 1
            assert archive.conversation(OTIS)["message_count"] == 4  # type: ignore[index]
        assert totals_match_messages(archive_path)


class TestInNoConversation:
    def test_no_list_and_no_count_holds_them(self, archive: Archive) -> None:
        assert [(c["id"], c["message_count"]) for c in archive.conversations(include_empty=True)] == [(OTIS, 3)]
        assert archive.conversation("") is None
        assert archive.count() == 3
        assert archive.count(since=DAY, until=DAY) == 3
        assert archive.count(from_me=True) == 0
        assert [row["key"] for row in archive.count(group_by="conversation")] == [OTIS]  # type: ignore[union-attr]
        assert archive.count(since=DAY, group_by="conversation") == [
            {"key": OTIS, "total": 3, "sent": 0, "received": 3}
        ]
        assert sum(row["total"] for row in archive.count(group_by="day")) == 3  # type: ignore[union-attr]
        assert sum(row["total"] for row in archive.count(group_by="sender")) == 3  # type: ignore[union-attr]

    def test_no_read_by_conversation_gives_them(self, archive: Archive) -> None:
        assert archive.messages("")["messages"] == []
        assert list(archive.iter_messages("")) == []
        assert archive.stats("")["total_messages"] == 0
        assert archive.day("", DAY)["total"] == 0
        assert [m["text"] for m in archive.messages(OTIS)["messages"]] == ["filed 0", "filed 1", "filed 2"]
        assert archive.day(OTIS, DAY)["total"] == 3
        assert archive.stats(OTIS)["total_messages"] == 3

    def test_search_does_not_find_them(self, archive: Archive) -> None:
        assert archive.search("orphan") == {"total": 0, "hits": []}
        assert archive.count(text="orphan") == 0
        assert archive.search("filed")["total"] == 3


class TestOnExplicitRequest:
    def test_count_and_search_and_the_stream_give_only_them(self, archive: Archive) -> None:
        assert archive.count(no_conversation=True) == 2
        assert archive.count(no_conversation=True, from_me=True) == 1
        assert archive.count(no_conversation=True, group_by="conversation") == [
            {"key": "", "total": 2, "sent": 1, "received": 1}
        ]
        assert archive.count(no_conversation=True, group_by="day") == [
            {"key": DAY, "total": 2, "sent": 1, "received": 1}
        ]
        found = archive.search("orphan", no_conversation=True, order="oldest")
        assert found["total"] == 2
        assert [hit["message"]["conversation"] for hit in found["hits"]] == ["", ""]
        assert archive.search("filed", no_conversation=True)["total"] == 0
        stream = list(archive.iter_messages(None, no_conversation=True))
        assert [(m["text"], m["conversation"], m["sender"]) for m in stream] == [
            ("orphan from Otis", "", OTIS),
            ("orphan from me", "", ""),
        ]
        assert [m["text"] for m in archive.iter_messages(None, no_conversation=True, since="2024-09-02")] == []

    def test_an_id_gives_the_message(self, archive: Archive, builder: ChatDbBuilder, world: dict[str, int]) -> None:
        guid = builder.guid_of(world["lost"])
        found = archive.message(guid)
        assert found is not None and (found["conversation"], found["text"]) == ("", "orphan from Otis")
        assert archive.context(guid) == [found]  # it has no neighbours
        assert archive.message_by_source_rowid(world["lost"]) == found

    def test_requests_that_contradict_each_other_are_refused(self, archive: Archive) -> None:
        with pytest.raises(ValueError):
            archive.iter_messages(OTIS, no_conversation=True)
        with pytest.raises(ValueError):
            archive.iter_messages(None)
        with pytest.raises(ValueError):
            archive.search("orphan", conversations=[OTIS], no_conversation=True)
        with pytest.raises(ValueError):
            archive.count(conversations=OTIS, no_conversation=True)
        with pytest.raises(ValueError):
            archive.count(no_conversation="yes")  # type: ignore[arg-type]


class TestOtherDoors:
    def test_the_frame_has_them_with_no_chat_id(self, archive: Archive) -> None:
        frame = messages_frame(archive)
        assert len(frame) == 5
        assert int(frame["chat_id"].isna().sum()) == 2

    def test_the_bridge_stores_a_row_with_no_chat_id_the_same_way(self, tmp_path: Path, archive_path: Path) -> None:
        export = tmp_path / "messages.csv"
        with export.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "message_id",
                    "timestamp",
                    "sender",
                    "text",
                    "chat_id",
                    "chat_display_name",
                    "is_from_me",
                    "service",
                    "attachment_types",
                    "attachment_files",
                ]
            )
            writer.writerow([1, "2024-09-01 09:00:00", OTIS, "filed", OTIS, "", 0, "iMessage", "", ""])
            writer.writerow([2, "2024-09-01 09:01:00", OTIS, "orphan", "", "", 0, "iMessage", "", ""])

        result = import_legacy_csv(export, archive_path)

        assert result.skipped == {}
        assert (result.messages_added, result.counts["without_conversation"]) == (2, 1)
        with Archive.open(archive_path) as archive:
            assert archive.summary()["messages"] == 1
            assert archive.summary()["messages_without_conversation"] == 1
            assert archive.count(no_conversation=True) == 1
            assert [c["id"] for c in archive.conversations()] == [OTIS]
