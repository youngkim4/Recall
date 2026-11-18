"""The filters `kind` and `attachments`, a search with no words, and the groupings week, weekday and hour.

The Harbor history is in tests/archive/worlds.py: a phone thread (10 messages),
an email thread (2) and a group (7). One message (a9) holds an image and a video.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from archive import Archive, import_chat_db
from tests.archive import worlds
from tests.archive.builders import ChatDbBuilder
from tests.archive.conftest import Harbor

A, B, G = worlds.ROWAN_PHONE, worlds.ROWAN_EMAIL, worlds.HARBOR_GROUP_ID

# from BY_WEEKDAY_HOUR in test_queries_figures.py, added up by hand
BY_WEEKDAY = [("0", 4, 3, 1), ("2", 2, 1, 1), ("4", 3, 1, 2), ("5", 1, 1, 0), ("6", 9, 2, 7)]
BY_HOUR = [
    ("00", 3, 1, 2),
    ("01", 1, 0, 1),
    ("03", 2, 1, 1),
    ("07", 1, 1, 0),
    ("08", 1, 1, 0),
    ("09", 2, 1, 1),
    ("12", 4, 1, 3),
    ("18", 2, 1, 1),
    ("21", 1, 0, 1),
    ("23", 2, 1, 1),
]
# from BY_DAY: each key is the Monday that starts the week
BY_WEEK = [
    ("2023-12-25", 1, 0, 1),
    ("2024-01-01", 2, 1, 1),
    ("2024-02-12", 2, 1, 1),
    ("2024-03-04", 12, 4, 8),
    ("2024-03-11", 1, 1, 0),
    ("2024-03-25", 1, 1, 0),
]


def rows(result: Any) -> list[tuple[str, int, int, int]]:
    return [(row["key"], row["total"], row["sent"], row["received"]) for row in result]


class TestGroupings:
    @pytest.mark.parametrize(("group_by", "expected"), [("weekday", BY_WEEKDAY), ("hour", BY_HOUR), ("week", BY_WEEK)])
    def test_week_weekday_and_hour(self, harbor: Harbor, group_by: str, expected: list[Any]) -> None:
        assert rows(harbor.archive.count(group_by=group_by)) == expected

    def test_a_grouping_follows_the_filters(self, harbor: Harbor) -> None:
        assert rows(harbor.archive.count(group_by="weekday", conversations=[B])) == [("0", 1, 1, 0), ("6", 1, 0, 1)]
        assert rows(harbor.archive.count(group_by="hour", text="clocks")) == [("01", 1, 0, 1), ("23", 1, 1, 0)]


class TestKind:
    def test_direct_and_group(self, harbor: Harbor) -> None:
        assert harbor.archive.count(kind="direct") == 12
        assert harbor.archive.count(kind="group") == 7
        assert rows(harbor.archive.count(kind="direct", group_by="conversation")) == [(A, 10, 5, 5), (B, 2, 1, 1)]

    def test_a_kind_with_conversations(self, harbor: Harbor) -> None:
        assert harbor.archive.count(kind="group", conversations=[A, G]) == 7
        assert harbor.archive.count(kind="group", conversations=[A]) == 0

    def test_a_search_with_a_kind(self, harbor: Harbor) -> None:
        assert harbor.archive.search("happy", kind="group")["total"] == 3
        assert harbor.archive.search("happy", kind="direct")["total"] == 0

    def test_an_unknown_kind_is_refused(self, harbor: Harbor) -> None:
        with pytest.raises(ValueError):
            harbor.archive.count(kind="person")


class TestAttachments:
    def test_harbor(self, harbor: Harbor) -> None:
        found = {choice: harbor.archive.count(attachments=choice) for choice in ("any", "none", "image", "video")}
        assert found == {"any": 1, "none": 18, "image": 1, "video": 1}
        assert harbor.archive.count(attachments="audio") == 0

    def test_each_kind(self, odd_files: Archive) -> None:
        found = {
            choice: harbor_labels(odd_files, attachments=choice)
            for choice in ("any", "none", "image", "video", "audio", "other")
        }
        assert found == {
            "any": ["gif", "photo", "song", "pdf", "blank"],
            "none": ["words"],
            "image": ["gif", "photo"],
            "video": [],
            "audio": ["song"],
            "other": ["pdf", "blank"],
        }

    def test_an_unknown_choice_is_refused(self, harbor: Harbor) -> None:
        with pytest.raises(ValueError):
            harbor.archive.count(attachments="sticker")
        with pytest.raises(ValueError):
            harbor.archive.search("x", attachments="sticker")

    def test_with_a_grouping_and_a_direction(self, harbor: Harbor) -> None:
        assert rows(harbor.archive.count(attachments="image", group_by="day")) == [("2024-03-10", 1, 0, 1)]
        assert harbor.archive.count(attachments="any", from_me=True) == 0


class TestBrowse:
    def test_no_words_list_the_filtered_messages_newest_first(self, harbor: Harbor) -> None:
        found = harbor.archive.search("", browse=True, conversations=[B])
        assert found["total"] == 2
        assert harbor.labels([hit["message"] for hit in found["hits"]]) == ["b12", "b11"]
        assert all(hit["ranges"] == [] for hit in found["hits"])

    def test_oldest_first_and_pages(self, harbor: Harbor) -> None:
        found = harbor.archive.search("  ", browse=True, kind="group", order="oldest", limit=2, offset=1)
        assert found["total"] == 7
        assert harbor.labels([hit["message"] for hit in found["hits"]]) == ["g14", "g15"]

    def test_relevance_with_no_words_is_newest_first(self, harbor: Harbor) -> None:
        found = harbor.archive.search("", browse=True, attachments="any", since="2024-03-01", order="relevance")
        assert harbor.labels([hit["message"] for hit in found["hits"]]) == ["a9"]

    def test_the_whole_archive(self, harbor: Harbor) -> None:
        found = harbor.archive.search("", browse=True, limit=1)
        assert found["total"] == 19
        assert harbor.labels([hit["message"] for hit in found["hits"]]) == ["a10"]

    def test_without_browse_no_words_still_give_nothing(self, harbor: Harbor) -> None:
        assert harbor.archive.search("", kind="group") == {"total": 0, "hits": []}

    def test_words_with_browse_are_a_search(self, harbor: Harbor) -> None:
        assert harbor.archive.search("happy", browse=True)["total"] == 3

    def test_browse_is_a_flag(self, harbor: Harbor) -> None:
        with pytest.raises(ValueError):
            harbor.archive.search("", browse="yes")  # type: ignore[arg-type]


def harbor_labels(archive: Archive, **filters: Any) -> list[str]:
    found = archive.search("", browse=True, order="oldest", **filters)
    return [hit["message"]["text"] for hit in found["hits"]]


@pytest.fixture
def odd_files(tmp_path: Path) -> Any:
    """One conversation: one message of words and five with an attachment of each kind."""
    source = tmp_path / "source" / "chat.db"
    source.parent.mkdir()
    made = ChatDbBuilder(source)
    friend = made.handle(worlds.ROWAN_PHONE)
    chat = made.chat(worlds.ROWAN_PHONE, members=[friend])
    kinds = [("words", None), ("gif", "image/gif"), ("photo", "IMAGE/JPEG"), ("song", "audio/mpeg")]
    kinds += [("pdf", "application/pdf"), ("blank", "")]
    for minute, (label, mime) in enumerate(kinds):
        rowid = made.message(chat, when=worlds.la(2024, 5, 1, 9, minute), text=label, handle=friend)
        if label != "words":
            made.attachment(rowid, mime_type=mime or None, transfer_name=f"{label}.bin")
    made.close()
    path = tmp_path / "data" / "archive.sqlite3"
    import_chat_db(source, path)
    with Archive.open(path) as opened:
        yield opened


def test_a_snapshot_reads_one_state_of_the_archive(odd_files: Archive) -> None:
    import sqlite3

    with odd_files.snapshot():
        before = odd_files.count()
        writer = sqlite3.connect(odd_files.path)
        writer.execute("UPDATE messages SET kind = 'system' WHERE text = 'words'")
        writer.commit()
        writer.close()
        inside = odd_files.count(group_by="day")
    assert (before, inside[0]["total"]) == (6, 6)
    # the count of each day reads the messages: after the block it sees the change
    assert odd_files.count(group_by="day")[0]["total"] == 5
