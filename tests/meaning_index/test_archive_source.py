"""The archive adapter, on a real archive that is made in a temp folder from invented rows.

The rows go through the importer of the package `archive`, so the adapter is
checked against the shapes that the archive gives, not against a copy of them.
The time zone is America/Los_Angeles: the times below are UTC.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from archive import Archive, import_chat_db
from meaning_index.archive_source import ArchiveSource
from meaning_index.build import build_index
from meaning_index.errors import SourceError
from meaning_index.fake_embedder import FakeEmbedder
from meaning_index.search import MeaningIndex
from meaning_index.segments import cut_segments
from meaning_index.store import IndexStore
from tests import guards
from tests.archive.builders import ChatDbBuilder, utc
from tests.meaning_index.support import words

PERSON = "+15550100123"
GROUP = "chat100200300"
QUIET = "+15550100777"


@dataclass(frozen=True)
class World:
    archive: Archive
    guids: dict[str, str]


def write_world(builder: ChatDbBuilder) -> dict[str, str]:
    """Two conversations with text, one with a group event only, and one message in no conversation."""
    guids: dict[str, str] = {}

    def add(label: str, rowid: int) -> None:
        guids[label] = builder.guid_of(rowid)

    rowan, mira, quiet = builder.handle(PERSON), builder.handle("+15550100456"), builder.handle(QUIET)
    person = builder.chat(PERSON, members=[rowan])
    group = builder.chat(GROUP, display_name="Lake Weekend", members=[rowan, mira])
    silent = builder.chat(QUIET, members=[quiet])

    said = builder.message
    add("ferry", said(person, when=utc(2024, 1, 15, 18, 0), text="Are we still on for the ferry?", handle=rowan))
    add("tickets", said(person, when=utc(2024, 1, 15, 18, 1), text="Yes, I bought two tickets", from_me=True))
    photo = builder.message(person, when=utc(2024, 1, 15, 18, 2), handle=rowan)
    builder.attachment(photo, mime_type="image/heic")
    add("photo", photo)
    builder.tapback(person, guids["ferry"], 2000, when=utc(2024, 1, 15, 18, 3), from_me=True)
    add("unsent", said(person, when=utc(2024, 1, 15, 18, 5), handle=rowan, retracted=utc(2024, 1, 15, 18, 6)))
    add("spaces", said(person, when=utc(2024, 1, 15, 18, 7), text="  \n ", handle=rowan))
    # 23:30 and 00:10 local: one exchange over midnight
    add("late", said(person, when=utc(2024, 1, 16, 7, 30), text="A late thought about the canoe", handle=rowan))
    add("tomorrow", said(person, when=utc(2024, 1, 16, 8, 10), text="Tell me tomorrow", from_me=True))

    add("renamed", said(group, when=utc(2024, 3, 1, 17, 0), handle=rowan, item_type=2, group_title="Lake Weekend"))
    add("canoe", said(group, when=utc(2024, 3, 1, 17, 1), text="Who brings the canoe?", handle=mira))
    add("dinner", said(group, when=utc(2024, 3, 1, 17, 2), text="I will cook dinner", from_me=True))

    builder.message(silent, when=utc(2024, 4, 1, 12, 0), handle=quiet, item_type=1, group_title="left")
    builder.message(None, when=utc(2024, 5, 1, 12, 0), text="A text in no conversation", handle=quiet)
    return guids


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    guards.pin_data_folder(monkeypatch)
    with guards.time_zone():
        source = tmp_path / "source" / "chat.db"
        source.parent.mkdir()
        builder = ChatDbBuilder(source)
        guids = write_world(builder)
        builder.close()
        path = tmp_path / "data" / "archive.sqlite3"
        import_chat_db(source, path)
        with Archive.open(path) as archive:
            yield World(archive, guids)


def test_it_lists_the_conversations_that_hold_a_message(world: World) -> None:
    listed = list(ArchiveSource(world.archive).conversation_ids())
    assert sorted(listed) == sorted([PERSON, GROUP])
    assert listed == [conversation["id"] for conversation in world.archive.conversations()]


def test_a_message_has_the_four_fields_of_the_port(world: World) -> None:
    first = next(iter(ArchiveSource(world.archive).messages(PERSON)))
    assert first == {
        "id": world.guids["ferry"],
        "timestamp": "2024-01-15 10:00:00",
        "text": "Are we still on for the ferry?",
        "is_from_me": False,
    }
    assert type(first["is_from_me"]) is bool


def test_messages_come_oldest_first_without_system_rows_and_rows_with_no_text(world: World) -> None:
    labels = {guid: label for label, guid in world.guids.items()}
    person = [labels[message["id"]] for message in ArchiveSource(world.archive).messages(PERSON)]
    group = [labels[message["id"]] for message in ArchiveSource(world.archive).messages(GROUP)]
    assert person == ["ferry", "tickets", "late", "tomorrow"]
    assert group == ["canoe", "dinner"]
    stamps = [message["timestamp"] for message in ArchiveSource(world.archive).messages(PERSON)]
    assert stamps == sorted(stamps)


def test_the_archive_holds_the_rows_that_the_adapter_leaves_out(world: World) -> None:
    kinds = {world.archive.message(world.guids[label])["kind"] for label in ("photo", "unsent", "spaces", "renamed")}
    assert kinds == {"attachment", "text", "system"}


def test_an_unknown_conversation_has_no_message(world: World) -> None:
    assert list(ArchiveSource(world.archive).messages("+15550100999")) == []
    assert list(ArchiveSource(world.archive).messages(QUIET)) == []


def test_the_segmenter_takes_what_the_adapter_gives(world: World) -> None:
    source = ArchiveSource(world.archive)
    first, second = cut_segments(PERSON, source.messages(PERSON))
    assert first.text == "2024-01-15\nThem: Are we still on for the ferry?\nMe: Yes, I bought two tickets"
    assert second.text == "2024-01-15\nThem: A late thought about the canoe\nMe: Tell me tomorrow"
    assert (second.first_timestamp, second.last_timestamp) == ("2024-01-15 23:30:00", "2024-01-16 00:10:00")
    assert (first.first_message_id, first.last_message_id) == (world.guids["ferry"], world.guids["tickets"])
    assert PERSON not in first.text and "Lake Weekend" not in first.text


def test_a_build_on_the_archive_is_searched_by_meaning(world: World, folder: Path) -> None:
    result = build_index(ArchiveSource(world.archive), FakeEmbedder(), IndexStore(folder), count_tokens=words)
    assert (result.segments, result.embedded) == (3, 3)
    index = MeaningIndex.load(IndexStore(folder))
    assert index is not None
    (boat,) = FakeEmbedder().embed(["boat ticket"])
    hit = index.search(boat)[0]
    assert hit.segment.conversation_id == PERSON
    assert hit.segment.first_message_id == world.guids["ferry"]
    between = world.archive.messages(PERSON, around=hit.segment.first_message_id, limit=1)["messages"]
    assert between[0]["text"] == "Are we still on for the ferry?"
    (supper,) = FakeEmbedder().embed(["supper"])
    assert index.search(supper)[0].segment.conversation_id == GROUP


def test_a_filter_by_day_means_the_same_day_as_in_the_archive(world: World, folder: Path) -> None:
    build_index(ArchiveSource(world.archive), FakeEmbedder(), IndexStore(folder), count_tokens=words)
    index = MeaningIndex.load(IndexStore(folder))
    assert index is not None
    (probe,) = FakeEmbedder().embed(["canoe"])
    days_with_hits: dict[str, list[str]] = {PERSON: [], GROUP: []}
    for conversation in (PERSON, GROUP):
        counted = {row["key"]: row["total"] for row in world.archive.count(conversations=conversation, group_by="day")}
        for day in ("2024-01-14", "2024-01-15", "2024-01-16", "2024-01-17", "2024-03-01", "2024-03-02"):
            hits = index.search(probe, conversations=[conversation], since=day, until=day)
            # the index finds a segment on a day exactly when the archive counts a message on that day
            assert bool(hits) == (counted.get(day, 0) > 0)
            assert world.archive.count(conversations=conversation, since=day, until=day) == counted.get(day, 0)
            days_with_hits[conversation] += [day] if hits else []
    assert days_with_hits == {PERSON: ["2024-01-15", "2024-01-16"], GROUP: ["2024-03-01"]}
    midnight = index.search(probe, conversations=[PERSON], since="2024-01-16", until="2024-01-16")
    assert [hit.segment.last_message_id for hit in midnight] == [world.guids["tomorrow"]]
    assert world.archive.count(conversations=PERSON, since="2024-01-16", until="2024-01-16") == 1


def test_an_incremental_build_follows_a_pull_of_the_archive(world: World, folder: Path) -> None:
    store = IndexStore(folder)
    build_index(ArchiveSource(world.archive), FakeEmbedder(), store, count_tokens=words)
    again = FakeEmbedder()
    result = build_index(ArchiveSource(world.archive), again, store, count_tokens=words)
    assert (result.embedded, result.reused, result.removed) == (0, 3, 0)
    assert again.calls == []


class Rows:
    """Stands in for an archive that breaks its contract."""

    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def conversations(self) -> list[dict[str, Any]]:
        return [{"id": "chat-x"}]

    def iter_messages(self, conversation_id: str) -> Iterator[Any]:
        yield from self._rows


@pytest.mark.parametrize(
    "row",
    [
        {"id": "a", "text": "MARKED-TEXT-0042", "is_from_me": False, "kind": "text"},
        {"id": None, "timestamp": "2024-01-15 10:00:00", "text": "MARKED-TEXT-0042", "is_from_me": False},
        {"id": "a", "timestamp": "2024-01-15 10:00:00", "text": "MARKED-TEXT-0042", "is_from_me": "yes"},
        {"id": "a", "timestamp": "2024-01-15 10:00:00", "text": "MARKED-TEXT-0042"},
        "MARKED-TEXT-0042",
    ],
)
def test_a_row_that_breaks_the_contract_is_a_typed_error_with_no_text(row: Any) -> None:
    with pytest.raises(SourceError) as caught:
        list(ArchiveSource(Rows([row])).messages("chat-x"))
    assert "MARKED" not in str(caught.value)


def test_a_conversation_with_no_id_is_a_typed_error() -> None:
    class Odd(Rows):
        def conversations(self) -> list[dict[str, Any]]:
            return [{"name": "nothing"}]

    with pytest.raises(SourceError):
        ArchiveSource(Odd([])).conversation_ids()


def test_whole_numbers_for_the_direction_are_read_as_the_archive_means_them() -> None:
    rows = [
        {"id": "a", "timestamp": "2024-01-15 10:00:00", "text": "one", "is_from_me": 1, "kind": "text"},
        {"id": "b", "timestamp": "2024-01-15 10:01:00", "text": "two", "is_from_me": 0, "kind": "text"},
    ]
    assert [m["is_from_me"] for m in ArchiveSource(Rows(rows)).messages("chat-x")] == [True, False]
