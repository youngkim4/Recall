"""Additions to the query API: one day, the oldest page, the source ROWID, and the sender buckets."""

from __future__ import annotations

import gc
import threading
import weakref
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from archive import Archive, ArchiveError, ArchiveNotFoundError, import_chat_db
from tests.archive.builders import ChatDbBuilder
from tests.archive.worlds import la

MIRA = "+15550100141"
GROUP = "chat800100"
# local days (America/Los_Angeles) and how many messages each holds
DAYS = {"2024-06-01": 3, "2024-06-03": 7, "2024-06-10": 2}


@pytest.fixture
def archive(builder: ChatDbBuilder, source_path: Path, archive_path: Path) -> Iterator[Archive]:
    mira = builder.handle(MIRA)
    chat = builder.chat(MIRA, members=[mira])
    group = builder.chat(GROUP, display_name="Tide watchers", members=[mira])
    for day, count in DAYS.items():
        start = datetime.fromisoformat(day)
        for index in range(count):
            when = la(start.year, start.month, start.day, 8) + timedelta(hours=index)
            builder.message(chat, when=when, text=f"{day} #{index}", handle=mira, from_me=index % 2 == 1)
    builder.message(chat, when=la(2024, 6, 3, 23, 59, 59), text="last second of the third", handle=mira)
    builder.message(chat, when=la(2024, 6, 4, 0, 0, 0), text="first second of the fourth", handle=mira)
    builder.message(group, when=la(2024, 6, 3, 12), text="group note from me", from_me=True)
    builder.message(group, when=la(2024, 6, 3, 13), text="group note with no sender", handle=0)
    builder.message(group, when=la(2024, 6, 3, 14), text="group note from Mira", handle=mira)
    builder.commit()
    import_chat_db(source_path, archive_path)
    with Archive.open(archive_path) as opened:
        yield opened


class TestDay:
    def test_one_day_in_order_with_its_neighbours(self, archive: Archive) -> None:
        found = archive.day(MIRA, "2024-06-03")

        assert (found["day"], found["total"], found["offset"]) == ("2024-06-03", 8, 0)
        assert [m["text"] for m in found["messages"]] == [
            *(f"2024-06-03 #{index}" for index in range(7)),
            "last second of the third",
        ]
        assert (found["previous_day"], found["next_day"]) == ("2024-06-01", "2024-06-04")

    def test_the_first_and_the_last_day_have_one_neighbour(self, archive: Archive) -> None:
        first, last = archive.day(MIRA, "2024-06-01"), archive.day(MIRA, "2024-06-10")
        assert (first["previous_day"], first["next_day"]) == (None, "2024-06-03")
        assert (last["previous_day"], last["next_day"]) == ("2024-06-04", None)

    def test_a_day_with_no_messages_still_names_its_neighbours(self, archive: Archive) -> None:
        found = archive.day(MIRA, "2024-06-07")
        assert (found["total"], found["messages"]) == (0, [])
        assert (found["previous_day"], found["next_day"]) == ("2024-06-04", "2024-06-10")

    def test_a_long_day_is_given_in_part(self, archive: Archive) -> None:
        part = archive.day(MIRA, "2024-06-03", limit=3, offset=2)
        assert (part["total"], part["offset"]) == (8, 2)
        assert [m["text"] for m in part["messages"]] == [f"2024-06-03 #{index}" for index in (2, 3, 4)]

    def test_a_named_place_is_used_as_it_is(self, archive: Archive) -> None:
        last_rows = archive.day(MIRA, "2024-06-03", limit=3, offset=7)
        assert (last_rows["offset"], len(last_rows["messages"])) == (7, 1)  # fewer rows than the limit
        past_the_end = archive.day(MIRA, "2024-06-03", limit=3, offset=50)
        assert (past_the_end["offset"], past_the_end["messages"], past_the_end["total"]) == (8, [], 8)

    def test_a_window_around_a_time_never_runs_past_the_end(self, archive: Archive) -> None:
        part = archive.day(MIRA, "2024-06-03", limit=3, around="2024-06-03 23:59:59")
        assert part["offset"] == 5
        assert len(part["messages"]) == 3

    def test_a_limit_above_the_most_is_the_most(self, archive: Archive) -> None:
        assert len(archive.day(MIRA, "2024-06-03", limit=10_000)["messages"]) == 8

    def test_around_puts_that_time_in_the_middle(self, archive: Archive) -> None:
        part = archive.day(MIRA, "2024-06-03", limit=3, around="2024-06-03 12:00:00")  # the time of "#4"
        assert part["offset"] == 3
        assert [m["text"] for m in part["messages"]] == [f"2024-06-03 #{index}" for index in (3, 4, 5)]

    def test_around_that_is_not_a_time_starts_at_the_beginning(self, archive: Archive) -> None:
        assert archive.day(MIRA, "2024-06-03", limit=3, around="soon")["offset"] == 0

    def test_the_count_agrees_with_count(self, archive: Archive) -> None:
        for day in ("2024-06-01", "2024-06-03", "2024-06-04", "2024-06-07"):
            assert archive.day(MIRA, day)["total"] == archive.count(conversations=MIRA, since=day, until=day)

    def test_an_unknown_conversation_gives_an_empty_day(self, archive: Archive) -> None:
        found = archive.day("+1555010014", "2024-06-03")  # a part of an id is not the id
        assert found == {
            "day": "2024-06-03",
            "total": 0,
            "offset": 0,
            "messages": [],
            "previous_day": None,
            "next_day": None,
        }

    def test_several_conversations_are_one_day(self, archive: Archive) -> None:
        found = archive.day([MIRA, GROUP], "2024-06-03")
        assert found["total"] == 11
        stamps = [m["timestamp"] for m in found["messages"]]
        assert stamps == sorted(stamps)

    @pytest.mark.parametrize("day", ["2024-6-3", "June 3", "2024-02-30", ""])
    def test_text_that_is_not_a_day_is_refused(self, archive: Archive, day: str) -> None:
        with pytest.raises(ValueError):
            archive.day(MIRA, day)

    @pytest.mark.parametrize("bad", [{"limit": 0}, {"offset": -1}, {"limit": "3"}, {"include_system": 1}])
    def test_bad_arguments_are_refused(self, archive: Archive, bad: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            archive.day(MIRA, "2024-06-03", **bad)  # type: ignore[arg-type]


class TestOldestPage:
    def test_the_oldest_page_is_the_start_of_the_conversation(self, archive: Archive) -> None:
        page = archive.messages(MIRA, oldest=True, limit=2)
        assert [m["text"] for m in page["messages"]] == ["2024-06-01 #0", "2024-06-01 #1"]
        assert (page["has_older"], page["has_newer"]) == (False, True)

    def test_the_default_page_is_still_the_newest(self, archive: Archive) -> None:
        page = archive.messages(MIRA, limit=2)
        assert [m["text"] for m in page["messages"]] == ["2024-06-10 #0", "2024-06-10 #1"]

    def test_oldest_cannot_be_given_with_an_anchor(self, archive: Archive) -> None:
        anchor = archive.messages(MIRA, limit=1)["messages"][0]["id"]
        with pytest.raises(ValueError):
            archive.messages(MIRA, oldest=True, after=anchor)


class TestSourceRowid:
    def test_a_rowid_gives_its_message(self, archive: Archive, builder: ChatDbBuilder) -> None:
        found = archive.message_by_source_rowid(1)
        assert found is not None
        assert (found["id"], found["text"]) == (builder.guid_of(1), "2024-06-01 #0")

    def test_an_unknown_rowid_gives_none(self, archive: Archive) -> None:
        assert archive.message_by_source_rowid(987_654) is None

    @pytest.mark.parametrize("bad", [-1, "1", 1.0, True, 2**63])
    def test_a_bad_rowid_is_refused(self, archive: Archive, bad: object) -> None:
        with pytest.raises(ValueError):
            archive.message_by_source_rowid(bad)  # type: ignore[arg-type]


class TestSenderBuckets:
    def test_a_received_message_with_no_sender_is_not_the_owner(self, archive: Archive) -> None:
        buckets = {row["key"]: row for row in archive.count(conversations=GROUP, group_by="sender")}  # type: ignore[union-attr]
        assert buckets == {
            "": {"key": "", "total": 1, "sent": 1, "received": 0},
            "unknown": {"key": "unknown", "total": 1, "sent": 0, "received": 1},
            MIRA: {"key": MIRA, "total": 1, "sent": 0, "received": 1},
        }


class TestArgumentChecks:
    def test_numbers_too_large_for_the_database_are_refused(self, archive: Archive) -> None:
        with pytest.raises(ValueError):
            archive.conversations(limit=2**63)
        with pytest.raises(ValueError):
            archive.search("note", offset=2**63)
        with pytest.raises(ValueError):
            archive.person(2**70)
        with pytest.raises(ValueError):
            archive.context("x", radius=2**64)

    def test_an_id_that_is_not_text_is_refused(self, archive: Archive) -> None:
        with pytest.raises(TypeError):
            archive.message(["x"])  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            archive.context(5)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            archive.messages(MIRA, before=7)  # type: ignore[arg-type]

    @pytest.mark.parametrize("day", ["0001-01-01", "1969-12-31", "9999-12-31", "2300-01-01"])
    def test_a_far_day_is_a_bound_and_not_an_error(self, archive: Archive, day: str) -> None:
        total = archive.count()
        assert archive.count(since=day) + archive.count(until=day) == total  # all of it is on one side
        assert archive.count(since="0001-01-01", until="9999-12-31") == total
        assert archive.search("note", until=day)["total"] in (0, 3)
        assert len(list(archive.iter_messages(MIRA, since=day))) in (0, archive.count(conversations=MIRA))

    @pytest.mark.parametrize("flag", [1, "yes", None])
    def test_a_flag_must_be_a_boolean(self, archive: Archive, flag: object) -> None:
        with pytest.raises(ValueError):
            archive.count(include_system=flag)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            archive.conversations(include_empty=flag)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            archive.messages(MIRA, oldest=flag)  # type: ignore[arg-type]


class TestLifetime:
    def test_a_thread_that_queried_does_not_keep_the_archive_alive(self, archive: Archive) -> None:
        other = Archive.open(archive.path)
        watcher = weakref.ref(other)
        holder = [other]
        ready, release = threading.Event(), threading.Event()

        def work() -> None:
            opened = holder.pop()  # the thread must not be what keeps the archive alive
            opened.summary()
            del opened
            ready.set()
            release.wait(10)

        thread = threading.Thread(target=work)
        thread.start()
        assert ready.wait(10)
        del other
        gc.collect()
        try:
            assert watcher() is None  # the connection of the live thread does not point back at the archive
        finally:
            release.set()
            thread.join()

    def test_an_empty_file_is_not_an_archive_and_stays_empty(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.sqlite3"
        empty.touch()

        with pytest.raises(ArchiveNotFoundError):
            Archive.open(empty)

        assert empty.stat().st_size == 0

    def test_the_path_is_fixed_when_the_archive_is_opened(
        self, archive: Archive, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(archive.path.parent)
        relative = Archive.open(archive.path.name)
        monkeypatch.chdir(tmp_path)
        try:
            assert relative.path.is_absolute()
            assert relative.summary()["messages"] == archive.summary()["messages"]
        finally:
            relative.close()


class TestCloseWhileOthersQuery:
    def test_close_waits_for_a_running_query_and_later_queries_fail_cleanly(self, archive: Archive) -> None:
        """Closing a connection in the middle of a query of another thread used to crash the interpreter."""
        for _ in range(3):
            opened = Archive.open(archive.path)
            errors: list[BaseException] = []
            started = threading.Barrier(4)

            def work(
                target: Archive = opened, seen: list[BaseException] = errors, gate: threading.Barrier = started
            ) -> None:
                gate.wait(10)
                try:
                    while True:
                        target.count(group_by="day", since="2024-06-01")
                        target.stats(MIRA)
                        target.search("note")
                        sum(1 for _ in target.iter_messages(MIRA))
                except ArchiveError:
                    return
                except Exception as error:  # noqa: BLE001 - anything else is the defect
                    seen.append(error)

            threads = [threading.Thread(target=work) for _ in range(3)]
            for thread in threads:
                thread.start()
            started.wait(10)
            opened.close()
            for thread in threads:
                thread.join(20)
            assert not any(thread.is_alive() for thread in threads)
            assert errors == []
            assert opened.open_connections() == 0
            with pytest.raises(ArchiveError):
                opened.summary()


class TestReviewCases:
    def test_a_message_far_outside_the_conversations_gets_the_right_hour(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        mira = builder.handle(MIRA)
        chat = builder.chat(MIRA, members=[mira])
        builder.message(chat, when=la(2020, 1, 10, 9), text="filed, years before", handle=mira)
        builder.message(None, when=la(2030, 7, 14, 20, 30), text="no conversation, summer of 2030", handle=mira)
        builder.message(chat, when=la(2030, 7, 14, 20, 30), item_type=2, group_title="event in 2030", handle=mira)
        builder.commit()
        import_chat_db(source_path, archive_path)

        with Archive.open(archive_path) as opened:
            lone = next(opened.iter_messages(None, no_conversation=True))
            assert lone["timestamp"] == "2030-07-14 20:30:00"
            assert opened.count(no_conversation=True, group_by="weekday_hour") == [
                {"key": "6-20", "total": 1, "sent": 0, "received": 1}  # a Sunday, in summer time
            ]
            assert opened.count(no_conversation=True, group_by="day")[0]["key"] == "2030-07-14"  # type: ignore[index]
            with_events = opened.count(group_by="weekday_hour", include_system=True)
            assert {"key": "6-20", "total": 1, "sent": 0, "received": 1} in with_events  # type: ignore[operator]

    def test_the_window_around_a_time_follows_the_wall_clock_also_in_the_repeated_hour(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        mira = builder.handle(MIRA)
        chat = builder.chat(MIRA, members=[mira])
        first = datetime(2024, 11, 3, 7, 0, tzinfo=UTC)  # midnight in Los Angeles; the clocks go back at 09:00 UTC
        for index in range(40):
            builder.message(chat, when=first + timedelta(minutes=6 * index), text=f"m{index}", handle=mira)
        builder.commit()
        import_chat_db(source_path, archive_path)

        with Archive.open(archive_path) as opened:
            stamps = [m["timestamp"] for m in opened.day(MIRA, "2024-11-03", limit=400)["messages"]]
            assert len(stamps) == 40 and len(set(stamps)) < 40  # some wall-clock times happen twice
            limit = 5
            for stamp in stamps:
                before = sum(1 for other in stamps if other < stamp)  # the rule of day_window in day_view.py
                expected = max(0, min(before - limit // 2, len(stamps) - limit))
                assert opened.day(MIRA, "2024-11-03", limit=limit, around=stamp)["offset"] == expected

    def test_around_must_be_text(self, archive: Archive) -> None:
        with pytest.raises(TypeError):
            archive.day(MIRA, "2024-06-03", around=5)  # type: ignore[arg-type]

    def test_a_very_long_list_of_conversation_ids_is_read_in_parts(self, archive: Archive) -> None:
        many = [f"unknown-{index}" for index in range(40_000)]
        assert archive.count(conversations=[*many, MIRA], since="2024-06-01") == archive.count(conversations=MIRA)
        assert archive.count(conversations=many, since="2024-06-01") == 0

    def test_text_with_a_lone_surrogate_never_raises(self, archive: Archive) -> None:
        broken = "a" + chr(0xD800)
        assert archive.search(broken) == {"total": 0, "hits": []}
        assert archive.search(chr(0xDFFF)) == {"total": 0, "hits": []}
        assert archive.count(text=broken) == 0
        assert archive.message(broken) is None
        assert archive.conversation(broken) is None
        assert archive.messages([broken])["messages"] == []
        assert archive.reactions([broken]) == {"a?": []}
