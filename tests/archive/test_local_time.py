"""Local time without a per-row conversion: UTC offset changes and day buckets.

count() and stats() group by local day, month, weekday and hour. They do it
with integer arithmetic on a table of UTC offset changes. These tests check
that table and the grouped figures against Python's own local time, in time
zones with daylight saving, half-hour offsets and short-lived changes.
"""

from __future__ import annotations

import os
import random
import time
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from archive import Archive, import_chat_db, timeconv
from tests.archive import worlds
from tests.archive.builders import ChatDbBuilder, utc

ZONES = [
    "America/Los_Angeles",
    "Asia/Seoul",
    "Asia/Kolkata",
    "Asia/Kathmandu",
    "Australia/Adelaide",
    "Australia/Lord_Howe",
    "Europe/London",
    "America/Santiago",
    "Africa/Casablanca",
    "Pacific/Apia",
    "UTC",
]
YEAR_2005 = 1_104_537_600
YEAR_2027 = 1_798_761_600


@pytest.fixture
def zone() -> Iterator[None]:
    """Let a test change TZ. The autouse fixture restores the test zone afterwards."""
    yield
    os.environ["TZ"] = "America/Los_Angeles"
    time.tzset()


def use_zone(name: str) -> None:
    os.environ["TZ"] = name
    time.tzset()


def offset_from_table(changes: list[tuple[int, int]], unix: int) -> int:
    found = changes[0][1]
    for start, offset in changes:
        if unix >= start:
            found = offset
    return found


class TestOffsetChanges:
    def test_los_angeles_2024(self) -> None:
        changes = timeconv.offset_changes()
        assert (1_710_064_800, -25_200) in changes  # 2024-03-10 10:00 UTC, clocks go forward
        assert (1_730_624_400, -28_800) in changes  # 2024-11-03 09:00 UTC, clocks go back
        assert changes == sorted(changes)
        assert all(a[1] != b[1] for a, b in zip(changes, changes[1:]))

    def test_zone_without_daylight_saving_has_one_offset(self, zone: None) -> None:
        use_zone("Asia/Seoul")
        assert [offset for _, offset in timeconv.offset_changes()] == [32_400]

    @pytest.mark.parametrize("name", ZONES)
    def test_table_matches_the_c_library_everywhere(self, zone: None, name: str) -> None:
        use_zone(name)
        changes = timeconv.offset_changes()
        rng = random.Random(name)
        instants = [rng.randrange(YEAR_2005, YEAR_2027) for _ in range(3000)]
        for start, _ in changes[1:]:
            instants.extend((start - 1, start, start + 1))
        for unix in instants:
            assert offset_from_table(changes, unix) == time.localtime(unix).tm_gmtoff, (name, unix)

    def test_a_zone_change_while_running_is_noticed(self, zone: None) -> None:
        before = timeconv.offset_changes()
        use_zone("Asia/Kolkata")
        assert [offset for _, offset in timeconv.offset_changes()] == [19_800]
        use_zone("America/Los_Angeles")
        assert timeconv.offset_changes() == before


class TestLocalSecondsSql:
    @pytest.mark.parametrize("name", ZONES)
    def test_sql_expression_gives_local_wall_clock_seconds(self, zone: None, name: str) -> None:
        import sqlite3

        use_zone(name)
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE m(sent_ns INTEGER)")
        rng = random.Random(name)
        instants = [rng.randrange(YEAR_2005, YEAR_2027) for _ in range(2000)]
        conn.executemany("INSERT INTO m VALUES (?)", [(unix * 1_000_000_000 + 7,) for unix in instants])
        expression = timeconv.sql_local_seconds("m.sent_ns")
        found = dict(conn.execute(f"SELECT m.sent_ns / 1000000000, {expression} FROM m").fetchall())
        for unix in instants:
            assert found[unix] == unix + time.localtime(unix).tm_gmtoff
        conn.close()

    def test_range_keeps_the_expression_short(self) -> None:
        whole = timeconv.sql_local_seconds("m.sent_ns")
        one_month = timeconv.sql_local_seconds("m.sent_ns", timeconv.day_start("2024-01-01"), timeconv.day_end("2024-01-31"))
        assert "CASE" in whole and "CASE" not in one_month
        assert "-28800" in one_month


@pytest.fixture(scope="module")
def scattered(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[tuple[int, bool]]]:
    """An archive with 4,000 messages at random moments from 2019 to 2025."""
    folder = tmp_path_factory.mktemp("scattered")
    source = folder / "chat.db"
    builder = ChatDbBuilder(source)
    rowan = builder.handle(worlds.ROWAN_PHONE)
    chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
    rng = random.Random(2024)
    low, high = int(utc(2019, 1, 1).timestamp()), int(utc(2025, 12, 31).timestamp())
    moments = sorted((rng.randrange(low, high), rng.random() < 0.4) for _ in range(4000))
    for unix, from_me in moments:
        builder.message(chat, when=unix - 978_307_200, text="x", handle=rowan, from_me=from_me)
    builder.close()
    archive_path = folder / "archive.sqlite3"
    import_chat_db(source, archive_path)
    return archive_path, moments


def reference(moments: list[tuple[int, bool]], pattern: str) -> list[tuple[str, int, int, int]]:
    total: Counter[str] = Counter()
    sent: Counter[str] = Counter()
    for unix, from_me in moments:
        local = datetime.fromtimestamp(unix)
        key = f"{local.weekday()}-{local.hour:02d}" if pattern == "weekday_hour" else local.strftime(pattern)
        total[key] += 1
        sent[key] += int(from_me)
    return [(key, total[key], sent[key], total[key] - sent[key]) for key in sorted(total)]


class TestGroupedCountsInOtherZones:
    @pytest.mark.parametrize("name", ZONES)
    def test_buckets_match_python_local_time(self, zone: None, scattered: tuple[Path, list[tuple[int, bool]]], name: str) -> None:
        archive_path, moments = scattered
        use_zone(name)
        with Archive.open(archive_path) as archive:
            for group_by, pattern in (("day", "%Y-%m-%d"), ("month", "%Y-%m"), ("year", "%Y"), ("weekday_hour", "weekday_hour")):
                found = [(row["key"], row["total"], row["sent"], row["received"]) for row in archive.count(group_by=group_by)]
                assert found == reference(moments, pattern), (name, group_by)

    @pytest.mark.parametrize("name", ["America/Los_Angeles", "Australia/Lord_Howe", "Africa/Casablanca"])
    def test_day_filters_and_stats_match_python_local_time(
        self, zone: None, scattered: tuple[Path, list[tuple[int, bool]]], name: str
    ) -> None:
        archive_path, moments = scattered
        use_zone(name)
        days = reference(moments, "%Y-%m-%d")
        with Archive.open(archive_path) as archive:
            for day, total, sent, _ in days[:: len(days) // 40]:
                assert archive.count(since=day, until=day) == total
                assert archive.count(since=day, until=day, from_me=True) == sent
            stats = archive.stats(worlds.ROWAN_PHONE)
            messages = list(archive.iter_messages(worlds.ROWAN_PHONE))
        assert stats["active_days"] == len(days)
        top = max(total for _, total, _, _ in days)
        assert stats["busiest_day_count"] == top
        assert stats["busiest_day"] == next(day for day, total, _, _ in days if total == top)
        assert stats["avg_messages_per_day"] == pytest.approx(len(moments) / len(days))
        for message in messages[::97]:
            assert message["timestamp"] == datetime.fromtimestamp(message["sent_at"]).strftime("%Y-%m-%d %H:%M:%S")


class TestClocksGoBack:
    def test_day_with_25_hours_is_one_bucket(self, builder: ChatDbBuilder, source_path: Path, archive_path: Path) -> None:
        rowan = builder.handle(worlds.ROWAN_PHONE)
        chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
        # 2024-11-03 in Los Angeles: 01:30 happens twice
        first = int(utc(2024, 11, 3, 8, 30).timestamp())  # 01:30 daylight time
        second = int(utc(2024, 11, 3, 9, 30).timestamp())  # 01:30 standard time
        late = int(utc(2024, 11, 4, 7, 59, 59).timestamp())  # 23:59:59 standard time
        next_day = int(utc(2024, 11, 4, 8, 0, 0).timestamp())  # 00:00 the next day
        for unix in (first, second, late, next_day):
            builder.message(chat, when=unix - 978_307_200, text="tick", handle=rowan)
        builder.close()
        import_chat_db(source_path, archive_path)
        with Archive.open(archive_path) as archive:
            by_day = [(row["key"], row["total"]) for row in archive.count(group_by="day")]
            by_hour = [(row["key"], row["total"]) for row in archive.count(group_by="weekday_hour")]
            assert by_day == [("2024-11-03", 3), ("2024-11-04", 1)]
            assert by_hour == [("0-00", 1), ("6-01", 2), ("6-23", 1)]
            assert archive.count(since="2024-11-03", until="2024-11-03") == 3
            stamps = [message["timestamp"] for message in archive.iter_messages(worlds.ROWAN_PHONE)]
            assert stamps[:2] == ["2024-11-03 01:30:00", "2024-11-03 01:30:00"]
