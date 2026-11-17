"""Review findings in the time rules.

- The offset table follows a change of the time zone, also between two zones that look alike.
- The table goes up to the end of stored time (2200).
- A day outside the supported range is a bound, not an error.
- A value that cannot be a date gives None. Every date is before the end of stored time.
- The column of the SQL expression may be an expression.
- Local time text goes back to unix seconds.
"""

from __future__ import annotations

import os
import random
import sqlite3
import time
from bisect import bisect_right
from collections.abc import Callable, Iterator
from datetime import date, datetime
from decimal import Decimal
from itertools import pairwise

import pytest

from archive import timeconv
from tests.archive.builders import utc
from tests.archive.conftest import TEST_TIME_ZONE

NS = timeconv.NS_PER_SECOND
DAY = 86_400
START = timeconv.APPLE_EPOCH_UNIX  # 2001-01-01
END = timeconv.MAX_UNIX_NS // NS  # 2200-01-01, the end of stored time: every stored moment is before it
FAR_ZONES = ["America/Los_Angeles", "Australia/Lord_Howe", "Europe/London", "Asia/Kolkata"]
# Zones in the order of their names. With a cache key of a few offsets, a zone of each group got the table of
# the zone before it.
SWEEP = [
    "Africa/Sao_Tome",
    "Africa/Timbuktu",
    "America/Araguaina",
    "America/Argentina/Buenos_Aires",
    "America/Argentina/San_Luis",
    "America/Argentina/Tucuman",
    "America/Catamarca",
    "America/Cayenne",
    "America/Indiana/Indianapolis",
    "America/Indiana/Marengo",
    "America/Indiana/Petersburg",
    "America/Indiana/Vevay",
    "America/Indiana/Vincennes",
    "America/Indiana/Winamac",
    "America/New_York",
    "America/North_Dakota/Center",
    "America/North_Dakota/New_Salem",
    "America/Rainy_River",
    "America/Rankin_Inlet",
    "America/Swift_Current",
    "America/Tegucigalpa",
    "America/Toronto",
    "Asia/Aqtau",
    "Asia/Aqtobe",
    "Asia/Ashgabat",
    "Asia/Famagusta",
    "Asia/Gaza",
    "Asia/Hebron",
    "Asia/Kolkata",
    "Australia/Canberra",
    "Australia/Currie",
    "Australia/Sydney",
    "Australia/Tasmania",
    "Europe/Lisbon",
    "Europe/London",
    "UTC",
]

ZoneSetter = Callable[[str | None], None]


@pytest.fixture
def use_zone() -> Iterator[ZoneSetter]:
    """Change the time zone of the process inside a test. The test zone comes back afterwards."""

    def change(name: str | None) -> None:
        if name is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = name
        time.tzset()

    yield change
    change(TEST_TIME_ZONE)


def offset_reader(changes: list[tuple[int, int]]) -> Callable[[int], int]:
    """A function that gives the offset a table holds for a moment."""
    starts = [start for start, _ in changes]
    return lambda unix: changes[max(bisect_right(starts, unix) - 1, 0)][1]


def other_digits(text: str) -> str:
    """The same text with every ASCII digit written as an Arabic-Indic digit."""
    return "".join(chr(0x0660 + int(char)) if "0" <= char <= "9" else char for char in text)


class TestZoneChange:
    def test_two_zones_that_differ_only_in_two_years_get_their_own_table(self, use_zone: ZoneSetter) -> None:
        summer_2006 = int(utc(2006, 7, 15, 12).timestamp())
        use_zone("America/Indiana/Indianapolis")
        indianapolis = timeconv.offset_changes()
        use_zone("America/Indiana/Petersburg")
        petersburg = timeconv.offset_changes()

        assert petersburg != indianapolis
        assert petersburg == timeconv._build_offset_table(END)
        assert offset_reader(indianapolis)(summer_2006) == -4 * 3600  # Eastern daylight time
        assert offset_reader(petersburg)(summer_2006) == -5 * 3600  # Central daylight time in 2006 and 2007

    def test_a_sweep_over_many_zones_never_gives_a_stale_table(self, use_zone: ZoneSetter) -> None:
        for name in SWEEP:
            use_zone(name)
            assert timeconv.offset_changes() == timeconv._build_offset_table(END), name

    def test_going_back_to_a_zone_gives_its_table_again(self, use_zone: ZoneSetter) -> None:
        tables: dict[str, list[tuple[int, int]]] = {}
        for name in ("America/Indiana/Marengo", "America/Indiana/Petersburg", "America/Indiana/Marengo"):
            use_zone(name)
            found = timeconv.offset_changes()
            assert tables.setdefault(name, found) == found
        assert tables["America/Indiana/Marengo"] != tables["America/Indiana/Petersburg"]

    def test_the_system_zone_is_used_when_tz_is_not_set(self, use_zone: ZoneSetter) -> None:
        use_zone(None)
        assert timeconv.offset_changes() == timeconv._build_offset_table(END)

    def test_a_system_with_no_localtime_link(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_link(path: object) -> str:
            raise OSError(22, "Invalid argument")

        monkeypatch.setattr(os, "readlink", no_link)
        assert timeconv.offset_changes() == timeconv._build_offset_table(END)

    def test_the_table_is_built_once_for_a_zone(self, use_zone: ZoneSetter, monkeypatch: pytest.MonkeyPatch) -> None:
        use_zone("Europe/London")
        first = timeconv.offset_changes()
        asked: list[int] = []
        real_offset = timeconv.utc_offset

        def counted_offset(unix_seconds: int) -> int:
            asked.append(unix_seconds)
            return real_offset(unix_seconds)

        def no_build(end: int) -> list[tuple[int, int]]:
            raise AssertionError("the table was built again")

        monkeypatch.setattr(timeconv, "utc_offset", counted_offset)
        monkeypatch.setattr(timeconv, "_build_offset_table", no_build)

        assert timeconv.offset_changes() == first
        assert timeconv.offset_changes(START, START + DAY) == first[:1]
        # the check that the zone is the same asks for one offset in each year from 2001 to next year
        years = time.gmtime().tm_year - 2001 + 2
        assert 0 < len(asked) <= 2 * (years + 1)

    def test_the_caller_cannot_change_the_cached_table(self) -> None:
        first = timeconv.offset_changes()
        first.clear()
        again = timeconv.offset_changes()
        assert again and again == timeconv._build_offset_table(END)


class TestFarDates:
    @pytest.mark.parametrize("name", FAR_ZONES)
    def test_the_table_matches_the_c_library_from_2001_to_2199(self, use_zone: ZoneSetter, name: str) -> None:
        use_zone(name)
        changes = timeconv.offset_changes()
        offset_at = offset_reader(changes)
        rng = random.Random(name)
        instants = [rng.randrange(START, END) for _ in range(4000)]
        instants.extend((START, END - 1, END))
        for start, _ in changes[1:]:
            instants.extend((start - 1, start, start + 1))
        for unix in instants:
            assert offset_at(unix) == time.localtime(unix).tm_gmtoff, (name, unix)

    def test_the_table_goes_on_after_this_decade(self) -> None:
        changes = timeconv.offset_changes()
        assert changes[-1][0] > int(utc(2199, 1, 1).timestamp())  # Los Angeles: the clocks still change in 2199
        assert changes == sorted(changes)
        assert all(before[1] != after[1] for before, after in pairwise(changes))

    @pytest.mark.parametrize("name", FAR_ZONES)
    def test_sql_gives_the_local_wall_clock_of_late_dates(self, use_zone: ZoneSetter, name: str) -> None:
        use_zone(name)
        low, high = int(utc(2040, 1, 1).timestamp()), END
        rng = random.Random(name)
        instants = [rng.randrange(low, high) for _ in range(1500)]
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE m(sent_ns INTEGER)")
            conn.executemany("INSERT INTO m VALUES (?)", [(unix * NS + 7,) for unix in instants])
            expression = timeconv.sql_local_seconds("m.sent_ns", low, high)
            found = dict(conn.execute(f"SELECT m.sent_ns / {NS}, {expression} FROM m").fetchall())
        finally:
            conn.close()
        for unix in instants:
            local = time.localtime(unix)
            assert found[unix] == unix + local.tm_gmtoff, (name, unix)
            assert (found[unix] // 3600) % 24 == local.tm_hour  # the hour of the day that Python gives

    def test_a_range_of_late_days_keeps_the_expression_short(self) -> None:
        july_2150 = timeconv.sql_local_seconds(
            "m.sent_ns", timeconv.day_start("2150-07-01"), timeconv.day_end("2150-07-31")
        )
        assert "CASE" not in july_2150
        assert "-25200" in july_2150  # daylight time in Los Angeles


class TestDaysOutsideTheRange:
    @pytest.mark.parametrize("day", ["0001-01-01", "0001-01-02", "1900-06-15", "1969-12-31", "1970-01-01"])
    def test_a_day_before_the_range_gives_zero(self, day: str) -> None:
        assert timeconv.day_start(day) == 0
        assert timeconv.day_end(day) == 0

    @pytest.mark.parametrize("day", ["2199-12-31", "2200-01-01", "2300-01-01", "9999-12-30", "9999-12-31"])
    def test_a_day_after_the_range_gives_the_end_of_stored_time(self, day: str) -> None:
        assert timeconv.day_start(day) == END
        assert timeconv.day_end(day) == END

    @pytest.mark.parametrize("name", ["America/Los_Angeles", "Pacific/Kiritimati", "Etc/GMT+12", "Asia/Kolkata", "UTC"])
    def test_the_first_and_the_last_day_of_the_range_are_local_midnights(self, use_zone: ZoneSetter, name: str) -> None:
        use_zone(name)
        assert timeconv.day_start("1970-01-02") == int(datetime(1970, 1, 2).timestamp()) > 0
        assert timeconv.day_end("1970-01-02") == int(datetime(1970, 1, 3).timestamp())
        assert timeconv.day_start("2199-12-30") == int(datetime(2199, 12, 30).timestamp())
        assert timeconv.day_end("2199-12-30") == int(datetime(2199, 12, 31).timestamp()) < END

    def test_bounds_never_go_backwards(self) -> None:
        days = [
            "0001-01-01", "1969-12-31", "1970-01-01", "1970-01-02", "1970-01-03", "2001-01-01", "2024-03-10",
            "2024-11-03", "2199-12-29", "2199-12-30", "2199-12-31", "2200-01-01", "9999-12-31",
        ]  # fmt: skip
        starts = [timeconv.day_start(day) for day in days]
        ends = [timeconv.day_end(day) for day in days]
        assert starts == sorted(starts) and ends == sorted(ends)
        assert all(0 <= start <= end <= END for start, end in zip(starts, ends))
        assert all(type(value) is int for value in starts + ends)

    @pytest.mark.parametrize(
        "value",
        [
            "0000-01-01",
            "2024-02-30",
            "2023-02-29",
            "2024-00-10",
            "2024-13-01",
            "10000-01-01",
            "-001-01-01",
            "2024-01-15T00:00",
            " 2024-01-15",
            "2024-01-15\n",
            other_digits("2024-01-15"),
            None,
            20240115,
            b"2024-01-15",
            date(2024, 1, 15),
        ],
    )
    def test_text_that_is_not_a_day_still_raises_value_error(self, value: object) -> None:
        for bound in (timeconv.day_start, timeconv.day_end):
            with pytest.raises(ValueError):
                bound(value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("error", [OverflowError("timestamp out of range"), OSError(22, "Invalid argument")])
    def test_a_c_library_with_no_local_time_for_a_day_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        class NoLocalTime(datetime):
            def timestamp(self) -> float:
                raise error

        monkeypatch.setattr(timeconv, "datetime", NoLocalTime)
        # the standard time of Los Angeles is 8 hours behind UTC
        assert timeconv.day_start("2024-01-15") == int(utc(2024, 1, 15, 8).timestamp())
        assert timeconv.day_end("2024-01-15") == int(utc(2024, 1, 16, 8).timestamp())
        assert timeconv.day_start("1970-01-02") == DAY + 8 * 3600
        assert timeconv.day_end("2199-12-30") == END - DAY + 8 * 3600


class TestValuesThatCannotBeADate:
    @pytest.mark.parametrize(
        "value",
        [
            float("inf"),
            float("-inf"),
            float("nan"),
            1e308,
            -1e308,
            10**400,
            -(10**400),
            Decimal("Infinity"),
            Decimal("NaN"),
            "inf",
            "nan",
            "1e400",
            "9" * 5000,
            "",
            "   ",
            b"",
            b"  ",
            b"\xff",
            [1],
            (1,),
            {},
            1j,
            object(),
        ],
    )
    def test_gives_none_and_does_not_raise(self, value: object) -> None:
        assert timeconv.apple_to_unix_ns(value) is None
        assert timeconv.apple_to_unix_seconds(value) is None

    def test_every_date_is_before_the_end_of_stored_time(self) -> None:
        end = timeconv.MAX_UNIX_NS - START * NS  # 2200-01-01T00:00:00Z as an Apple date
        assert timeconv.apple_to_unix_ns(end - 1) == timeconv.MAX_UNIX_NS - 1
        assert timeconv.apple_to_unix_seconds(end - 1) == END - 1
        assert timeconv.apple_to_unix_ns(end) is None
        assert timeconv.apple_to_unix_ns(END - START) is None  # the same moment in seconds
        assert timeconv.apple_to_unix_ns(END - START - 1) == (END - 1) * NS

    def test_a_day_after_the_range_is_a_bound_with_no_date_beyond_it(self) -> None:
        last_date = timeconv.apple_to_unix_ns(timeconv.MAX_UNIX_NS - START * NS - 1)
        assert last_date is not None
        assert last_date < timeconv.day_end("9999-12-31") * NS  # `until` a far day leaves nothing out
        assert last_date < timeconv.day_start("9999-12-31") * NS  # `since` a far day finds nothing

    def test_numbers_of_another_type_are_still_read(self) -> None:
        moment = int(utc(2024, 1, 15, 18, 30, 5).timestamp())
        apple_seconds = moment - START
        assert timeconv.apple_to_unix_seconds(float(apple_seconds)) == moment
        assert timeconv.apple_to_unix_seconds(str(apple_seconds)) == moment
        assert timeconv.apple_to_unix_seconds(f" {apple_seconds} ".encode()) == moment
        assert timeconv.apple_to_unix_ns(float(apple_seconds * NS)) == moment * NS


class TestSqlColumn:
    @staticmethod
    def table(instants: list[int], shift_ns: int) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE m(a INTEGER, b INTEGER)")
        conn.executemany("INSERT INTO m VALUES (?, ?)", [(unix * NS + shift_ns, shift_ns) for unix in instants])
        return conn

    def test_the_column_may_be_an_expression(self) -> None:
        rng = random.Random(5)
        instants = [rng.randrange(START, END) for _ in range(500)]
        conn = self.table(instants, 5 * NS)
        try:
            expression = timeconv.sql_local_seconds("m.a - m.b")
            found = conn.execute(f"SELECT (m.a - m.b) / {NS}, {expression} FROM m").fetchall()
        finally:
            conn.close()
        assert len(found) == len(instants)
        for unix, local in found:
            assert local == unix + time.localtime(unix).tm_gmtoff

    def test_the_column_is_put_in_parentheses_in_a_short_expression_too(self) -> None:
        january = timeconv.day_start("2024-01-01"), timeconv.day_end("2024-01-31")
        moment = int(utc(2024, 1, 15, 18, 30, 5).timestamp())
        conn = self.table([moment], 5 * NS)
        try:
            expression = timeconv.sql_local_seconds("m.a - m.b", *january)
            assert "CASE" not in expression
            assert conn.execute(f"SELECT {expression} FROM m").fetchone() == (moment - 8 * 3600,)
        finally:
            conn.close()

    def test_an_offset_table_with_no_entry_is_refused(self) -> None:
        with pytest.raises(ValueError):
            timeconv._offset_sql("m.sent_ns", [])


class TestLocalTimeText:
    def test_known_moments(self) -> None:
        # the test time zone is America/Los_Angeles
        assert timeconv.local_to_unix("2024-01-15 10:30:05") == int(utc(2024, 1, 15, 18, 30, 5).timestamp())
        assert timeconv.local_to_unix("2024-07-04 00:00:00") == int(utc(2024, 7, 4, 7).timestamp())
        assert timeconv.local_to_unix("2001-01-01 00:00:00") == int(utc(2001, 1, 1, 8).timestamp())

    def test_a_fraction_of_a_second_is_dropped(self) -> None:
        whole = timeconv.local_to_unix("2024-01-15 10:30:05")
        assert timeconv.local_to_unix("2024-01-15 10:30:05.250000") == whole
        assert timeconv.local_to_unix("2024-01-15 10:30:05.9") == whole
        assert timeconv.local_to_unix("2024-01-15 10:30:05.999999999") == whole

    def test_outer_white_space_is_ignored(self) -> None:
        assert timeconv.local_to_unix("  2024-01-15 10:30:05\n") == timeconv.local_to_unix("2024-01-15 10:30:05")

    def test_the_result_is_a_whole_number(self) -> None:
        assert type(timeconv.local_to_unix("2024-01-15 10:30:05.5")) is int

    @pytest.mark.parametrize("name", [*FAR_ZONES, "UTC"])
    def test_text_from_local_timestamp_comes_back(self, use_zone: ZoneSetter, name: str) -> None:
        use_zone(name)
        rng = random.Random(name)
        for unix in [rng.randrange(START, END - 3 * DAY) for _ in range(1500)]:
            text = timeconv.local_timestamp(unix)
            found = timeconv.local_to_unix(text)
            assert found is not None and timeconv.local_timestamp(found) == text, (name, unix)
            # another moment with the same text is the earlier one of an hour that happens twice
            assert found == unix or (found < unix and time.localtime(found).tm_gmtoff > time.localtime(unix).tm_gmtoff)

    def test_an_hour_that_happens_twice_gives_the_first_moment(self) -> None:
        # 2024-11-03 01:30 in Los Angeles: 08:30 UTC in daylight time, then 09:30 UTC in standard time
        first, second = int(utc(2024, 11, 3, 8, 30).timestamp()), int(utc(2024, 11, 3, 9, 30).timestamp())
        assert timeconv.local_timestamp(first) == timeconv.local_timestamp(second) == "2024-11-03 01:30:00"
        assert timeconv.local_to_unix("2024-11-03 01:30:00") == first

    def test_an_hour_that_does_not_exist_gives_the_moment_one_hour_later(self) -> None:
        # 2024-03-10 in Los Angeles: the clocks go from 02:00 to 03:00
        found = timeconv.local_to_unix("2024-03-10 02:30:00")
        assert found == int(utc(2024, 3, 10, 10, 30).timestamp())
        assert timeconv.local_timestamp(found) == "2024-03-10 03:30:00"

    @pytest.mark.parametrize(
        "text",
        [
            "soon",
            "not a time",
            "",
            "   ",
            "2024-01-15",
            "2024-01-15 10:30",
            "2024-01-15T10:30:05",
            "2024-01-15 10:30:05Z",
            "2024-01-15 10:30:05+00:00",
            "2024-01-15 10:30:05 PST",
            "2024-01-15  10:30:05",
            "2024-01-15\n10:30:05",
            "2024-01-15 10:30:05.",
            "2024-01-15 10:30:05,5",
            "2024-1-5 10:30:05",
            "24-01-15 10:30:05",
            "2024-13-01 00:00:00",
            "2024-02-30 00:00:00",
            "2023-02-29 00:00:00",
            "2024-00-10 00:00:00",
            "2024-01-15 24:00:00",
            "2024-01-15 10:60:00",
            "2024-01-15 10:30:60",
            "0000-01-01 00:00:00",
            "20240115103005",
            "1705343405",
            other_digits("2024-01-15 10:30:05"),
        ],
    )
    def test_text_that_is_not_a_local_time_gives_none(self, text: str) -> None:
        assert timeconv.local_to_unix(text) is None

    @pytest.mark.parametrize(
        "value",
        [None, 5, 1_705_343_405, 1.5, b"2024-01-15 10:30:05", ["2024-01-15 10:30:05"], datetime(2024, 1, 15, 10, 30)],
    )
    def test_a_value_that_is_not_text_gives_none(self, value: object) -> None:
        assert timeconv.local_to_unix(value) is None

    @pytest.mark.parametrize(
        "text",
        [
            "0001-01-01 00:00:00",
            "1969-12-31 23:59:59",
            "1970-01-01 23:59:59",
            "2199-12-31 00:00:00",
            "2200-01-01 00:00:00",
            "9999-12-31 23:59:59",
        ],
    )
    def test_a_time_outside_the_supported_days_gives_none(self, text: str) -> None:
        assert timeconv.local_to_unix(text) is None

    @pytest.mark.parametrize("name", ["America/Los_Angeles", "Pacific/Kiritimati", "Etc/GMT+12", "UTC"])
    def test_the_first_and_the_last_second_of_the_supported_days(self, use_zone: ZoneSetter, name: str) -> None:
        use_zone(name)
        first = timeconv.local_to_unix("1970-01-02 00:00:00")
        last = timeconv.local_to_unix("2199-12-30 23:59:59")
        assert first == timeconv.day_start("1970-01-02") > 0
        assert last == timeconv.day_end("2199-12-30") - 1 < END

    @pytest.mark.parametrize("error", [OverflowError("timestamp out of range"), OSError(22, "Invalid argument")])
    def test_a_c_library_with_no_local_time_for_a_moment_gives_none(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        class NoLocalTime(datetime):
            def timestamp(self) -> float:
                raise error

        monkeypatch.setattr(timeconv, "datetime", NoLocalTime)
        assert timeconv.local_to_unix("2024-01-15 10:30:05") is None
