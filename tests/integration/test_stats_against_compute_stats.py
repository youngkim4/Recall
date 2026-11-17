"""Archive.stats() against compute_stats in conversation.py, on one invented export.

Every figure that both give is equal while no clock change is in play. The
last tests state what a clock change does: the archive keeps true UTC and
measures elapsed time, pandas subtracts the times that the clock showed.
The time zone of the tests is America/Los_Angeles (see guards.py):
the clocks go forward on 2024-03-10 and back on 2024-11-03.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from archive.compat import messages_frame
from conversation import compute_stats, filter_conversation, within_dates
from tests.integration import bridge_data as data
from tests.integration import guards
from tests.integration.bridge_data import GROUP, MIRA, NEIGHBOUR, ROWAN, Loaded, export_row

HOUR_IN_DAYS = 1 / 24  # 0.0417 days
# day -> messages on it, one an hour from 09:00. Two days share the largest count, and the two
# longest gaps are equal (13.75 days): a tie goes to the earliest day and to the earliest gap.
DAYS = {"2024-01-05": 3, "2024-01-06": 7, "2024-01-20": 7, "2024-02-03": 2}
FRACTIONS = ("avg_messages_per_day", "longest_gap_days")
SHARED = (
    "total_messages",
    "sent_count",
    "received_count",
    "first_timestamp",
    "last_timestamp",
    "active_days",
    "avg_messages_per_day",
    "busiest_day",
    "busiest_day_count",
    "longest_gap_days",
    "longest_gap_start",
    "longest_gap_end",
)


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day, count in DAYS.items():
        for stamp in data.stamps(f"{day} 09:00:00", count, step_seconds=3600):
            number = len(rows) + 1
            rows.append(export_row(number, stamp, f"message {number}", from_me=number % 3 == 0))
    rows += [
        export_row(101, "2024-01-06 12:30:00", "another person", NEIGHBOUR),
        export_row(102, "2024-01-12 12:30:00", "the same person, later", NEIGHBOUR, from_me=True),
        export_row(111, "2024-01-06 13:30:00", "a group", GROUP, sender=MIRA),
        export_row(112, "2024-01-06 13:31:00", "the group again", GROUP, sender=ROWAN),
        export_row(113, "2024-01-09 08:00:00", "an answer in the group", GROUP, from_me=True),
        export_row(121, "2024-01-13 12:00:00", "a row with no conversation", "", sender=ROWAN),
    ]
    return rows


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Loaded]:
    # a module fixture is made before the time zone guard of a test: the bridge reads local times
    with guards.time_zone():
        both = data.load_both(tmp_path_factory.mktemp("stats"), _rows())
    yield both
    both.archive.close()


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def frame_figures(loaded: Loaded, chat: str, since: str | None = None, until: str | None = None) -> dict[str, Any]:
    """compute_stats on the rows that the preview of the app takes, in the words of the archive."""
    conv = filter_conversation(within_dates(loaded.frame, since, until), chat)
    stats = compute_stats(conv)
    return {
        "total_messages": stats.total_messages,
        "sent_count": stats.sent_count,
        "received_count": stats.received_count,
        "first_timestamp": _text(stats.first_timestamp),
        "last_timestamp": _text(stats.last_timestamp),
        "active_days": stats.active_days,
        "avg_messages_per_day": stats.avg_messages_per_day,
        "busiest_day": None if stats.busiest_day is None else stats.busiest_day.strftime("%Y-%m-%d"),
        "busiest_day_count": stats.busiest_day_count,
        "longest_gap_days": stats.longest_gap_days,
        "longest_gap_start": _text(stats.longest_gap_start),
        "longest_gap_end": _text(stats.longest_gap_end),
    }


def archive_figures(loaded: Loaded, chat: str, since: str | None = None, until: str | None = None) -> dict[str, Any]:
    stats = loaded.archive.stats(chat, since=since, until=until)
    return {name: stats[name] for name in SHARED}


def nearly(figures: dict[str, Any]) -> dict[str, Any]:
    """The same figures, with room for the last digit of the two that are fractions."""
    return {**figures, **{name: pytest.approx(figures[name]) for name in FRACTIONS}}


def test_every_shared_figure_is_equal(loaded: Loaded) -> None:
    figures = archive_figures(loaded, ROWAN)
    assert figures == nearly(frame_figures(loaded, ROWAN))
    assert figures == {
        "total_messages": 19,
        "sent_count": 6,
        "received_count": 13,
        "first_timestamp": "2024-01-05 09:00:00",
        "last_timestamp": "2024-02-03 10:00:00",
        "active_days": 4,
        "avg_messages_per_day": 4.75,  # for each active day, in both
        "busiest_day": "2024-01-06",  # the earliest of the two days with 7 messages
        "busiest_day_count": 7,
        "longest_gap_days": 13.75,
        "longest_gap_start": "2024-01-06 15:00:00",  # the earliest of the two gaps of 13.75 days
        "longest_gap_end": "2024-01-20 09:00:00",
    }


def test_the_archive_names_the_conversation_and_leaves_three_figures_out(loaded: Loaded) -> None:
    stats = loaded.archive.stats(ROWAN)
    assert stats["conversation"] == ROWAN  # ConversationStats calls it chat_id
    assert set(stats) == {"conversation", *SHARED}
    assert compute_stats(loaded.conversation(ROWAN)).chat_id == ROWAN
    for missing in ("unknown_direction_count", "attachments", "reactions"):
        assert missing not in stats


@pytest.mark.parametrize(
    ("since", "until"),
    [
        ("2024-01-06", "2024-01-20"),
        ("2024-01-06", "2024-01-06"),
        ("2024-01-07", None),
        (None, "2024-01-05"),
        (None, "2024-01-19"),
        ("2024-01-20", "2024-02-03"),
        ("2024-01-10", "2024-01-15"),
        ("2025-01-01", None),
        (None, "2001-01-02"),
    ],
)
def test_the_figures_of_a_range_of_days_are_equal(loaded: Loaded, since: str | None, until: str | None) -> None:
    """Both include all of the first day and all of the last day (within_dates in conversation.py)."""
    assert archive_figures(loaded, ROWAN, since, until) == nearly(frame_figures(loaded, ROWAN, since, until))


def test_a_range_with_no_message_gives_zeros_and_no_times(loaded: Loaded) -> None:
    figures = archive_figures(loaded, ROWAN, "2024-01-10", "2024-01-15")
    assert figures == nearly(frame_figures(loaded, ROWAN, "2024-01-10", "2024-01-15"))
    assert figures["total_messages"] == 0 and figures["longest_gap_days"] == 0.0
    assert [figures[name] for name in ("first_timestamp", "busiest_day", "longest_gap_start")] == [None, None, None]


@pytest.mark.parametrize("chat", [NEIGHBOUR, GROUP])
def test_other_conversations_have_their_own_figures(loaded: Loaded, chat: str) -> None:
    assert archive_figures(loaded, chat) == nearly(frame_figures(loaded, chat))


@pytest.mark.parametrize("chat", ["+19995550000", ROWAN[:-1], ""])
def test_an_unknown_conversation_gives_zeros_on_both_sides(loaded: Loaded, chat: str) -> None:
    figures = archive_figures(loaded, chat)
    assert figures == nearly(frame_figures(loaded, chat))
    assert figures["total_messages"] == 0


def test_one_message_has_no_gap(loaded: Loaded) -> None:
    figures = archive_figures(loaded, GROUP, "2024-01-09", "2024-01-09")
    assert figures == nearly(frame_figures(loaded, GROUP, "2024-01-09", "2024-01-09"))
    assert (figures["total_messages"], figures["longest_gap_days"], figures["longest_gap_start"]) == (1, 0.0, None)


def test_compute_stats_on_the_frame_of_the_archive_gives_the_figures_of_the_archive(loaded: Loaded) -> None:
    """After the hook-up the figures on pandas read archive.compat.messages_frame."""
    frame = messages_frame(loaded.archive)
    stats = compute_stats(filter_conversation(frame, ROWAN))
    own = loaded.archive.stats(ROWAN)
    assert (stats.total_messages, stats.sent_count, stats.received_count) == (19, 6, 13)
    assert stats.unknown_direction_count == 0
    assert str(stats.first_timestamp) == own["first_timestamp"]
    assert stats.longest_gap_days == pytest.approx(own["longest_gap_days"])


def test_a_row_that_does_not_say_who_sent_it_is_not_in_the_archive(tmp_path: Path) -> None:
    """A difference: compute_stats counts such a row in the total and in neither direction.

    README, "Legacy export bridge": a row "with no `is_from_me` is not stored: reason
    `bad_row`". The exporter always writes the direction, so its files hold no such row.
    """
    rows = [
        export_row(1, "2024-01-05 09:00:00", "received"),
        export_row(2, "2024-01-05 09:01:00", "sent", from_me=True),
        export_row(3, "2024-01-05 09:02:00", "nobody knows", is_from_me=""),
    ]
    both = data.load_both(tmp_path, rows)
    try:
        stats = compute_stats(both.conversation(ROWAN))
        assert (stats.total_messages, stats.sent_count, stats.received_count) == (3, 1, 1)
        assert stats.unknown_direction_count == 1
        own = both.archive.stats(ROWAN)
        assert (own["total_messages"], own["sent_count"], own["received_count"]) == (2, 1, 1)
        assert both.result.skipped == {"bad_row": 1}
    finally:
        both.archive.close()


def _gap(both: Loaded) -> tuple[dict[str, Any], dict[str, Any]]:
    """The longest gap as compute_stats gives it, and as the archive gives it."""
    wanted = ("longest_gap_days", "longest_gap_start", "longest_gap_end")
    frame, own = frame_figures(both, ROWAN), archive_figures(both, ROWAN)
    return {name: frame[name] for name in wanted}, {name: own[name] for name in wanted}


@pytest.mark.parametrize(
    ("day_before", "day_of_change", "elapsed_hours"),
    [("2024-03-09", "2024-03-10", 23), ("2024-11-02", "2024-11-03", 25)],
    ids=["clocks go forward", "clocks go back"],
)
def test_a_gap_across_a_clock_change_differs_by_one_hour(
    tmp_path: Path, day_before: str, day_of_change: str, elapsed_hours: int
) -> None:
    """THE DIFFERENCE: noon to noon is 24 hours on the clock and 23 or 25 hours of elapsed time.

    README, "stats()": "`longest_gap_days` is elapsed time. `compute_stats` subtracts
    wall-clock times. The two differ by one hour (0.0417 days) when the longest gap
    crosses a clock change."
    """
    rows = [
        export_row(1, f"{day_before} 11:00:00", "an hour before noon"),
        export_row(2, f"{day_before} 12:00:00", "noon", from_me=True),
        export_row(3, f"{day_of_change} 12:00:00", "noon of the next day"),
        export_row(4, f"{day_of_change} 13:00:00", "an hour later", from_me=True),
    ]
    both = data.load_both(tmp_path, rows)
    try:
        frame, own = frame_figures(both, ROWAN), archive_figures(both, ROWAN)
        assert frame["longest_gap_days"] == 1.0
        assert own["longest_gap_days"] == pytest.approx(elapsed_hours / 24)
        assert abs(frame["longest_gap_days"] - own["longest_gap_days"]) == pytest.approx(HOUR_IN_DAYS)
        assert round(HOUR_IN_DAYS, 4) == 0.0417
        # the gap is between the same two messages on both sides, and every other figure is equal
        assert own["longest_gap_start"] == f"{day_before} 12:00:00"
        assert own["longest_gap_end"] == f"{day_of_change} 12:00:00"
        others = [name for name in SHARED if name != "longest_gap_days"]
        assert [own[name] for name in others] == [nearly(frame)[name] for name in others]
    finally:
        both.archive.close()


def test_a_clock_change_can_make_another_gap_the_longest(tmp_path: Path) -> None:
    """A consequence of the same rule: the two sides can name two different gaps.

    The first gap is 23.5 hours with no clock change. The second gap is 24 hours on
    the clock and 23 hours of elapsed time: the clocks go forward inside it.
    """
    rows = [
        export_row(1, "2024-03-08 12:00:00", "start"),
        export_row(2, "2024-03-09 11:30:00", "23.5 hours later", from_me=True),
        export_row(3, "2024-03-10 11:30:00", "24 hours on the clock later"),
    ]
    both = data.load_both(tmp_path, rows)
    try:
        frame, own = _gap(both)
        assert frame == {
            "longest_gap_days": 1.0,
            "longest_gap_start": "2024-03-09 11:30:00",
            "longest_gap_end": "2024-03-10 11:30:00",
        }
        assert own == {
            "longest_gap_days": pytest.approx(23.5 / 24),
            "longest_gap_start": "2024-03-08 12:00:00",
            "longest_gap_end": "2024-03-09 11:30:00",
        }
    finally:
        both.archive.close()


def test_a_local_time_that_does_not_exist_is_read_as_the_time_one_hour_later(tmp_path: Path) -> None:
    """A difference: on 2024-03-10 the clock goes from 01:59:59 to 03:00:00. No clock showed 02:30.

    README, "Legacy export bridge": "A local time that does not exist (the clocks went
    forward) is read as the time one hour later. The message then comes after the
    messages of that later time with a lower ROWID." pandas keeps the time as written.
    """
    rows = [
        export_row(1, "2024-03-10 01:50:00", "before the change"),
        export_row(2, "2024-03-10 03:10:00", "after the change", from_me=True),
        export_row(3, "2024-03-10 02:30:00", "a time that no clock showed"),
    ]
    both = data.load_both(tmp_path, rows)
    try:
        stored = [both.archive.message(data.legacy_id(number)) for number in (1, 2, 3)]
        assert [message["timestamp"] for message in stored] == [
            "2024-03-10 01:50:00",
            "2024-03-10 03:10:00",
            "2024-03-10 03:30:00",
        ]
        assert stored[2]["sent_at"] - stored[1]["sent_at"] == 20 * 60
        assert stored[1]["sent_at"] - stored[0]["sent_at"] == 20 * 60  # 01:50 to 03:10 is 20 minutes of elapsed time
        day = both.archive.day(ROWAN, "2024-03-10")
        assert data.archive_ids(day["messages"]) == list(map(data.legacy_id, (1, 2, 3)))
        in_frame = filter_conversation(both.frame, ROWAN)
        assert data.frame_ids(in_frame) == list(map(data.legacy_id, (1, 3, 2)))
        assert data.frame_times(in_frame)[1] == "2024-03-10 02:30:00"
        frame, own = frame_figures(both, ROWAN), archive_figures(both, ROWAN)
        assert (frame["last_timestamp"], own["last_timestamp"]) == ("2024-03-10 03:10:00", "2024-03-10 03:30:00")
        # 01:50 to 02:30 on the clock, against 01:50 to 03:10 of true time: 40 minutes on both sides
        assert frame["longest_gap_days"] == pytest.approx(40 / 1440)
        assert own["longest_gap_days"] == pytest.approx(20 / 1440)
    finally:
        both.archive.close()


def test_a_local_time_that_happens_twice_is_read_as_its_first_pass(tmp_path: Path) -> None:
    """On 2024-11-03 the clock shows 01:00 to 01:59 two times. The text of the time stays as written.

    README, "Legacy export bridge": "A local time that occurs two times (the clocks
    went back) is read as its first pass."
    """
    rows = [
        export_row(1, "2024-11-03 00:30:00", "before the hour that repeats"),
        export_row(2, "2024-11-03 01:30:00", "inside it", from_me=True),
        export_row(3, "2024-11-03 02:30:00", "after it"),
    ]
    both = data.load_both(tmp_path, rows)
    try:
        stored = [both.archive.message(data.legacy_id(number)) for number in (1, 2, 3)]
        assert [message["timestamp"] for message in stored] == [row["timestamp"] for row in rows]
        assert stored[1]["sent_at"] - stored[0]["sent_at"] == 3600  # the first pass: one hour after 00:30
        assert stored[2]["sent_at"] - stored[1]["sent_at"] == 2 * 3600  # 01:30 to 02:30 holds the second pass
        frame, own = _gap(both)
        assert frame["longest_gap_days"] == pytest.approx(HOUR_IN_DAYS)  # two gaps of one hour on the clock
        assert own["longest_gap_days"] == pytest.approx(2 * HOUR_IN_DAYS)
        assert (frame["longest_gap_start"], own["longest_gap_start"]) == ("2024-11-03 00:30:00", "2024-11-03 01:30:00")
    finally:
        both.archive.close()
