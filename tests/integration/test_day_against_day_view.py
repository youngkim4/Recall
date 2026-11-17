"""Archive.day() against day_window in day_view.py, on one invented export.

The export is read the two ways the app can read it: by the loader of
conversation.py, which /api/day uses today, and into an archive through the
legacy bridge. Each test asks both for the same day and compares the answers.

Where the two differ on purpose, a test states the difference and names the
rule of archive/README.md.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import day_view
from archive import days as archive_days
from archive.compat import messages_frame
from conversation import in_conversation
from tests.integration import bridge_data as data
from tests.integration import guards
from tests.integration.bridge_data import GROUP, MIRA, NEIGHBOUR, ROWAN, Loaded, export_row

SHORT_DAY = "2024-03-04"
QUIET_DAY = "2024-03-05"
LONG_DAY = "2024-03-06"
TWIN_DAY = "2024-03-08"
LAST_DAY = "2024-03-09"
LONG_DAY_MESSAGES = 450
LONG_DAY_TIMES = data.stamps(f"{LONG_DAY} 08:00:00", LONG_DAY_MESSAGES, step_seconds=97)
PART = 200  # app/src/lib/dayParts.ts: how many messages one press brings


def _rows() -> list[dict[str, Any]]:
    short = [
        export_row(101 + place, stamp, f"short day {place}", from_me=place % 2 == 1)
        for place, stamp in enumerate(data.stamps(f"{SHORT_DAY} 09:00:00", 5))
    ]
    elsewhere = [
        export_row(201, f"{SHORT_DAY} 09:00:30", "another person, same day", NEIGHBOUR),
        export_row(202, f"{QUIET_DAY} 10:00:00", "another person, quiet day", NEIGHBOUR),
        export_row(203, f"{QUIET_DAY} 10:05:00", "and an answer", NEIGHBOUR, from_me=True),
        export_row(204, f"{QUIET_DAY} 10:06:00", "and one more", NEIGHBOUR),
        export_row(211, f"{QUIET_DAY} 11:00:00", "a group, quiet day", GROUP, sender=MIRA),
        export_row(212, f"{LONG_DAY} 11:00:00", "a group, long day", GROUP, from_me=True),
        export_row(221, f"{QUIET_DAY} 12:00:00", "a row with no conversation", "", sender=ROWAN),
        export_row(222, f"{LONG_DAY} 12:00:00", "a sent row with no conversation", "", from_me=True),
    ]
    long = [
        export_row(1001 + place, stamp, f"long day {place}", from_me=place % 3 == 0)
        for place, stamp in enumerate(LONG_DAY_TIMES)
    ]
    twins = [export_row(2001 + place, f"{TWIN_DAY} 10:00:00", f"same second {place}") for place in range(4)]
    twins.append(export_row(2005, f"{TWIN_DAY} 10:00:01", "one second later", from_me=True))
    last = [
        export_row(3001 + place, stamp, f"last day {place}")
        for place, stamp in enumerate(data.stamps(f"{LAST_DAY} 21:00:00", 3))
    ]
    return [*short, *elsewhere, *long, *twins, *last]


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Loaded]:
    # a module fixture is made before the time zone guard of a test: the bridge reads local times
    with guards.time_zone():
        both = data.load_both(tmp_path_factory.mktemp("day"), _rows())
    yield both
    both.archive.close()


@dataclass(frozen=True)
class Part:
    """What one answer says, in the same words for both sides."""

    day: str
    total: int
    offset: int
    ids: tuple[str, ...]
    times: tuple[str, ...]
    previous_day: str | None
    next_day: str | None


def frame_part(loaded: Loaded, chat: str, day: str, **ask: Any) -> Part:
    window = day_view.day_window(loaded.conversation(chat), day, **ask)
    ids, times = data.frame_ids(window.rows), data.frame_times(window.rows)
    return Part(window.day, window.total, window.offset, tuple(ids), tuple(times), window.previous_day, window.next_day)


def archive_part(loaded: Loaded, chat: str, day: str, **ask: Any) -> Part:
    answer = loaded.archive.day(chat, day, **ask)
    times = tuple(message["timestamp"] for message in answer["messages"])
    return Part(
        answer["day"],
        answer["total"],
        answer["offset"],
        tuple(data.archive_ids(answer["messages"])),
        times,
        answer["previous_day"],
        answer["next_day"],
    )


def long_day_ids(first: int, count: int) -> tuple[str, ...]:
    return tuple(data.legacy_id(1001 + place) for place in range(first, min(first + count, LONG_DAY_MESSAGES)))


def test_the_two_modules_use_the_same_limits() -> None:
    assert archive_days.DAY_LIMIT == day_view.DAY_LIMIT == PART
    assert archive_days.DAY_LIMIT_MAX == day_view.DAY_LIMIT_MAX == 400


def test_the_bridge_stored_every_row_of_the_export(loaded: Loaded) -> None:
    assert loaded.result.skipped == {}
    assert loaded.result.messages_added == len(loaded.frame)


def test_a_short_day_is_the_same_both_ways(loaded: Loaded) -> None:
    part = archive_part(loaded, ROWAN, SHORT_DAY)
    assert part == frame_part(loaded, ROWAN, SHORT_DAY)
    assert part.ids == tuple(data.legacy_id(number) for number in range(101, 106))
    assert part.times[0] == f"{SHORT_DAY} 09:00:00"
    assert (part.day, part.total, part.offset) == (SHORT_DAY, 5, 0)


@pytest.mark.parametrize(
    ("day", "previous_day", "next_day"),
    [
        ("2024-01-01", None, SHORT_DAY),
        (SHORT_DAY, None, LONG_DAY),
        (QUIET_DAY, SHORT_DAY, LONG_DAY),
        (LONG_DAY, SHORT_DAY, TWIN_DAY),
        ("2024-03-07", LONG_DAY, TWIN_DAY),
        (TWIN_DAY, LONG_DAY, LAST_DAY),
        (LAST_DAY, TWIN_DAY, None),
        ("2025-01-01", LAST_DAY, None),
    ],
)
def test_the_neighbours_are_the_nearest_days_with_messages(
    loaded: Loaded, day: str, previous_day: str | None, next_day: str | None
) -> None:
    part = archive_part(loaded, ROWAN, day)
    assert part == frame_part(loaded, ROWAN, day)
    assert (part.previous_day, part.next_day) == (previous_day, next_day)


def test_a_day_with_no_message_is_empty_and_keeps_its_neighbours(loaded: Loaded) -> None:
    part = archive_part(loaded, ROWAN, QUIET_DAY)
    assert part == frame_part(loaded, ROWAN, QUIET_DAY)
    assert (part.total, part.offset, part.ids) == (0, 0, ())
    # the day is not empty for the others: an id never matches in part, and a row with no chat is in no day
    assert archive_part(loaded, NEIGHBOUR, QUIET_DAY) == frame_part(loaded, NEIGHBOUR, QUIET_DAY)
    assert archive_part(loaded, NEIGHBOUR, QUIET_DAY).total == 3
    assert archive_part(loaded, GROUP, QUIET_DAY).ids == (data.legacy_id(211),)


@pytest.mark.parametrize("chat", ["+19995550000", ROWAN[:-1], ROWAN + " ", "chat", ""])
def test_an_unknown_conversation_has_no_day(loaded: Loaded, chat: str) -> None:
    part = archive_part(loaded, chat, LONG_DAY)
    assert part == frame_part(loaded, chat, LONG_DAY)
    assert part == Part(LONG_DAY, 0, 0, (), (), None, None)


@pytest.mark.parametrize("day", ["2024-13-40", "2024-3-6", "yesterday", "", "2024-03-06 12:00:00"])
def test_text_that_is_not_a_day_is_refused_by_both(loaded: Loaded, day: str) -> None:
    with pytest.raises(ValueError):
        day_view.day_window(loaded.conversation(ROWAN), day)
    with pytest.raises(ValueError):
        loaded.archive.day(ROWAN, day)


def test_a_long_day_starts_with_its_first_part(loaded: Loaded) -> None:
    part = archive_part(loaded, ROWAN, LONG_DAY)
    assert part == frame_part(loaded, ROWAN, LONG_DAY)
    assert (part.total, part.offset, part.ids) == (LONG_DAY_MESSAGES, 0, long_day_ids(0, PART))


@pytest.mark.parametrize("limit", [1, 50, 399, 400, 401, 1000])
def test_a_limit_above_the_largest_part_is_the_largest_part(loaded: Loaded, limit: int) -> None:
    part = archive_part(loaded, ROWAN, LONG_DAY, limit=limit)
    assert part == frame_part(loaded, ROWAN, LONG_DAY, limit=limit)
    assert len(part.ids) == min(limit, 400)


@pytest.mark.parametrize("limit", [0, -1])
def test_a_limit_below_one_is_refused_by_both(loaded: Loaded, limit: int) -> None:
    with pytest.raises(ValueError):
        day_view.day_window(loaded.conversation(ROWAN), LONG_DAY, limit=limit)
    with pytest.raises(ValueError):
        loaded.archive.day(ROWAN, LONG_DAY, limit=limit)


@pytest.mark.parametrize("offset", [0, 1, 199, 250, 440, 449, 450, 451, 10_000])
def test_a_place_named_with_offset_is_used_as_it_is(loaded: Loaded, offset: int) -> None:
    part = archive_part(loaded, ROWAN, LONG_DAY, offset=offset, limit=PART)
    assert part == frame_part(loaded, ROWAN, LONG_DAY, offset=offset, limit=PART)
    # the answer can hold fewer rows than the limit, and a place past the end gives none
    assert part.offset == min(offset, LONG_DAY_MESSAGES)
    assert part.ids == long_day_ids(offset, PART)


def test_a_negative_offset_is_moved_to_the_start_by_day_window_and_refused_by_the_archive(loaded: Loaded) -> None:
    """A difference: the caller of Archive.day() must not pass a place below 0."""
    assert frame_part(loaded, ROWAN, LONG_DAY, offset=-5).offset == 0
    with pytest.raises(ValueError):
        loaded.archive.day(ROWAN, LONG_DAY, offset=-5)


@pytest.mark.parametrize(
    "around",
    [
        f"{LONG_DAY} 00:00:00",
        f"{LONG_DAY} 07:59:59",
        LONG_DAY_TIMES[0],
        LONG_DAY_TIMES[100],
        LONG_DAY_TIMES[101],
        f"{LONG_DAY} 12:00:00",
        f"{LONG_DAY} 12:00:00.500",
        LONG_DAY_TIMES[-1],
        f"{LONG_DAY} 23:59:59",
        f"{SHORT_DAY} 09:00:00",
        f"{LAST_DAY} 21:00:00",
        "nonsense",
        "",
    ],
)
@pytest.mark.parametrize("limit", [50, PART, 400])
def test_with_no_offset_the_part_is_a_window_around_a_time(loaded: Loaded, around: str, limit: int) -> None:
    part = archive_part(loaded, ROWAN, LONG_DAY, around=around, limit=limit)
    assert part == frame_part(loaded, ROWAN, LONG_DAY, around=around, limit=limit)
    # a whole window that never runs past either end of the day
    assert len(part.ids) == limit
    assert 0 <= part.offset <= LONG_DAY_MESSAGES - limit


def test_the_window_holds_the_message_it_was_asked_around(loaded: Loaded) -> None:
    around = LONG_DAY_TIMES[300]
    part = archive_part(loaded, ROWAN, LONG_DAY, around=around, limit=PART)
    assert part.offset == 300 - PART // 2
    assert data.legacy_id(1301) in part.ids
    last = archive_part(loaded, ROWAN, LONG_DAY, around=LONG_DAY_TIMES[-1], limit=PART)
    assert last.offset == LONG_DAY_MESSAGES - PART


@pytest.mark.parametrize("around", [f"{LONG_DAY}T12:00:00", f"{LONG_DAY} 12:00:00+00:00", "2024/03/06 12:00:00"])
def test_a_time_that_is_not_written_as_the_app_writes_it(loaded: Loaded, around: str) -> None:
    """A difference: day_window reads any spelling of a time. The archive reads "YYYY-MM-DD HH:MM:SS".

    README, "day()": "Text that is not such a time is ignored." The app sends the
    `timestamp` of a message, which has that form.
    """
    assert frame_part(loaded, ROWAN, LONG_DAY, around=around).offset == 49
    assert archive_part(loaded, ROWAN, LONG_DAY, around=around).offset == 0
    assert archive_part(loaded, ROWAN, LONG_DAY, around=f"{LONG_DAY} 12:00:00").offset == 49


def _walk(loaded: Loaded, read: Any, around: str) -> tuple[str, ...]:
    """Read a long day as the reader of the app does: the part around a time, then each part beside it."""
    shown = read(loaded, ROWAN, LONG_DAY, around=around, limit=PART)
    ids = list(shown.ids)
    first, end = shown.offset, shown.offset + len(shown.ids)
    while first > 0:  # partToAsk(shown, earlier=true)
        start = max(0, first - PART)
        part = read(loaded, ROWAN, LONG_DAY, offset=start, limit=first - start)
        assert part.offset == start and len(part.ids) == first - start
        ids, first = [*part.ids, *ids], start
    while end < shown.total:  # partToAsk(shown, earlier=false)
        size = min(PART, shown.total - end)
        part = read(loaded, ROWAN, LONG_DAY, offset=end, limit=size)
        assert part.offset == end and len(part.ids) == size
        ids, end = [*ids, *part.ids], end + size
    return tuple(ids)


@pytest.mark.parametrize("around", [LONG_DAY_TIMES[0], LONG_DAY_TIMES[225], LONG_DAY_TIMES[-1]])
def test_the_parts_of_a_long_day_join_to_the_whole_day(loaded: Loaded, around: str) -> None:
    whole = long_day_ids(0, LONG_DAY_MESSAGES)
    assert _walk(loaded, archive_part, around) == whole
    assert _walk(loaded, frame_part, around) == whole


def test_messages_of_one_second_are_in_the_order_of_the_export(loaded: Loaded) -> None:
    part = archive_part(loaded, ROWAN, TWIN_DAY)
    assert part == frame_part(loaded, ROWAN, TWIN_DAY)
    assert part.ids == tuple(data.legacy_id(number) for number in range(2001, 2006))
    assert part.times[:4] == (f"{TWIN_DAY} 10:00:00",) * 4


def test_an_export_that_lists_one_second_in_another_order(tmp_path: Path) -> None:
    """A difference: inside one second the archive orders by source ROWID, the frame by row of the file.

    README, "messages() and context()": "Messages with the same send time are in
    the order of their source ROWID." The exporter writes rows in ROWID order,
    so the two orders are the same for a file it wrote.
    """
    rows = [export_row(number, f"{SHORT_DAY} 10:00:00", f"row {number}") for number in (12, 11, 13)]
    both = data.load_both(tmp_path, rows)
    try:
        assert frame_part(both, ROWAN, SHORT_DAY).ids == tuple(map(data.legacy_id, (12, 11, 13)))
        assert archive_part(both, ROWAN, SHORT_DAY).ids == tuple(map(data.legacy_id, (11, 12, 13)))
    finally:
        both.archive.close()


def test_a_reaction_phrase_is_a_row_of_the_export_and_no_message_of_the_archive(tmp_path: Path) -> None:
    """A difference: today's day view counts a tapback row as a message. The archive hides it.

    README, "Legacy export bridge": a row whose whole text is a reaction phrase is
    stored as kind `system`: "it is hidden by default and is not counted."
    """
    phrase = f"Loved {data.CURLY_OPEN}the ferry leaves at nine{data.CURLY_CLOSE}"
    rows = [
        export_row(1, f"{SHORT_DAY} 09:00:00", "the ferry leaves at nine"),
        export_row(2, f"{SHORT_DAY} 09:01:00", phrase, from_me=True),
        export_row(3, f"{SHORT_DAY} 09:02:00", data.PLACEHOLDER, attachment_types="image/jpeg"),
    ]
    both = data.load_both(tmp_path, rows)
    try:
        assert frame_part(both, ROWAN, SHORT_DAY).ids == tuple(map(data.legacy_id, (1, 2, 3)))
        hidden = archive_part(both, ROWAN, SHORT_DAY)
        assert (hidden.total, hidden.ids) == (2, tuple(map(data.legacy_id, (1, 3))))
        shown = both.archive.day(ROWAN, SHORT_DAY, include_system=True)
        assert data.archive_ids(shown["messages"]) == list(map(data.legacy_id, (1, 2, 3)))
        assert [message["kind"] for message in shown["messages"]] == ["text", "system", "attachment"]
    finally:
        both.archive.close()


@pytest.mark.parametrize("day", [SHORT_DAY, QUIET_DAY, LONG_DAY, TWIN_DAY, LAST_DAY])
@pytest.mark.parametrize("chat", [ROWAN, NEIGHBOUR, GROUP])
def test_day_window_on_the_frame_of_the_archive_gives_the_day_of_the_archive(
    loaded: Loaded, chat: str, day: str
) -> None:
    """After the hook-up the code on pandas reads archive.compat.messages_frame: the two then agree."""
    frame = messages_frame(loaded.archive)
    window = day_view.day_window(frame[in_conversation(frame, chat)], day, around=f"{day} 12:00:00", limit=PART)
    answer = loaded.archive.day(chat, day, around=f"{day} 12:00:00", limit=PART)
    assert list(window.rows["message_id"]) == data.archive_ids(answer["messages"])
    assert (window.total, window.offset) == (answer["total"], answer["offset"])
    assert (window.previous_day, window.next_day) == (answer["previous_day"], answer["next_day"])
