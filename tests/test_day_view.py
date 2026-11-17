"""Tests for one day of a conversation: the messages of that day, the part of
a long day that is shown, and the days with messages on either side."""
import pandas as pd
import pytest

import ui_server
from day_view import DAY_LIMIT_MAX, day_window
from tests.export_kit import (  # noqa: F401 - archives and contacts_cache are fixtures
    AVERY,
    BLAKE,
    MODEL,
    NAMES,
    archives,
    contacts_cache,
    get,
    loaded,
    row,
    save_names,
)

pytestmark = pytest.mark.usefixtures("archives", "contacts_cache")


def conversation(stamps):
    return pd.DataFrame({"timestamp": pd.to_datetime(stamps), "text": [f"m{index}" for index in range(len(stamps))]})


def minutes(day, count):
    return [str(stamp) for stamp in pd.date_range(start=f"{day} 09:00:00", periods=count, freq="min")]


def test_a_day_holds_the_messages_of_that_day_in_order():
    conv = conversation(["2024-11-15 23:59:00", "2024-11-16 00:00:00", "2024-11-16 23:59:59", "2024-11-17 00:00:00"])

    window = day_window(conv, "2024-11-16")

    assert list(window.rows["text"]) == ["m1", "m2"]
    assert (window.total, window.offset) == (2, 0)


def test_the_days_on_either_side_are_the_nearest_days_with_messages():
    conv = conversation(["2024-11-01 10:00:00", "2024-11-16 10:00:00", "2024-12-25 10:00:00"])

    window = day_window(conv, "2024-11-16")

    assert (window.previous_day, window.next_day) == ("2024-11-01", "2024-12-25")


def test_the_first_and_last_days_have_no_day_beyond_them():
    conv = conversation(["2024-11-01 10:00:00", "2024-11-16 10:00:00"])

    assert day_window(conv, "2024-11-01").previous_day is None
    assert day_window(conv, "2024-11-16").next_day is None


def test_a_day_with_nothing_still_says_where_the_messages_are():
    conv = conversation(["2024-11-01 10:00:00", "2024-12-25 10:00:00"])

    window = day_window(conv, "2024-11-16")

    assert window.total == 0
    assert window.rows.empty
    assert (window.previous_day, window.next_day) == ("2024-11-01", "2024-12-25")


def test_a_long_day_is_shown_around_the_message_asked_for():
    conv = conversation(minutes("2024-11-16", 500))

    # the 300th message of the day, in a window of 100
    window = day_window(conv, "2024-11-16", around="2024-11-16 13:59:00", limit=100)

    assert (window.total, window.offset) == (500, 249)
    assert len(window.rows) == 100
    assert "m299" in set(window.rows["text"])


def test_a_long_day_with_no_message_asked_for_starts_at_its_start():
    conv = conversation(minutes("2024-11-16", 500))

    window = day_window(conv, "2024-11-16", limit=100)

    assert window.offset == 0
    assert list(window.rows["text"])[:2] == ["m0", "m1"]


def test_a_window_around_a_message_never_runs_past_either_end_of_the_day():
    conv = conversation(minutes("2024-11-16", 500))

    near_the_end = day_window(conv, "2024-11-16", around="2024-11-16 17:19:00", limit=100)
    near_the_start = day_window(conv, "2024-11-16", around="2024-11-16 09:02:00", limit=100)

    assert (near_the_end.offset, len(near_the_end.rows)) == (400, 100)
    assert (near_the_start.offset, len(near_the_start.rows)) == (0, 100)


def test_a_place_asked_for_is_used_as_it_is():
    conv = conversation(minutes("2024-11-16", 500))

    # the part is not pulled back to fill a whole window: the rows that follow
    # what the reader already has are the only ones wanted
    near_the_end = day_window(conv, "2024-11-16", offset=480, limit=100)
    before_the_start = day_window(conv, "2024-11-16", offset=-20, limit=100)

    assert (near_the_end.offset, list(near_the_end.rows["text"])) == (480, [f"m{index}" for index in range(480, 500)])
    assert (before_the_start.offset, len(before_the_start.rows)) == (0, 100)


def test_a_place_past_the_end_of_the_day_gives_no_messages():
    conv = conversation(minutes("2024-11-16", 500))

    window = day_window(conv, "2024-11-16", offset=999, limit=100)

    assert (window.total, window.offset, len(window.rows)) == (500, 500, 0)


def test_a_long_day_read_in_parts_gives_every_message_once():
    conv = conversation(minutes("2024-11-16", 500))
    part = 200
    # opened in the middle, then read to both ends the way the day view does
    opened = day_window(conv, "2024-11-16", around="2024-11-16 13:10:00", limit=part)
    shown = list(opened.rows["text"])
    start, end = opened.offset, opened.offset + len(opened.rows)

    while end < opened.total:
        later = day_window(conv, "2024-11-16", offset=end, limit=part)
        assert later.offset == end
        shown = shown + list(later.rows["text"])
        end = later.offset + len(later.rows)
    while start > 0:
        place = max(0, start - part)
        earlier = day_window(conv, "2024-11-16", offset=place, limit=start - place)
        assert (earlier.offset, len(earlier.rows)) == (place, start - place)
        shown = list(earlier.rows["text"]) + shown
        start = earlier.offset

    assert shown == [f"m{index}" for index in range(500)]


@pytest.mark.parametrize("around", ["2024-11-16T13:59:00Z", "2024-11-16T13:59:00+05:00"])
def test_a_time_that_names_a_zone_is_read_as_the_time_on_the_clock(around):
    conv = conversation(minutes("2024-11-16", 500))

    window = day_window(conv, "2024-11-16", around=around, limit=100)

    assert window.offset == 249


@pytest.mark.parametrize("limit", [0, -3])
def test_a_limit_below_one_is_refused(limit):
    with pytest.raises(ValueError):
        day_window(conversation(["2024-11-16 10:00:00"]), "2024-11-16", limit=limit)


def test_no_more_than_the_largest_part_is_given_at_once():
    conv = conversation(minutes("2024-11-16", 500))

    assert len(day_window(conv, "2024-11-16", limit=5000).rows) == DAY_LIMIT_MAX == 400


def test_a_message_with_no_readable_time_is_in_no_day():
    conv = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-11-15 10:00:00", None, "2024-11-16 10:00:00", None]),
        "text": ["the day before", "no time", "that day", "no time either"],
    })

    window = day_window(conv, "2024-11-16")

    assert list(window.rows["text"]) == ["that day"]
    assert (window.total, window.previous_day, window.next_day) == (1, "2024-11-15", None)


def test_a_conversation_that_is_not_in_order_is_given_in_order():
    conv = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-11-16 12:00:00", "2024-11-16 09:00:00", "2024-11-16 10:30:00"]),
        "text": ["noon", "morning", "mid-morning"],
    })

    assert list(day_window(conv, "2024-11-16").rows["text"]) == ["morning", "mid-morning", "noon"]


@pytest.mark.parametrize("day", ["2024-13-45", "yesterday", "", "2024-02-30"])
def test_a_day_that_is_not_a_day_is_refused(day):
    with pytest.raises(ValueError):
        day_window(conversation(["2024-11-16 10:00:00"]), day)


# ---- the route: what the address may hold


def ask_for_day(server, **query):
    return get(server, "/api/day", **query)


@pytest.fixture
def saved_names():
    save_names(NAMES)


@pytest.fixture
def one_day(tmp_path, saved_names):
    return loaded(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-16 09:00:00", "morning", from_me=1),
        row(AVERY, "2024-11-16 09:01:00", "hello back"),
    ])


def test_the_route_gives_a_day(one_day, server):
    status, payload = ask_for_day(server, messagesPath=one_day, contact=AVERY, day="2024-11-16")

    assert status == 200
    assert [message["text"] for message in payload["messages"]] == ["morning", "hello back"]
    assert [message["isFromMe"] for message in payload["messages"]] == [True, False]
    # an id of the archive is text, never a number
    assert [message["messageId"] for message in payload["messages"]] == ["legacy:1", "legacy:2"]


def test_the_route_reads_a_time_that_names_a_zone(one_day, server):
    status, payload = ask_for_day(
        server, messagesPath=one_day, contact=AVERY, day="2024-11-16", around="2024-11-16T09:01:00Z",
    )

    assert (status, payload["total"]) == (200, 2)


@pytest.mark.parametrize("query", [
    {"limit": "0"},
    {"limit": "-3"},
    {"limit": "many"},
    {"offset": "later"},
    {"day": "yesterday"},
    {"day": ""},
    {"contact": ""},
])
def test_the_route_refuses_what_it_cannot_read(one_day, server, query):
    asked = {"messagesPath": one_day, "contact": AVERY, "day": "2024-11-16", **query}

    status, payload = ask_for_day(server, **asked)

    assert status == 400
    assert payload["error"]


def test_the_route_says_when_there_is_no_archive(tmp_path, server):
    absent = tmp_path / "absent.csv"

    status, payload = ask_for_day(server, messagesPath=absent, contact=AVERY, day="2024-11-16")
    preview_status, preview = get(server, "/api/preview", messagesPath=absent, contact=AVERY, model=MODEL)

    assert (status, payload["code"]) == (400, "no_archive")
    assert "no archive" in payload["error"]
    # the same answer the preview gives when there is nothing to read
    assert (preview_status, preview) == (status, payload)


def test_a_moment_says_which_day_it_is_and_when_its_message_was_sent(tmp_path, saved_names, server):
    now = pd.Timestamp("2026-11-16 12:00:00")
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-16 09:00:00", "ok"),
        row(AVERY, "2024-11-16 21:15:00", "a line long enough to be worth quoting back"),
    ])

    (moment,) = ui_server.memories_payload(csv, now=now)["onThisDay"]
    _status, day = ask_for_day(server, messagesPath=csv, contact=AVERY, day=moment["day"], around=moment["previewAt"])

    assert moment["day"] == "2024-11-16"
    assert moment["previewAt"] == "2024-11-16 21:15:00"
    # the time and the id are written as the day writes them: the app finds the quoted message by either
    quoted = [message for message in day["messages"] if message["messageId"] == moment["previewId"]]
    assert [(message["timestamp"], message["text"]) for message in quoted] == [
        (moment["previewAt"], "a line long enough to be worth quoting back"),
    ]


@pytest.mark.parametrize("contact", ["<NA>", "nan", "None"])
def test_messages_that_no_conversation_holds_are_not_served_as_one(tmp_path, saved_names, server, contact):
    """Rows with no conversation id come from different threads. No id, and no
    text that stands for "no value", opens them as one conversation."""
    csv = loaded(tmp_path / "messages.csv", [
        row("", "2024-11-16 09:00:00", "from one thread", sender=AVERY),
        row("", "2024-11-16 09:01:00", "from another", sender=BLAKE),
        row(AVERY, "2024-11-16 09:02:00", "a real conversation"),
    ])

    status, payload = ask_for_day(server, messagesPath=csv, contact=contact, day="2024-11-16")

    assert status == 200
    assert payload["total"] == 0
    assert payload["messages"] == []
    assert (payload["previousDay"], payload["nextDay"]) == (None, None)
