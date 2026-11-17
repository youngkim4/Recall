"""Tests for one day of a conversation: the messages of that day, the part of
a long day that is shown, and the days with messages on either side."""
import json
from urllib.parse import urlencode

import pandas as pd
import pytest

import ui_server
from day_view import DAY_LIMIT_MAX, day_window
from tests.loopback import fetch

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache")

COLUMNS = ["message_id", "timestamp", "sender", "text", "chat_id", "chat_display_name", "is_from_me"]
AVERY = "+15550000001"
BLAKE = "+15550000002"
NAMES = {AVERY: "Avery Stone", BLAKE: "Blake Reyes"}


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


# ---- through the server: the same shape every message has elsewhere


def write_export(path, rows):
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["message_id"] = range(1, len(frame) + 1)
    frame.to_csv(path, index=False)
    return path


def row(chat_id, timestamp, text, from_me=0, sender=None):
    return (0, timestamp, sender if sender is not None else chat_id, text, chat_id, "", from_me)


@pytest.fixture
def saved_names(monkeypatch):
    monkeypatch.setattr(
        ui_server,
        "resolve_contact_names",
        lambda handles, **_: {handle: NAMES[handle] for handle in handles if handle in NAMES},
    )


def test_a_day_of_one_conversation_is_given_as_messages(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-15 22:00:00", "the night before"),
        row(AVERY, "2024-11-16 09:00:00", "morning", from_me=1),
        row(AVERY, "2024-11-16 09:01:00", "hello back"),
        row(BLAKE, "2024-11-16 09:02:00", "another conversation"),
        row(AVERY, "2024-11-18 08:00:00", "two days on"),
    ])

    payload = ui_server.day_payload(csv, AVERY, "2024-11-16")

    assert payload["day"] == "2024-11-16"
    assert [message["text"] for message in payload["messages"]] == ["morning", "hello back"]
    assert [message["isFromMe"] for message in payload["messages"]] == [True, False]
    assert (payload["total"], payload["offset"]) == (2, 0)
    assert (payload["previousDay"], payload["nextDay"]) == ("2024-11-15", "2024-11-18")


def test_in_a_group_each_message_names_who_wrote_it(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row("chat77", "2024-11-16 09:00:00", "from Avery", sender=AVERY),
        row("chat77", "2024-11-16 09:01:00", "from someone unsaved", sender="+15550000009"),
    ])

    messages = ui_server.day_payload(csv, "chat77", "2024-11-16")["messages"]

    assert [message["senderName"] for message in messages] == ["Avery Stone", ""]


def test_a_conversation_whose_id_starts_another_is_not_that_other(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [row(AVERY, "2024-11-16 09:00:00", "hello")])

    payload = ui_server.day_payload(csv, AVERY[:-1], "2024-11-16")

    assert (payload["total"], payload["messages"]) == (0, [])


def test_messages_sent_in_the_same_second_keep_the_order_of_the_export(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-16 09:00:00", "first"),
        row(AVERY, "2024-11-16 09:00:00", "second", from_me=1),
        row(AVERY, "2024-11-16 09:00:00", "third"),
    ])

    messages = ui_server.day_payload(csv, AVERY, "2024-11-16")["messages"]

    assert [message["text"] for message in messages] == ["first", "second", "third"]


@pytest.mark.parametrize("limit", [0, -3])
def test_the_server_refuses_a_limit_below_one(tmp_path, saved_names, limit):
    csv = write_export(tmp_path / "messages.csv", [row(AVERY, "2024-11-16 09:00:00", "hello")])

    with pytest.raises(ValueError):
        ui_server.day_payload(csv, AVERY, "2024-11-16", limit=limit)


def test_the_server_gives_the_part_of_a_long_day_that_was_asked_for(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, stamp, f"m{index}") for index, stamp in enumerate(minutes("2024-11-16", 450))
    ])

    payload = ui_server.day_payload(csv, AVERY, "2024-11-16", offset=400, limit=200)

    assert (payload["total"], payload["offset"]) == (450, 400)
    assert [message["text"] for message in payload["messages"]] == [f"m{index}" for index in range(400, 450)]


# ---- the route: what the address may hold


def ask_for_day(server, **query):
    status, _headers, body = fetch(server, f"/api/day?{urlencode(query)}")
    return status, json.loads(body)


@pytest.fixture
def one_day(tmp_path, saved_names):
    return write_export(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-16 09:00:00", "morning", from_me=1),
        row(AVERY, "2024-11-16 09:01:00", "hello back"),
    ])


def test_the_route_gives_a_day(one_day, server):
    status, payload = ask_for_day(server, messagesPath=one_day, contact=AVERY, day="2024-11-16")

    assert status == 200
    assert [message["text"] for message in payload["messages"]] == ["morning", "hello back"]


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


def test_the_route_says_when_the_export_is_not_there(tmp_path, server):
    status, payload = ask_for_day(server, messagesPath=tmp_path / "absent.csv", contact=AVERY, day="2024-11-16")

    # the same answer the preview gives for an export that is gone
    assert status == 400
    assert "not found" in payload["error"]


def test_a_moment_says_which_day_it_is_and_when_its_message_was_sent(tmp_path, saved_names):
    now = pd.Timestamp("2026-11-16 12:00:00")
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-16 09:00:00", "ok"),
        row(AVERY, "2024-11-16 21:15:00", "a line long enough to be worth quoting back"),
    ])

    (moment,) = ui_server.memories_payload(csv, now=now)["onThisDay"]

    assert moment["day"] == "2024-11-16"
    assert moment["previewAt"] == "2024-11-16 21:15:00"
