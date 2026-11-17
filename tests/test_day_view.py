"""Tests for one day of a conversation: the messages of that day, the part of
a long day that is shown, and the days with messages on either side."""
import pandas as pd
import pytest

import ui_server
from day_view import day_window

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


def test_the_window_never_runs_past_either_end_of_the_day():
    conv = conversation(minutes("2024-11-16", 500))

    near_the_end = day_window(conv, "2024-11-16", around="2024-11-16 17:19:00", limit=100)
    by_place = day_window(conv, "2024-11-16", offset=480, limit=100)
    before_the_start = day_window(conv, "2024-11-16", offset=-20, limit=100)

    assert (near_the_end.offset, len(near_the_end.rows)) == (400, 100)
    assert (by_place.offset, len(by_place.rows)) == (400, 100)
    assert before_the_start.offset == 0


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


def test_a_moment_says_which_day_it_is_and_when_its_message_was_sent(tmp_path, saved_names):
    now = pd.Timestamp("2026-11-16 12:00:00")
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-16 09:00:00", "ok"),
        row(AVERY, "2024-11-16 21:15:00", "a line long enough to be worth quoting back"),
    ])

    (moment,) = ui_server.memories_payload(csv, now=now)["onThisDay"]

    assert moment["day"] == "2024-11-16"
    assert moment["previewAt"] == "2024-11-16 21:15:00"
