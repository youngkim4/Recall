"""Tests for the archive summary and the moments the overview shows."""
import pandas as pd
import pytest

import ui_server
from conversation import month_points
from tests.export_kit import (  # noqa: F401 - archives and contacts_cache are fixtures
    AVERY,
    BLAKE,
    NAMES,
    PLACEHOLDER,
    UNSAVED,
    archives,
    contacts_cache,
    get,
    loaded,
    row,
    save_names,
    thread,
)

pytestmark = pytest.mark.usefixtures("archives", "contacts_cache")

NOW = pd.Timestamp("2026-11-16 12:00:00")
PHOTO = "image/heic"


@pytest.fixture
def saved_names():
    save_names(NAMES)


def conversation_list(export, limit=10):
    status, answer = ui_server.contacts_answer({"messagesPath": str(export), "limit": str(limit)})
    assert status == 200, answer
    return {listed["chat_id"]: listed for listed in answer["contacts"]}


def day_of(server, export, contact, day):
    status, answer = get(server, "/api/day", messagesPath=export, contact=contact, day=day)
    assert status == 200, answer
    return answer


def test_archive_summary_counts_every_conversation(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2023-04-10 09:00:00", "first"),
        row(AVERY, "2024-01-01 09:00:00", "second"),
        row(BLAKE, "2024-06-01 09:00:00", "third"),
        row("chat77", "2025-04-03 21:30:00", "last", sender=BLAKE),
    ])

    # the conversation list is capped; the summary is not
    assert len(conversation_list(csv, limit=1)) == 1
    assert ui_server.archive_summary(csv) == {
        "messages": 4,
        "conversations": 3,
        "firstMessage": "2023-04-10T09:00:00",
        "lastMessage": "2025-04-03T21:30:00",
    }


def test_archive_summary_with_no_archive(tmp_path):
    assert ui_server.archive_summary(tmp_path / "absent.csv") == {
        "messages": 0,
        "conversations": 0,
        "firstMessage": None,
        "lastMessage": None,
    }


def test_quiet_is_measured_from_the_end_of_the_archive(tmp_path, saved_names):
    # Blake wrote on the archive's last day. An archive that has not been
    # updated for a year does not make that conversation quiet.
    csv = loaded(
        tmp_path / "messages.csv",
        thread(AVERY, "2024-09-15 20:00:00", 1000) + thread(BLAKE, "2025-04-03 20:00:00", 1000),
    )

    reconnect = ui_server.memories_payload(csv, now=NOW)["reconnect"]

    assert [item["chatId"] for item in reconnect] == [AVERY]
    assert reconnect[0]["quietDays"] == 200
    assert reconnect[0]["lastDate"] == "2024-09-15"
    assert reconnect[0]["name"] == "Avery Stone"


def test_on_this_day_quote_says_who_sent_it(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-16 09:00:00", "morning"),
        row(AVERY, "2024-11-16 09:05:00", "this one is long enough to be the quote", from_me=1),
        row(BLAKE, "2023-11-16 18:00:00", "a message that Blake wrote on this day"),
    ])

    by_chat = {item["chatId"]: item for item in ui_server.memories_payload(csv, now=NOW)["onThisDay"]}

    assert by_chat[AVERY]["preview"] == "this one is long enough to be the quote"
    assert by_chat[AVERY]["previewFromMe"] is True
    assert by_chat[AVERY]["previewSender"] == ""
    assert by_chat[BLAKE]["previewFromMe"] is False
    assert by_chat[BLAKE]["previewSender"] == ""


def test_group_quote_names_the_person_who_wrote_it(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row("chat77", "2024-11-16 09:00:00", "who is bringing the tent this weekend", sender=BLAKE, group_name="Road Trip"),
    ])

    (moment,) = ui_server.memories_payload(csv, now=NOW)["onThisDay"]

    assert moment["name"] == "Road Trip"
    assert moment["previewFromMe"] is False
    assert moment["previewSender"] == "Blake Reyes"


def test_unsaved_conversation_is_sent_without_a_label(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(UNSAVED, "2024-11-16 09:00:00", "a message from a number that is not saved"),
    ])

    (moment,) = ui_server.memories_payload(csv, now=NOW)["onThisDay"]

    # the interface names unsaved conversations; the server never invents a label
    assert moment["name"] == ""
    assert moment["chatId"] == UNSAVED


def test_long_quote_is_cut_at_a_word_and_marked(tmp_path, saved_names):
    text = " ".join(["remember"] * 60)
    csv = loaded(tmp_path / "messages.csv", [row(AVERY, "2024-11-16 09:00:00", text)])

    (moment,) = ui_server.memories_payload(csv, now=NOW)["onThisDay"]
    preview = moment["preview"]

    assert preview.endswith("remember…")
    assert len(preview) <= ui_server.MEMORY_QUOTE_LIMIT + 1
    assert text.startswith(preview[:-1] + " ")


def test_unnamed_group_is_listed_under_its_people(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row("chat77", "2025-01-01 09:00:00", "one", sender=AVERY),
        row("chat77", "2025-01-01 09:01:00", "two", sender=AVERY),
        row("chat77", "2025-01-01 09:02:00", "three", sender=BLAKE),
        row("chat88", "2025-01-02 09:00:00", "named", sender=BLAKE, group_name="Road Trip"),
    ])

    contacts = conversation_list(csv)

    assert contacts["chat77"]["display_name"] == "Avery Stone and Blake Reyes"
    assert contacts["chat88"]["display_name"] == "Road Trip"


def test_conversation_list_counts_what_you_sent(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2025-01-01 09:00:00", "hi"),
        row(AVERY, "2025-01-01 09:01:00", "hello", from_me=1),
        row(AVERY, "2025-01-01 09:02:00", "how are you", from_me=1),
        row(BLAKE, "2025-01-02 09:00:00", "hey"),
    ])

    contacts = conversation_list(csv)

    assert (contacts[AVERY]["message_count"], contacts[AVERY]["sent_count"], contacts[AVERY]["received_count"]) == (3, 2, 1)
    assert (contacts[BLAKE]["sent_count"], contacts[BLAKE]["received_count"]) == (0, 1)


def test_sent_mask_agrees_with_parse_boolish():
    from conversation import parse_boolish, sent_mask

    values = [1, 0, 1.0, 0.0, True, False, "1", "0", "true", "False", " yes ", "n", "", None, float("nan"), "maybe"]
    series = pd.Series(values, dtype="object")

    expected = [(not pd.isna(parse_boolish(value))) and int(parse_boolish(value)) == 1 for value in values]

    assert sent_mask(series).tolist() == expected


def test_the_patterns_of_an_archive_count_both_voices_by_month(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-04 21:10:00", "one"),
        row(AVERY, "2024-11-04 21:40:00", "two", from_me=1),
        row(BLAKE, "2025-01-07 08:05:00", "three"),
    ])

    patterns = ui_server.archive_patterns(csv)

    # a month with nothing in it keeps its place between two that have
    assert patterns["monthly"] == [
        {"key": "2024-11", "total": 2, "sent": 1, "received": 1},
        {"key": "2024-12", "total": 0, "sent": 0, "received": 0},
        {"key": "2025-01", "total": 1, "sent": 0, "received": 1},
    ]


def test_the_patterns_of_an_archive_count_every_day(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2024-12-30 09:00:00", "one"),
        row(BLAKE, "2024-12-30 23:59:00", "two"),
        row(AVERY, "2025-01-01 00:00:00", "three"),
    ])

    assert ui_server.archive_patterns(csv)["daily"] == {"start": "2024-12-30", "counts": [2, 0, 1]}


def test_the_patterns_of_an_archive_count_each_hour_of_each_weekday(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        # 4 November 2024 is a Monday; 10 November is a Sunday
        row(AVERY, "2024-11-04 21:10:00", "one"),
        row(AVERY, "2024-11-04 21:40:00", "two", from_me=1),
        row(BLAKE, "2024-11-10 00:05:00", "three"),
    ])

    rhythm = ui_server.archive_patterns(csv)["rhythm"]

    # seven rows from Monday, twenty-four hours from midnight
    assert [len(day) for day in rhythm] == [24] * 7
    assert rhythm[0][21] == 2
    assert rhythm[6][0] == 1
    assert sum(sum(day) for day in rhythm) == 3


def test_the_patterns_with_no_archive_are_empty(tmp_path):
    assert ui_server.archive_patterns(tmp_path / "absent.csv") == {
        "monthly": [],
        "daily": {"start": "", "counts": []},
        "rhythm": [[0] * 24 for _ in range(7)],
    }


def test_moments_of_an_export_with_no_readable_times_are_empty(tmp_path, saved_names):
    # a row whose time cannot be read is not loaded, so the archive holds no message
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "not a time", "hello"),
        row(AVERY, "", "again"),
    ])

    assert ui_server.archive_summary(csv)["messages"] == 0
    assert ui_server.memories_payload(csv, now=NOW) == {"onThisDay": [], "anniversaries": [], "reconnect": []}


def test_the_months_leave_a_row_that_does_not_say_out_of_both_voices():
    # an archive holds no such row; the figures on a frame still keep it out of both voices
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-11-04 21:10:00", "2024-11-04 21:40:00", "2024-11-05 08:00:00"]),
        "is_from_me": pd.Series([0, 1, ""], dtype="object"),
    })

    assert month_points(frame) == [{"key": "2024-11", "total": 3, "sent": 1, "received": 1}]


NO_CONVERSATION = [
    row(AVERY, "2024-11-04 21:10:00", "one"),
    row(AVERY, "2024-11-04 21:40:00", "two", from_me=1),
    # a message that no conversation holds: it is shown nowhere, so it is in no count
    row("", "2024-11-05 08:00:00", "three", sender=BLAKE),
]


def test_the_summary_leaves_out_a_message_that_no_conversation_holds(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", NO_CONVERSATION)

    assert ui_server.archive_summary(csv) == {
        "messages": 2,
        "conversations": 1,
        "firstMessage": "2024-11-04T21:10:00",
        "lastMessage": "2024-11-04T21:40:00",
    }


def test_the_patterns_add_up_to_the_count_of_the_summary(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", NO_CONVERSATION)

    patterns = ui_server.archive_patterns(csv)
    counted = ui_server.archive_summary(csv)["messages"]

    assert sum(month["total"] for month in patterns["monthly"]) == counted
    assert sum(patterns["daily"]["counts"]) == counted
    assert sum(sum(day) for day in patterns["rhythm"]) == counted


# ---- a moment and the day it opens count the same messages


def test_a_moment_counts_every_message_of_its_day(tmp_path, saved_names, server):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-16 09:00:00", "a line long enough to be worth quoting back"),
        # a picture with no words is a message of that day as well
        row(AVERY, "2024-11-16 09:01:00", PLACEHOLDER, attachments=PHOTO),
        row(AVERY, "2024-11-16 09:02:00", PLACEHOLDER, from_me=1, attachments=PHOTO),
        row(AVERY, "2024-11-16 23:59:59", "late"),
        row(AVERY, "2024-11-17 00:00:00", "the next day"),
        row(BLAKE, "2024-11-16 10:00:00", "another conversation on the same day"),
    ])

    by_chat = {item["chatId"]: item for item in ui_server.memories_payload(csv, now=NOW)["onThisDay"]}
    moment = by_chat[AVERY]
    day = day_of(server, csv, AVERY, moment["day"])

    assert moment["count"] == day["total"] == 4
    assert moment["preview"] == "a line long enough to be worth quoting back"


def test_a_day_with_no_words_is_not_a_moment(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-16 09:00:00", PLACEHOLDER, attachments=PHOTO),
        row(AVERY, "2024-11-16 09:01:00", PLACEHOLDER, attachments=PHOTO),
        row(BLAKE, "2025-03-01 10:00:00", "some other day"),
    ])

    assert ui_server.memories_payload(csv, now=NOW)["onThisDay"] == []


def test_the_busiest_days_come_first_by_every_message_they_hold(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-16 09:00:00", "two messages with words"),
        row(AVERY, "2024-11-16 09:01:00", "and this is the second"),
        row(BLAKE, "2023-11-16 09:00:00", "one message with words"),
        row(BLAKE, "2023-11-16 09:01:00", PLACEHOLDER, attachments=PHOTO),
        row(BLAKE, "2023-11-16 09:02:00", PLACEHOLDER, attachments=PHOTO),
    ])

    moments = ui_server.memories_payload(csv, now=NOW)["onThisDay"]

    assert [(item["chatId"], item["count"]) for item in moments] == [(BLAKE, 3), (AVERY, 2)]


def test_a_moment_names_the_message_it_quotes(tmp_path, saved_names, server):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-16 09:00:00", "ok"),
        row(AVERY, "2024-11-16 21:15:00", "short"),
        row(AVERY, "2024-11-16 21:15:00", "a line long enough to be worth quoting back"),
        row(AVERY, "2024-11-16 21:15:00", "also short"),
    ])

    (moment,) = ui_server.memories_payload(csv, now=NOW)["onThisDay"]
    messages = day_of(server, csv, AVERY, moment["day"])["messages"]

    # three messages share that second, so the time does not say which was quoted
    assert sum(message["timestamp"] == moment["previewAt"] for message in messages) == 3
    assert [message["text"] for message in messages if message["messageId"] == moment["previewId"]] == [
        "a line long enough to be worth quoting back",
    ]


def test_a_moment_with_nothing_to_quote_names_no_message(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [row(AVERY, "2024-11-16 09:00:00", "https://example.com/a")])

    (moment,) = ui_server.memories_payload(csv, now=NOW)["onThisDay"]

    assert (moment["preview"], moment["previewId"], moment["previewAt"]) == ("", "", None)


# ---- anniversaries


def hourly(chat_id, first, count):
    """count messages, one an hour, starting at first."""
    return [row(chat_id, str(stamp), "ok") for stamp in pd.date_range(start=first, periods=count, freq="h")]


def test_an_anniversary_early_in_january_is_seen_from_late_december(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", hourly(AVERY, "2020-01-02 09:00:00", 300))

    (item,) = ui_server.memories_payload(csv, now=pd.Timestamp("2026-12-28 12:00:00"))["anniversaries"]

    assert (item["chatId"], item["date"], item["years"], item["inDays"]) == (AVERY, "2020-01-02", 7, 5)


def test_a_first_anniversary_is_seen_before_the_year_turns(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", hourly(AVERY, "2026-01-02 09:00:00", 300))

    (item,) = ui_server.memories_payload(csv, now=pd.Timestamp("2026-12-28 12:00:00"))["anniversaries"]

    assert (item["years"], item["inDays"]) == (1, 5)


def test_an_anniversary_is_seen_for_a_week_and_on_its_day(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", hourly(AVERY, "2020-11-23 09:00:00", 300))

    def in_days(now):
        return [item["inDays"] for item in ui_server.memories_payload(csv, now=pd.Timestamp(now))["anniversaries"]]

    assert in_days("2026-11-15 12:00:00") == []
    assert in_days("2026-11-16 12:00:00") == [7]
    assert in_days("2026-11-23 23:00:00") == [0]
    # the day after, the next one is a year away
    assert in_days("2026-11-24 00:30:00") == []


def test_a_conversation_that_started_this_year_has_no_anniversary_yet(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", hourly(AVERY, "2026-11-10 09:00:00", 300))

    assert ui_server.memories_payload(csv, now=pd.Timestamp("2026-11-23 12:00:00"))["anniversaries"] == []


# ---- times that cannot be read


def test_a_conversation_with_no_readable_times_does_not_stop_the_moments(tmp_path, saved_names):
    # Avery's first message has its anniversary two days after NOW.
    # The rows of Blake are not loaded: a row whose time cannot be read is no message of the archive.
    readable = thread(AVERY, "2024-11-18 20:00:00", 1000)
    unreadable = [row(BLAKE, "not a time", "ok") for _ in range(1000)]
    csv = loaded(tmp_path / "messages.csv", readable + unreadable)

    memories = ui_server.memories_payload(csv, now=NOW)

    assert [item["chatId"] for item in memories["anniversaries"]] == [AVERY]
    assert memories["reconnect"] == []


def test_messages_that_no_conversation_holds_are_not_a_conversation(tmp_path, saved_names):
    """A row of the export with no conversation id belongs to no conversation.
    Such rows were read as one conversation whose id was the text of "no value"."""
    csv = loaded(tmp_path / "messages.csv", [
        row("", "2024-11-16 09:00:00", "a line long enough to be worth quoting back", sender=BLAKE),
        row("", "2024-11-16 09:01:00", "another line long enough to be quoted back", sender=BLAKE),
        *thread("", "2020-01-01 10:00:00", 1200),
        row(AVERY, "2026-11-10 09:00:00", "hello"),
    ])

    memories = ui_server.memories_payload(csv, now=NOW)

    ids = [moment["chatId"] for moments in memories.values() for moment in moments]
    assert all(cid == AVERY for cid in ids), ids
