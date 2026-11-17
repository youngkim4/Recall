"""Tests for three rules that every payload follows: who sent a message is
yes, no or unknown; an unnamed conversation is labelled one way; and a group
is an id made of "chat" and digits."""
import pandas as pd
import pytest

import ui_server
from conversation import group_chat_mask, is_group_chat

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache")

COLUMNS = ["message_id", "timestamp", "sender", "text", "chat_id", "chat_display_name", "is_from_me"]
NOW = pd.Timestamp("2026-11-16 12:00:00")
AVERY = "+15550000001"
BLAKE = "+15550000002"
UNSAVED = "+15550000009"
GROUP = "chat4471"
# one person whose address starts with the letters a group id starts with
CHATTER = "chatterbox@example.com"
NAMES = {AVERY: "Avery Stone", BLAKE: "Blake Reyes"}


def write_export(path, rows):
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["message_id"] = range(1, len(frame) + 1)
    frame.to_csv(path, index=False)
    return path


def row(chat_id, timestamp, text, from_me=0, sender=None):
    return (0, timestamp, sender if sender is not None else chat_id, text, chat_id, "", from_me)


def thread(chat_id, last, count):
    stamps = pd.date_range(end=last, periods=count, freq="min")
    return [row(chat_id, str(stamp), "ok then") for stamp in stamps]


@pytest.fixture
def saved_names(monkeypatch):
    monkeypatch.setattr(
        ui_server,
        "resolve_contact_names",
        lambda handles, **_: {handle: NAMES[handle] for handle in handles if handle in NAMES},
    )


# ---- who sent it ----


@pytest.mark.parametrize("stored, expected", [
    (1, True),
    (0, False),
    ("yes", True),
    ("no", False),
    ("T", True),
    ("maybe", None),
    ("", None),
])
def test_a_found_message_says_who_sent_it_as_yes_no_or_unknown(tmp_path, saved_names, stored, expected):
    csv = write_export(tmp_path / "messages.csv", [row(AVERY, "2025-01-01 09:00:00", "pizza night", from_me=stored)])

    (result,) = ui_server.search_messages_payload(csv, query="pizza")["results"]

    assert result["isFromMe"] is expected


def test_a_found_message_reads_direction_words_from_a_merged_export(tmp_path, saved_names):
    # a column that holds words and numbers is read back as text
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, "2025-01-01 09:00:00", "pizza one", from_me="true"),
        row(AVERY, "2025-01-02 09:00:00", "pizza two", from_me="0"),
        row(AVERY, "2025-01-03 09:00:00", "pizza three", from_me="TRUE"),
    ])

    results = ui_server.search_messages_payload(csv, query="pizza")["results"]

    assert {result["text"]: result["isFromMe"] for result in results} == {
        "pizza one": True, "pizza two": False, "pizza three": True,
    }


def test_your_own_message_in_a_group_carries_no_sender_name(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(GROUP, "2025-01-01 09:00:00", "pizza at mine", sender=AVERY),
        row(GROUP, "2025-01-01 09:01:00", "pizza from me", from_me="yes", sender=AVERY),
    ])

    by_text = {result["text"]: result for result in ui_server.search_messages_payload(csv, query="pizza")["results"]}

    assert by_text["pizza at mine"]["senderName"] == "Avery Stone"
    # a row you sent can carry the handle it went to; that is not who wrote it
    assert by_text["pizza from me"]["senderName"] == ""


def test_the_model_is_told_which_lines_are_yours():
    lines = ui_server.citation_context([
        {"chatId": AVERY, "displayName": "Avery Stone", "isFromMe": True, "text": "on my way", "timestamp": "2025-01-01"},
        {"chatId": AVERY, "displayName": "Avery Stone", "isFromMe": False, "text": "ok", "timestamp": "2025-01-01"},
        {"chatId": AVERY, "displayName": "Avery Stone", "isFromMe": None, "text": "hm", "timestamp": "2025-01-01"},
    ]).splitlines()

    assert lines[0] == "[1] You in Avery Stone (2025-01-01): on my way"
    assert lines[1] == "[2] Avery Stone (2025-01-01): ok"
    assert lines[2] == "[3] Avery Stone (2025-01-01): hm"


def test_the_conversation_around_a_match_marks_your_lines(tmp_path, saved_names):
    frame = pd.DataFrame([
        row(AVERY, "2025-01-01 09:00:00", "are we on for pizza"),
        row(AVERY, "2025-01-01 09:01:00", "yes at eight", from_me="yes"),
        row(AVERY, "2025-01-01 09:02:00", "see you then", from_me="no"),
    ], columns=COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["message_id"] = range(1, len(frame) + 1)

    windows = ui_server.build_context_windows(frame, frame.iloc[[0]])
    text = ui_server.format_context_windows(windows, {AVERY: "Avery Stone"}, {})

    assert "[2025-01-01] You: yes at eight" in text
    assert "[2025-01-01] Avery Stone: see you then" in text


# ---- what an unnamed conversation is called ----


@pytest.mark.parametrize("saved, chat_id, expected", [
    ("Avery Stone", AVERY, "Avery Stone"),
    ("", AVERY, "Number ending in 0001"),
    ("  ", " +1 (555) 000-0001 ", "Number ending in 0001"),
    ("", "sam99@example.com", "sam99@example.com"),
    ("", "72345", "72345"),
    ("", "RECALLCO", "RECALLCO"),
    ("", GROUP, "Unnamed group"),
    ("", CHATTER, CHATTER),
    ("", "", "Unknown conversation"),
    (None, None, "Unknown conversation"),
])
def test_text_the_user_reads_names_a_conversation_the_way_the_interface_does(saved, chat_id, expected):
    # the same cases as conversationName in app/src/lib/conversations.ts
    assert ui_server.conversation_label(saved, chat_id) == expected


def test_a_local_answer_names_conversations_the_way_the_interface_does():
    answer = ui_server.local_ask_answer(["pizza"], [
        {"chatId": "sam99@example.com", "displayName": "", "text": "pizza tonight"},
        {"chatId": "72345", "displayName": "", "text": "your pizza code"},
        {"chatId": UNSAVED, "displayName": "", "text": "pizza?"},
    ])

    assert "- sam99@example.com: pizza tonight" in answer
    assert "- 72345: your pizza code" in answer
    assert "- Number ending in 0009: pizza?" in answer
    assert UNSAVED not in answer


@pytest.mark.parametrize("citation, expected", [
    ({"displayName": "Avery Stone", "chatId": AVERY}, "Avery Stone"),
    ({"chatId": AVERY}, "a number ending in 0001"),
    ({"chatId": "sam99@example.com"}, "an unsaved email contact"),
    ({"chatId": CHATTER}, "an unsaved email contact"),
    ({"chatId": "72345"}, "the short code 72345"),
    ({"chatId": GROUP}, "a group chat"),
    ({"chatId": "RECALLCO"}, "an unsaved contact"),
    ({"chatId": ""}, "an unsaved contact"),
    ({}, "an unsaved contact"),
])
def test_text_the_model_reads_never_holds_a_whole_handle(citation, expected):
    assert ui_server.contact_label(citation) == expected


@pytest.mark.parametrize("label", [ui_server.contact_label, lambda c: ui_server.conversation_label("", c["chatId"])])
def test_a_phone_number_is_never_written_out(label):
    for handle in (AVERY, "15550000001", "(555) 000-0001", "+44 20 7946 0958"):
        digits = "".join(ch for ch in handle if ch.isdigit())

        text = label({"chatId": handle})

        assert digits not in text
        assert digits[-5:] not in text
        assert text.endswith(digits[-4:])


# ---- what a group is ----


@pytest.mark.parametrize("chat_id, expected", [
    ("chat4471", True),
    ("chat123456789012345678", True),
    (CHATTER, False),
    ("chat", False),
    ("chat-a", False),
    ("chat12a", False),
    ("Chat4471", False),
    ("chat１２", False),
    (" chat4471", False),
    ("chat4471\n", False),
    (AVERY, False),
    ("", False),
    (None, False),
    (pd.NA, False),
    (float("nan"), False),
])
def test_a_group_id_is_chat_and_digits(chat_id, expected):
    assert is_group_chat(chat_id) is expected


def test_a_column_of_ids_is_read_the_same_way_as_one_id():
    values = ["chat4471", CHATTER, "chat", "chat-a", "Chat4471", AVERY, "", None, pd.NA, "chat12a", "chat１２"]

    assert group_chat_mask(pd.Series(values, dtype="object")).tolist() == [is_group_chat(value) for value in values]
    assert group_chat_mask(pd.Series(values, dtype="string")).tolist() == [is_group_chat(value) for value in values]


@pytest.fixture
def chatter_export(tmp_path):
    """An archive whose busiest thread is one person with an address that
    starts with "chat", beside a real group."""
    return write_export(tmp_path / "messages.csv", [
        row(CHATTER, "2021-03-01 09:00:00", "hello from the new address"),
        row(CHATTER, "2021-03-01 09:05:00", "got it, saving this one", from_me=1),
        *thread(CHATTER, "2024-01-01 12:00:00", 1100),
        row(GROUP, "2025-06-01 09:00:00", "who is in", sender=AVERY),
        row(GROUP, "2025-06-01 09:01:00", "me", sender=BLAKE),
        *thread(AVERY, "2026-11-01 12:00:00", 5),
    ])


def test_a_person_whose_address_starts_with_chat_is_listed_as_a_person(chatter_export, saved_names):
    contacts = ui_server.contacts_from_messages(chatter_export, name_lookup=lambda handles: {})
    by_id = contacts.set_index("chat_id")

    assert int(by_id.loc[CHATTER, "is_group"]) == 0
    assert int(by_id.loc[GROUP, "is_group"]) == 1
    # only a group is named after the people in it
    assert by_id.loc[CHATTER, "display_name"] == ""
    assert by_id.loc[GROUP, "display_name"] == "Avery Stone and Blake Reyes"


def test_their_name_is_not_made_up_from_who_wrote_there(chatter_export, saved_names):
    _df, _catalog, names = ui_server.cached_messages_bundle(chatter_export)

    assert not names.get(CHATTER)


def test_their_first_words_are_shown(chatter_export, saved_names):
    entries = ui_server.first_words_payload(chatter_export)["entries"]

    assert CHATTER in {entry["chatId"] for entry in entries}
    assert GROUP not in {entry["chatId"] for entry in entries}


def test_they_can_be_someone_to_reconnect_with(chatter_export, saved_names):
    reconnect = ui_server.memories_payload(chatter_export, now=NOW)["reconnect"]

    assert [item["chatId"] for item in reconnect] == [CHATTER]


def test_their_recent_messages_are_not_given_a_group_sender(chatter_export, saved_names):
    recent = ui_server.preview_payload(chatter_export, CHATTER, "gpt-5.4-mini")["recentMessages"]

    assert {message["sender"] for message in recent} == {""}


def test_a_match_in_their_thread_is_not_given_a_group_sender(chatter_export, saved_names, monkeypatch):
    asked = []
    monkeypatch.setattr(
        ui_server, "resolve_contact_names", lambda handles, **_: asked.append(list(handles)) or {},
    )

    (result,) = ui_server.search_messages_payload(chatter_export, query="new address")["results"]

    assert result["chatId"] == CHATTER
    # they are not looked up as someone who wrote in a group
    assert [CHATTER] not in asked


def test_the_model_is_not_told_their_thread_is_a_group():
    lines = ui_server.citation_context([
        {"chatId": CHATTER, "displayName": "", "sender": CHATTER, "isFromMe": False, "text": "hello", "timestamp": "2021-03-01"},
    ]).splitlines()

    assert lines == ["[1] an unsaved email contact (2021-03-01): hello"]
