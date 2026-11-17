"""Tests for three rules that every payload follows: who sent a message is
yes, no or unknown; an unnamed conversation is labelled one way; and a group
is an id made of "chat" and digits."""
import numpy as np
import pandas as pd
import pytest

import ui_server
from conversation import (
    contact_label,
    conversation_label,
    count_directions,
    direction_label,
    group_chat_mask,
    is_group_chat,
    received_mask,
    sent_mask,
)
from tests.export_kit import (  # noqa: F401 - archives and contacts_cache are fixtures
    AVERY,
    BLAKE,
    GROUP as SMALL_GROUP,
    MODEL,
    NAMES,
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
GROUP = "chat4471"
# one person whose address starts with the letters a group id starts with
CHATTER = "chatterbox@example.com"


@pytest.fixture
def saved_names():
    save_names(NAMES)


def found(server, export, **query):
    status, answer = get(server, "/api/search", messagesPath=export, **query)
    assert status == 200, answer
    return answer["results"]


# ---- who sent it ----


def test_a_found_message_reads_direction_words_from_a_merged_export(tmp_path, saved_names, server):
    # a column that holds words and numbers is read back as text
    export = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2025-01-01 09:00:00", "pizza one", from_me="true"),
        row(AVERY, "2025-01-02 09:00:00", "pizza two", from_me="0"),
        row(AVERY, "2025-01-03 09:00:00", "pizza three", from_me="TRUE"),
    ])

    results = found(server, export, query="pizza")

    assert {result["text"]: result["isFromMe"] for result in results} == {
        "pizza one": True, "pizza two": False, "pizza three": True,
    }


def test_your_own_message_in_a_group_carries_no_sender_name(tmp_path, saved_names, server):
    export = loaded(tmp_path / "messages.csv", [
        row(SMALL_GROUP, "2025-01-01 09:00:00", "pizza at mine", sender=AVERY),
        row(SMALL_GROUP, "2025-01-01 09:01:00", "pizza from me", from_me="1", sender=AVERY),
    ])

    by_text = {result["text"]: result for result in found(server, export, query="pizza")}

    assert by_text["pizza at mine"]["senderName"] == "Avery Stone"
    # a row you sent can carry the handle it went to; that is not who wrote it
    assert (by_text["pizza from me"]["sender"], by_text["pizza from me"]["senderName"]) == ("", "")
    assert by_text["pizza from me"]["isFromMe"] is True


# every way a direction can be written, and what it says
DIRECTIONS = [
    (1, True), (0, False), (True, True), (False, False), (1.0, True), (0.0, False),
    ("1", True), ("0", False), ("y", True), ("N", False), (" Yes ", True), ("false", False),
    (np.int64(1), True), (np.int64(0), False), (np.float64(1.0), True),
    # a value that is neither says nothing: it is not a message received
    (2, None), (0.5, None), (-1, None), (np.int64(2), None), (np.float64(0.5), None),
    ("2", None), ("maybe", None), ("", None), (None, None), (float("nan"), None), (pd.NA, None), (-0.0, None),
    (np.bool_(True), True), (np.bool_(False), False),
]


@pytest.mark.parametrize("value, sent", DIRECTIONS)
def test_one_value_says_what_a_column_of_them_says(value, sent):
    column = pd.Series([value], dtype="object")

    assert ui_server.sent_by_me(value) is sent
    assert bool(sent_mask(column).iloc[0]) is (sent is True)
    assert bool(received_mask(column).iloc[0]) is (sent is False)


def test_a_column_of_every_way_is_counted_one_way():
    column = pd.Series([value for value, _sent in DIRECTIONS], dtype="object")
    said = [sent for _value, sent in DIRECTIONS]

    assert count_directions(column) == (said.count(True), said.count(False))
    assert int(sent_mask(column).sum()) == said.count(True)
    assert int(received_mask(column).sum()) == said.count(False)
    assert [direction_label(value) for value in (2, 0.5, -1)] == ["??", "??", "??"]


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
    assert conversation_label(saved, chat_id) == expected


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
    assert contact_label(citation) == expected


@pytest.mark.parametrize("label", [contact_label, lambda c: conversation_label("", c["chatId"])])
def test_a_phone_number_is_never_written_out(label):
    for handle in (AVERY, "15550000001", "(555) 000-0001", "+44 20 7946 0958"):
        digits = "".join(ch for ch in handle if ch.isdigit())

        text = label({"chatId": handle})

        assert digits not in text
        assert digits[-5:] not in text
        assert text.endswith(digits[-4:])


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA, "", "  "])
def test_a_name_or_an_id_that_is_missing_is_none_at_all(missing):
    assert conversation_label(missing, AVERY) == "Number ending in 0001"
    assert contact_label({"displayName": missing, "chatId": AVERY}) == "a number ending in 0001"
    assert conversation_label("Avery Stone", missing) == "Avery Stone"
    assert conversation_label(missing, missing) == "Unknown conversation"
    assert contact_label({"displayName": missing, "chatId": missing}) == "an unsaved contact"


def test_a_report_of_the_command_line_labels_with_the_same_two_functions():
    import analysis
    import conversation

    assert conversation.contact_label is analysis.contact_label
    assert conversation.conversation_label is analysis.conversation_label


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
    return loaded(tmp_path / "messages.csv", [
        row(CHATTER, "2021-03-01 09:00:00", "hello from the new address"),
        row(CHATTER, "2021-03-01 09:05:00", "got it, saving this one", from_me=1),
        *thread(CHATTER, "2024-01-01 12:00:00", 1100, text="ok then"),
        row(GROUP, "2025-06-01 09:00:00", "who is in", sender=AVERY),
        row(GROUP, "2025-06-01 09:01:00", "me", sender=BLAKE),
        *thread(AVERY, "2026-11-01 12:00:00", 5, text="ok then"),
    ])


def test_a_person_whose_address_starts_with_chat_is_listed_as_a_person(chatter_export, saved_names):
    status, answer = ui_server.contacts_answer({"messagesPath": str(chatter_export)})
    by_id = {listed["chat_id"]: listed for listed in answer["contacts"]}

    assert status == 200
    assert by_id[CHATTER]["is_group"] == 0
    assert by_id[GROUP]["is_group"] == 1
    # only a group is named after the people in it
    assert by_id[CHATTER]["display_name"] == ""
    assert by_id[GROUP]["display_name"] == "Avery Stone and Blake Reyes"


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
    recent = ui_server.preview_payload(chatter_export, CHATTER, MODEL)["recentMessages"]

    assert recent
    assert {message["sender"] for message in recent} == {""}


def test_a_match_in_their_thread_is_not_given_a_group_sender(chatter_export, server):
    # also when their address is saved: the name of who wrote is for a group only
    save_names({**NAMES, CHATTER: "Casey Hatter"})

    (result,) = found(server, chatter_export, query="new address")

    assert (result["chatId"], result["displayName"]) == (CHATTER, "Casey Hatter")
    assert result["senderName"] == ""
