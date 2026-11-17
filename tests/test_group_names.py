"""Tests for naming unnamed group chats after the people in them."""
import pandas as pd

from group_names import derive_group_names, join_names

NAMES = {
    "+15550000001": "Avery Stone",
    "+15550000002": "Blake Reyes",
    "+15550000003": "Casey Lin",
    "avery@example.com": "Avery Stone",
}


def lookup(handles):
    return {handle: NAMES[handle] for handle in handles if handle in NAMES}


def frame(rows):
    return pd.DataFrame(rows, columns=["chat_id", "sender", "is_from_me"])


def inbound(chat_id, sender, times):
    return [(chat_id, sender, 0)] * times


def test_join_names_reads_as_a_sentence_list():
    assert join_names(["Avery"]) == "Avery"
    assert join_names(["Avery", "Blake"]) == "Avery and Blake"
    assert join_names(["Avery", "Blake", "Casey"]) == "Avery, Blake and Casey"


def test_large_group_names_its_two_busiest_people_and_counts_the_rest():
    df = frame(
        inbound("chat1", "+15550000002", 2)
        + inbound("chat1", "+15550000001", 3)
        + inbound("chat1", "+15550000003", 1)
        + inbound("chat1", "+15550000009", 1)
    )

    assert derive_group_names(df, ["chat1"], lookup) == {"chat1": "Avery Stone, Blake Reyes and 2 others"}


def test_small_group_names_everyone():
    df = frame(
        inbound("chat1", "+15550000001", 3)
        + inbound("chat1", "+15550000002", 2)
        + inbound("chat1", "+15550000003", 1)
        + inbound("chat2", "+15550000002", 5)
        + inbound("chat2", "+15550000003", 1)
    )

    assert derive_group_names(df, ["chat1", "chat2"], lookup) == {
        "chat1": "Avery Stone, Blake Reyes and Casey Lin",
        "chat2": "Blake Reyes and Casey Lin",
    }


def test_unsaved_people_are_counted_but_not_named():
    df = frame(inbound("chat1", "+15550000001", 3) + inbound("chat1", "+15550000009", 9))

    assert derive_group_names(df, ["chat1"], lookup) == {"chat1": "Avery Stone and 1 other"}


def test_one_voice_still_reads_as_a_group():
    df = frame(inbound("chat1", "+15550000001", 4))

    assert derive_group_names(df, ["chat1"], lookup) == {"chat1": "Avery Stone and others"}


def test_group_with_no_saved_people_gets_no_name():
    df = frame(inbound("chat1", "+15550000008", 4) + inbound("chat1", "+15550000009", 2))

    assert derive_group_names(df, ["chat1"], lookup) == {}


def test_one_person_with_two_handles_counts_once():
    df = frame(
        inbound("chat1", "+15550000001", 2)
        + inbound("chat1", "avery@example.com", 2)
        + inbound("chat1", "+15550000002", 1)
    )

    assert derive_group_names(df, ["chat1"], lookup) == {"chat1": "Avery Stone and Blake Reyes"}


def test_messages_you_sent_do_not_add_a_person():
    # an outbound row carries the handle it was sent to, not who wrote it
    df = frame(inbound("chat1", "+15550000001", 2) + [("chat1", "+15550000002", 1), ("chat1", None, 1)])

    assert derive_group_names(df, ["chat1"], lookup) == {"chat1": "Avery Stone and others"}


def test_only_the_requested_groups_are_named():
    df = frame(inbound("chat1", "+15550000001", 2) + inbound("chat2", "+15550000002", 2))

    assert derive_group_names(df, ["chat2"], lookup) == {"chat2": "Blake Reyes and others"}
    assert derive_group_names(df, [], lookup) == {}
