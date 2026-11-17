"""Tests for searching the archive for exact words."""
import pandas as pd
import pytest

import ui_server
from ui_server import search_messages_payload

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache")

COLUMNS = ["message_id", "timestamp", "sender", "text", "chat_id", "chat_display_name", "is_from_me"]
AVERY = "+15550000001"
BLAKE = "+15550000002"
UNSAVED = "+15550000009"
GROUP = "chat77"
NAMES = {AVERY: "Avery Stone", BLAKE: "Blake Reyes"}


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


def test_search_returns_the_newest_matches_and_counts_them_all(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, f"2025-01-0{day} 09:00:00", f"pizza night {day}") for day in range(1, 6)
    ] + [row(AVERY, "2025-01-09 09:00:00", "something else")])

    payload = search_messages_payload(csv, query="pizza", limit=2)

    assert [result["text"] for result in payload["results"]] == ["pizza night 5", "pizza night 4"]
    assert payload["count"] == 2
    # how many there are in all, so the list can say when it is not showing everything
    assert payload["total"] == 5


def test_a_search_with_no_matches_counts_none(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [row(AVERY, "2025-01-01 09:00:00", "hello")])

    assert search_messages_payload(csv, query="pizza") == {
        "query": "pizza", "contact": "", "results": [], "count": 0, "total": 0,
    }


def test_a_match_in_a_group_names_the_group_and_who_wrote_it(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(GROUP, "2025-01-01 09:00:00", "pizza at mine", sender=AVERY),
        row(GROUP, "2025-01-01 09:01:00", "pizza sounds good", sender=UNSAVED),
        row(GROUP, "2025-01-01 09:02:00", "bringing pizza", from_me=1, sender=""),
        row(GROUP, "2025-01-01 09:03:00", "ok", sender=BLAKE),
    ])

    results = search_messages_payload(csv, query="pizza")["results"]

    # the group goes by the name the conversation list gives it
    assert {result["displayName"] for result in results} == {"Avery Stone, Blake Reyes and 1 other"}
    by_text = {result["text"]: result for result in results}
    assert by_text["pizza at mine"]["senderName"] == "Avery Stone"
    # someone not in Contacts has no name here; the interface labels them
    assert by_text["pizza sounds good"]["senderName"] == ""
    assert by_text["bringing pizza"]["senderName"] == ""


def test_browsing_with_no_words_counts_the_whole_conversation(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, f"2025-01-0{day} 09:00:00", f"message {day}") for day in range(1, 6)
    ])

    payload = search_messages_payload(csv, contact=AVERY, limit=2)

    assert [result["text"] for result in payload["results"]] == ["message 5", "message 4"]
    assert payload["count"] == 2
    assert payload["total"] == 5
