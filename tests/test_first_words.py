import pandas as pd
import pytest

import ui_server


def write_messages_csv(path, rows):
    """Rows name a direction for readability; the export itself stores is_from_me."""
    frame = pd.DataFrame(
        rows,
        columns=["message_id", "chat_id", "timestamp", "direction", "text", "sender"],
    )
    frame = frame.assign(is_from_me=frame["direction"].eq("outgoing").astype(int)).drop(columns="direction")
    frame.to_csv(path, index=False)
    return path


pytestmark = pytest.mark.usefixtures("fresh_bundle_cache")


def test_first_words_basic(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_server, "resolve_contact_names", lambda handles, **_: {
        "+15550000001": "Mom",
        "+15550000002": "Sam Park",
        "+15550000003": "Lee Min",
    })
    csv = write_messages_csv(tmp_path / "messages.csv", [
        ("a1", "+15550000001", "2014-06-12 09:00:00", "incoming", "call me when you land, I love you", "+15550000001"),
        ("a2", "+15550000001", "2014-06-12 09:05:00", "outgoing", "landed safe, love you too", "me"),
        ("a3", "+15550000001", "2024-01-01 10:00:00", "incoming", "happy new year", "+15550000001"),
        ("b1", "+15550000002", "2016-02-01 12:00:00", "outgoing", "yo is this sam?", "me"),
        ("b2", "+15550000002", "2016-02-01 12:10:00", "incoming", "yeah who dis", "+15550000002"),
        ("c1", "+15550000003", "2018-03-03 08:00:00", "incoming", "[Attachment: IMG_1.heic]", "+15550000003"),
        ("c2", "+15550000003", "2018-03-03 08:01:00", "incoming", "https://example.com/x", "+15550000003"),
        ("c3", "+15550000003", "2018-03-03 08:02:00", "incoming", "hey it's lee from class", "+15550000003"),
        ("g1", "chat100200", "2013-01-01 00:00:00", "incoming", "group hello", "+15550000009"),
    ])
    payload = ui_server.first_words_payload(csv)
    entries = payload["entries"]
    by_person = {e["person"]: e for e in entries}

    assert "Group" not in str(entries)
    assert by_person["Mom"]["text"].startswith("call me when you land")
    assert by_person["Mom"]["direction"] == "incoming"
    assert by_person["Mom"]["reply"]["text"].startswith("landed safe")
    assert by_person["Mom"]["yearsAgo"] >= 10
    # attachment-only and bare-link lines never count as first words
    assert by_person["Lee Min"]["text"] == "hey it's lee from class"
    assert payload["totals"]["messages"] == 9
    assert payload["totals"]["firstYear"] == 2013


def test_first_words_merges_handles_by_name(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_server, "resolve_contact_names", lambda handles, **_: {
        "+15550000001": "Mom",
        "mom@icloud.com": "Mom",
        "+15550000002": "Sam Park",
        "+15550000003": "Lee Min",
    })
    csv = write_messages_csv(tmp_path / "messages.csv", [
        ("a1", "mom@icloud.com", "2012-05-01 09:00:00", "incoming", "testing this email thing", "mom@icloud.com"),
        ("a2", "+15550000001", "2014-06-12 09:00:00", "incoming", "switched to my new phone", "+15550000001"),
        ("b1", "+15550000002", "2016-02-01 12:00:00", "outgoing", "yo is this sam?", "me"),
        ("c1", "+15550000003", "2018-03-03 08:00:00", "incoming", "hey it's lee", "+15550000003"),
    ])
    entries = ui_server.first_words_payload(csv)["entries"]
    moms = [e for e in entries if e["person"] == "Mom"]
    assert len(moms) == 1
    # the earliest handle's first message wins
    assert moms[0]["text"] == "testing this email thing"
    assert moms[0]["messageCount"] == 2


def test_first_words_unnamed_fallback_masks_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_server, "resolve_contact_names", lambda handles, **_: {})
    csv = write_messages_csv(tmp_path / "messages.csv", [
        ("a1", "+15550000001", "2020-01-01 09:00:00", "incoming", "first hello", "+15550000001"),
        ("b1", "+15550000002", "2021-01-01 09:00:00", "outgoing", "second hello", "me"),
    ])
    entries = ui_server.first_words_payload(csv)["entries"]
    assert entries, "unnamed fallback should still produce entries"
    for entry in entries:
        # someone not in Contacts has no name here; the interface labels them
        # from the id, the same way everywhere, and never with the full number
        assert entry["person"] == ""
        assert entry["chatId"].startswith("+1555000000")


def test_first_words_reads_direction_from_a_real_export(tmp_path, monkeypatch):
    # a real export has is_from_me and no direction column
    monkeypatch.setattr(ui_server, "resolve_contact_names", lambda handles, **_: {
        "+15550000001": "Mom",
        "+15550000002": "Sam Park",
    })
    frame = pd.DataFrame(
        [
            (1, "2014-06-12 09:00:00", "+15550000001", "call me when you land", "+15550000001", "", 0),
            (2, "2014-06-12 09:05:00", "+15550000001", "landed safe, love you", "+15550000001", "", 1),
            (3, "2016-02-01 12:00:00", "+15550000002", "yo is this sam?", "+15550000002", "", 1),
            (4, "2016-02-01 12:10:00", "+15550000002", "yeah who is this", "+15550000002", "", 0),
        ],
        columns=["message_id", "timestamp", "sender", "text", "chat_id", "chat_display_name", "is_from_me"],
    )
    csv = tmp_path / "messages.csv"
    frame.to_csv(csv, index=False)

    by_person = {entry["person"]: entry for entry in ui_server.first_words_payload(csv)["entries"]}

    assert by_person["Mom"]["direction"] == "incoming"
    assert by_person["Mom"]["reply"]["direction"] == "outgoing"
    assert by_person["Mom"]["reply"]["text"] == "landed safe, love you"
    assert by_person["Sam Park"]["direction"] == "outgoing"
    assert by_person["Sam Park"]["reply"]["direction"] == "incoming"


def test_first_words_missing_csv(tmp_path):
    payload = ui_server.first_words_payload(tmp_path / "absent.csv")
    assert payload == {
        "entries": [],
        "signature": "",
        "totals": {"messages": 0, "people": 0},
    }


def test_setup_marker_roundtrip(tmp_path, monkeypatch):
    marker_path = tmp_path / "setup.json"
    monkeypatch.setattr(ui_server, "SETUP_MARKER_PATH", marker_path)
    monkeypatch.setattr(ui_server, "SAVES_DIR", tmp_path)
    assert ui_server.read_setup_marker() == {}
    ui_server.write_setup_marker({"completed": True})
    ui_server.write_setup_marker({"firstWordsShown": True})
    assert ui_server.read_setup_marker() == {"completed": True, "firstWordsShown": True}
