import pytest

import ui_server
from tests.export_kit import archives, contacts_cache, loaded, row, save_names  # noqa: F401 - two are fixtures

pytestmark = pytest.mark.usefixtures("archives", "contacts_cache")

MOM = "+15550000001"
SAM = "+15550000002"
LEE = "+15550000003"
MOM_EMAIL = "mom@example.com"


def test_first_words_basic(tmp_path):
    save_names({MOM: "Mom", SAM: "Sam Park", LEE: "Lee Min"})
    csv = loaded(tmp_path / "messages.csv", [
        row(MOM, "2014-06-12 09:00:00", "call me when you land, I love you"),
        row(MOM, "2014-06-12 09:05:00", "landed safe, love you too", from_me=1),
        row(MOM, "2024-01-01 10:00:00", "happy new year"),
        row(SAM, "2016-02-01 12:00:00", "yo is this sam?", from_me=1),
        row(SAM, "2016-02-01 12:10:00", "yeah who dis"),
        row(LEE, "2018-03-03 08:00:00", "[Attachment: IMG_1.heic]"),
        row(LEE, "2018-03-03 08:01:00", "https://example.com/x"),
        row(LEE, "2018-03-03 08:02:00", "hey it's lee from class"),
        row("chat100200", "2013-01-01 00:00:00", "group hello", sender="+15550000009"),
    ])
    payload = ui_server.first_words_payload(csv)
    entries = payload["entries"]
    by_person = {e["person"]: e for e in entries}

    assert "chat100200" not in {entry["chatId"] for entry in entries}
    assert by_person["Mom"]["text"].startswith("call me when you land")
    assert by_person["Mom"]["direction"] == "incoming"
    assert by_person["Mom"]["reply"]["text"].startswith("landed safe")
    assert by_person["Mom"]["yearsAgo"] >= 10
    # attachment-only and bare-link lines never count as first words
    assert by_person["Lee Min"]["text"] == "hey it's lee from class"
    assert payload["totals"]["messages"] == 9
    assert payload["totals"]["firstYear"] == 2013


def test_first_words_merges_handles_by_name(tmp_path):
    save_names({MOM: "Mom", MOM_EMAIL: "Mom", SAM: "Sam Park", LEE: "Lee Min"})
    csv = loaded(tmp_path / "messages.csv", [
        row(MOM_EMAIL, "2012-05-01 09:00:00", "testing this email thing"),
        row(MOM, "2014-06-12 09:00:00", "switched to my new phone"),
        row(SAM, "2016-02-01 12:00:00", "yo is this sam?", from_me=1),
        row(LEE, "2018-03-03 08:00:00", "hey it's lee"),
    ])
    entries = ui_server.first_words_payload(csv)["entries"]
    moms = [e for e in entries if e["person"] == "Mom"]
    assert len(moms) == 1
    # the earliest handle's first message wins
    assert moms[0]["text"] == "testing this email thing"
    assert moms[0]["messageCount"] == 2


def test_first_words_unnamed_fallback_masks_labels(tmp_path):
    csv = loaded(tmp_path / "messages.csv", [
        row(MOM, "2020-01-01 09:00:00", "first hello"),
        row(SAM, "2021-01-01 09:00:00", "second hello", from_me=1),
    ])
    entries = ui_server.first_words_payload(csv)["entries"]
    assert entries, "unnamed fallback should still produce entries"
    for entry in entries:
        # someone not in Contacts has no name here; the interface labels them
        # from the id, the same way everywhere, and never with the full number
        assert entry["person"] == ""
        assert entry["chatId"].startswith("+1555000000")


def test_first_words_say_who_wrote_first_and_who_answered(tmp_path):
    save_names({MOM: "Mom", SAM: "Sam Park"})
    csv = loaded(tmp_path / "messages.csv", [
        row(MOM, "2014-06-12 09:00:00", "call me when you land"),
        row(MOM, "2014-06-12 09:05:00", "landed safe, love you", from_me=1),
        row(SAM, "2016-02-01 12:00:00", "yo is this sam?", from_me=1),
        row(SAM, "2016-02-01 12:10:00", "yeah who is this"),
    ])

    by_person = {entry["person"]: entry for entry in ui_server.first_words_payload(csv)["entries"]}

    assert by_person["Mom"]["direction"] == "incoming"
    assert by_person["Mom"]["reply"]["direction"] == "outgoing"
    assert by_person["Mom"]["reply"]["text"] == "landed safe, love you"
    assert by_person["Sam Park"]["direction"] == "outgoing"
    assert by_person["Sam Park"]["reply"]["direction"] == "incoming"


def test_first_words_with_no_archive(tmp_path):
    payload = ui_server.first_words_payload(tmp_path / "absent.csv")
    assert payload == {
        "entries": [],
        "signature": "",
        "totals": {"messages": 0, "people": 0},
    }


def test_the_signature_of_first_words_changes_with_the_archive(tmp_path):
    # the app shows the first words once for one content of the archive
    rows = [row(MOM, "2020-01-01 09:00:00", "first hello")]
    csv = loaded(tmp_path / "messages.csv", rows)
    before = ui_server.first_words_payload(csv)["signature"]

    again = ui_server.first_words_payload(csv)["signature"]
    loaded(csv, [*rows, row(SAM, "2021-01-01 09:00:00", "second hello")])
    after = ui_server.first_words_payload(csv)["signature"]

    assert before and before == again
    assert after and after != before


def test_first_words_count_the_messages_that_the_overview_counts(tmp_path):
    csv = loaded(tmp_path / "messages.csv", [
        row(MOM, "2020-01-01 09:00:00", "first hello"),
        row(MOM, "2020-01-01 09:01:00", "hello back", from_me=1),
        # a message that no conversation holds is shown nowhere, so it is in no count
        row("", "2020-01-02 09:00:00", "from no conversation", sender=SAM),
    ])

    assert ui_server.archive_summary(csv)["messages"] == 2
    assert ui_server.first_words_payload(csv)["totals"]["messages"] == 2


def test_setup_marker_roundtrip(tmp_path, monkeypatch):
    marker_path = tmp_path / "setup.json"
    monkeypatch.setattr(ui_server, "SETUP_MARKER_PATH", marker_path)
    monkeypatch.setattr(ui_server, "SAVES_DIR", tmp_path)
    assert ui_server.read_setup_marker() == {}
    ui_server.write_setup_marker({"completed": True})
    ui_server.write_setup_marker({"firstWordsShown": True})
    assert ui_server.read_setup_marker() == {"completed": True, "firstWordsShown": True}
