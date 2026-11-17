"""Tests that a name saved in Contacts shows everywhere without a new export."""
import json

import pandas as pd
import pytest

import ui_server
from local_store import get_cache_entry as store_get_cache_entry
from local_store import set_cache_entry as store_set_cache_entry

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache")

COLUMNS = ["message_id", "timestamp", "sender", "text", "chat_id", "chat_display_name", "is_from_me"]
AVERY = "+15550000001"
BLAKE = "+15550000002"
GROUP = "chat77"
MODEL = "gpt-5.4-mini"


def row(chat_id, timestamp, text, from_me=0, sender=None):
    return (0, timestamp, sender if sender is not None else chat_id, text, chat_id, "", from_me)


@pytest.fixture
def export(tmp_path):
    frame = pd.DataFrame([
        row(AVERY, "2024-11-16 09:00:00", "see you at the lake house"),
        row(AVERY, "2024-11-16 09:01:00", "on my way", from_me=1),
        row(GROUP, "2025-01-01 09:00:00", "who is driving", sender=AVERY),
        row(GROUP, "2025-01-01 09:01:00", "not me", sender=BLAKE),
        row(GROUP, "2025-01-01 09:02:00", "I can", from_me=1, sender=""),
    ], columns=COLUMNS)
    frame["message_id"] = range(1, len(frame) + 1)
    path = tmp_path / "messages.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def contacts(tmp_path, monkeypatch):
    """The Contacts cache file and the names in it. Calling it saves names, the
    way loading Contacts rewrites the file."""
    cache = tmp_path / "contact_names.json"
    saved: dict[str, str] = {}

    def save(names: dict[str, str]):
        saved.update(names)
        cache.write_text(json.dumps(saved), encoding="utf-8")

    monkeypatch.setattr(ui_server, "CONTACTS_CACHE_PATH", cache)
    monkeypatch.setattr(
        ui_server,
        "resolve_contact_names",
        lambda handles, **_: {handle: saved[handle] for handle in handles if handle in saved},
    )
    return save


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Stored payloads of this test only, and nothing kept in memory from another."""
    path = tmp_path / "recall_store.sqlite3"
    monkeypatch.setattr(ui_server, "get_cache_entry", lambda namespace, key: store_get_cache_entry(namespace, key, path))
    monkeypatch.setattr(
        ui_server,
        "set_cache_entry",
        lambda namespace, key, value, metadata=None, limit=None: store_set_cache_entry(
            namespace, key, value, metadata=metadata, path=path, limit=limit,
        ),
    )
    ui_server.PREVIEW_CACHE.clear()
    ui_server.ANALYSIS_CACHE.clear()
    yield path
    ui_server.PREVIEW_CACHE.clear()
    ui_server.ANALYSIS_CACHE.clear()


def test_a_saved_name_reaches_the_archive_without_a_new_export(export, contacts):
    before_df, _catalog, before = ui_server.cached_messages_bundle(export)
    assert before.get(AVERY) is None

    contacts({AVERY: "Avery Stone"})
    after_df, _catalog, after = ui_server.cached_messages_bundle(export)

    assert after[AVERY] == "Avery Stone"
    # only the names were worked out again; the export was not read a second time
    assert after_df is before_df


def test_the_archive_is_kept_while_neither_file_changes(export, contacts):
    contacts({AVERY: "Avery Stone"})

    first = ui_server.cached_messages_bundle(export)
    second = ui_server.cached_messages_bundle(export)

    assert all(a is b for a, b in zip(first, second))


def test_a_new_export_is_read_again(export, contacts):
    before_df, _catalog, _names = ui_server.cached_messages_bundle(export)

    frame = pd.read_csv(export, dtype={"chat_id": "string", "sender": "string"})
    extra = pd.DataFrame([row(BLAKE, "2025-02-01 08:00:00", "new here")], columns=COLUMNS)
    pd.concat([frame, extra]).to_csv(export, index=False)
    after_df, catalog, _names = ui_server.cached_messages_bundle(export)

    assert after_df is not before_df
    assert BLAKE in set(catalog["chat_id"])


def test_a_group_named_after_its_people_follows_their_names(export, contacts):
    contacts({AVERY: "Avery Stone"})
    _df, _catalog, before = ui_server.cached_messages_bundle(export)

    contacts({BLAKE: "Blake Reyes"})
    _df, _catalog, after = ui_server.cached_messages_bundle(export)

    assert before.get(GROUP) != after.get(GROUP)
    assert after[GROUP] == "Avery Stone and Blake Reyes"


def test_search_shows_the_name_as_soon_as_it_is_saved(export, contacts):
    before = ui_server.search_messages_payload(export, query="lake house")
    contacts({AVERY: "Avery Stone"})
    after = ui_server.search_messages_payload(export, query="lake house")

    assert before["results"][0]["displayName"] == ""
    assert after["results"][0]["displayName"] == "Avery Stone"


def test_moments_show_the_name_as_soon_as_it_is_saved(export, contacts):
    now = pd.Timestamp("2026-11-16 12:00:00")

    before = ui_server.memories_payload(export, now=now)
    contacts({AVERY: "Avery Stone"})
    after = ui_server.memories_payload(export, now=now)

    assert [moment["name"] for moment in before["onThisDay"]] == [""]
    assert [moment["name"] for moment in after["onThisDay"]] == ["Avery Stone"]


def test_a_preview_stored_before_the_name_was_saved_is_not_served(export, contacts, store):
    def speaker_names(payload):
        return {speaker["handle"]: speaker["name"] for speaker in payload["dynamics"]["speakers"]}

    first, first_cached = ui_server.cached_preview_payload(export, GROUP, MODEL)
    again, again_cached = ui_server.cached_preview_payload(export, GROUP, MODEL)
    contacts({BLAKE: "Blake Reyes"})
    after, after_cached = ui_server.cached_preview_payload(export, GROUP, MODEL)

    assert (first_cached, again_cached, after_cached) == (False, True, False)
    assert speaker_names(first)[BLAKE] == ""
    assert speaker_names(after)[BLAKE] == "Blake Reyes"
    assert [message["senderName"] for message in after["recentMessages"]] == ["", "Blake Reyes", ""]


def test_an_opened_report_shows_the_name_as_soon_as_it_is_saved(export, contacts, store, tmp_path):
    report = tmp_path / "analysis_chat77.md"
    report.write_text("# Conversation Analysis - chat77\n\n## Summary\nA story.\n", encoding="utf-8")

    def opened():
        payload, _cached = ui_server.cached_analysis_payload(export, GROUP, MODEL, report_path=str(report))
        return payload

    before = opened()
    contacts({AVERY: "Avery Stone", BLAKE: "Blake Reyes"})
    after = opened()

    assert before["contactDisplayName"] == ""
    assert after["contactDisplayName"] == "Avery Stone and Blake Reyes"
    assert {speaker["name"] for speaker in after["dynamics"]["speakers"]} == {"Avery Stone", "Blake Reyes"}
