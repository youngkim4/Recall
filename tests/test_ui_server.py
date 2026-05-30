"""Tests for the local Recall UI server helpers."""
import sqlite3

import pandas as pd

import ui_server
from local_store import get_cache_entry as store_get_cache_entry
from local_store import set_cache_entry as store_set_cache_entry
from ui_server import (
    ANALYSIS_CACHE,
    PREVIEW_CACHE,
    ROOT,
    REACT_DIST_DIR,
    UI_DIR,
    ask_messages_payload,
    build_analysis_payload,
    cached_analysis_payload,
    cached_preview_payload,
    companion_report_paths,
    contacts_from_messages,
    ensure_under,
    markdown_section,
    search_messages_payload,
    selected_static_root,
)


def test_ensure_under_accepts_paths_inside_base():
    assert ensure_under(UI_DIR / "index.html", UI_DIR) is True


def test_ensure_under_rejects_prefix_sibling_paths():
    assert ensure_under(ROOT / "ui_evil" / "index.html", UI_DIR) is False


def test_selected_static_root_uses_react_dist_when_built(tmp_path, monkeypatch):
    react_dist = tmp_path / "app" / "dist"
    legacy_ui = tmp_path / "ui"
    react_dist.mkdir(parents=True)
    legacy_ui.mkdir()
    (react_dist / "index.html").write_text("<div id='root'></div>", encoding="utf-8")

    monkeypatch.setattr(ui_server, "REACT_DIST_DIR", react_dist)
    monkeypatch.setattr(ui_server, "UI_DIR", legacy_ui)
    monkeypatch.delenv("RECALL_UI_FRONTEND", raising=False)

    assert selected_static_root() == react_dist


def test_selected_static_root_falls_back_to_legacy_when_react_missing(tmp_path, monkeypatch):
    react_dist = tmp_path / "app" / "dist"
    legacy_ui = tmp_path / "ui"
    legacy_ui.mkdir()

    monkeypatch.setattr(ui_server, "REACT_DIST_DIR", react_dist)
    monkeypatch.setattr(ui_server, "UI_DIR", legacy_ui)
    monkeypatch.delenv("RECALL_UI_FRONTEND", raising=False)

    assert selected_static_root() == legacy_ui


def test_selected_static_root_can_force_legacy(monkeypatch):
    monkeypatch.setenv("RECALL_UI_FRONTEND", "legacy")

    assert selected_static_root() == UI_DIR


def test_contacts_from_messages_uses_export_metadata(tmp_path):
    messages_path = tmp_path / "messages.csv"
    pd.DataFrame({
        "timestamp": ["2025-01-02", "2025-01-01", "2025-02-03"],
        "chat_id": ["+15550001", "+15550001", "chat123"],
        "text": ["two", "one", "group"],
        "is_from_me": [1, 0, 1],
    }).to_csv(messages_path, index=False)

    contacts = contacts_from_messages(messages_path, limit=10)

    first = contacts.iloc[0]
    second = contacts.iloc[1]
    assert first["chat_id"] == "+15550001"
    assert first["message_count"] == 2
    assert str(first["first_msg"]).startswith("2025-01-01")
    assert str(first["last_msg"]).startswith("2025-01-02")
    assert first["is_group"] == 0
    assert first["display_name"] == ""
    assert second["chat_id"] == "chat123"
    assert second["is_group"] == 1


def test_contacts_from_messages_can_add_display_names(tmp_path):
    messages_path = tmp_path / "messages.csv"
    pd.DataFrame({
        "timestamp": ["2025-01-02", "2025-01-01"],
        "chat_id": ["+15550001", "+15550001"],
        "text": ["two", "one"],
        "is_from_me": [1, 0],
    }).to_csv(messages_path, index=False)

    contacts = contacts_from_messages(
        messages_path,
        limit=10,
        name_lookup=lambda handles: {"+15550001": "Avery Stone"},
    )

    assert contacts.iloc[0]["display_name"] == "Avery Stone"


def test_contacts_from_messages_uses_exported_group_names(tmp_path):
    messages_path = tmp_path / "messages.csv"
    pd.DataFrame({
        "timestamp": ["2025-01-02", "2025-01-01", "2025-01-03"],
        "chat_id": ["chat123", "chat123", "+15550001"],
        "chat_display_name": ["Road Trip", "Road Trip", ""],
        "text": ["two", "one", "direct"],
        "is_from_me": [1, 0, 1],
    }).to_csv(messages_path, index=False)

    contacts = contacts_from_messages(messages_path, limit=10)

    group = contacts[contacts["chat_id"] == "chat123"].iloc[0]
    assert group["display_name"] == "Road Trip"
    assert group["is_group"] == 1


def test_contacts_from_messages_fills_group_names_from_database(tmp_path):
    db_path = tmp_path / "chat.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, chat_identifier TEXT, display_name TEXT)")
    conn.execute("INSERT INTO chat VALUES (1, 'chat123', 'Project Crew')")
    conn.commit()
    conn.close()

    messages_path = tmp_path / "messages.csv"
    pd.DataFrame({
        "timestamp": ["2025-01-01"],
        "chat_id": ["chat123"],
        "text": ["group"],
        "is_from_me": [1],
    }).to_csv(messages_path, index=False)

    contacts = contacts_from_messages(messages_path, limit=10, db_path=db_path)

    assert contacts.iloc[0]["display_name"] == "Project Crew"


def test_search_messages_payload_filters_text_and_contact(tmp_path):
    messages_path = tmp_path / "messages.csv"
    pd.DataFrame({
        "message_id": [1, 2, 3],
        "timestamp": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "sender": ["me", "them", "me"],
        "chat_id": ["chat-a", "chat-a", "chat-b"],
        "text": ["lunch plans", "project launch", "launch notes"],
        "is_from_me": [1, 0, 1],
    }).to_csv(messages_path, index=False)

    payload = search_messages_payload(messages_path, query="launch", contact="chat-a")

    assert payload["count"] == 1
    assert payload["results"][0]["text"] == "project launch"
    assert payload["results"][0]["chatId"] == "chat-a"


def test_ask_messages_payload_returns_citations(tmp_path):
    messages_path = tmp_path / "messages.csv"
    pd.DataFrame({
        "message_id": [1, 2, 3],
        "timestamp": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "sender": ["me", "them", "me"],
        "chat_id": ["chat-a", "chat-a", "chat-a"],
        "text": ["we talked about school", "nothing special", "school trip details"],
        "is_from_me": [1, 0, 1],
    }).to_csv(messages_path, index=False)

    payload = ask_messages_payload(messages_path, "what happened with school?", contact="chat-a")

    assert payload["terms"] == ["happened", "school"]
    assert len(payload["citations"]) == 2
    assert payload["citations"][0]["text"] == "school trip details"


def test_markdown_section_extracts_single_section():
    text = "# Report\n\n## Overview\nA\n\n## Summary\nStory\n\nMore story\n\n## Other\nB\n"

    assert markdown_section(text, "Summary") == "Story\n\nMore story"


def test_markdown_section_can_include_nested_headings():
    text = "# Report\n\n## Summary\n## Arc\nStory\n\n## Dynamics\nMore story\n"

    assert markdown_section(text, "Summary", stop_at_next_heading=False) == "## Arc\nStory\n\n## Dynamics\nMore story"


def test_build_analysis_payload_surfaces_report_outputs(tmp_path):
    messages_path = tmp_path / "messages.csv"
    pd.DataFrame({
        "message_id": [1, 2, 3],
        "timestamp": ["2025-01-01", "2025-01-02", "2025-02-01"],
        "sender": ["me", "them", "me"],
        "text": ["hello", "hi", "remember this"],
        "chat_id": ["chat-test", "chat-test", "chat-test"],
        "is_from_me": [1, 0, 1],
        "service": ["iMessage", "iMessage", "iMessage"],
    }).to_csv(messages_path, index=False)

    report_path = tmp_path / "analysis.md"
    report_path.write_text("# Report\n\n## Summary\n## Arc\nA useful story.\n", encoding="utf-8")
    events_path = tmp_path / "events.csv"
    pd.DataFrame({
        "date": ["2025-02-01"],
        "title": ["Remembered something"],
        "detail": ["A detail"],
        "category": ["memory"],
        "score": [8.5],
        "quote": ["remember this"],
    }).to_csv(events_path, index=False)

    payload = build_analysis_payload(
        messages_path,
        "chat-test",
        "gpt-5.5",
        report_path=str(report_path),
        events_path=str(events_path),
    )

    assert payload["summary"] == "## Arc\nA useful story."
    assert payload["stats"]["totalMessages"] == 3
    assert payload["events"][0]["title"] == "Remembered something"
    assert payload["events"][0]["score"] == 8.5
    assert {file["kind"] for file in payload["files"]} == {"md", "csv"}


def test_cached_analysis_payload_reuses_unchanged_report(tmp_path):
    ANALYSIS_CACHE.clear()
    messages_path = tmp_path / "messages.csv"
    pd.DataFrame({
        "message_id": [1, 2],
        "timestamp": ["2025-01-01", "2025-01-02"],
        "sender": ["me", "them"],
        "text": ["hello", "hi"],
        "chat_id": ["chat-test", "chat-test"],
        "is_from_me": [1, 0],
        "service": ["iMessage", "iMessage"],
    }).to_csv(messages_path, index=False)
    report_path = tmp_path / "analysis_chat-test.md"
    report_path.write_text("# Report\n\n## Summary\nFirst summary.\n", encoding="utf-8")

    first, first_cached = cached_analysis_payload(
        messages_path,
        "chat-test",
        "gpt-5.5",
        report_path=str(report_path),
    )
    second, second_cached = cached_analysis_payload(
        messages_path,
        "chat-test",
        "gpt-5.5",
        report_path=str(report_path),
    )

    assert first_cached is False
    assert second_cached is True
    assert second["summary"] == first["summary"]

    report_path.write_text("# Report\n\n## Summary\nSecond summary with more text.\n", encoding="utf-8")
    third, third_cached = cached_analysis_payload(
        messages_path,
        "chat-test",
        "gpt-5.5",
        report_path=str(report_path),
    )

    assert third_cached is False
    assert third["summary"] == "Second summary with more text."


def test_cached_preview_payload_reuses_unchanged_messages(tmp_path):
    PREVIEW_CACHE.clear()
    messages_path = tmp_path / "messages.csv"
    pd.DataFrame({
        "message_id": [1, 2],
        "timestamp": ["2025-01-01", "2025-01-02"],
        "sender": ["me", "them"],
        "text": ["hello", "hi"],
        "chat_id": ["chat-test", "chat-test"],
        "is_from_me": [1, 0],
        "service": ["iMessage", "iMessage"],
    }).to_csv(messages_path, index=False)

    first, first_cached = cached_preview_payload(messages_path, "chat-test", "gpt-5.5")
    second, second_cached = cached_preview_payload(messages_path, "chat-test", "gpt-5.5")

    assert first["stats"]["totalMessages"] == 2
    assert second["stats"]["totalMessages"] == 2
    assert first_cached is False
    assert second_cached is True


def test_cached_analysis_payload_persists_after_memory_cache_clear(tmp_path, monkeypatch):
    ANALYSIS_CACHE.clear()
    store_path = tmp_path / "recall_store.sqlite3"
    monkeypatch.setattr(ui_server, "get_cache_entry", lambda namespace, key: store_get_cache_entry(namespace, key, store_path))
    monkeypatch.setattr(
        ui_server,
        "set_cache_entry",
        lambda namespace, key, value, metadata=None, limit=None: store_set_cache_entry(
            namespace,
            key,
            value,
            metadata=metadata,
            path=store_path,
            limit=limit,
        ),
    )

    messages_path = tmp_path / "messages.csv"
    pd.DataFrame({
        "message_id": [1, 2],
        "timestamp": ["2025-01-01", "2025-01-02"],
        "sender": ["me", "them"],
        "text": ["hello", "hi"],
        "chat_id": ["chat-test", "chat-test"],
        "is_from_me": [1, 0],
        "service": ["iMessage", "iMessage"],
    }).to_csv(messages_path, index=False)
    report_path = tmp_path / "analysis_chat-test.md"
    report_path.write_text("# Report\n\n## Summary\nStored summary.\n", encoding="utf-8")

    first, first_cached = cached_analysis_payload(
        messages_path,
        "chat-test",
        "gpt-5.5",
        report_path=str(report_path),
    )
    ANALYSIS_CACHE.clear()
    second, second_cached = cached_analysis_payload(
        messages_path,
        "chat-test",
        "gpt-5.5",
        report_path=str(report_path),
    )

    assert first_cached is False
    assert second_cached is True
    assert second["summary"] == first["summary"]
    assert store_path.exists()


def test_companion_report_paths_finds_existing_files(tmp_path):
    report_path = tmp_path / "analysis_+15550001.md"
    events_path = tmp_path / "events_timeline_+15550001.csv"
    html_path = tmp_path / "analysis_+15550001.html"
    report_path.write_text("# Report", encoding="utf-8")
    events_path.write_text("date,title\n", encoding="utf-8")
    html_path.write_text("<html></html>", encoding="utf-8")

    assert companion_report_paths(report_path, "+15550001") == (events_path, html_path)
