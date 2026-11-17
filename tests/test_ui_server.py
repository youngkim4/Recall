"""Tests for the local Recall UI server helpers."""
import pandas as pd
import pytest

import local_store
import ui_server
from tests.export_kit import AVERY, MODEL, archives, contacts_cache, loaded, row  # noqa: F401 - two are fixtures
from ui_server import (
    ANALYSIS_CACHE,
    ROOT,
    UI_DIR,
    build_analysis_payload,
    cached_analysis_payload,
    cached_preview_payload,
    companion_report_paths,
    ensure_under,
    markdown_section,
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


def test_markdown_section_extracts_single_section():
    text = "# Report\n\n## Overview\nA\n\n## Summary\nStory\n\nMore story\n\n## Other\nB\n"

    assert markdown_section(text, "Summary") == "Story\n\nMore story"


def test_markdown_section_can_include_nested_headings():
    text = "# Report\n\n## Summary\n## Arc\nStory\n\n## Dynamics\nMore story\n"

    assert markdown_section(text, "Summary", stop_at_next_heading=False) == "## Arc\nStory\n\n## Dynamics\nMore story"


@pytest.fixture
def two_messages(tmp_path, archives, contacts_cache):
    return loaded(tmp_path / "messages.csv", [
        row(AVERY, "2025-01-01 09:00:00", "hello", from_me=1),
        row(AVERY, "2025-01-02 09:00:00", "hi"),
    ])


def test_build_analysis_payload_surfaces_report_outputs(tmp_path, archives, contacts_cache):
    export = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2025-01-01 09:00:00", "hello", from_me=1),
        row(AVERY, "2025-01-02 09:00:00", "hi"),
        row(AVERY, "2025-02-01 09:00:00", "remember this", from_me=1),
    ])
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
        export,
        AVERY,
        MODEL,
        report_path=str(report_path),
        events_path=str(events_path),
    )

    assert payload["summary"] == "## Arc\nA useful story."
    assert payload["stats"]["totalMessages"] == 3
    assert payload["events"] == [{
        "date": "2025-02-01",
        "title": "Remembered something",
        "detail": "A detail",
        "category": "memory",
        "quote": "remember this",
        "score": 8.5,
    }]
    assert {file["kind"] for file in payload["files"]} == {"md", "csv"}


def test_cached_analysis_payload_reuses_unchanged_report(tmp_path, two_messages):
    report_path = tmp_path / "analysis_+15550000001.md"
    report_path.write_text("# Report\n\n## Summary\nFirst summary.\n", encoding="utf-8")

    first, first_cached = cached_analysis_payload(two_messages, AVERY, MODEL, report_path=str(report_path))
    second, second_cached = cached_analysis_payload(two_messages, AVERY, MODEL, report_path=str(report_path))

    assert first_cached is False
    assert second_cached is True
    assert second["summary"] == first["summary"]

    report_path.write_text("# Report\n\n## Summary\nSecond summary with more text.\n", encoding="utf-8")
    third, third_cached = cached_analysis_payload(two_messages, AVERY, MODEL, report_path=str(report_path))

    assert third_cached is False
    assert third["summary"] == "Second summary with more text."


def test_cached_preview_payload_reuses_unchanged_messages(two_messages):
    first, first_cached = cached_preview_payload(two_messages, AVERY, MODEL)
    second, second_cached = cached_preview_payload(two_messages, AVERY, MODEL)

    assert first["stats"]["totalMessages"] == 2
    assert second["stats"]["totalMessages"] == 2
    assert first_cached is False
    assert second_cached is True


def test_cached_analysis_payload_persists_after_memory_cache_clear(tmp_path, two_messages):
    report_path = tmp_path / "analysis_+15550000001.md"
    report_path.write_text("# Report\n\n## Summary\nStored summary.\n", encoding="utf-8")

    first, first_cached = cached_analysis_payload(two_messages, AVERY, MODEL, report_path=str(report_path))
    ANALYSIS_CACHE.clear()
    second, second_cached = cached_analysis_payload(two_messages, AVERY, MODEL, report_path=str(report_path))

    assert first_cached is False
    assert second_cached is True
    assert second["summary"] == first["summary"]
    # the store of this test (the `archives` fixture), not the one of the app
    assert local_store.DEFAULT_STORE_PATH.exists()


def test_companion_report_paths_finds_existing_files(tmp_path):
    report_path = tmp_path / "analysis_+15550001.md"
    events_path = tmp_path / "events_timeline_+15550001.csv"
    html_path = tmp_path / "analysis_+15550001.html"
    report_path.write_text("# Report", encoding="utf-8")
    events_path.write_text("date,title\n", encoding="utf-8")
    html_path.write_text("<html></html>", encoding="utf-8")

    assert companion_report_paths(report_path, "+15550001") == (events_path, html_path)
