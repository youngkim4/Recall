"""The server against the services that it calls.

`ui_server` is imported here and in no service module. It is imported when
this file is collected, as the base tests do: imported inside a test, it would
keep the output stream of that one test.

Each test loads one invented export into its archive. It asks a route of the
server on a loopback port, and it asks the service for the same thing. The
route must answer with the payload of the service: with the names of the
Contacts cache, with what the address holds, and for a search with the last
word as a prefix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import ui_server
from archive import Archive
from recall_services import conversations, day, messages, places, reports, search
from recall_services.names import Names
from tests.export_kit import archives, contacts_cache, get, loaded, row, save_names  # noqa: F401 - two are fixtures

pytestmark = pytest.mark.usefixtures("archives", "contacts_cache")

AVERY = "+15550180001"
BLAKE = "+15550180002"
UNSAVED = "+15550189999"
GROUP = "chat880011"
TITLED = "chat880022"
SAVED = {AVERY: "Avery Stone", BLAKE: "Blake Reyes"}
PHOTO = "￼"
LONG_TEXT = "pizza " + " ".join(["oven"] * 400)

ROWS = [
    row(AVERY, "2025-01-01 09:00:00", "pizza night at mine"),
    # the old exporter wrote the handle of the other person on an own message of a one-to-one thread
    row(AVERY, "2025-01-01 09:05:00", "bringing the pizza stone", from_me=1),
    row(AVERY, "2025-01-01 09:06:00", PHOTO, attachments="image/heic"),
    row(AVERY, "2025-01-02 19:30:00", "  that was a good pizza  "),
    row(AVERY, "2025-01-04 08:00:00", "back to salad"),
    row(UNSAVED, "2025-01-03 10:00:00", "your pizza order is ready"),
    row(GROUP, "2025-01-05 12:00:00", "pizza on friday?", sender=AVERY),
    row(GROUP, "2025-01-05 12:01:00", "yes to pizza", sender=BLAKE),
    row(GROUP, "2025-01-05 12:02:00", "count me in for pizza", sender=UNSAVED),
    row(GROUP, "2025-01-05 12:03:00", "booking a table", from_me=1, sender=""),
    row(TITLED, "2025-01-06 12:00:00", "minutes of the meeting", sender=BLAKE, group_name="Allotment Board"),
    row(TITLED, "2025-01-07 12:00:00", LONG_TEXT, sender=BLAKE, group_name="Allotment Board"),
]


@pytest.fixture
def export(tmp_path: Path) -> Path:
    return loaded(tmp_path / "messages.csv", ROWS)


@pytest.fixture
def saved() -> Names:
    """The names of the Contacts cache that the server reads, as the service gets them."""
    cache = save_names(SAVED)
    return Names.load(cache)


@pytest.fixture
def bridged(export: Path) -> Archive:
    """The archive that the server reads for this export, as the service gets it."""
    return places.state_for(export).archive()


def as_json(value: Any) -> Any:
    """A payload as the route sends it."""
    return json.loads(json.dumps(value, default=ui_server.json_default))


# --- the routes that answer with the payload of a service ----------------------


@pytest.mark.parametrize(
    "asked",
    [
        {"query": "pizza"},
        {"query": "pizza", "contact": AVERY, "limit": "2"},
        {"query": "pizza", "contact": AVERY, "limit": "2", "offset": "2"},
        {"query": "pizza", "contact": GROUP},
        {"query": "friday pizza"},
        {"query": '"pizza night"'},
        {"query": "salad"},
        {"query": "zeppelin"},
        {"contact": AVERY, "limit": "3"},
        {},
    ],
)
def test_the_search_route_answers_with_the_payload_of_the_service(
    server: Any, export: Path, bridged: Archive, saved: Names, asked: dict[str, str]
) -> None:
    status, answer = get(server, "/api/search", messagesPath=export, **asked)
    expected = search.payload(
        bridged, saved, asked.get("query", ""), asked.get("contact", ""), asked.get("limit", ""), asked.get("offset", "")
    )
    assert status == 200
    assert answer == as_json(expected)


def test_the_search_route_reads_the_last_word_as_a_prefix_when_asked(
    server: Any, export: Path, bridged: Archive, saved: Names
) -> None:
    # the reader may still be typing: whole words unless the app sends prefix=true
    assert get(server, "/api/search", messagesPath=export, query="piz")[1]["total"] == 0
    status, answer = get(server, "/api/search", messagesPath=export, query="piz", prefix="true")
    assert (status, answer["total"], answer["query"]) == (200, 8, "piz")
    assert answer == as_json(search.payload(bridged, saved, "piz", as_you_type=True))


@pytest.mark.parametrize(
    ("contact", "when", "more"),
    [
        (AVERY, "2025-01-01", {}),
        (AVERY, "2025-01-02", {}),
        (AVERY, "2025-01-03", {}),
        (GROUP, "2025-01-05", {}),
        (GROUP, "2025-01-05", {"offset": "1", "limit": "2"}),
        (GROUP, "2025-01-05", {"around": "2025-01-05 12:02:00", "limit": "2"}),
        (TITLED, "2025-01-06", {}),
        ("+15550180000", "2025-01-01", {}),
    ],
)
def test_the_day_route_answers_with_the_payload_of_the_service(
    server: Any, export: Path, bridged: Archive, saved: Names, contact: str, when: str, more: dict[str, str]
) -> None:
    status, answer = get(server, "/api/day", messagesPath=export, contact=contact, day=when, **more)
    expected = day.payload(
        bridged, saved, contact, when, around=more.get("around"), offset=more.get("offset", ""), limit=more.get("limit", "")
    )
    assert status == 200
    assert answer == as_json(expected)


def test_the_conversation_list_route_answers_with_the_payload_of_the_service(
    server: Any, export: Path, bridged: Archive, saved: Names
) -> None:
    status, answer = get(server, "/api/contacts", messagesPath=export, limit="3")
    # the answer names the archive file itself, as /api/defaults does
    expected = conversations.payload(bridged, saved, "3", str(places.archive_path_for(export)))
    assert status == 200
    assert answer == as_json(expected)
    assert answer["messagesPath"] == str(export.with_suffix(".sqlite3"))
    assert (len(answer["contacts"]), answer["archive"]["conversations"]) == (3, 4)


def test_a_text_is_cut_at_the_limit_of_the_service_on_every_route(server: Any, export: Path, saved: Names) -> None:
    _status, found = get(server, "/api/search", messagesPath=export, query="oven")
    _status, opened = get(server, "/api/day", messagesPath=export, contact=TITLED, day="2025-01-07")
    for shown in (found["results"][0]["text"], opened["messages"][0]["text"]):
        assert len(shown) <= messages.MESSAGE_TEXT_LIMIT + 1
        assert shown.endswith("…") and LONG_TEXT.startswith(shown[:-1])
        assert len(shown) > messages.MESSAGE_TEXT_LIMIT - len("oven ")


def test_the_totals_of_the_server_are_those_of_the_service(export: Path, bridged: Archive) -> None:
    assert as_json(ui_server.archive_summary(export)) == conversations.summary(bridged)


def test_the_name_of_each_conversation_is_the_one_of_the_service(export: Path, bridged: Archive, saved: Names) -> None:
    _frame, _catalog, held = ui_server.cached_messages_bundle(export)
    assert held == saved.conversations(bridged.conversations())
    assert held[GROUP] == "Avery Stone, Blake Reyes and 1 other"
    assert held[TITLED] == "Allotment Board"
    assert (held[AVERY], held[UNSAVED]) == ("Avery Stone", "")


# --- the files of a report, read by the server --------------------------------


def test_the_server_reads_the_files_that_a_run_writes(desk: Any) -> None:
    from tests.recall_services.report_world import MODEL
    from tests.report_engine.synth import MINA

    done = reports.run(MINA, MODEL, "2023-01-01", "2023-02-28", **desk.more())
    text = done.report_path.read_text(encoding="utf-8")
    assert ui_server.report_contact(done.report_path) == MINA
    assert ui_server.report_scope(text) == ("2023-01-01", "2023-02-28")
    assert ui_server.companion_report_paths(done.report_path, MINA) == (done.events_path, None)
    story = ui_server.markdown_section(text, "Summary", stop_at_next_heading=False)
    assert reports.story_of(text) == story and story.startswith("### The Arc")
    assert ui_server.markdown_to_html(story).count("<h3>") == 4
    # the events file, read as the server reads the file of a report of the older kind
    read_back = ui_server.events_of_file(str(done.events_path))
    assert [set(event) for event in read_back] == [set(reports.EVENT_COLUMNS)] * len(done.report.moments)
    assert [event["title"] for event in read_back] == [moment.title for moment in done.report.moments]
    assert [event["score"] for event in read_back] == [moment.score for moment in done.report.moments]
    for name in (done.report_path.name, done.events_path.name):
        assert ui_server._REPORT_FILE_NAME.fullmatch(name), name


def test_the_server_gives_the_moments_of_a_saved_report_what_opens_their_day(desk: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.recall_services.report_world import MODEL
    from tests.report_engine.synth import MINA

    done = reports.run(MINA, MODEL, "2023-01-01", "2023-02-28", **desk.more())
    text = done.report_path.read_text(encoding="utf-8")
    # the server finds the saved report in the data folder: here the places of this desk
    monkeypatch.setattr(reports, "default_places", lambda: desk.places)

    moments = ui_server.report_moments(MINA, text, str(done.events_path), desk.archive)

    assert moments == reports.moments_for(done.report, desk.archive)
    assert moments and all(moment["chatId"] == MINA and moment["day"] for moment in moments)
    # a report of the older kind has no saved report: its events file is read, and nothing opens a day
    older = ui_server.report_moments(MINA, "# Conversation Analysis - older", str(done.events_path), desk.archive)
    assert [set(event) for event in older] == [set(reports.EVENT_COLUMNS)] * len(moments)


@pytest.mark.parametrize(
    "text",
    ["", "no story here", "## Summary", "## Summary\n", "# T\n\n## Overview\n- x\n\n## Summary\n### The Arc\nwords\n\n## Not a section\nmore\n"],
)
def test_the_story_of_a_report_file_is_cut_as_the_server_cuts_it(text: str) -> None:
    assert reports.story_of(text) == ui_server.markdown_section(text, "Summary", stop_at_next_heading=False)
