"""Tests that a name saved in Contacts shows everywhere without a new import."""
import pandas as pd
import pytest

import ui_server
from tests.export_kit import (  # noqa: F401 - archives and contacts_cache are fixtures
    AVERY,
    BLAKE,
    GROUP,
    MODEL,
    archives,
    contacts_cache,
    get,
    loaded,
    row,
    save_names,
)

pytestmark = pytest.mark.usefixtures("archives", "contacts_cache")

ROWS = [
    row(AVERY, "2024-11-16 09:00:00", "see you at the lake house"),
    row(AVERY, "2024-11-16 09:01:00", "on my way", from_me=1),
    row(GROUP, "2025-01-01 09:00:00", "who is driving", sender=AVERY),
    row(GROUP, "2025-01-01 09:01:00", "not me", sender=BLAKE),
    row(GROUP, "2025-01-01 09:02:00", "I can", from_me=1, sender=""),
]


@pytest.fixture
def export(tmp_path):
    return loaded(tmp_path / "messages.csv", ROWS)


def test_a_saved_name_reaches_the_archive_without_a_new_import(export):
    before_df, _catalog, before = ui_server.cached_messages_bundle(export)
    assert before[AVERY] == ""

    save_names({AVERY: "Avery Stone"})
    after_df, _catalog, after = ui_server.cached_messages_bundle(export)

    assert after[AVERY] == "Avery Stone"
    # only the names were worked out again; the archive was not read a second time
    assert after_df is before_df


def test_the_archive_is_kept_while_neither_file_changes(export):
    save_names({AVERY: "Avery Stone"})

    first = ui_server.cached_messages_bundle(export)
    second = ui_server.cached_messages_bundle(export)

    assert all(a is b for a, b in zip(first, second))


def test_an_archive_that_was_loaded_again_is_read_again(export):
    before_df, _catalog, _names = ui_server.cached_messages_bundle(export)

    loaded(export, [*ROWS, row(BLAKE, "2025-02-01 08:00:00", "new here")])
    after_df, catalog, _names = ui_server.cached_messages_bundle(export)

    assert after_df is not before_df
    assert BLAKE in set(catalog["chat_id"])


def test_a_group_named_after_its_people_follows_their_names(export):
    save_names({AVERY: "Avery Stone"})
    _df, _catalog, before = ui_server.cached_messages_bundle(export)

    save_names({AVERY: "Avery Stone", BLAKE: "Blake Reyes"})
    _df, _catalog, after = ui_server.cached_messages_bundle(export)

    assert before.get(GROUP) != after.get(GROUP)
    assert after[GROUP] == "Avery Stone and Blake Reyes"


def test_search_shows_the_name_as_soon_as_it_is_saved(export, server):
    def lake_house():
        status, answer = get(server, "/api/search", messagesPath=export, query="lake house")
        assert status == 200, answer
        return answer["results"][0]["displayName"]

    before = lake_house()
    save_names({AVERY: "Avery Stone"})
    after = lake_house()

    assert (before, after) == ("", "Avery Stone")


def test_a_day_shows_the_name_as_soon_as_it_is_saved(export, server):
    def who_wrote():
        status, answer = get(server, "/api/day", messagesPath=export, contact=GROUP, day="2025-01-01")
        assert status == 200, answer
        return [message["senderName"] for message in answer["messages"]]

    before = who_wrote()
    save_names({BLAKE: "Blake Reyes"})
    after = who_wrote()

    assert (before, after) == (["", "", ""], ["", "Blake Reyes", ""])


def test_the_conversation_list_shows_the_name_as_soon_as_it_is_saved(export):
    def listed():
        _status, answer = ui_server.contacts_answer({"messagesPath": str(export)})
        return {entry["chat_id"]: entry["display_name"] for entry in answer["contacts"]}, answer["contactNameCount"]

    before = listed()
    save_names({AVERY: "Avery Stone"})
    after = listed()

    assert before == ({AVERY: "", GROUP: ""}, 0)
    assert after == ({AVERY: "Avery Stone", GROUP: "Avery Stone and 1 other"}, 2)


def test_moments_show_the_name_as_soon_as_it_is_saved(export):
    now = pd.Timestamp("2026-11-16 12:00:00")

    before = ui_server.memories_payload(export, now=now)
    save_names({AVERY: "Avery Stone"})
    after = ui_server.memories_payload(export, now=now)

    assert [moment["name"] for moment in before["onThisDay"]] == [""]
    assert [moment["name"] for moment in after["onThisDay"]] == ["Avery Stone"]


def test_a_preview_stored_before_the_name_was_saved_is_not_served(export):
    def speaker_names(payload):
        return {speaker["handle"]: speaker["name"] for speaker in payload["dynamics"]["speakers"]}

    first, first_cached = ui_server.cached_preview_payload(export, GROUP, MODEL)
    again, again_cached = ui_server.cached_preview_payload(export, GROUP, MODEL)
    save_names({BLAKE: "Blake Reyes"})
    after, after_cached = ui_server.cached_preview_payload(export, GROUP, MODEL)

    assert (first_cached, again_cached, after_cached) == (False, True, False)
    assert speaker_names(first)[BLAKE] == ""
    assert speaker_names(after)[BLAKE] == "Blake Reyes"
    assert [message["senderName"] for message in after["recentMessages"]] == ["", "Blake Reyes", ""]


def test_an_opened_report_shows_the_name_as_soon_as_it_is_saved(export, tmp_path):
    report = tmp_path / "analysis_chat77.md"
    report.write_text("# Conversation Analysis - chat77\n\n## Summary\nA story.\n", encoding="utf-8")

    def opened():
        payload, _cached = ui_server.cached_analysis_payload(export, GROUP, MODEL, report_path=str(report))
        return payload

    before = opened()
    save_names({AVERY: "Avery Stone", BLAKE: "Blake Reyes"})
    after = opened()

    assert before["contactDisplayName"] == ""
    assert after["contactDisplayName"] == "Avery Stone and Blake Reyes"
    assert {speaker["name"] for speaker in after["dynamics"]["speakers"]} == {"Avery Stone", "Blake Reyes"}
