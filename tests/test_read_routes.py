"""The routes that read the archive, asked on a loopback port as the app asks them.

No request here names a file: each one reads the archive of the data folder,
which is a temp folder of the test (`data_folder`). The archive is loaded from
an invented export, as the app loads the export of an earlier build.
"""
import pandas as pd
import pytest

import ui_server
from tests.export_kit import (  # noqa: F401 - data_folder is a fixture
    AVERY,
    BLAKE,
    GROUP,
    MODEL,
    NAMES,
    PLACEHOLDER,
    UNSAVED,
    data_folder,
    get,
    jobs_ended,
    loaded,
    row,
    save_names,
    write_export,
)
from tests.loopback import post

LONG_DAY = "2024-03-02"
LONG_DAY_MESSAGES = 450
ROWS = [
    row(AVERY, "2025-01-01 09:00:00", "pizza night at mine"),
    row(AVERY, "2025-01-01 09:05:00", "bringing the pizza stone", from_me=1),
    row(AVERY, "2025-01-01 09:06:00", PLACEHOLDER, attachments="image/heic"),
    row(AVERY, "2025-01-02 19:30:00", "that was a good pizza"),
    row(AVERY, "2025-01-04 08:00:00", "back to salad"),
    row(UNSAVED, "2025-01-03 10:00:00", "your pizza order is ready"),
    row(GROUP, "2025-01-05 12:00:00", "pizza on friday?", sender=AVERY),
    row(GROUP, "2025-01-05 12:01:00", "yes to pizza", sender=BLAKE),
    row(GROUP, "2025-01-05 12:02:00", "booking a table", from_me=1, sender=""),
]
MESSAGES = len(ROWS)
EMPTY_MOMENTS = {"onThisDay": [], "anniversaries": [], "reconnect": []}
EMPTY_PATTERNS = {"monthly": [], "daily": {"start": "", "counts": []}, "rhythm": [[0] * 24 for _ in range(7)]}
EMPTY_FIRST_WORDS = {"entries": [], "signature": "", "totals": {"messages": 0, "people": 0}}
READ_ROUTES = [
    ("/api/contacts", {}),
    ("/api/search", {"query": "pizza"}),
    ("/api/day", {"contact": AVERY, "day": "2025-01-01"}),
    ("/api/preview", {"contact": AVERY, "model": MODEL}),
    ("/api/memories", {}),
    ("/api/overview", {}),
    ("/api/first-words", {}),
]


@pytest.fixture
def app_archive(data_folder):
    """The archive of the app, loaded from an export in the data folder. Gives the data folder."""
    loaded(data_folder / "messages.csv", ROWS)
    return data_folder


@pytest.fixture
def named(app_archive):
    save_names(NAMES)


def texts(answer):
    return [result["text"] for result in answer["results"]]


# ---- /api/defaults


def test_the_defaults_offer_three_models_from_low_to_high(data_folder, server):
    status, answer = get(server, "/api/defaults")

    assert status == 200
    assert [choice["level"] for choice in answer["modelChoices"]] == ["low", "medium", "high"]
    assert all(choice["id"] and choice["label"] for choice in answer["modelChoices"])
    assert answer["models"] == [choice["id"] for choice in answer["modelChoices"]]
    assert answer["defaultModel"] in answer["models"]


def test_the_defaults_name_the_archive_file_in_the_data_folder(data_folder, server):
    _status, answer = get(server, "/api/defaults")

    assert answer["messagesPath"] == str(data_folder / "saves" / "archive.sqlite3")
    assert answer["outDir"] == str(data_folder / "out")
    assert (answer["reports"], answer["setupCompleted"]) == ([], False)


def test_the_defaults_say_whether_there_is_something_to_read(data_folder, server):
    def has_messages():
        return get(server, "/api/defaults")[1]["hasMessages"]

    nothing = has_messages()
    # a header and no row is no export
    write_export(data_folder / "messages.csv", [])
    no_rows = has_messages()
    # an export of an earlier build waits to be loaded: the app has something to read
    write_export(data_folder / "messages.csv", ROWS)
    an_export = has_messages()

    assert (nothing, no_rows, an_export) == (False, False, True)
    # asking for the defaults starts no load
    assert not (data_folder / "saves" / "archive.sqlite3").exists()
    assert ui_server.list_jobs() == []


def test_the_defaults_say_that_the_archive_is_there(app_archive, server):
    (app_archive / "messages.csv").unlink()

    assert get(server, "/api/defaults")[1]["hasMessages"] is True


# ---- /api/contacts


def test_the_conversation_list_is_read_from_the_archive(named, server, data_folder):
    status, answer = get(server, "/api/contacts")

    assert status == 200
    assert (answer["source"], answer["messagesPath"]) == ("messages", str(data_folder / "saves" / "archive.sqlite3"))
    assert [(listed["chat_id"], listed["message_count"]) for listed in answer["contacts"]] == [
        (AVERY, 5), (GROUP, 3), (UNSAVED, 1),
    ]
    assert answer["contacts"][0] == {
        "chat_id": AVERY,
        "display_name": "Avery Stone",
        "is_group": 0,
        "first_msg": "2025-01-01T09:00:00",
        "last_msg": "2025-01-04T08:00:00",
        "message_count": 5,
        "sent_count": 1,
        "received_count": 4,
    }
    assert [listed["display_name"] for listed in answer["contacts"]] == ["Avery Stone", "Avery Stone and Blake Reyes", ""]
    assert answer["contactNameCount"] == 2


def test_the_totals_of_the_list_are_those_of_the_whole_archive(app_archive, server):
    status, answer = get(server, "/api/contacts", limit="1")

    assert status == 200
    # the list is cut at the limit; the totals are not
    assert len(answer["contacts"]) == 1
    assert answer["archive"] == {
        "messages": MESSAGES,
        "conversations": 3,
        "firstMessage": "2025-01-01T09:00:00",
        "lastMessage": "2025-01-05T12:02:00",
    }


def test_a_limit_of_the_list_that_is_no_number_is_refused(app_archive, server):
    status, answer = get(server, "/api/contacts", limit="many")

    assert status == 400
    assert answer["error"]


def test_with_only_an_export_the_list_waits_for_the_archive_and_then_gives_it(data_folder, server):
    write_export(data_folder / "messages.csv", ROWS)

    status, waiting = get(server, "/api/contacts")

    assert (status, waiting["code"]) == (503, "archive_building")
    assert waiting["error"] == "Recall is getting your archive ready. This takes a moment."
    assert (waiting["job"]["action"], waiting["job"]["status"]) == ("bridge", "queued")

    assert jobs_ended()
    _status, watched = get(server, f"/api/jobs/{waiting['job']['id']}")
    status, answer = get(server, "/api/contacts")

    assert (watched["job"]["status"], watched["job"]["error"]) == ("completed", None)
    assert (watched["job"]["result"]["messages"], watched["job"]["result"]["conversations"]) == (MESSAGES, 3)
    assert status == 200
    assert (answer["source"], answer["archive"]["messages"]) == ("messages", MESSAGES)
    assert [listed["chat_id"] for listed in answer["contacts"]] == [AVERY, GROUP, UNSAVED]
    # the export stays where it is
    assert (data_folder / "messages.csv").exists()


@pytest.mark.parametrize(("path", "query"), READ_ROUTES)
def test_every_read_route_waits_while_the_archive_is_built(data_folder, server, path, query):
    write_export(data_folder / "messages.csv", ROWS)

    status, first = get(server, path, **query)
    again_status, again = get(server, "/api/search", query="pizza")

    assert (status, first["code"]) == (503, "archive_building")
    assert first["job"]["action"] == "bridge"
    # one load for every request that waits: a second one is told of the same job, or finds the archive
    if again_status == 503:
        assert again["job"]["id"] == first["job"]["id"]
    assert jobs_ended()
    assert [job["action"] for job in ui_server.list_jobs()] == ["bridge"]
    assert get(server, path, **query)[0] == 200


def test_with_no_archive_and_no_messages_the_list_says_so(data_folder, server):
    status, answer = get(server, "/api/contacts")

    assert status == 400
    assert answer == {"error": "There is no archive yet, and Messages cannot be read.", "code": "no_archive"}
    assert ui_server.list_jobs() == []


@pytest.mark.parametrize(("path", "query"), [route for route in READ_ROUTES if route[0] in ("/api/search", "/api/day", "/api/preview")])
def test_a_route_that_needs_the_archive_says_when_there_is_none(data_folder, server, path, query):
    status, answer = get(server, path, **query)

    assert status == 400
    assert answer == {
        "error": "There is no archive yet. Open the setup guide to read your messages.",
        "code": "no_archive",
    }
    assert ui_server.list_jobs() == []


def test_an_export_with_no_row_is_no_archive_and_starts_no_load(data_folder, server):
    write_export(data_folder / "messages.csv", [])

    status, answer = get(server, "/api/search", query="pizza")

    assert (status, answer["code"]) == (400, "no_archive")
    assert ui_server.list_jobs() == []
    assert not (data_folder / "saves" / "archive.sqlite3").exists()


# ---- /api/search


def test_a_search_counts_every_match_and_gives_the_newest_first(app_archive, server):
    status, answer = get(server, "/api/search", query="pizza", limit="2")

    assert status == 200
    assert texts(answer) == ["yes to pizza", "pizza on friday?"]
    # `count` is what this answer holds, `total` is every match: the screen shows `total`
    assert (answer["count"], answer["total"], answer["offset"]) == (2, 6, 0)
    assert (answer["query"], answer["contact"]) == ("pizza", "")


def test_a_search_gives_the_part_that_starts_at_an_offset(app_archive, server):
    _status, first = get(server, "/api/search", query="pizza", limit="2")
    status, second = get(server, "/api/search", query="pizza", limit="2", offset="2")
    _status, past = get(server, "/api/search", query="pizza", limit="2", offset="50")

    assert status == 200
    assert texts(second) == ["your pizza order is ready", "that was a good pizza"]
    assert (second["count"], second["total"], second["offset"]) == (2, 6, 2)
    assert not set(texts(first)) & set(texts(second))
    assert (past["results"], past["count"], past["total"]) == ([], 0, 6)


def test_several_words_match_in_any_order(app_archive, server):
    _status, answer = get(server, "/api/search", query="mine night pizza")
    _status, other = get(server, "/api/search", query="pizza salad")

    assert texts(answer) == ["pizza night at mine"]
    # each word must be in the message
    assert other["total"] == 0


def test_a_phrase_in_quotes_matches_as_a_phrase(app_archive, server):
    _status, phrase = get(server, "/api/search", query='"pizza night"')
    _status, turned = get(server, "/api/search", query='"night pizza"')

    assert texts(phrase) == ["pizza night at mine"]
    assert turned["total"] == 0


def test_the_last_word_of_a_search_is_read_as_a_prefix(app_archive, server):
    _status, typing = get(server, "/api/search", query="good piz")
    _status, whole_word = get(server, "/api/search", query="piz good")
    _status, middle = get(server, "/api/search", query="izz")

    # the reader may still be typing the last word
    assert texts(typing) == ["that was a good pizza"]
    assert typing["query"] == "good piz"
    # a word before the last one is whole, and a part from the middle of a word matches nothing
    assert texts(whole_word) == []
    assert middle["total"] == 0


def test_a_search_in_one_conversation_stays_in_it(app_archive, server):
    _status, answer = get(server, "/api/search", query="pizza", contact=GROUP)
    _status, part = get(server, "/api/search", query="pizza", contact=GROUP[:-1])

    assert texts(answer) == ["yes to pizza", "pizza on friday?"]
    assert {result["chatId"] for result in answer["results"]} == {GROUP}
    # a part of an id is no conversation
    assert (part["results"], part["total"]) == ([], 0)


@pytest.mark.parametrize("query", [{"limit": "many"}, {"limit": "2.5"}, {"offset": "later"}])
def test_a_search_refuses_a_limit_or_an_offset_that_is_no_number(app_archive, server, query):
    status, answer = get(server, "/api/search", **{"query": "pizza", **query})

    assert status == 400
    assert answer["error"]
    assert "results" not in answer


def test_a_found_message_has_the_shape_the_app_reads(named, server):
    _status, answer = get(server, "/api/search", query="friday")

    assert answer["results"] == [{
        # an id of the archive is text: `legacy:` and the number of the row in the export
        "messageId": "legacy:7",
        "chatId": GROUP,
        "displayName": "Avery Stone and Blake Reyes",
        "senderName": "Avery Stone",
        "timestamp": "2025-01-05 12:00:00",
        "sender": AVERY,
        "text": "pizza on friday?",
        "attachment": "",
        "isFromMe": False,
    }]


def test_the_names_of_a_search_follow_the_contacts_cache(app_archive, server):
    def names():
        (result,) = get(server, "/api/search", query="friday")[1]["results"]
        return result["displayName"], result["senderName"]

    before = names()
    save_names({AVERY: "Avery Stone"})
    one_saved = names()
    save_names({AVERY: "Avery S.", BLAKE: "Blake Reyes"})
    changed = names()

    assert before == ("", "")
    assert one_saved == ("Avery Stone and 1 other", "Avery Stone")
    assert changed == ("Avery S. and Blake Reyes", "Avery S.")


def test_no_words_and_no_conversation_give_nothing(app_archive, server):
    status, answer = get(server, "/api/search")

    assert status == 200
    assert (answer["results"], answer["count"], answer["total"]) == ([], 0, 0)


# ---- /api/day


@pytest.fixture
def long_day(data_folder):
    """A day of 450 messages, one a minute from 08:00, with a day on each side."""
    stamps = pd.date_range(start=f"{LONG_DAY} 08:00:00", periods=LONG_DAY_MESSAGES, freq="min")
    loaded(data_folder / "messages.csv", [
        row(AVERY, "2024-03-01 09:00:00", "the day before"),
        *[row(AVERY, str(stamp), f"note {place}" if place != 300 else "the ferry leaves at noon") for place, stamp in enumerate(stamps)],
        row(AVERY, "2024-03-05 09:00:00", "three days on"),
    ])
    return data_folder


def test_a_day_is_opened_from_a_search_hit_at_the_message_that_was_found(long_day, server):
    (hit,) = get(server, "/api/search", query="ferry")[1]["results"]

    status, day = get(server, "/api/day", contact=hit["chatId"], day=hit["timestamp"][:10], around=hit["timestamp"])

    assert status == 200
    assert (day["contact"], day["day"], day["total"]) == (AVERY, LONG_DAY, LONG_DAY_MESSAGES)
    # a part of the long day, with the found message in it
    assert len(day["messages"]) == 200 and 0 < day["offset"] <= 300
    found = [message for message in day["messages"] if message["messageId"] == hit["messageId"]]
    assert found == [hit]
    assert day["messages"].index(hit) == 300 - day["offset"]
    assert (day["previousDay"], day["nextDay"]) == ("2024-03-01", "2024-03-05")


def test_a_day_with_no_time_asked_for_starts_at_its_start(long_day, server):
    _status, day = get(server, "/api/day", contact=AVERY, day=LONG_DAY)

    assert (day["offset"], len(day["messages"])) == (0, 200)
    assert [message["text"] for message in day["messages"][:2]] == ["note 0", "note 1"]


def test_a_place_in_a_day_is_used_as_it_is(long_day, server):
    status, day = get(server, "/api/day", contact=AVERY, day=LONG_DAY, offset="400", limit="200")
    _status, past = get(server, "/api/day", contact=AVERY, day=LONG_DAY, offset="999")

    assert status == 200
    # the part is not pulled back to fill a whole window
    assert (day["total"], day["offset"]) == (LONG_DAY_MESSAGES, 400)
    assert [message["text"] for message in day["messages"]] == [f"note {place}" for place in range(400, 450)]
    assert (past["total"], past["messages"]) == (LONG_DAY_MESSAGES, [])


def test_an_offset_wins_over_a_time(long_day, server):
    _status, day = get(server, "/api/day", contact=AVERY, day=LONG_DAY, offset="10", limit="5", around=f"{LONG_DAY} 13:00:00")

    assert (day["offset"], [message["text"] for message in day["messages"]]) == (10, [f"note {place}" for place in range(10, 15)])


def test_a_day_with_no_message_says_where_the_messages_are(long_day, server):
    status, day = get(server, "/api/day", contact=AVERY, day="2024-03-03")

    assert status == 200
    assert (day["total"], day["messages"]) == (0, [])
    assert (day["previousDay"], day["nextDay"]) == (LONG_DAY, "2024-03-05")


@pytest.mark.parametrize("query", [
    {"day": "yesterday"},
    {"day": "2024-02-30"},
    {"day": ""},
    {"contact": ""},
    {"contact": "   "},
    {"limit": "0"},
    {"limit": "many"},
    {"offset": "later"},
])
def test_a_day_refuses_what_it_cannot_read(long_day, server, query):
    status, answer = get(server, "/api/day", **{"contact": AVERY, "day": LONG_DAY, **query})

    assert status == 400
    assert answer["error"]
    assert "messages" not in answer


# ---- /api/preview


def preview_of(server, contact=AVERY, **more):
    return get(server, "/api/preview", contact=contact, model=MODEL, **more)


def test_a_preview_gives_the_figures_of_one_conversation(named, server):
    status, answer = preview_of(server)
    stats = answer["stats"]

    assert status == 200
    assert (stats["chatId"], stats["totalMessages"], stats["sentCount"], stats["receivedCount"]) == (AVERY, 5, 1, 4)
    assert stats["unknownDirectionCount"] == 0
    assert (stats["firstTimestamp"], stats["lastTimestamp"]) == ("2025-01-01T09:00:00", "2025-01-04T08:00:00")
    assert (stats["activeDays"], stats["busiestDayCount"]) == (3, 3)
    assert (stats["attachments"]["total"], stats["attachments"]["photos"]) == (1, 1)
    assert answer["daily"] == {"start": "2025-01-01", "counts": [3, 1, 0, 1]}
    assert answer["monthly"] == [{"month": "2025-01", "total": 5, "sent": 1, "received": 4, "sentRatio": 0.2}]
    assert [message["text"] for message in answer["recentMessages"]][-2:] == ["that was a good pizza", "back to salad"]
    assert answer["recentMessages"][2]["attachment"] == "Photo"


def test_a_preview_follows_the_dates_that_are_asked_for(app_archive, server):
    _status, answer = preview_of(server, since="2025-01-02", until="2025-01-04")

    assert answer["stats"]["totalMessages"] == 2
    assert answer["estimate"]["msg_count"] == 2
    # the calendar is the whole conversation, whatever the dates
    assert answer["daily"] == {"start": "2025-01-01", "counts": [3, 1, 0, 1]}


def test_a_preview_carries_the_plan_of_a_report(app_archive, server):
    _status, answer = preview_of(server)
    estimate = answer["estimate"]

    assert "estimateError" not in answer
    assert (estimate["msg_count"], estimate["periods"], estimate["periods_cached"]) == (5, 1, 0)
    assert estimate["needs_chunking"] is False
    assert estimate["input_tokens"] > 0 and estimate["estimated_cost"] > 0


def test_a_preview_for_a_model_that_is_not_offered_has_no_plan_and_says_why(app_archive, server):
    status, answer = get(server, "/api/preview", contact=AVERY, model="not-a-model")

    assert status == 200
    assert answer["estimate"] is None
    assert answer["estimateError"]
    assert answer["stats"]["totalMessages"] == 5


def test_a_second_preview_is_served_from_what_was_kept(app_archive, server):
    first = preview_of(server)[1]
    second = preview_of(server)[1]
    other = preview_of(server, contact=GROUP)[1]

    assert (first["cached"], second["cached"], other["cached"]) == (False, True, False)
    assert {**second, "cached": False} == first


def test_a_preview_is_forgotten_when_the_previews_are_forgotten(app_archive, server):
    preview_of(server)
    kept = preview_of(server)[1]["cached"]

    ui_server.forget_previews()
    after = preview_of(server)[1]["cached"]

    assert (kept, after) == (True, False)


def test_the_route_that_clears_the_previews_forgets_them(app_archive, server):
    preview_of(server)

    status, answer = post(server, "/api/cache/preview", {})

    assert (status, answer) == (200, {"cleared": True})
    assert preview_of(server)[1]["cached"] is False


def test_a_preview_of_another_version_is_not_served(app_archive, server, monkeypatch):
    preview_of(server)
    assert ui_server.preview_cache_key(None, AVERY, MODEL)[0] == ui_server.PREVIEW_PAYLOAD_VERSION == 8

    monkeypatch.setattr(ui_server, "PREVIEW_PAYLOAD_VERSION", 9)

    assert preview_of(server)[1]["cached"] is False


def test_a_preview_kept_before_the_archive_changed_is_not_served(app_archive, server):
    before = preview_of(server)[1]
    # the archive is loaded again with one more message; nothing here forgets the previews
    loaded(app_archive / "messages.csv", [*ROWS, row(AVERY, "2025-01-06 08:00:00", "pizza again?", from_me=1)])
    after = preview_of(server)[1]

    assert (before["cached"], before["stats"]["totalMessages"]) == (False, 5)
    assert (after["cached"], after["stats"]["totalMessages"]) == (False, 6)


def test_a_preview_kept_before_a_name_was_saved_is_not_served(app_archive, server):
    before = preview_of(server, contact=GROUP)[1]
    save_names(NAMES)
    after = preview_of(server, contact=GROUP)[1]

    assert (before["cached"], after["cached"]) == (False, False)
    assert [speaker["name"] for speaker in before["dynamics"]["speakers"]] == ["", ""]
    assert {speaker["name"] for speaker in after["dynamics"]["speakers"]} == {"Avery Stone", "Blake Reyes"}


# ---- /api/memories, /api/overview, /api/first-words


def years_ago(years, clock="09:00:00"):
    """This day of an earlier year, as the time stamp of an export."""
    today = pd.Timestamp.now().normalize()
    try:
        day = today.replace(year=today.year - years)
    except ValueError:  # 29 February: the nearest year that has one
        day = today.replace(year=today.year - 4 * years)
    return f"{day.date()} {clock}"


def test_the_moments_are_read_from_the_archive(data_folder, server):
    quoted = years_ago(2, "21:15:00")
    loaded(data_folder / "messages.csv", [
        row(AVERY, years_ago(2), "ok"),
        row(AVERY, quoted, "a line long enough to be worth quoting back"),
        row(BLAKE, "2020-06-01 10:00:00", "some other day"),
    ])
    save_names(NAMES)

    status, answer = get(server, "/api/memories")
    (moment,) = answer["memories"]["onThisDay"]
    _status, day = get(server, "/api/day", contact=moment["chatId"], day=moment["day"])

    assert status == 200
    assert (moment["chatId"], moment["name"], moment["day"]) == (AVERY, "Avery Stone", quoted[:10])
    assert (moment["preview"], moment["previewFromMe"]) == ("a line long enough to be worth quoting back", False)
    # the moment counts its day as the day counts it, and names the quoted message as the day names it
    assert moment["count"] == day["total"] == 2
    assert (moment["previewAt"], moment["previewId"]) == (day["messages"][1]["timestamp"], day["messages"][1]["messageId"])
    assert (answer["memories"]["anniversaries"], answer["memories"]["reconnect"]) == ([], [])


def test_the_overview_is_read_from_the_archive(app_archive, server):
    status, answer = get(server, "/api/overview")
    patterns = answer["overview"]

    assert status == 200
    assert patterns["monthly"] == [{"key": "2025-01", "total": MESSAGES, "sent": 2, "received": 7}]
    assert patterns["daily"] == {"start": "2025-01-01", "counts": [3, 1, 1, 1, 3]}
    # 1 January 2025 is a Wednesday: the third row from Monday
    assert [len(day) for day in patterns["rhythm"]] == [24] * 7
    assert patterns["rhythm"][2][9] == 3
    assert sum(sum(day) for day in patterns["rhythm"]) == MESSAGES


def test_the_first_words_are_read_from_the_archive(named, server):
    status, answer = get(server, "/api/first-words")
    first_words = answer["firstWords"]

    assert status == 200
    # one person with a name: nearly nobody is named, so the unnamed person is shown too; a group never is
    assert [(entry["chatId"], entry["person"]) for entry in first_words["entries"]] == [(AVERY, "Avery Stone"), (UNSAVED, "")]
    first = first_words["entries"][0]
    assert (first["text"], first["direction"], first["timestamp"]) == ("pizza night at mine", "incoming", "2025-01-01 09:00:00")
    assert first["reply"] == {"text": "bringing the pizza stone", "direction": "outgoing", "timestamp": "2025-01-01 09:05:00"}
    assert first_words["signature"]
    assert first_words["totals"] == {"messages": MESSAGES, "people": 2, "firstYear": 2025, "lastYear": 2025}


@pytest.mark.parametrize(("path", "key", "empty"), [
    ("/api/memories", "memories", EMPTY_MOMENTS),
    ("/api/overview", "overview", EMPTY_PATTERNS),
    ("/api/first-words", "firstWords", EMPTY_FIRST_WORDS),
])
def test_a_screen_of_the_overview_is_empty_with_no_archive_and_no_error(data_folder, server, path, key, empty):
    status, answer = get(server, path)

    assert status == 200
    assert answer == {key: empty}
    assert ui_server.list_jobs() == []


# ---- /api/analysis: a report of the older kind


OLDER_REPORT = f"""# Conversation Analysis - {AVERY}

## Overview
- Scope: all dates
- Total messages: 5
- Sent / Received: 1 / 4
- First / Last: 2025-01-01 09:00:00 / 2025-01-04 08:00:00

## Summary
### The Arc
They planned a pizza night.

### Turning Points
The stone arrived.
"""
OLDER_EVENTS = (
    "date,title,detail,category,score,quote\n"
    "2025-01-01,NA,5,plans,7.5,pizza night at mine\n"
    "2025-01-02,null,None,food,not a number,\n"
)


@pytest.fixture
def older_report(named, data_folder):
    """A report that an earlier build wrote: a Markdown file and its events file in the reports folder."""
    report = data_folder / "out" / f"analysis_{AVERY}.md"
    report.write_text(OLDER_REPORT, encoding="utf-8")
    (data_folder / "out" / f"events_timeline_{AVERY}.csv").write_text(OLDER_EVENTS, encoding="utf-8")
    return report


def test_an_older_report_gives_its_events_as_the_file_holds_them(older_report, server):
    status, answer = get(server, "/api/analysis", reportPath=older_report, model=MODEL)
    analysis = answer["analysis"]

    assert status == 200
    # every cell is the text that it is: a title "NA" is no missing value, and "5" is no number
    assert analysis["events"] == [
        {"date": "2025-01-01", "title": "NA", "detail": "5", "category": "plans", "quote": "pizza night at mine", "score": 7.5},
        {"date": "2025-01-02", "title": "null", "detail": "None", "category": "food", "quote": "", "score": None},
    ]
    # nothing in an event of an older report opens a day
    assert not {"chatId", "day", "at", "pickedId"} & set(analysis["events"][0])


def test_an_older_report_is_opened_with_its_story_and_the_figures_of_its_conversation(older_report, server):
    status, answer = get(server, "/api/analysis", reportPath=older_report, model=MODEL)
    analysis = answer["analysis"]

    assert (status, answer["cached"]) == (200, False)
    # the conversation is read from the heading of the report
    assert (analysis["contact"], analysis["contactDisplayName"]) == (AVERY, "Avery Stone")
    assert analysis["summary"].startswith("### The Arc")
    assert analysis["summaryHtml"].count("<h3>") == 2
    assert analysis["shownPlain"] is False
    assert analysis["stats"]["totalMessages"] == 5
    assert [(file["kind"], file["name"]) for file in analysis["files"]] == [
        ("md", f"analysis_{AVERY}.md"), ("csv", f"events_timeline_{AVERY}.csv"),
    ]


def test_an_opened_report_is_kept_until_its_file_changes(older_report, server):
    def opened():
        return get(server, "/api/analysis", reportPath=older_report, model=MODEL)[1]

    first, second = opened(), opened()
    older_report.write_text(OLDER_REPORT.replace("The stone arrived.", "The stone arrived late."), encoding="utf-8")
    third = opened()

    assert (first["cached"], second["cached"], third["cached"]) == (False, True, False)
    assert "arrived late" in third["analysis"]["summary"]


def test_a_report_that_is_not_in_the_data_folder_is_not_opened(older_report, server, tmp_path):
    outside = tmp_path / f"analysis_{AVERY}.md"
    outside.write_text(OLDER_REPORT, encoding="utf-8")

    for path in (outside, older_report.with_name("analysis_nobody.md")):
        status, answer = get(server, "/api/analysis", reportPath=path, model=MODEL)
        assert (status, answer) == (404, {"error": "Report not found"})


def test_the_list_of_reports_names_the_conversation_of_each(older_report, server):
    _status, answer = get(server, "/api/reports")

    assert [(report["name"], report["contact"], report["displayName"]) for report in answer["reports"]] == [
        (f"analysis_{AVERY}.md", AVERY, "Avery Stone"),
    ]
