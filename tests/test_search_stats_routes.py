"""/api/search with filters and /api/stats, asked on a loopback port as the app asks them.

Each request reads the archive of the data folder, loaded from an invented export.
"""
import pytest

import ui_server
from tests.export_kit import (  # noqa: F401 - data_folder is a fixture
    AVERY,
    BLAKE,
    GROUP,
    NAMES,
    PLACEHOLDER,
    UNSAVED,
    data_folder,
    get,
    loaded,
    row,
    save_names,
)

ROWS = [
    row(AVERY, "2025-01-01 09:00:00", "🍕 pizza night at mine"),
    row(AVERY, "2025-01-01 09:05:00", "bringing the pizza stone", from_me=1),
    row(AVERY, "2025-01-01 09:06:00", PLACEHOLDER, attachments="image/heic"),
    row(AVERY, "2025-01-03 19:30:00", "this is which hi", from_me=1),
    row(UNSAVED, "2025-01-03 10:00:00", "hi, your pizza order is ready"),
    row(GROUP, "2025-01-05 12:00:00", "pizza on friday?", sender=AVERY),
    row(GROUP, "2025-01-05 12:01:00", "yes to pizza", sender=BLAKE),
]
NO_FILTERS = {"contacts": [], "since": "", "until": "", "from": "", "kind": "", "attachments": ""}


@pytest.fixture
def app_archive(data_folder):
    loaded(data_folder / "messages.csv", ROWS)
    save_names(NAMES)
    return data_folder


def texts(answer):
    return [result["text"] for result in answer["results"]]


# ---- /api/search


def test_a_word_is_whole_and_its_ranges_are_given(app_archive, server):
    status, answer = get(server, "/api/search", query="hi", prefix="false")

    assert status == 200
    # "this" and "which" hold "hi" but are other words
    assert {result["text"]: result["ranges"] for result in answer["results"]} == {
        "this is which hi": [[14, 16]],
        "hi, your pizza order is ready": [[0, 2]],
    }
    assert (answer["filters"], answer["order"], answer["prefix"]) == (NO_FILTERS, "newest", False)


HI_ROWS = [
    row(AVERY, "2025-02-01 09:00:00", "hi"),
    row(AVERY, "2025-02-01 09:01:00", "this one"),
    row(AVERY, "2025-02-01 09:02:00", "which one"),
    row(AVERY, "2025-02-01 09:03:00", "that was a hit"),
    row(AVERY, "2025-02-01 09:04:00", "tell him", from_me=1),
    row(AVERY, "2025-02-01 09:05:00", "Hi there"),
]


@pytest.fixture
def hi_archive(data_folder):
    loaded(data_folder / "messages.csv", HI_ROWS)
    return data_folder


def test_hi_with_prefix_false_is_the_word_hi_only(hi_archive, server):
    _status, whole = get(server, "/api/search", query="Hi", prefix="false")
    _status, counted = get(server, "/api/stats", query="Hi", prefix="false")

    # no "this", no "which", no "hit", no "him"
    assert texts(whole) == ["Hi there", "hi"]
    assert {result["text"]: result["ranges"] for result in whole["results"]} == {"Hi there": [[0, 2]], "hi": [[0, 2]]}
    assert (whole["total"], whole["prefix"]) == (2, False)
    assert (counted["totals"]["total"], counted["prefix"]) == (2, False)


def test_with_no_prefix_parameter_stats_count_whole_words(hi_archive, server):
    _status, counted = get(server, "/api/stats", query="Hi")
    _status, typing = get(server, "/api/stats", query="Hi", prefix="true")

    # the word "hi" only: no "this", "which", "hit" or "him"
    assert (counted["totals"]["total"], counted["prefix"]) == (2, False)
    # a prefix only when asked: then "hit" and "him" count too
    assert (typing["totals"]["total"], typing["prefix"]) == (4, True)


def test_with_no_prefix_parameter_search_reads_the_last_word_as_a_prefix_as_today(app_archive, server):
    _status, answer = get(server, "/api/search", query="piz")
    _status, hi = get(server, "/api/search", query="Hi")

    assert (answer["total"], answer["prefix"]) == (5, True)
    assert "🍕 pizza night at mine" in texts(answer)
    # as the reader types "Hi", the words that start with it match, never a part inside a word
    assert sorted(texts(hi)) == ["hi, your pizza order is ready", "this is which hi"]


def test_ranges_count_utf16_units(app_archive, server):
    (result,) = get(server, "/api/search", query="night")[1]["results"]

    # the emoji is two units of JavaScript
    assert result["ranges"] == [[9, 14]]


def test_the_filters_are_applied_and_said(app_archive, server):
    _status, groups = get(server, "/api/search", query="pizza", kind="group", order="oldest")
    _status, mine = get(server, "/api/search", query="pizza", **{"from": "me"})
    _status, two = get(server, "/api/search", query="pizza", contacts=f"{AVERY},{UNSAVED}", until="2025-01-02")

    assert texts(groups) == ["pizza on friday?", "yes to pizza"]
    assert (groups["filters"]["kind"], groups["order"]) == ("group", "oldest")
    assert texts(mine) == ["bringing the pizza stone"]
    assert texts(two) == ["bringing the pizza stone", "🍕 pizza night at mine"]
    assert two["filters"]["contacts"] == [AVERY, UNSAVED]


def test_no_words_and_a_filter_list_those_messages(app_archive, server):
    _status, photos = get(server, "/api/search", attachments="image")
    _status, nothing = get(server, "/api/search")

    assert [result["attachment"] for result in photos["results"]] == ["Photo"]
    assert photos["results"][0]["ranges"] == []
    assert (nothing["results"], nothing["total"]) == ([], 0)


@pytest.mark.parametrize(
    ("query", "sentence"),
    [
        ({"kind": "self"}, "kind must be direct, group or empty."),
        ({"order": "best"}, "order must be newest, oldest or relevance."),
        ({"prefix": "yes"}, "prefix must be true or false."),
        ({"since": "2025-13-01"}, "since must be a day written as YYYY-MM-DD."),
    ],
)
def test_a_search_refuses_a_value_that_is_not_allowed(app_archive, server, query, sentence):
    status, answer = get(server, "/api/search", **{"query": "pizza", **query})

    assert (status, answer) == (400, {"error": sentence})


# ---- /api/stats


def test_the_totals_and_a_row_for_each_day(app_archive, server):
    status, answer = get(server, "/api/stats", by="day")

    assert status == 200
    assert answer["totals"] == {"total": 7, "sent": 2, "received": 5}
    assert (answer["first"], answer["last"]) == ("2025-01-01", "2025-01-05")
    assert [(row["key"], row["total"]) for row in answer["rows"]] == [
        ("2025-01-01", 3),
        ("2025-01-02", 0),
        ("2025-01-03", 2),
        ("2025-01-04", 0),
        ("2025-01-05", 2),
    ]


def test_people_are_named_as_the_app_names_them(app_archive, server):
    _status, answer = get(server, "/api/stats", by="person", query="pizza")

    assert [(row["key"], row["label"], row["total"]) for row in answer["rows"]] == [
        (AVERY, "Avery Stone", 2),
        (UNSAVED, "Number ending in 0009", 1),
    ]
    # no full number of a person who is not saved
    assert UNSAVED not in {row["label"] for row in answer["rows"]}
    # a group chat is no person: it is listed by conversation only
    _status, threads = get(server, "/api/stats", by="conversation", query="pizza")
    assert (GROUP, "Avery Stone and Blake Reyes", 2) in [(row["key"], row["label"], row["total"]) for row in threads["rows"]]


def test_stats_refuse_a_value_that_is_not_allowed(app_archive, server):
    status, answer = get(server, "/api/stats", by="minute")

    assert status == 400
    assert answer == {
        "error": "by must be day, week, month, year, weekday, hour, weekday_hour, person, conversation or empty."
    }


def test_stats_need_the_archive(data_folder, server):
    status, answer = get(server, "/api/stats")

    assert (status, answer["code"]) == (400, "no_archive")
    assert ui_server.list_jobs() == []
