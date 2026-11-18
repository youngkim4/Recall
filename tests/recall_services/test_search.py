"""search.py: the /api/search payload from Archive.search."""

from __future__ import annotations

import json

import pytest

from archive import Archive
from recall_services import search
from recall_services.names import Names
from tests.recall_services import world
from tests.recall_services.test_messages import RESULT_KEYS

# SearchResponse in app/src/types.ts at 89ab504, plus `offset`, `filters`, `order` and `prefix`
RESPONSE_KEYS = {"query", "contact", "count", "total", "results", "offset", "filters", "order", "prefix"}
NO_FILTERS = {"contacts": [], "since": "", "until": "", "from": "", "kind": "", "attachments": ""}


def texts(found: dict[str, object]) -> list[str]:
    return [str(result["text"]) for result in found["results"]]  # type: ignore[union-attr]


def test_the_payload_has_the_keys_the_app_reads(archive: Archive, saved: Names) -> None:
    found = search.payload(archive, saved, "wheelbarrow")
    assert set(found) == RESPONSE_KEYS
    # each result also has the ranges of its words
    assert all(set(result) == RESULT_KEYS | {"ranges"} for result in found["results"])
    json.dumps(found)


def test_the_newest_matches_come_first_and_all_are_counted(archive: Archive, saved: Names) -> None:
    found = search.payload(archive, saved, "wheelbarrow", limit=1)
    assert texts(found) == ["The wheelbarrow has a flat tire"]
    # `count` is what the response holds, `total` is every match: CONTRACTS.md "Figures"
    assert (found["count"], found["total"], found["offset"]) == (1, 2, 0)
    assert (found["query"], found["contact"]) == ("wheelbarrow", "")


def test_a_search_with_no_match(archive: Archive, saved: Names) -> None:
    assert search.payload(archive, saved, "zeppelin") == {
        "query": "zeppelin",
        "contact": "",
        "results": [],
        "count": 0,
        "total": 0,
        "offset": 0,
        "filters": NO_FILTERS,
        "order": "newest",
        "prefix": False,
    }


def test_a_match_in_a_group_names_the_group_and_who_wrote_it(archive: Archive, saved: Names) -> None:
    by_text = {result["text"]: result for result in search.payload(archive, saved, "wheelbarrow")["results"]}
    asked = by_text["Who has the wheelbarrow?"]
    assert (asked["displayName"], asked["senderName"], asked["chatId"]) == ("Garden Club", "Ines Calder", world.GARDEN)
    assert by_text["The wheelbarrow has a flat tire"]["senderName"] == ""


def test_several_words_must_all_be_in_the_message(archive: Archive, saved: Names) -> None:
    assert texts(search.payload(archive, saved, "tire wheelbarrow")) == ["The wheelbarrow has a flat tire"]
    assert search.payload(archive, saved, "tire ladder")["total"] == 0


def test_a_quoted_phrase_is_matched_as_a_phrase(archive: Archive, saved: Names) -> None:
    assert texts(search.payload(archive, saved, '"flat tire"')) == ["The wheelbarrow has a flat tire"]
    assert search.payload(archive, saved, '"tire flat"')["total"] == 0


def test_a_star_makes_a_prefix(archive: Archive, saved: Names) -> None:
    assert search.payload(archive, saved, "wheelb")["total"] == 0
    assert search.payload(archive, saved, "wheelb*")["total"] == 2


def test_case_does_not_matter(archive: Archive, saved: Names) -> None:
    assert search.payload(archive, saved, "WHEELBARROW")["total"] == 2


def test_a_search_in_one_conversation(archive: Archive, saved: Names) -> None:
    found = search.payload(archive, saved, "mulch", contact=world.UNSAVED)
    assert texts(found) == ["Your delivery of mulch is on its way"]
    assert found["contact"] == world.UNSAVED
    assert search.payload(archive, saved, "mulch")["total"] == 4


def test_a_conversation_is_never_matched_by_a_part_of_its_id(archive: Archive, saved: Names) -> None:
    part = world.UNSAVED[:-1]
    assert search.payload(archive, saved, "mulch", contact=part) == {
        "query": "mulch",
        "contact": part,
        "results": [],
        "count": 0,
        "total": 0,
        "offset": 0,
        "filters": {**NO_FILTERS, "contacts": [part]},
        "order": "newest",
        "prefix": False,
    }


def test_pages(archive: Archive, saved: Names) -> None:
    first = search.payload(archive, saved, "pruning note", contact=world.INES_PHONE, limit=3)
    second = search.payload(archive, saved, "pruning note", contact=world.INES_PHONE, limit=3, offset=3)
    assert first["total"] == second["total"] == world.LONG_DAY_MESSAGES
    assert texts(first) == ["pruning note 449", "pruning note 448", "pruning note 447"]
    assert texts(second) == ["pruning note 446", "pruning note 445", "pruning note 444"]
    assert (second["offset"], second["count"]) == (3, 3)
    past = search.payload(archive, saved, "pruning note", contact=world.INES_PHONE, limit=3, offset=10_000)
    assert (past["results"], past["count"], past["total"]) == ([], 0, world.LONG_DAY_MESSAGES)


@pytest.mark.parametrize(("limit", "held"), [(0, 60), (None, 60), (-5, 1), (1, 1), (200, 200), (5000, 200), ("7", 7)])
def test_the_limit_is_held_between_one_and_two_hundred(
    archive: Archive, saved: Names, limit: object, held: int
) -> None:
    found = search.payload(archive, saved, "pruning", limit=limit)  # type: ignore[arg-type]
    assert found["count"] == min(held, world.LONG_DAY_MESSAGES)


@pytest.mark.parametrize("offset", [-1, None, "", "0"])
def test_an_offset_below_zero_or_missing_is_zero(archive: Archive, saved: Names, offset: object) -> None:
    assert search.payload(archive, saved, "wheelbarrow", offset=offset)["offset"] == 0  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["many", 2.5, [1]])
def test_a_limit_or_an_offset_that_is_no_number_is_refused(archive: Archive, saved: Names, bad: object) -> None:
    with pytest.raises(ValueError):
        search.payload(archive, saved, "wheelbarrow", limit=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        search.payload(archive, saved, "wheelbarrow", offset=bad)  # type: ignore[arg-type]


def test_text_that_looks_like_query_syntax_never_raises(archive: Archive, saved: Names) -> None:
    for typed in ('"unclosed AND ( *', "NEAR(a b)", "a:b OR c", "***", '""', "\x00", "🙂", "' OR 1=1 --"):
        found = search.payload(archive, saved, typed)
        assert found["count"] == len(found["results"])


def test_a_word_past_the_shown_part_of_a_long_text_is_still_found(archive: Archive, saved: Names) -> None:
    found = search.payload(archive, saved, world.LONG_TEXT_WORD)
    assert found["total"] == 1
    # the text is cut at 600 characters from its start, as today: the word itself is past the cut
    assert world.LONG_TEXT_WORD not in found["results"][0]["text"]


def test_a_tapback_and_a_group_event_are_never_results(archive: Archive, saved: Names) -> None:
    assert search.payload(archive, saved, "Loved")["total"] == 0
    assert search.payload(archive, saved, world.GARDEN_TITLE)["total"] == 0


def test_a_message_with_no_conversation_is_never_a_result(archive: Archive, saved: Names) -> None:
    assert search.payload(archive, saved, "conversation holds")["total"] == 0


# --- no words: the newest messages of one conversation -----------------------


def test_no_words_and_a_conversation_browse_it_from_its_newest_message(archive: Archive, saved: Names) -> None:
    found = search.payload(archive, saved, "", contact=world.GARDEN, limit=2)
    assert texts(found) == ["Tire fixed", "I will bring a pump"]
    # the whole conversation is counted, as the server does today
    assert (found["count"], found["total"], found["query"]) == (2, 5, "")
    deeper = search.payload(archive, saved, "   ", contact=world.GARDEN, limit=2, offset=2)
    assert texts(deeper) == ["The wheelbarrow has a flat tire", "I do, behind the shed"]
    assert deeper["offset"] == 2


def test_no_words_and_no_conversation_give_nothing(archive: Archive, saved: Names) -> None:
    assert search.payload(archive, saved, "") == {
        "query": "",
        "contact": "",
        "results": [],
        "count": 0,
        "total": 0,
        "offset": 0,
        "filters": NO_FILTERS,
        "order": "newest",
        "prefix": False,
    }
    assert search.payload(archive, saved, None)["total"] == 0  # type: ignore[arg-type]


def test_browsing_an_unknown_conversation_gives_nothing(archive: Archive, saved: Names) -> None:
    found = search.payload(archive, saved, "", contact="+15550000000")
    assert (found["results"], found["total"]) == ([], 0)


# --- words as they are typed -------------------------------------------------


@pytest.mark.parametrize(
    ("typed", "asked"),
    [
        ("wheelb", "wheelb*"),
        ("flat ti", "flat ti*"),
        ("wheelbarrow ", "wheelbarrow"),
        ("wheelb*", "wheelb*"),
        ('"flat tire"', '"flat tire"'),
        ('"flat ti', '"flat ti*'),
        ("  lemons", "lemons*"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_the_word_that_is_still_typed_is_a_prefix(typed: str, asked: str) -> None:
    assert search.as_typed(typed) == asked


def test_a_search_as_typed_finds_a_word_before_it_is_whole(archive: Archive, saved: Names) -> None:
    found = search.payload(archive, saved, "flat ti", as_you_type=True)
    assert texts(found) == ["The wheelbarrow has a flat tire"]
    # the payload gives back what the reader typed
    assert found["query"] == "flat ti"
    assert search.payload(archive, saved, "flat ti")["total"] == 0


def test_a_word_followed_by_a_space_is_whole(archive: Archive, saved: Names) -> None:
    assert search.payload(archive, saved, "wheelb ", as_you_type=True)["total"] == 0
