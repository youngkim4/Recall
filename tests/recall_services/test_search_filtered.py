"""search.py: filters, order, whole words or a prefix, and the answer to a query string.

The Orchard history is in tests/recall_services/world.py.
"""

from __future__ import annotations

import json

import pytest

from archive import Archive
from recall_services import search
from recall_services.names import Names
from recall_services.search_filters import Filters, read_filters
from tests.recall_services import world

NOTHING_APPLIED = {"contacts": [], "since": "", "until": "", "from": "", "kind": "", "attachments": ""}


def texts(found: dict[str, object]) -> list[str]:
    return [str(result["text"]) for result in found["results"]]  # type: ignore[union-attr]


def asked(archive: Archive, saved: Names, **params: str) -> dict[str, object]:
    return search.answer(archive, saved, params)


def test_the_payload_says_what_it_applied(archive: Archive, saved: Names) -> None:
    found = asked(archive, saved, query="wheelbarrow")
    assert (found["filters"], found["order"], found["prefix"]) == (NOTHING_APPLIED, "newest", False)
    narrowed = asked(archive, saved, query="wheelbarrow", kind="group", since="2024-04-01", order="oldest")
    assert narrowed["filters"] == {**NOTHING_APPLIED, "kind": "group", "since": "2024-04-01"}
    assert narrowed["order"] == "oldest"
    json.dumps(narrowed)


def test_whole_words_unless_the_reader_is_typing(archive: Archive, saved: Names) -> None:
    assert asked(archive, saved, query="wheelb")["total"] == 0
    typing = asked(archive, saved, query="wheelb", prefix="true")
    assert (typing["total"], typing["prefix"], typing["query"]) == (2, True, "wheelb")
    # a star typed by the reader is always a prefix
    assert asked(archive, saved, query="wheelb*", prefix="false")["total"] == 2


def test_several_conversations(archive: Archive, saved: Names) -> None:
    both = asked(archive, saved, query="mulch", contacts=f"{world.UNSAVED},{world.UNTITLED}")
    assert both["total"] == 4
    one = asked(archive, saved, query="mulch", contacts=world.UNSAVED)
    assert texts(one) == ["Your delivery of mulch is on its way"]
    # the one `contact` of today still works, and adds to the list
    joined = asked(archive, saved, query="mulch", contact=world.UNSAVED, contacts=world.UNTITLED)
    assert (joined["total"], joined["contact"]) == (4, world.UNSAVED)
    assert joined["filters"]["contacts"] == [world.UNSAVED, world.UNTITLED]  # type: ignore[index]


def test_days_include_both_ends(archive: Archive, saved: Names) -> None:
    assert asked(archive, saved, query="pruning")["total"] == world.LONG_DAY_MESSAGES + 1
    same_day = asked(archive, saved, query="pruning", since=world.LONG_DAY, until=world.LONG_DAY)
    assert same_day["total"] == world.LONG_DAY_MESSAGES
    assert asked(archive, saved, query="pruning", until="2024-03-01")["total"] == 1


def test_who_wrote_it(archive: Archive, saved: Names) -> None:
    # every third note of the long day is the owner's, and "the pruning saw"
    assert asked(archive, saved, query="pruning", **{"from": "me"})["total"] == 151
    assert asked(archive, saved, query="pruning", **{"from": "them"})["total"] == 300


def test_direct_or_group(archive: Archive, saved: Names) -> None:
    assert asked(archive, saved, query="wheelbarrow", kind="group")["total"] == 2
    assert asked(archive, saved, query="wheelbarrow", kind="direct")["total"] == 0


def test_no_words_and_a_filter_list_those_messages_newest_first(archive: Archive, saved: Names) -> None:
    found = asked(archive, saved, attachments="image")
    assert [result["attachment"] for result in found["results"]] == ["2 photos", "Photo"]  # type: ignore[index]
    assert (found["total"], found["count"], found["query"]) == (2, 2, "")
    after = asked(archive, saved, attachments="any", since="2024-03-02")
    assert after["total"] == 1


def test_no_words_and_no_filter_give_nothing(archive: Archive, saved: Names) -> None:
    found = asked(archive, saved)
    assert (found["results"], found["total"]) == ([], 0)
    # a query of stars has no word: it gives nothing too
    assert asked(archive, saved, query="***")["total"] == 0


def test_no_words_list_a_conversation_deeper_than_before(archive: Archive, saved: Names) -> None:
    found = search.payload(archive, saved, "", contact=world.INES_PHONE, limit=2, offset=457)
    assert found["total"] == 459
    assert texts(found) == ["Yes, and the pruning saw", "Are you bringing the ladder on Saturday?"]


def test_order(archive: Archive, saved: Names) -> None:
    oldest = asked(archive, saved, query="wheelbarrow", order="oldest")
    assert texts(oldest) == ["Who has the wheelbarrow?", "The wheelbarrow has a flat tire"]
    best = asked(archive, saved, query="wheelbarrow tire", order="relevance")
    assert texts(best) == ["The wheelbarrow has a flat tire"]
    no_words = search.payload(archive, saved, "", filters=Filters(contacts=(world.GARDEN,)), order="relevance")
    assert texts(no_words)[0] == "Tire fixed"


def test_filters_given_as_an_object(archive: Archive, saved: Names) -> None:
    filters = read_filters({"kind": "group", "from": "me"})
    found = search.payload(archive, saved, "", filters=filters)
    assert texts(found) == ["Almost, two more weeks", "I will bring a pump"]


@pytest.mark.parametrize(
    "params",
    [
        {"order": "best"},
        {"prefix": "yes"},
        {"since": "yesterday"},
        {"from": "everyone"},
        {"kind": "self"},
        {"attachments": "gif"},
        {"limit": "many"},
        {"offset": "later"},
    ],
)
def test_a_value_that_is_not_allowed_is_refused(archive: Archive, saved: Names, params: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        asked(archive, saved, query="wheelbarrow", **params)
