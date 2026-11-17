"""Reading messages: the Message shape, iter_messages, messages() pages and context()."""

from __future__ import annotations

import json
import types
from datetime import datetime

import pytest

from archive import paging
from tests.archive import worlds
from tests.archive.conftest import Harbor

A, B, G = worlds.ROWAN_PHONE, worlds.ROWAN_EMAIL, worlds.HARBOR_GROUP_ID
A_ALL = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"]
MESSAGE_KEYS = [
    "id", "conversation", "timestamp", "sent_at", "is_from_me", "sender", "text", "kind",
    "attachments", "reply_to", "edited", "unsent",
]  # fmt: skip
EMPTY_PAGE = {"messages": [], "has_older": False, "has_newer": False}


def flags(page: dict[str, object]) -> tuple[bool, bool]:
    return bool(page["has_older"]), bool(page["has_newer"])


class TestMessageShape:
    def test_received_text_message(self, harbor: Harbor) -> None:
        message = next(harbor.archive.iter_messages(A))
        assert list(message) == MESSAGE_KEYS
        assert message == {
            "id": harbor.id("a1"),
            "conversation": A,
            "timestamp": "2024-03-08 09:00:00",
            "sent_at": int(worlds.la(2024, 3, 8, 9, 0).timestamp()),
            "is_from_me": False,
            "sender": A,
            "text": "Morning. Coffee later?",
            "kind": "text",
            "attachments": [],
            "reply_to": None,
            "edited": False,
            "unsent": False,
        }
        assert type(message["is_from_me"]) is bool and type(message["edited"]) is bool
        json.dumps(message)

    def test_own_message_has_an_empty_sender(self, harbor: Harbor) -> None:
        message = harbor.archive.context(harbor.id("a2"), radius=0)[0]
        assert (message["is_from_me"], message["sender"]) == (True, "")

    def test_attachment_message_lists_mime_types_in_order(self, harbor: Harbor) -> None:
        message = harbor.archive.context(harbor.id("a9"), radius=0)[0]
        assert (message["kind"], message["text"]) == ("attachment", "")
        assert message["attachments"] == ["image/heic", "video/quicktime"]

    def test_reply_and_edit_fields(self, harbor: Harbor) -> None:
        reply = harbor.archive.context(harbor.id("b12"), radius=0)[0]
        edited = harbor.archive.context(harbor.id("g17"), radius=0)[0]
        assert reply["reply_to"] == harbor.id("b11") and reply["edited"] is False
        assert edited["edited"] is True and edited["reply_to"] is None

    def test_local_time_follows_the_daylight_saving_change(self, harbor: Harbor) -> None:
        before, after = (harbor.archive.message(harbor.id(label)) for label in ("g16", "g17"))
        assert before is not None and after is not None
        assert (before["timestamp"], after["timestamp"]) == ("2024-03-10 01:59:59", "2024-03-10 03:00:00")
        assert after["sent_at"] - before["sent_at"] == 1

    def test_timestamp_matches_sent_at_in_local_time(self, harbor: Harbor) -> None:
        for message in harbor.archive.iter_messages(G):
            assert message["timestamp"] == datetime.fromtimestamp(message["sent_at"]).strftime("%Y-%m-%d %H:%M:%S")

    def test_message_by_id(self, harbor: Harbor) -> None:
        found = harbor.archive.message(harbor.id("a5"))
        assert found is not None and found["text"] == "Good night."
        assert harbor.archive.message("no-such-guid") is None


class TestIterMessages:
    def test_oldest_first_and_complete(self, harbor: Harbor) -> None:
        assert harbor.labels(list(harbor.archive.iter_messages(A))) == A_ALL

    def test_it_is_a_generator(self, harbor: Harbor) -> None:
        stream = harbor.archive.iter_messages(A)
        assert isinstance(stream, types.GeneratorType)
        assert next(stream)["id"] == harbor.id("a1")

    def test_small_chunks_give_the_same_stream(self, harbor: Harbor, monkeypatch: pytest.MonkeyPatch) -> None:
        for size in (1, 2, 3, 7):
            monkeypatch.setattr(paging, "STREAM_CHUNK", size)
            assert harbor.labels(list(harbor.archive.iter_messages(A))) == A_ALL

    def test_day_range_includes_all_of_both_days(self, harbor: Harbor) -> None:
        one_day = harbor.archive.iter_messages(A, since="2024-03-10", until="2024-03-10")
        assert harbor.labels(list(one_day)) == ["a5", "a6", "a7", "a8", "a9"]
        assert (
            harbor.labels(list(harbor.archive.iter_messages(A, since="2024-03-09", until="2024-03-10"))) == A_ALL[3:9]
        )
        assert harbor.labels(list(harbor.archive.iter_messages(A, since="2024-03-11"))) == ["a10"]
        assert harbor.labels(list(harbor.archive.iter_messages(A, until="2024-03-08"))) == ["a1", "a2", "a3"]
        assert list(harbor.archive.iter_messages(A, since="2024-04-01")) == []

    def test_group_events_only_on_request(self, harbor: Harbor) -> None:
        with_events = list(harbor.archive.iter_messages(A, include_system=True))
        assert harbor.labels(with_events) == [*A_ALL, "a_event"]
        assert with_events[-1]["kind"] == "system"
        named = list(harbor.archive.iter_messages(G, include_system=True))[0]
        assert (named["kind"], named["text"]) == ("system", worlds.HARBOR_GROUP_TITLE)

    def test_unknown_conversation_is_empty(self, harbor: Harbor) -> None:
        assert list(harbor.archive.iter_messages("+15550109999")) == []

    def test_bad_day_raises_when_the_stream_starts(self, harbor: Harbor) -> None:
        with pytest.raises(ValueError):
            list(harbor.archive.iter_messages(A, since="March 10"))

    def test_several_conversations_are_merged_by_time(self, harbor: Harbor) -> None:
        merged = harbor.labels(list(harbor.archive.iter_messages([A, B])))
        assert merged == ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "b11", "b12", "a10"]
        assert harbor.labels(list(harbor.archive.iter_messages([B, A], since="2024-03-11"))) == ["b12", "a10"]


class TestMessagesPages:
    def test_newest_page_by_default(self, harbor: Harbor) -> None:
        page = harbor.archive.messages(A, limit=4)
        assert list(page) == ["messages", "has_older", "has_newer"]
        assert harbor.labels(page["messages"]) == ["a7", "a8", "a9", "a10"]
        assert flags(page) == (True, False)

    def test_default_limit_is_fifty(self, harbor: Harbor) -> None:
        page = harbor.archive.messages(A)
        assert harbor.labels(page["messages"]) == A_ALL
        assert flags(page) == (False, False)

    def test_before_an_anchor(self, harbor: Harbor) -> None:
        page = harbor.archive.messages(A, before=harbor.id("a7"), limit=3)
        assert harbor.labels(page["messages"]) == ["a4", "a5", "a6"]
        assert flags(page) == (True, True)

    def test_before_the_first_message(self, harbor: Harbor) -> None:
        page = harbor.archive.messages(A, before=harbor.id("a1"))
        assert page == {"messages": [], "has_older": False, "has_newer": True}

    def test_after_an_anchor(self, harbor: Harbor) -> None:
        page = harbor.archive.messages(A, after=harbor.id("a8"), limit=5)
        assert harbor.labels(page["messages"]) == ["a9", "a10"]
        assert flags(page) == (True, False)
        middle = harbor.archive.messages(A, after=harbor.id("a2"), limit=2)
        assert harbor.labels(middle["messages"]) == ["a3", "a4"]
        assert flags(middle) == (True, True)

    def test_after_the_last_message(self, harbor: Harbor) -> None:
        page = harbor.archive.messages(A, after=harbor.id("a10"))
        assert page == {"messages": [], "has_older": True, "has_newer": False}

    def test_around_an_anchor(self, harbor: Harbor) -> None:
        page = harbor.archive.messages(A, around=harbor.id("a5"), limit=5)
        assert harbor.labels(page["messages"]) == ["a3", "a4", "a5", "a6", "a7"]
        assert flags(page) == (True, True)

    def test_around_with_an_even_limit_puts_the_extra_message_after(self, harbor: Harbor) -> None:
        page = harbor.archive.messages(A, around=harbor.id("a5"), limit=4)
        assert harbor.labels(page["messages"]) == ["a4", "a5", "a6", "a7"]

    def test_around_the_first_message_fills_from_the_newer_side(self, harbor: Harbor) -> None:
        page = harbor.archive.messages(A, around=harbor.id("a1"), limit=5)
        assert harbor.labels(page["messages"]) == ["a1", "a2", "a3", "a4", "a5"]
        assert flags(page) == (False, True)

    def test_around_the_last_message_fills_from_the_older_side(self, harbor: Harbor) -> None:
        page = harbor.archive.messages(A, around=harbor.id("a10"), limit=4)
        assert harbor.labels(page["messages"]) == ["a7", "a8", "a9", "a10"]
        assert flags(page) == (True, False)

    def test_around_with_limit_one_is_the_anchor(self, harbor: Harbor) -> None:
        page = harbor.archive.messages(A, around=harbor.id("a5"), limit=1)
        assert harbor.labels(page["messages"]) == ["a5"]
        assert flags(page) == (True, True)

    def test_pages_walk_the_whole_conversation_without_gaps(self, harbor: Harbor) -> None:
        seen: list[str] = []
        page = harbor.archive.messages(A, limit=3)
        while True:
            seen = harbor.labels(page["messages"]) + seen
            if not page["has_older"]:
                break
            page = harbor.archive.messages(A, before=page["messages"][0]["id"], limit=3)
        assert seen == A_ALL
        forward = harbor.archive.messages(A, around=harbor.id("a1"), limit=3)
        walked = harbor.labels(forward["messages"])
        while forward["has_newer"]:
            forward = harbor.archive.messages(A, after=forward["messages"][-1]["id"], limit=3)
            walked += harbor.labels(forward["messages"])
        assert walked == A_ALL

    def test_messages_of_the_same_second_keep_their_order(self, harbor: Harbor) -> None:
        page = harbor.archive.messages(A, after=harbor.id("a7"), limit=1)
        assert harbor.labels(page["messages"]) == ["a8"]
        assert harbor.labels(harbor.archive.messages(A, before=harbor.id("a8"), limit=1)["messages"]) == ["a7"]

    def test_group_events_are_left_out_unless_asked(self, harbor: Harbor) -> None:
        assert harbor.labels(harbor.archive.messages(A, limit=2)["messages"]) == ["a9", "a10"]
        with_events = harbor.archive.messages(A, limit=2, include_system=True)
        assert harbor.labels(with_events["messages"]) == ["a10", "a_event"]
        assert harbor.archive.messages(A, after=harbor.id("a10"))["has_newer"] is False
        assert (
            harbor.archive.messages(A, after=harbor.id("a10"), include_system=True)["messages"][0]["kind"] == "system"
        )

    def test_unknown_conversation_or_anchor_gives_an_empty_page(self, harbor: Harbor) -> None:
        assert harbor.archive.messages("+15550109999") == EMPTY_PAGE
        assert harbor.archive.messages(A, around="no-such-guid") == EMPTY_PAGE
        assert harbor.archive.messages(A, before=harbor.id("b11")) == EMPTY_PAGE
        assert harbor.archive.messages(worlds.HARBOR_EMPTY_ID) == EMPTY_PAGE

    @pytest.mark.parametrize("limit", [0, -3, "10", 2.5, True])
    def test_bad_limit_raises(self, harbor: Harbor, limit: object) -> None:
        with pytest.raises(ValueError):
            harbor.archive.messages(A, limit=limit)  # type: ignore[arg-type]

    def test_more_than_one_anchor_raises(self, harbor: Harbor) -> None:
        with pytest.raises(ValueError):
            harbor.archive.messages(A, before=harbor.id("a5"), after=harbor.id("a2"))

    def test_merged_thread_of_one_person(self, harbor: Harbor) -> None:
        newest = harbor.archive.messages([A, B], limit=3)
        assert harbor.labels(newest["messages"]) == ["b11", "b12", "a10"]
        assert flags(newest) == (True, False)
        middle = harbor.archive.messages([A, B], around=harbor.id("b11"), limit=5)
        assert harbor.labels(middle["messages"]) == ["a8", "a9", "b11", "b12", "a10"]
        assert flags(middle) == (True, False)
        older = harbor.archive.messages([A, B], before=harbor.id("b12"), limit=2)
        assert harbor.labels(older["messages"]) == ["a9", "b11"]
        assert flags(older) == (True, True)


class TestContext:
    def test_neighbours_on_both_sides(self, harbor: Harbor) -> None:
        assert harbor.labels(harbor.archive.context(harbor.id("a5"), radius=2)) == ["a3", "a4", "a5", "a6", "a7"]

    def test_default_radius_is_six(self, harbor: Harbor) -> None:
        assert harbor.labels(harbor.archive.context(harbor.id("a5"))) == A_ALL

    def test_edges_and_zero_radius(self, harbor: Harbor) -> None:
        assert harbor.labels(harbor.archive.context(harbor.id("a1"), radius=2)) == ["a1", "a2", "a3"]
        assert harbor.labels(harbor.archive.context(harbor.id("a10"), radius=1)) == ["a9", "a10"]
        assert harbor.labels(harbor.archive.context(harbor.id("a5"), radius=0)) == ["a5"]

    def test_neighbours_are_from_the_same_conversation(self, harbor: Harbor) -> None:
        around = harbor.archive.context(harbor.id("g16"), radius=3)
        assert {message["conversation"] for message in around} == {G}
        assert harbor.labels(around) == ["g15", "g18", "g19", "g16", "g17"]

    def test_unknown_message_is_an_empty_list(self, harbor: Harbor) -> None:
        assert harbor.archive.context("no-such-guid") == []

    def test_group_events_only_on_request(self, harbor: Harbor) -> None:
        assert harbor.labels(harbor.archive.context(harbor.id("g13"), radius=1)) == ["g13", "g14"]
        assert harbor.labels(harbor.archive.context(harbor.id("g13"), radius=1, include_system=True)) == [
            "g_renamed",
            "g13",
            "g14",
        ]
        assert harbor.archive.context(harbor.id("g_renamed"), radius=1) == harbor.archive.context(
            harbor.id("g13"), radius=0
        )

    def test_bad_radius_raises(self, harbor: Harbor) -> None:
        with pytest.raises(ValueError):
            harbor.archive.context(harbor.id("a5"), radius=-1)
