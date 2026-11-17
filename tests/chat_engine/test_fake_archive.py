"""The fake must follow the archive contract, or the engine tests prove nothing."""

import pytest

from chat_engine.fake_archive import FakeArchive, make_message
from tests.chat_engine.tiny import (
    GROUP,
    INES_PHONE,
    JONAS_PHONE,
    MAYA_EMAIL,
    MAYA_PHONE,
    UNSAVED_PHONE,
    tiny_archive,
    tiny_messages,
)


def _ids(result: dict) -> list[str]:
    return [hit["message"]["id"] for hit in result["hits"]]


def test_summary_spans_the_archive() -> None:
    summary = tiny_archive().summary()
    assert summary == {
        "messages": len(tiny_messages()),
        "conversations": 7,
        "first_timestamp": "2019-06-01 09:00:00",
        "last_timestamp": "2026-08-01 10:00:00",
    }


def test_conversations_by_count_and_recent() -> None:
    archive = tiny_archive()
    by_count = archive.conversations(order="count")
    assert by_count[0]["id"] == MAYA_PHONE
    assert by_count[0]["message_count"] == 9
    assert by_count[0]["sent_count"] == 4 and by_count[0]["received_count"] == 5
    by_recent = archive.conversations(order="recent")
    assert by_recent[0]["id"] == JONAS_PHONE
    assert [c["id"] for c in archive.conversations(order="count", limit=2, offset=1)] == [
        c["id"] for c in by_count[1:3]
    ]
    assert archive.conversation(GROUP)["members"] == [MAYA_PHONE, JONAS_PHONE, UNSAVED_PHONE]
    assert archive.conversation("nope") is None


def test_search_ands_words_and_ignores_case_and_diacritics() -> None:
    archive = tiny_archive()
    assert _ids(archive.search("LEASE", order="oldest")) == ["g4", "a2", "b1", "a5"]
    assert _ids(archive.search("lease lisbon", order="newest")) == ["g4"]
    assert _ids(archive.search("apartamento otimo")) == ["p3"]
    assert archive.search("zanzibar")["total"] == 0


def test_search_phrase_prefix_and_bad_syntax() -> None:
    archive = tiny_archive()
    assert _ids(archive.search('"lease got signed"')) == ["a2"]
    assert _ids(archive.search('"signed lease"')) == []
    assert _ids(archive.search("sourd*", order="oldest")) == ["j1", "j4"]
    assert _ids(archive.search('"lease got', order="oldest")) == ["a2"]
    assert archive.search("")["total"] == 0
    assert archive.search('"" ***')["total"] == 0


def test_search_filters_limit_offset_and_ranges() -> None:
    archive = tiny_archive()
    scoped = archive.search("lease", conversations=[MAYA_PHONE], order="newest")
    assert _ids(scoped) == ["a5", "a2"]
    assert _ids(archive.search("lease", since="2022-01-01", until="2022-12-31")) == ["b1"]
    assert _ids(archive.search("sourdough", from_me=True)) == ["j4"]
    page = archive.search("lease", order="oldest", limit=2, offset=1)
    assert page["total"] == 4 and _ids(page) == ["a2", "b1"]
    hit = archive.search("lease", conversations=[MAYA_EMAIL])["hits"][0]
    start, end = hit["ranges"][0]
    assert hit["message"]["text"][start:end] == "lease"


def test_search_relevance_prefers_more_matches() -> None:
    messages = [
        make_message("x1", "c", "2024-01-01 10:00:00", "tea"),
        make_message("x2", "c", "2024-01-02 10:00:00", "tea tea tea and more tea"),
        make_message("x3", "c", "2024-01-03 10:00:00", "tea time"),
    ]
    archive = FakeArchive(messages)
    assert _ids(archive.search("tea", order="relevance"))[0] == "x2"


def test_count_totals_and_filters() -> None:
    archive = tiny_archive()
    assert archive.count() == len(tiny_messages())
    assert archive.count(conversations=[MAYA_PHONE]) == 9
    assert archive.count(conversations=[MAYA_PHONE], from_me=True) == 4
    assert archive.count(text="lease") == 4
    assert archive.count(since="2025-01-01", until="2025-12-31") == 6


def test_count_groups() -> None:
    archive = tiny_archive()
    by_year = archive.count(conversations=[MAYA_PHONE], group_by="year")
    assert by_year == [
        {"key": "2021", "total": 4, "sent": 2, "received": 2},
        {"key": "2023", "total": 3, "sent": 1, "received": 2},
        {"key": "2025", "total": 2, "sent": 1, "received": 1},
    ]
    by_month = archive.count(conversations=[JONAS_PHONE], group_by="month")
    assert [row["key"] for row in by_month] == ["2024-02", "2026-08"]
    by_day = archive.count(conversations=[INES_PHONE], group_by="day")
    assert [row["key"] for row in by_day] == ["2024-06-01", "2024-06-02"]
    by_conversation = archive.count(text="lease", group_by="conversation")
    assert {row["key"]: row["total"] for row in by_conversation} == {
        MAYA_PHONE: 2,
        MAYA_EMAIL: 1,
        GROUP: 1,
    }
    by_sender = archive.count(conversations=[GROUP], group_by="sender")
    assert {row["key"]: row["total"] for row in by_sender} == {
        "": 1,
        MAYA_PHONE: 2,
        JONAS_PHONE: 1,
        UNSAVED_PHONE: 1,
    }
    by_hour = archive.count(conversations=[GROUP], since="2019-06-01", until="2019-06-01", group_by="weekday_hour")
    assert by_hour == [{"key": "5-09", "total": 4, "sent": 1, "received": 3}]


def test_context_stays_in_one_conversation() -> None:
    archive = tiny_archive()
    around = archive.context("a5", radius=1)
    assert [m["id"] for m in around] == ["a4", "a5", "a6"]
    assert [m["id"] for m in archive.context("a1", radius=2)] == ["a1", "a2", "a3"]
    assert archive.context("missing", radius=3) == []


def test_messages_pages_by_id_cursor() -> None:
    archive = tiny_archive()
    newest = archive.messages(MAYA_PHONE, limit=3)
    assert [m["id"] for m in newest["messages"]] == ["a7", "a8", "a9"]
    assert newest["has_older"] and not newest["has_newer"]
    older = archive.messages(MAYA_PHONE, before="a7", limit=3)
    assert [m["id"] for m in older["messages"]] == ["a4", "a5", "a6"]
    newer = archive.messages(MAYA_PHONE, after="a1", limit=2)
    assert [m["id"] for m in newer["messages"]] == ["a2", "a3"]
    assert newer["has_older"] and newer["has_newer"]


def test_messages_around_keeps_the_anchor_and_fills_the_page() -> None:
    archive = tiny_archive()
    middle = archive.messages(MAYA_PHONE, around="a5", limit=4)
    assert [m["id"] for m in middle["messages"]] == ["a4", "a5", "a6", "a7"]
    at_start = archive.messages(MAYA_PHONE, around="a1", limit=4)
    assert [m["id"] for m in at_start["messages"]] == ["a1", "a2", "a3", "a4"]
    assert not at_start["has_older"] and at_start["has_newer"]
    at_end = archive.messages(MAYA_PHONE, around="a9", limit=4)
    assert [m["id"] for m in at_end["messages"]] == ["a6", "a7", "a8", "a9"]


def test_messages_with_unknown_anchor_or_two_cursors() -> None:
    archive = tiny_archive()
    empty = {"messages": [], "has_older": False, "has_newer": False}
    assert archive.messages(MAYA_PHONE, around="nope") == empty
    assert archive.messages("no-such-conversation") == empty
    with pytest.raises(ValueError):
        archive.messages(MAYA_PHONE, before="a2", after="a1")


def test_stats_use_the_field_names_of_the_archive() -> None:
    stats = tiny_archive().stats(MAYA_PHONE)
    assert stats["conversation"] == MAYA_PHONE
    assert stats["total_messages"] == 9 and stats["sent_count"] == 4 and stats["received_count"] == 5
    assert stats["first_timestamp"] == "2021-03-04 18:20:00"
    assert stats["last_timestamp"] == "2025-12-24 20:01:00"
    assert stats["active_days"] == 3
    assert stats["avg_messages_per_day"] == 3.0
    assert stats["busiest_day"] == "2021-03-04" and stats["busiest_day_count"] == 4
    assert 897.4 < stats["longest_gap_days"] < 897.5
    assert stats["longest_gap_start"] == "2023-07-11 09:05:00"
    assert stats["longest_gap_end"] == "2025-12-24 20:00:00"
    ranged = tiny_archive().stats(MAYA_PHONE, since="2023-01-01", until="2023-12-31")
    assert ranged["total_messages"] == 3
    empty = tiny_archive().stats("nobody")
    assert empty["total_messages"] == 0 and empty["busiest_day"] is None and empty["longest_gap_days"] == 0.0


def test_every_call_is_recorded_and_inputs_are_not_shared() -> None:
    archive = tiny_archive()
    archive.search("lease", conversations=[MAYA_PHONE], order="newest")
    archive.count(text="lease")
    assert [name for name, _ in archive.calls] == ["search", "count"]
    assert archive.calls[0][1]["conversations"] == [MAYA_PHONE]
    hit = archive.search("lease")["hits"][0]["message"]
    hit["text"] = "tampered"
    assert archive.search("tampered")["total"] == 0


def test_system_messages_are_left_out_unless_asked_for() -> None:
    messages = [
        make_message("s1", "chat9", "2024-01-01 10:00:00", "hello all", sender="+15550000001"),
        make_message("s2", "chat9", "2024-01-01 10:01:00", "Maya named the group hello", kind="system"),
        make_message("s3", "chat9", "2024-01-01 10:02:00", "hello again", sender="+15550000001"),
    ]
    archive = FakeArchive(messages, [{"id": "chat9", "is_group": True}, {"id": "empty", "is_group": False}])
    assert archive.count() == 2 and archive.count(include_system=True) == 3
    assert archive.search("hello")["total"] == 2
    assert [m["id"] for m in archive.context("s1", radius=3)] == ["s1", "s3"]
    assert archive.context("s2", radius=3) == []
    assert [m["id"] for m in archive.messages("chat9")["messages"]] == ["s1", "s3"]
    assert archive.stats("chat9")["total_messages"] == 2
    assert [c["id"] for c in archive.conversations()] == ["chat9"]
    assert archive.summary()["conversations"] == 1


def test_a_negative_offset_reads_from_the_start() -> None:
    archive = tiny_archive()
    assert archive.conversations(limit=2, offset=-3) == archive.conversations(limit=2, offset=0)
    assert archive.conversations(limit=-1) == []
    first = archive.search("lease", limit=2, offset=-5)
    assert first == archive.search("lease", limit=2, offset=0)


def test_an_unknown_group_is_refused_also_when_nothing_matches() -> None:
    archive = tiny_archive()
    with pytest.raises(ValueError):
        archive.count(text="zzzznothing", group_by="decade")
    with pytest.raises(ValueError):
        archive.count(conversations=[], group_by="decade")
