"""One invented export in the chat package's `FrameArchive` and in the real archive.

`FrameArchive` is the stand-in that the chat engine was built against. The two
are compared only where equality has a meaning: totals, counts, the pages of a
search in time order, message pages, cursors and context.

Three things differ by design. For each one a test states the rule of the real
archive, and the two are not compared:

- Member order. The archive puts the member who wrote the most first.
  `FrameArchive` keeps the order of the first messages. The label of a group in
  a tool result differs for this reason.
- Clock changes. The archive stores true UTC: `sent_at` and the longest gap are
  elapsed time, and a local time that the clocks skip is read one hour later.
  `FrameArchive` reads a local time as if it were UTC.
- Relevance order. It is the archive's own rank.
"""

from __future__ import annotations

import calendar
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from tests.integration import engine_data as data
from tests.integration.engine_data import Seam

SECONDS_PER_DAY = 86_400
NO_SUCH_WORD = "zzzzqqqqxxxx"
SAME_IN_BOTH = ("conversation", "is_from_me", "sender", "text", "kind", "attachments", "reply_to", "edited", "unsent")
SAME_STATS = (
    "conversation",
    "total_messages",
    "sent_count",
    "received_count",
    "active_days",
    "avg_messages_per_day",
    "busiest_day",
    "busiest_day_count",
)
Arguments = Callable[[Any], dict[str, Any]]

COUNTS: dict[str, Arguments] = {
    "all": lambda picks: {},
    "by year": lambda picks: {"group_by": "year"},
    "by month": lambda picks: {"group_by": "month"},
    "by day": lambda picks: {"group_by": "day"},
    "by conversation": lambda picks: {"group_by": "conversation"},
    "by sender": lambda picks: {"group_by": "sender"},
    "largest conversation": lambda picks: {"conversations": [picks.largest]},
    "largest by day": lambda picks: {"conversations": [picks.largest], "group_by": "day"},
    "largest by sender": lambda picks: {"conversations": [picks.largest], "group_by": "sender"},
    "group by sender": lambda picks: {"conversations": [picks.group], "group_by": "sender"},
    "a sender with no handle": lambda picks: {"conversations": [data.CLOCK_CHAT], "group_by": "sender"},
    "one year": lambda picks: {"since": picks.since, "until": picks.until},
    "one year, sent": lambda picks: {"since": picks.since, "until": picks.until, "from_me": True},
    "one year, received": lambda picks: {"since": picks.since, "until": picks.until, "from_me": False},
    "a word": lambda picks: {"text": picks.medium},
    "a word by year": lambda picks: {"text": picks.common, "group_by": "year"},
    "a phrase": lambda picks: {"text": picks.phrase},
    "a prefix": lambda picks: {"text": picks.medium[:4] + "*"},
    "a year with no message": lambda picks: {
        "conversations": [picks.largest],
        "since": picks.empty_since,
        "until": picks.empty_until,
    },
    "no conversation at all": lambda picks: {"conversations": []},
}

SEARCHES: dict[str, Arguments] = {
    "common word": lambda picks: {"query": picks.common},
    "common word, oldest first": lambda picks: {"query": picks.common, "order": "oldest"},
    "medium word": lambda picks: {"query": picks.medium},
    "medium word, oldest first": lambda picks: {"query": picks.medium, "order": "oldest"},
    "rare word": lambda picks: {"query": picks.rare},
    "two words": lambda picks: {"query": f"{picks.medium} {picks.common}"},
    "phrase": lambda picks: {"query": picks.phrase},
    "prefix": lambda picks: {"query": picks.medium[:4] + "*"},
    "upper case": lambda picks: {"query": picks.medium.upper()},
    "second page": lambda picks: {"query": picks.common, "offset": 50},
    "in the largest conversation": lambda picks: {"query": picks.common, "conversations": [picks.largest]},
    "in one year": lambda picks: {"query": picks.common, "since": picks.since, "until": picks.until},
    "sent by the owner": lambda picks: {"query": picks.common, "from_me": True},
    "received": lambda picks: {"query": picks.common, "from_me": False},
    "word of the longest text": lambda picks: {"query": picks.longest_word, "limit": 20},
    "no match": lambda picks: {"query": NO_SUCH_WORD},
    "empty query": lambda picks: {"query": ""},
    "text that is no query syntax": lambda picks: {"query": '"unclosed AND ( *'},
    "no conversation at all": lambda picks: {"query": picks.common, "conversations": []},
}


@pytest.fixture(scope="module")
def seam(request: pytest.FixtureRequest) -> Seam:
    return data.shared_seam(request)


def _listed(archive: Any, order: str = "count") -> list[dict[str, Any]]:
    return archive.conversations(limit=None, offset=0, order=order)


def _by_id(conversations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(conversation["id"]): conversation for conversation in conversations}


def _page(page: Mapping[str, Any]) -> tuple[list[str], bool, bool]:
    return [data.plain_id(message["id"]) for message in page["messages"]], page["has_older"], page["has_newer"]


def _ids(messages: Any) -> list[str]:
    return [data.plain_id(message["id"]) for message in messages]


# --- totals --------------------------------------------------------------------


def test_the_bridge_stores_every_row_of_the_export(seam: Seam) -> None:
    counts = seam.imported.counts
    assert (seam.imported.status, seam.imported.mode) == ("ok", "legacy_csv")
    assert counts["rows_seen"] == len(seam.frame) == data.ROWS + len(data.CLOCK_ROWS)
    assert dict(seam.imported.skipped) == {}
    assert counts["messages_seen"] + counts["reaction_phrases"] == counts["rows_seen"]
    assert counts["reaction_phrases"] > 0
    assert counts["oversize_texts"] > 0
    # each package has its own rule for a reaction phrase: on this export the two rules agree
    assert seam.frame_archive.dropped == {"bad_rows": 0, "duplicates": 0, "reactions": counts["reaction_phrases"]}


def test_summary_totals_agree(seam: Seam) -> None:
    frame, real = seam.frame_archive.summary(), seam.adapter.summary()
    assert real["messages"] == frame["messages"] == seam.imported.counts["messages_seen"]
    assert real["conversations"] == frame["conversations"] == len(_listed(seam.adapter))
    assert real["first_timestamp"] == data.as_archive_reads(frame["first_timestamp"])
    assert real["last_timestamp"] == data.as_archive_reads(frame["last_timestamp"])
    assert real["messages_without_conversation"] == 0


def test_conversations_have_the_same_counts_and_people(seam: Seam) -> None:
    frame, real = _by_id(_listed(seam.frame_archive)), _by_id(_listed(seam.adapter))
    assert set(real) == set(frame)
    assert data.CLOCK_CHAT in real
    for conversation_id, mine in real.items():
        theirs = frame[conversation_id]
        for key in ("is_group", "name", "message_count", "sent_count", "received_count"):
            assert mine[key] == theirs[key], (conversation_id, key)
        assert mine["first_timestamp"] == data.as_archive_reads(theirs["first_timestamp"])
        assert mine["last_timestamp"] == data.as_archive_reads(theirs["last_timestamp"])
        assert set(mine["members"]) == set(theirs["members"]), conversation_id
        if not mine["is_group"]:
            assert mine["members"] == theirs["members"] == [conversation_id]
        # only the archive knows people: a conversation with one person names that person
        assert (mine["person_id"] is None) == mine["is_group"]


def test_the_archive_puts_the_busiest_member_first(seam: Seam) -> None:
    """Member order. The member who wrote the most comes first; the same number goes by the text of the handle.

    `FrameArchive` has the order of the first messages, so the two lists are not compared.
    """
    received = seam.frame[seam.frame["is_from_me"] == "0"]
    written = data.message_rows(received).groupby(["chat_id", "sender"]).size()
    groups = [conversation for conversation in _listed(seam.adapter) if conversation["is_group"]]
    assert len(groups) >= 3
    for group in groups:
        seen = set(received.loc[received["chat_id"] == group["id"], "sender"]) - {""}
        by_count = sorted(seen, key=lambda handle: (-int(written.get((group["id"], handle), 0)), handle))
        assert group["members"] == by_count, group["id"]


def test_conversations_come_in_the_order_of_the_archive(seam: Seam) -> None:
    """The archive lists the same sizes as `FrameArchive`. Two conversations of one size have its own order:
    the later last message first. `FrameArchive` has them by id, so the ids are not compared.
    """
    by_chat = data.message_rows(seam.frame).groupby("chat_id")["message_id"]
    last = {str(chat): max(seam.moments[row_id] for row_id in row_ids) for chat, row_ids in by_chat}
    real, frame = _listed(seam.adapter), _listed(seam.frame_archive)
    assert [conversation["message_count"] for conversation in real] == [c["message_count"] for c in frame]
    by_size = sorted(real, key=lambda c: (-c["message_count"], -last[c["id"]], c["id"]))
    assert [conversation["id"] for conversation in real] == [conversation["id"] for conversation in by_size]
    recent = [conversation["id"] for conversation in _listed(seam.adapter, "recent")]
    assert recent == sorted(last, key=lambda conversation_id: (-last[conversation_id], conversation_id))
    assert recent == [conversation["id"] for conversation in _listed(seam.frame_archive, "recent")]


# --- counts --------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(COUNTS))
def test_count_agrees(seam: Seam, case: str) -> None:
    arguments = COUNTS[case](seam.picks)
    real = seam.adapter.count(**arguments)
    assert real == seam.frame_archive.count(**arguments)
    if case not in ("a year with no message", "no conversation at all"):
        assert real, "the case counts nothing: it proves nothing"


def test_a_received_message_with_no_handle_has_its_own_sender_key(seam: Seam) -> None:
    rows = seam.adapter.count(conversations=[data.CLOCK_CHAT], group_by="sender")
    assert {row["key"]: row["total"] for row in rows} == {"": 4, data.CLOCK_CHAT: 4, "unknown": 1}


def test_weekday_and_hour_follow_the_clock_of_the_zone(seam: Seam) -> None:
    """Clock changes. The archive counts a message in the hour that the clocks showed.

    `FrameArchive` counts the hour that the text of the export names, also an
    hour that the clocks skipped. So the two tables are not compared.
    """
    stamps = list(data.message_rows(seam.frame)["timestamp"])
    on_the_clock = Counter(data.weekday_hour(data.as_archive_reads(stamp)) for stamp in stamps)
    real = {row["key"]: row["total"] for row in seam.adapter.count(group_by="weekday_hour")}
    assert real == dict(on_the_clock)
    as_written = Counter(data.weekday_hour(stamp) for stamp in stamps)
    frame = {row["key"]: row["total"] for row in seam.frame_archive.count(group_by="weekday_hour")}
    assert frame == dict(as_written)
    assert frame != real


# --- search --------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(SEARCHES))
def test_a_search_in_time_order_agrees(seam: Seam, case: str) -> None:
    arguments = {"limit": 50, "offset": 0, "order": "newest", **SEARCHES[case](seam.picks)}
    query = arguments.pop("query")
    frame, real = seam.frame_archive.search(query, **arguments), seam.adapter.search(query, **arguments)
    assert real["total"] == frame["total"]
    assert _ids(hit["message"] for hit in real["hits"]) == _ids(hit["message"] for hit in frame["hits"])
    assert [hit["ranges"] for hit in real["hits"]] == [hit["ranges"] for hit in frame["hits"]]
    nothing = case in ("no match", "empty query", "text that is no query syntax", "no conversation at all")
    assert (real["total"] == 0) == nothing


def test_relevance_order_is_the_archives_own(seam: Seam) -> None:
    """Relevance order. The page holds matches only, each one time. Its order is the rank of the
    archive's index, and it can change, so it is not compared with the order of `FrameArchive`.
    """
    word = seam.picks.medium
    every = seam.adapter.search(word, limit=10_000, order="oldest")
    ranked = seam.adapter.search(word, limit=50, order="relevance")
    assert ranked["total"] == every["total"] == seam.frame_archive.search(word, order="relevance")["total"]
    assert every["total"] > 50
    ids = [hit["message"]["id"] for hit in ranked["hits"]]
    assert len(ids) == len(set(ids)) == 50
    assert set(ids) <= {hit["message"]["id"] for hit in every["hits"]}
    for hit in ranked["hits"]:
        marked = {hit["message"]["text"][start:end].lower() for start, end in hit["ranges"]}
        assert marked == {word}


# --- pages, cursors, context ---------------------------------------------------


def test_pages_and_cursors_agree(seam: Seam) -> None:
    frame, real, largest = seam.frame_archive, seam.adapter, seam.picks.largest
    newest = frame.messages(largest, limit=50)
    assert _page(real.messages(largest, limit=50)) == _page(newest)
    assert newest["has_older"] and not newest["has_newer"]
    first = newest["messages"][0]["id"]
    older = frame.messages(largest, before=first, limit=50)
    assert _page(real.messages(largest, before=data.legacy_id(first), limit=50)) == _page(older)
    anchor = older["messages"][len(older["messages"]) // 2]["id"]
    for cursor in ("after", "around"):
        mine = real.messages(largest, limit=50, **{cursor: data.legacy_id(anchor)})
        assert _page(mine) == _page(frame.messages(largest, limit=50, **{cursor: anchor})), cursor
        assert len(mine["messages"]) == 50


@pytest.mark.parametrize("radius", [1, 6, 15])
def test_context_agrees(seam: Seam, radius: int) -> None:
    page = seam.frame_archive.messages(seam.picks.largest, limit=200)["messages"]
    anchor = page[len(page) // 2]["id"]
    real = seam.adapter.context(data.legacy_id(anchor), radius=radius)
    assert _ids(real) == _ids(seam.frame_archive.context(anchor, radius=radius))
    assert len(real) == 2 * radius + 1


def test_an_unknown_anchor_or_conversation_gives_nothing(seam: Seam) -> None:
    empty = {"messages": [], "has_older": False, "has_newer": False}
    for archive in (seam.adapter, seam.frame_archive):
        assert archive.messages(seam.picks.largest, around="nope", limit=5) == empty
        assert archive.messages("nobody", limit=5) == empty
        assert archive.context("nope") == []
        assert archive.conversation("nobody") is None


def test_a_walk_of_pages_reads_a_whole_conversation(seam: Seam) -> None:
    real, middle = seam.adapter, seam.picks.middle
    page = real.messages(middle, limit=40)
    walked = list(page["messages"])
    while page["has_older"]:
        page = real.messages(middle, before=page["messages"][0]["id"], limit=40)
        walked = [*page["messages"], *walked]
    assert len(walked) == real.conversation(middle)["message_count"] > 80
    assert _ids(walked) == _ids(real.iter_messages(middle)) == _ids(seam.frame_archive.iter_messages(middle))


# --- messages ------------------------------------------------------------------


def test_every_conversation_holds_the_same_messages(seam: Seam) -> None:
    """The order of the archive is (send time, ROWID of the export). `sent_at` is true UTC."""
    for conversation_id in seam.conversation_ids():
        frame = {message["id"]: message for message in seam.frame_archive.iter_messages(conversation_id)}
        real = list(seam.adapter.iter_messages(conversation_id))
        in_time = sorted(frame, key=lambda row_id: (seam.moments[row_id], int(row_id)))
        assert _ids(real) == in_time, conversation_id
        for message in real:
            theirs = frame[data.plain_id(message["id"])]
            assert {key: message[key] for key in SAME_IN_BOTH} == {key: theirs[key] for key in SAME_IN_BOTH}
            assert message["timestamp"] == data.as_archive_reads(theirs["timestamp"])
            assert message["sent_at"] == seam.moments[theirs["id"]]


def test_the_two_orders_agree_where_the_clocks_skip_no_time(seam: Seam) -> None:
    skipped = {row_id for row_id, stamp in seam.stamps.items() if data.as_archive_reads(stamp) != stamp}
    compared = 0
    for conversation_id in seam.conversation_ids():
        frame = _ids(seam.frame_archive.iter_messages(conversation_id))
        if skipped.isdisjoint(frame):
            compared += 1
            assert _ids(seam.adapter.iter_messages(conversation_id)) == frame, conversation_id
    assert compared >= len(seam.conversation_ids()) - len(skipped)


def test_a_time_that_the_clocks_skip_is_read_one_hour_later(seam: Seam) -> None:
    """Clock changes. 02:30 did not exist on 2021-03-14: the archive has the message at 03:30,
    after the message of 03:10. `FrameArchive` keeps the text of the export and its order.
    """
    first_id = data.ROWS + 1
    by_stamp = {stamp: str(first_id + place) for place, (stamp, *_rest) in enumerate(data.CLOCK_ROWS)}
    real = list(seam.adapter.iter_messages(data.CLOCK_CHAT))
    moved = next(message for message in real if message["id"] == data.legacy_id(by_stamp[data.SKIPPED_TIME]))
    assert moved["timestamp"] == "2021-03-14 03:30:00"
    assert moved["sent_at"] == calendar.timegm((2021, 3, 14, 10, 30, 0))
    order = _ids(real)
    assert order.index(by_stamp["2021-03-14 03:10:00"]) + 1 == order.index(by_stamp[data.SKIPPED_TIME])
    frame = list(seam.frame_archive.iter_messages(data.CLOCK_CHAT))
    assert [message["timestamp"] for message in frame] == sorted(stamp for stamp, *_rest in data.CLOCK_ROWS)
    assert sorted(_ids(real)) == sorted(_ids(frame))


def test_a_time_that_comes_twice_is_its_first_pass(seam: Seam) -> None:
    """Clock changes. 01:30 came two times on 2021-11-07. The archive reads the first one: summer time."""
    double = next(
        message for message in seam.adapter.iter_messages(data.CLOCK_CHAT) if message["timestamp"] == data.DOUBLE_TIME
    )
    assert double["sent_at"] == calendar.timegm((2021, 11, 7, 8, 30, 0))
    theirs = next(
        message
        for message in seam.frame_archive.iter_messages(data.CLOCK_CHAT)
        if message["timestamp"] == data.DOUBLE_TIME
    )
    assert theirs["sent_at"] == calendar.timegm((2021, 11, 7, 1, 30, 0))  # the local time read as UTC


# --- figures of one conversation -----------------------------------------------


def test_stats_agree_but_for_the_longest_gap(seam: Seam) -> None:
    picks = seam.picks
    periods: list[dict[str, Any]] = [{}, {"since": picks.chat_since, "until": picks.chat_until}]
    for conversation_id in seam.conversation_ids():
        for period in periods if conversation_id == picks.largest else periods[:1]:
            frame = seam.frame_archive.stats(conversation_id, **period)
            real = seam.adapter.stats(conversation_id, **period)
            assert set(real) == set(frame)
            assert {key: real[key] for key in SAME_STATS} == {key: frame[key] for key in SAME_STATS}
            assert real["first_timestamp"] == data.as_archive_reads(frame["first_timestamp"])
            assert real["last_timestamp"] == data.as_archive_reads(frame["last_timestamp"])
    assert seam.adapter.stats("nobody") == seam.frame_archive.stats("nobody")


def test_the_longest_gap_is_elapsed_time(seam: Seam) -> None:
    """Clock changes. The gap is the difference of two moments. `FrameArchive` subtracts the
    times on the clock, which is one hour more or less when the clocks changed in between.
    """
    for conversation_id in seam.conversation_ids():
        messages = list(seam.adapter.iter_messages(conversation_id))
        gaps = [later["sent_at"] - earlier["sent_at"] for earlier, later in zip(messages, messages[1:], strict=False)]
        stats = seam.adapter.stats(conversation_id)
        found = (stats["longest_gap_days"], stats["longest_gap_start"], stats["longest_gap_end"])
        if not gaps:
            assert found == (0.0, None, None)
            continue
        widest = gaps.index(max(gaps))  # the earliest of two equal gaps
        assert found[0] == pytest.approx(max(gaps) / SECONDS_PER_DAY)
        assert found[1:] == (messages[widest]["timestamp"], messages[widest + 1]["timestamp"])
    winter_to_summer = seam.adapter.stats(data.CLOCK_CHAT)
    assert winter_to_summer["longest_gap_days"] == pytest.approx(244 - 1 / 24)
    assert (winter_to_summer["longest_gap_start"], winter_to_summer["longest_gap_end"]) == (
        "2020-01-10 12:00:00",
        "2020-09-10 12:00:00",
    )
    assert seam.frame_archive.stats(data.CLOCK_CHAT)["longest_gap_days"] == 244.0
