"""Reading a conversation from its start or its end, with the paging calls only."""

import json

import pytest

from chat_engine import tool_browse
from chat_engine.browse import Browsed, browse
from chat_engine.fake_archive import FakeArchive, make_message
from chat_engine.registry import ConversationAliases, EvidenceRegistry
from chat_engine.render import Limits
from chat_engine.scope import Scope
from chat_engine.tools import ToolBox
from tests.chat_engine.tiny import GROUP, JONAS_PHONE, MAYA_PHONE, UNSAVED_PHONE, tiny_archive, tiny_labeler

LONG = "+15550000042"


def _long_archive(count: int = 1000) -> FakeArchive:
    rows = [
        make_message(
            f"L{i:04d}",
            LONG,
            f"{2020 + i // 300}-{1 + (i % 300) // 25:02d}-{1 + i % 25:02d} 10:{i % 60:02d}:00",
            f"note number {i} about the garden" if i % 7 else "",
            is_from_me=bool(i % 2),
            sender=LONG,
            kind="text" if i % 7 else "attachment",
            attachments=[] if i % 7 else ["image/jpeg"],
        )
        for i in range(count)
    ]
    return FakeArchive(rows)


def _ids(result) -> list[str]:
    return [message["id"] for message in result.messages]


def test_short_conversation_is_read_in_one_call() -> None:
    archive = tiny_archive()
    first = browse(archive, JONAS_PHONE, edge="oldest", limit=3)
    assert _ids(first) == ["j1", "j2", "j3"]
    assert first.reached_edge and first.anchored
    assert [name for name, _ in archive.calls] == ["messages"]
    last = browse(tiny_archive(), JONAS_PHONE, edge="newest", limit=2)
    assert _ids(last) == ["j3", "j4"]


def test_start_of_a_long_conversation_is_found_in_a_few_calls() -> None:
    archive = _long_archive()
    result = browse(archive, LONG, edge="oldest", limit=5)
    assert _ids(result) == ["L0000", "L0001", "L0002", "L0003", "L0004"]
    assert result.reached_edge
    assert len(archive.calls) <= 8


def test_end_of_a_long_conversation() -> None:
    archive = _long_archive()
    result = browse(archive, LONG, edge="newest", limit=3)
    assert _ids(result) == ["L0997", "L0998", "L0999"]
    assert result.reached_edge
    assert [name for name, _ in archive.calls] == ["messages"]


def test_walk_goes_on_when_the_probe_lands_far_from_the_start() -> None:
    rows = [make_message(f"e{i:03d}", LONG, f"2020-01-01 10:{i // 60:02d}:{i % 60:02d}", "") for i in range(450)]
    rows += [make_message(f"t{i:03d}", LONG, f"2021-01-01 10:{i // 60:02d}:{i % 60:02d}", "hello") for i in range(250)]
    archive = FakeArchive(rows)
    result = browse(archive, LONG, edge="oldest", limit=4)
    assert _ids(result) == ["e000", "e001", "e002", "e003"]
    assert result.reached_edge


def test_walk_stops_at_the_hop_cap_and_says_so() -> None:
    rows = [make_message(f"e{i:04d}", LONG, f"2020-01-{1 + i // 1440:02d} {(i // 60) % 24:02d}:{i % 60:02d}:00", "")
            for i in range(900)]
    archive = FakeArchive(rows)
    result = browse(archive, LONG, edge="oldest", limit=4, max_hops=2)
    assert not result.reached_edge
    assert len(result.messages) == 4


def test_period_is_read_from_its_first_day() -> None:
    archive = _long_archive()
    result = browse(archive, LONG, edge="oldest", limit=3, since="2021-03-01", until="2021-03-31")
    days = [message["timestamp"][:10] for message in result.messages]
    assert days == sorted(days) and all("2021-03-01" <= day <= "2021-03-31" for day in days)
    everything = [
        m for m in archive.messages(LONG, limit=5000)["messages"] if "2021-03-01" <= m["timestamp"][:10] <= "2021-03-31"
    ]
    assert _ids(result) == [m["id"] for m in everything[:3]]
    newest = browse(_long_archive(), LONG, edge="newest", limit=3, since="2021-03-01", until="2021-03-31")
    assert _ids(newest) == [m["id"] for m in everything[-3:]]


def test_period_inside_the_first_page_needs_no_search() -> None:
    archive = tiny_archive()
    result = browse(archive, GROUP, edge="oldest", limit=5, since="2019-06-02", until="2019-06-02")
    assert _ids(result) == ["g5"] and result.anchored and result.reached_edge
    assert [name for name, _ in archive.calls] == ["messages"]


def _gap_archive() -> FakeArchive:
    """Text in 2020 and 2022; five attachment-only messages in May 2021."""
    def stamp(day: str, i: int) -> str:
        return f"{day} 10:{i // 60:02d}:{i % 60:02d}"

    rows = [make_message(f"a{i:03d}", LONG, stamp("2020-01-01", i), "early words") for i in range(300)]
    rows += [make_message(f"p{i}", LONG, f"2021-05-0{i + 1} 09:00:00", "", kind="attachment") for i in range(5)]
    rows += [make_message(f"z{i:03d}", LONG, stamp("2022-01-01", i), "late words") for i in range(300)]
    return FakeArchive(rows)


def test_period_with_no_text_is_found_by_walking_the_pages() -> None:
    result = browse(_gap_archive(), LONG, edge="oldest", limit=5, since="2021-05-01", until="2021-05-31")
    assert _ids(result) == ["p0", "p1", "p2", "p3", "p4"] and result.anchored and result.reached_edge


def test_period_with_no_text_and_too_far_to_walk_is_not_anchored() -> None:
    result = browse(
        _gap_archive(), LONG, edge="oldest", limit=5, since="2021-05-01", until="2021-05-31", max_hops=0
    )
    assert result.messages == () and not result.anchored


def test_period_with_nothing_in_a_short_conversation_is_just_empty() -> None:
    result = browse(tiny_archive(), JONAS_PHONE, edge="oldest", limit=5, since="2010-01-01", until="2010-12-31")
    assert result.messages == () and result.anchored and result.reached_edge


def test_empty_or_unknown_conversation() -> None:
    result = browse(tiny_archive(), "nobody", edge="oldest", limit=5)
    assert result.messages == () and result.reached_edge


# --- through the search_messages tool ---


def _box(scope: Scope | None = None, archive: FakeArchive | None = None):
    registry = EvidenceRegistry()
    aliases = ConversationAliases()
    box = ToolBox(archive or tiny_archive(), tiny_labeler(), registry, aliases, scope or Scope(), Limits())
    return box, aliases, registry


def test_search_tool_with_no_query_reads_the_start_of_one_conversation() -> None:
    box, aliases, registry = _box()
    alias = aliases.alias(UNSAVED_PHONE)
    arguments = {"query": "", "conversations": [alias], "order": "oldest"}
    prepared = box.prepare("search_messages", json.dumps(arguments))
    assert prepared.status == "Reading the first messages with a number ending in 0199"
    outcome = prepared.run(3000)
    assert outcome.ok
    header = outcome.text.splitlines()[0]
    assert header == f"First 3 messages of {alias} · a number ending in 0199 · direct (all of it):"
    assert "hey this is Dana from the climbing gym" in outcome.text
    assert registry.ref_of("u1") == "m1"
    assert outcome.summary == "Read the first 3 messages with a number ending in 0199"


def test_search_tool_with_no_query_reads_the_latest_messages_by_default() -> None:
    box, aliases, _ = _box()
    alias = aliases.alias(MAYA_PHONE)
    prepared = box.prepare("search_messages", json.dumps({"query": None, "conversations": [alias], "limit": 2}))
    assert prepared.status == "Reading the latest messages with Maya Okafor"
    outcome = prepared.run(3000)
    assert outcome.text.splitlines()[0].startswith("Latest 2 messages of")
    assert "merry christmas you" in outcome.text
    assert "are we still on for tonight" not in outcome.text


def test_search_tool_with_no_query_needs_exactly_one_conversation() -> None:
    box, aliases, _ = _box()
    none = box.prepare("search_messages", json.dumps({"query": ""})).run(3000)
    assert not none.ok and "exactly one conversation" in none.text
    two = box.prepare(
        "search_messages",
        json.dumps({"query": "", "conversations": [aliases.alias(MAYA_PHONE), aliases.alias(JONAS_PHONE)]}),
    ).run(3000)
    assert not two.ok
    sender = box.prepare(
        "search_messages", json.dumps({"query": "", "conversations": [aliases.alias(MAYA_PHONE)], "from_me": True})
    ).run(3000)
    assert not sender.ok and "from_me" in sender.text


def test_search_tool_with_no_query_uses_the_scope_conversation() -> None:
    box, _, _ = _box(Scope(conversations=(JONAS_PHONE,)))
    prepared = box.prepare("search_messages", json.dumps({"query": "", "order": "oldest", "limit": 2}))
    assert prepared.status == "Reading the first messages with Jonas Weber"
    outcome = prepared.run(3000)
    assert "sourdough starter is alive" in outcome.text


def test_search_tool_reads_a_period_and_says_when_it_cannot(monkeypatch: pytest.MonkeyPatch) -> None:
    box, aliases, _ = _box(archive=_long_archive())
    alias = aliases.alias(LONG)
    arguments = {"query": "", "conversations": [alias], "order": "oldest", "since": "2021-03-01", "until": "2021-03-31"}
    outcome = box.prepare("search_messages", json.dumps(arguments)).run(3000)
    assert outcome.text.splitlines()[0].startswith("First 8 messages of")
    assert "2021-03-01 to 2021-03-31" in outcome.text.splitlines()[0]
    box, aliases, _ = _box(archive=_gap_archive())
    arguments = {"query": "", "conversations": [aliases.alias(LONG)], "since": "2021-05-01", "until": "2021-05-31"}
    monkeypatch.setattr(tool_browse, "browse", lambda *_args, **_kwargs: Browsed((), False, anchored=False))
    nothing = box.prepare("search_messages", json.dumps(arguments)).run(3000)
    assert not nothing.ok and "count_messages" in nothing.text
    monkeypatch.undo()
    box, aliases, _ = _box()
    jonas = aliases.alias(JONAS_PHONE)
    arguments = {"query": "", "conversations": [jonas], "since": "2010-01-01", "until": "2010-12-31"}
    empty = box.prepare("search_messages", json.dumps(arguments)).run(3000)
    assert empty.text.startswith("No messages in")


def test_group_status_reads_naturally() -> None:
    box, aliases, _ = _box()
    prepared = box.prepare("search_messages", json.dumps({"query": "", "conversations": [aliases.alias(GROUP)]}))
    assert prepared.status == "Reading the latest messages in Lisbon crew"
