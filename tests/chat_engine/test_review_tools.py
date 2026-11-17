"""Regression tests for the findings of the review of the tools."""

import json

import pytest

from chat_engine.browse import browse
from chat_engine.fake_archive import FakeArchive, make_message
from chat_engine.labeler import MaskingLabeler
from chat_engine.registry import ConversationAliases, EvidenceRegistry
from chat_engine.render import Limits
from chat_engine.scope import Scope
from chat_engine.tool_figures import stats_lines
from chat_engine.tools import ToolBox
from tests.chat_engine.tiny import (
    GROUP,
    JONAS_PHONE,
    MAYA_EMAIL,
    MAYA_PHONE,
    UNSAVED_PHONE,
    tiny_archive,
    tiny_labeler,
)

C = "+15550000042"


class Bench:
    def __init__(self, scope: Scope | None = None, archive: FakeArchive | None = None, labeler=None) -> None:
        self.archive = archive or tiny_archive()
        self.aliases = ConversationAliases()
        self.registry = EvidenceRegistry()
        self.box = ToolBox(
            self.archive, labeler or tiny_labeler(), self.registry, self.aliases, scope or Scope(), Limits()
        )

    def raw(self, tool: str, arguments: str, budget: int = 3000):
        return self.box.prepare(tool, arguments).run(budget)

    def run(self, tool: str, /, budget: int = 3000, **arguments: object):
        return self.raw(tool, json.dumps(arguments), budget)


def _daily(count: int, text: str = "note", conversation: str = C) -> FakeArchive:
    rows = []
    for i in range(count):
        year, day_of_year = 2021 + i // 336, i % 336
        stamp = f"{year}-{1 + day_of_year // 28:02d}-{1 + day_of_year % 28:02d} 12:00:00"
        rows.append(make_message(f"d{i:04d}", conversation, stamp, f"{text} {i}" if text else ""))
    return FakeArchive(rows)


# --- arguments ---------------------------------------------------------------


@pytest.mark.parametrize("value", ["Infinity", "-Infinity", "NaN", "1e400"])
def test_a_number_that_is_not_finite_is_refused_not_raised(value: str) -> None:
    bench = Bench()
    outcome = bench.raw("search_messages", f'{{"query": "lease", "limit": {value}}}')
    assert not outcome.ok and "Cannot run search_messages" in outcome.text
    listed = bench.raw("list_conversations", f'{{"limit": {value}}}')
    assert not listed.ok
    assert bench.archive.calls == []


def test_refusals_never_echo_a_handle_or_a_long_value() -> None:
    bench = Bench()
    bench.run("list_conversations")
    handle = bench.run("search_messages", query="x", conversations=[UNSAVED_PHONE])
    assert not handle.ok and "7205550199" not in handle.text
    long = bench.run("search_messages", query="x", conversations=["z" * 10_000])
    assert not long.ok and len(long.text) < 400
    tool = bench.raw("t" * 5000, "{}")
    assert not tool.ok and len(tool.text) < 200 and len(tool.summary) < 200
    reference = bench.run("read_around", reference="m" * 5000)
    assert not reference.ok and len(reference.text) < 300
    many = bench.run("search_messages", query="x", conversations=["c1"] * 500)
    assert not many.ok


def test_a_conversation_outside_the_scope_is_refused_even_next_to_one_inside() -> None:
    bench = Bench(scope=Scope(conversations=(JONAS_PHONE, MAYA_PHONE)))
    jonas, mail = bench.aliases.alias(JONAS_PHONE), bench.aliases.alias(MAYA_EMAIL)
    mixed = bench.run("count_messages", conversations=[jonas, mail])
    assert not mixed.ok and "outside the scope" in mixed.text
    assert bench.run("count_messages", conversations=[jonas, jonas]).text.endswith(": 4.")


# --- a scope of days ---------------------------------------------------------


def test_days_only_scope_lists_only_conversations_with_messages_in_the_days() -> None:
    bench = Bench(scope=Scope(since="2024-01-01", until="2024-12-31"))
    text = bench.run("list_conversations").text
    assert "2 conversations with the most messages (of 2)" in text
    assert "Jonas Weber" in text and "Inês Duarte" in text
    assert "Maya" not in text and "Lisbon" not in text and "0 messages" not in text
    assert "3 messages in the scope dates (you 1, them 2)" in text
    assert "No conversation matches" in bench.run("list_conversations", name="maya").text
    refused = bench.run("search_messages", query="lease", conversations=["Maya Okafor"])
    assert not refused.ok


# --- counts ------------------------------------------------------------------


def test_count_by_sender_gives_one_row_for_one_person_with_two_handles() -> None:
    rows = [
        make_message("s1", "chatx", "2024-01-01 10:00:00", "one", sender=MAYA_PHONE),
        make_message("s2", "chatx", "2024-01-01 10:01:00", "two", sender=MAYA_EMAIL),
        make_message("s3", "chatx", "2024-01-01 10:02:00", "three", sender=MAYA_PHONE),
        make_message("s4", "chatx", "2024-01-01 10:03:00", "four", sender=JONAS_PHONE),
        make_message("s5", "chatx", "2024-01-01 10:04:00", "five", is_from_me=True),
    ]
    bench = Bench(archive=FakeArchive(rows, [{"id": "chatx", "is_group": True, "name": "Team"}]))
    text = bench.run("count_messages", group_by="sender").text
    assert text.count("Maya Okafor") == 1 and "Maya Okafor: 3" in text
    assert text.index("Maya Okafor: 3") < text.index("Jonas Weber: 1")
    assert "You: 1" in text and "Total 5." in text


def test_figures_survive_odd_values_and_show_only_plain_extra_fields() -> None:
    lines = stats_lines({"total_messages": "5", "sent_count": None, "avg_messages_per_day": "n/a", "active_days": 2})
    assert lines[0] == "messages: 5 (you 0, others 5)"
    assert "active days: 2 (0.0 messages on an active day)" in lines
    extra = stats_lines({"total_messages": 1, "+17205550199": 5, "Streak Days": 3, "streak_days": 4})
    assert extra == ["messages: 1 (you 0, others 1)", "streak_days: 4"]
    received = stats_lines({"total_messages": 10, "sent_count": 3, "received_count": 6})
    assert received[0] == "messages: 10 (you 3, others 6)"


# --- what a search result says about what it leaves out -----------------------


def _burst() -> FakeArchive:
    rows = [make_message(f"p{i}", C, f"2024-05-05 10:{i:02d}:00", f"pizza number {i}") for i in range(6)]
    rows += [make_message(f"q{i}", C, f"202{i}-01-01 09:00:00", f"pizza again {i}") for i in range(3)]
    return FakeArchive(rows)


def test_every_visible_match_is_marked_and_the_footer_counts_only_what_is_hidden() -> None:
    outcome = Bench(archive=_burst()).run("search_messages", query="pizza", limit=3)
    assert "9 messages match" in outcome.text
    visible = outcome.text.count("<<< match")
    assert visible > 3
    assert f"{9 - visible} more matches not shown" in outcome.text
    for line in outcome.text.splitlines():
        if "pizza" in line and line.startswith("m"):
            assert line.endswith("<<< match"), line


def test_footer_does_not_tell_the_model_to_raise_a_limit_that_is_not_the_cause() -> None:
    bench = Bench(archive=_burst())
    wide = bench.run("search_messages", query="pizza", limit=20).text
    assert "raise limit" not in wide
    assert "read_around" in wide
    narrow = Bench(archive=_burst()).run("search_messages", query="pizza", limit=1).text
    assert "raise limit" in narrow


def test_any_mode_says_when_more_messages_hold_one_of_the_words() -> None:
    outcome = Bench().run("search_messages", query="lease sourdough", match="any", limit=1)
    assert outcome.text.count("<<< match") >= 1
    assert "more messages hold one of the words" in outcome.text
    everything = Bench().run("search_messages", query="lease sourdough", match="any", limit=20)
    assert "more messages hold" not in everything.text


def test_a_block_that_does_not_fit_is_shown_with_shorter_messages_or_alone() -> None:
    long = "match here " + "filler " * 900
    rows = [make_message(f"b{i}", C, f"2024-05-05 10:0{i}:00", long if i != 2 else "the needle") for i in range(5)]
    roomy = Bench(archive=FakeArchive(rows)).run("search_messages", budget=700, query="needle")
    assert "the needle  <<< match" in roomy.text
    assert roomy.text.count("[cut: ") == 4
    tight = Bench(archive=FakeArchive(rows)).run("search_messages", budget=120, query="needle")
    assert "the needle  <<< match" in tight.text and "[cut: " not in tight.text
    assert "more match" not in tight.text


# --- names by digits ---------------------------------------------------------


def test_the_model_cannot_find_a_number_by_guessing_its_digits() -> None:
    bench = Bench()
    assert "No conversation matches" in bench.run("list_conversations", name="720").text
    assert "No conversation matches" in bench.run("list_conversations", name="72055").text
    assert "a number ending in 0199 · direct, phone" in bench.run("list_conversations", name="0199").text
    assert "a number ending in 0199 · direct, phone" in bench.run("list_conversations", name="7205550199").text


def test_the_seed_still_finds_a_number_by_the_area_code_the_user_typed() -> None:
    directory = Bench().box.context.directory()
    assert directory.named_in("who is this 720 number") == ["720"]
    assert directory.named_in("did I spend 555 dollars or 150 euros") == []
    assert [entry.conversation["id"] for entry in directory.find("720", loose_digits=True)] == [GROUP, UNSAVED_PHONE]


# --- reading a conversation --------------------------------------------------


def test_latest_messages_up_to_a_day_are_filled_from_older_pages() -> None:
    archive = _daily(500)
    everything = archive.messages(C, limit=1000)["messages"]
    until = everything[300]["timestamp"][:10]
    found = browse(archive, C, edge="newest", limit=8, until=until)
    assert [m["id"] for m in found.messages] == [m["id"] for m in everything[293:301]]
    assert found.reached_edge and found.anchored


def test_first_messages_from_a_day_are_filled_from_newer_pages() -> None:
    archive = _daily(500)
    everything = archive.messages(C, limit=1000)["messages"]
    since = everything[195]["timestamp"][:10]
    found = browse(archive, C, edge="oldest", limit=8, since=since)
    assert [m["id"] for m in found.messages] == [m["id"] for m in everything[195:203]]


def test_a_period_before_or_after_the_conversation_is_empty_not_unknown() -> None:
    archive = _daily(500)
    after = browse(archive, C, edge="oldest", limit=5, since="2030-01-01")
    before = browse(archive, C, edge="newest", limit=5, until="2001-01-01")
    for found in (after, before):
        assert found.messages == () and found.anchored and found.reached_edge


def test_attachments_only_conversation_can_be_read_up_to_a_day() -> None:
    archive = _daily(500, text="")
    everything = archive.messages(C, limit=1000)["messages"]
    until = everything[100]["timestamp"][:10]
    found = browse(archive, C, edge="newest", limit=4, until=until)
    assert [m["id"] for m in found.messages] == [m["id"] for m in everything[97:101]]
    assert found.anchored


def test_period_in_a_gap_of_a_long_conversation_is_empty() -> None:
    def clock(i: int) -> str:
        return f"10:{i // 60:02d}:{i % 60:02d}"

    rows = [make_message(f"a{i:03d}", C, f"2020-01-01 {clock(i)}", "early words") for i in range(300)]
    rows += [make_message(f"z{i:03d}", C, f"2022-01-01 {clock(i)}", "late words") for i in range(300)]
    found = browse(FakeArchive(rows), C, edge="oldest", limit=5, since="2021-05-01", until="2021-05-31")
    assert found.messages == () and found.anchored and found.reached_edge


def test_reading_gives_up_honestly_when_the_period_is_too_far_to_walk() -> None:
    archive = _daily(1500, text="")
    everything = archive.messages(C, limit=5000)["messages"]
    since = everything[10]["timestamp"][:10]
    found = browse(archive, C, edge="oldest", limit=4, since=since, until=since, max_hops=2)
    assert found.messages == () and not found.anchored


def test_labeler_is_shared_so_tests_do_not_leak_between_benches() -> None:
    assert MaskingLabeler().redact(UNSAVED_PHONE) == UNSAVED_PHONE
