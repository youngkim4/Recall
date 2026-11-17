import json

import pytest

from chat_engine.fake_archive import FakeArchive, make_message
from chat_engine.labeler import MaskingLabeler
from chat_engine.registry import ConversationAliases, EvidenceRegistry
from chat_engine.render import Limits
from chat_engine.scope import Scope
from chat_engine.tools import TOOL_SPECS, ToolBox
from tests.chat_engine.tiny import (
    GROUP,
    JONAS_PHONE,
    LONG_TEXT,
    MAYA_EMAIL,
    MAYA_PHONE,
    SAVED_NAMES,
    UNSAVED_EMAIL,
    UNSAVED_PHONE,
    tiny_archive,
    tiny_labeler,
)

RAW_HANDLES = ["7205550199", "720555", "+1720", "rk.unsaved", "example.org"]


class Bench:
    def __init__(
        self, scope: Scope | None = None, limits: Limits | None = None, archive: FakeArchive | None = None
    ) -> None:
        self.archive = archive or tiny_archive()
        self.registry = EvidenceRegistry()
        self.aliases = ConversationAliases()
        self.box = ToolBox(self.archive, tiny_labeler(), self.registry, self.aliases, scope or Scope(), limits)

    def run(self, tool: str, /, budget: int = 3000, **arguments: object):
        return self.box.prepare(tool, json.dumps(arguments)).run(budget)

    def status(self, tool: str, /, **arguments: object) -> str:
        return self.box.prepare(tool, json.dumps(arguments)).status

    def alias(self, conversation_id: str) -> str:
        return self.aliases.alias(conversation_id)


def test_specs_are_strict_json_schemas() -> None:
    names = [spec.name for spec in TOOL_SPECS]
    assert names == [
        "search_messages",
        "read_around",
        "count_messages",
        "list_conversations",
        "conversation_stats",
    ]
    for spec in TOOL_SPECS:
        schema = spec.parameters
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert sorted(schema["required"]) == sorted(schema["properties"])
        assert spec.description


def test_search_returns_hits_with_references_and_the_exchange_around_them() -> None:
    bench = Bench()
    outcome = bench.run("search_messages", query="lease signed")
    assert outcome.ok
    assert '1 message matches' in outcome.text
    assert "Maya Okafor: yes!! the lease got signed btw  <<< match" in outcome.text
    assert "You: are we still on for tonight" in outcome.text
    assert "You: omg congrats" in outcome.text
    ref = bench.registry.ref_of("a2")
    assert ref and f"{ref} 18:22" in outcome.text
    assert "1 match" in outcome.summary


def test_search_status_is_plain_words() -> None:
    bench = Bench()
    alias = bench.alias(MAYA_PHONE)
    assert bench.status("search_messages", query="lease") == "Searching for “lease”"
    assert bench.status("search_messages", query="lease", conversations=[alias]) == (
        "Searching for “lease” in Maya Okafor"
    )


def test_search_with_no_hit_explains_and_counts_each_word() -> None:
    outcome = Bench().run("search_messages", query="lease zanzibar")
    assert "no message matches" in outcome.text
    assert '"lease" 4' in outcome.text and '"zanzibar" 0' in outcome.text
    assert "0 matches" in outcome.summary


def test_search_any_mode_shows_the_weight_of_each_word() -> None:
    outcome = Bench().run("search_messages", query="zanzibar sourdough starter", match="any")
    assert "any of" in outcome.text
    assert '"sourdough" 2' in outcome.text
    assert 'No message has: "zanzibar"' in outcome.text
    assert "sourdough starter is alive  <<< match" in outcome.text


def test_search_says_how_many_hits_are_not_shown() -> None:
    outcome = Bench().run("search_messages", query="lease", limit=2)
    assert "4 messages match" in outcome.text
    assert "2 more matches not shown" in outcome.text


def test_search_accepts_an_alias_or_an_exact_name_and_rejects_unknown_ones() -> None:
    bench = Bench()
    alias = bench.alias(MAYA_EMAIL)
    by_alias = bench.run("search_messages", query="lease", conversations=[alias])
    assert "sending the lease scan" in by_alias.text and "landlord" not in by_alias.text
    by_name = bench.run("search_messages", query="sourdough", conversations=["Jonas Weber"])
    assert "sourdough starter is alive" in by_name.text
    unknown = bench.run("search_messages", query="lease", conversations=["c99"])
    assert not unknown.ok and "c99" in unknown.text and "list_conversations" in unknown.text
    ambiguous = bench.run("search_messages", query="lease", conversations=["Maya Okafor"])
    assert not ambiguous.ok and "more than one" in ambiguous.text


def test_search_rejects_bad_arguments_without_touching_the_archive() -> None:
    bench = Bench()
    assert not bench.run("search_messages", query="   ").ok
    assert not bench.run("search_messages", query="lease", limit="many").ok
    assert not bench.run("search_messages", query="lease", from_me="yes").ok
    assert not bench.run("search_messages", query=7).ok
    assert not bench.run("search_messages", query="lease", conversations=[3]).ok
    assert not bench.run("search_messages", query="lease", since="last summer").ok
    assert not bench.run("search_messages", query="lease", order="sideways").ok
    assert not bench.box.prepare("search_messages", "{not json").run(3000).ok
    assert not bench.box.prepare("search_messages", "[1, 2]").run(3000).ok
    assert not bench.box.prepare("no_such_tool", "{}").run(3000).ok
    assert bench.archive.calls == []


def test_search_clamps_the_limit() -> None:
    bench = Bench()
    bench.run("search_messages", query="lease", limit=500)
    name, arguments = bench.archive.calls[0]
    assert name == "search" and arguments["limit"] == 60


def test_long_message_reaches_the_model_whole() -> None:
    outcome = Bench().run("search_messages", query="flowerpot")
    assert len(LONG_TEXT) > 1500
    assert LONG_TEXT in outcome.text
    assert "[cut" not in outcome.text


def test_very_long_message_is_cut_with_a_note() -> None:
    huge = "start " + "word " * 3000 + "needle end"
    archive = FakeArchive([make_message("h1", "+15550000009", "2024-01-01 10:00:00", huge)])
    bench = Bench(archive=archive, limits=Limits(max_message_chars=400, result_tokens=3000))
    outcome = bench.run("search_messages", query="needle")
    assert "a number ending in 0009: start word word" in outcome.text
    assert "needle end  <<< match" in outcome.text
    assert outcome.text.count("[cut: ") == 1 and "characters]" in outcome.text


def test_result_is_cut_at_the_token_budget_and_says_so() -> None:
    outcome = Bench().run("search_messages", budget=120, query="lease", limit=20)
    assert "Result cut to fit" in outcome.text
    assert outcome.text.count("<<< match") < 4


def test_read_around_shows_the_exchange_and_marks_the_message() -> None:
    bench = Bench()
    bench.run("search_messages", query="renew")
    ref = bench.registry.ref_of("a5")
    outcome = bench.run("read_around", reference=ref, radius=2)
    assert "we move in on the first of April" in outcome.text
    assert "fifty more a month" in outcome.text
    assert "the landlord wants to renew the lease  <<< " + ref in outcome.text
    assert bench.status("read_around", reference=ref) == "Reading a conversation from July 2023"
    assert "July 2023" in outcome.summary


def test_read_around_caps_the_radius_and_rejects_unknown_references() -> None:
    bench = Bench()
    bench.run("search_messages", query="renew")
    ref = bench.registry.ref_of("a5")
    bench.run("read_around", reference=ref, radius=99)
    assert bench.archive.calls[-1] == ("context", {"message_id": "a5", "radius": 15})
    unknown = bench.run("read_around", reference="m999")
    assert not unknown.ok and "m999" in unknown.text


def test_count_plain_and_with_text() -> None:
    bench = Bench()
    alias = bench.alias(MAYA_PHONE)
    total = bench.run("count_messages")
    assert "27" in total.text and total.summary == "Counted messages: 27"
    lease = bench.run("count_messages", conversations=[alias], text="lease", since="2021-01-01", until="2021-12-31")
    assert lease.text.rstrip().endswith(": 1.")
    assert "Maya Okafor" in lease.text and "2021-01-01 to 2021-12-31" in lease.text
    assert bench.status("count_messages", text="lease") == "Counting messages that mention “lease”"


def test_count_by_year_gives_exact_rows() -> None:
    bench = Bench()
    outcome = bench.run("count_messages", conversations=[bench.alias(MAYA_PHONE)], group_by="year")
    assert "2021: 4 (you 2, others 2)" in outcome.text
    assert "2023: 3 (you 1, others 2)" in outcome.text
    assert "Total 9 (you 4, others 5)" in outcome.text
    assert bench.status("count_messages", group_by="year") == "Counting messages by year"


def test_count_by_conversation_and_sender_uses_labels_only() -> None:
    bench = Bench()
    by_conversation = bench.run("count_messages", group_by="conversation")
    assert "Maya Okafor · direct: 9" in by_conversation.text
    assert "a number ending in 0199 · direct: 3" in by_conversation.text
    by_sender = bench.run("count_messages", conversations=[bench.alias(GROUP)], group_by="sender")
    assert "Maya Okafor: 2" in by_sender.text
    assert "You: 1" in by_sender.text
    assert "a number ending in 0199: 1" in by_sender.text
    for raw in RAW_HANDLES:
        assert raw not in by_conversation.text and raw not in by_sender.text


def test_count_caps_long_tables_and_says_so() -> None:
    rows = [
        make_message(f"d{i}", "+15550000009", f"2023-{1 + i // 28:02d}-{1 + i % 28:02d} 10:00:00", "hi")
        for i in range(200)
    ]
    outcome = Bench(archive=FakeArchive(rows)).run("count_messages", group_by="day")
    assert "busiest" in outcome.text and "of 200" in outcome.text


def test_list_conversations_by_name_shows_every_thread_of_a_person() -> None:
    bench = Bench()
    outcome = bench.run("list_conversations", name="maya")
    lines = outcome.text.splitlines()
    assert "3 of 7" in lines[0]
    assert "Maya Okafor · direct, phone · 9 messages (you 4, them 5) · 2021-03-04 to 2025-12-24" in lines[1]
    assert "Lisbon crew · group of 3: Maya Okafor, Jonas Weber, a number ending in 0199" in lines[2]
    assert "Maya Okafor · direct, email" in lines[3] and "same person as" in lines[3]
    assert bench.status("list_conversations", name="maya") == "Looking up “maya”"


def test_list_conversations_by_digits_never_shows_the_handle() -> None:
    outcome = Bench().run("list_conversations", name="720")
    assert "a number ending in 0199 · direct, phone" in outcome.text
    for raw in RAW_HANDLES:
        assert raw not in outcome.text


def test_list_conversations_without_a_name_lists_the_biggest() -> None:
    bench = Bench()
    outcome = bench.run("list_conversations", limit=2)
    assert "2 conversations with the most messages (of 7)" in outcome.text
    assert len(outcome.text.splitlines()) == 3
    assert "no conversation" in bench.run("list_conversations", name="zzz").text.lower()


def test_conversation_stats_renders_figures() -> None:
    bench = Bench()
    outcome = bench.run("conversation_stats", conversation=bench.alias(MAYA_PHONE))
    assert "messages: 9 (you 4, others 5)" in outcome.text
    assert "first message: 2021-03-04 18:20:00" in outcome.text
    assert "active days: 3 (3.0 messages on an active day)" in outcome.text
    assert "busiest day: 2021-03-04 (4 messages)" in outcome.text
    assert "longest gap: 897.5 days, from 2023-07-11 09:05:00 to 2025-12-24 20:00:00" in outcome.text
    assert MAYA_PHONE not in outcome.text and "conversation:" not in outcome.text
    assert not bench.run("conversation_stats", conversation="c99").ok


def test_conversation_stats_shows_unknown_numbers_but_never_unknown_text() -> None:
    from chat_engine.tool_figures import stats_lines

    lines = stats_lines({"total_messages": 2, "sent_count": 1, "streak_days": 4, "top_sender": "+17205550199"})
    assert lines == ["messages: 2 (you 1, others 1)", "streak_days: 4"]


def test_count_by_weekday_and_hour_reads_as_words() -> None:
    bench = Bench()
    outcome = bench.run(
        "count_messages", conversations=[bench.alias(GROUP)], since="2019-06-01", until="2019-06-01",
        group_by="weekday_hour",
    )
    assert "Saturday 09:00: 4 (you 1, others 3)" in outcome.text


def test_scope_limits_every_tool() -> None:
    bench = Bench(scope=Scope(conversations=(JONAS_PHONE,)))
    jonas = bench.alias(JONAS_PHONE)
    maya = bench.alias(MAYA_PHONE)

    assert "sourdough starter is alive" in bench.run("search_messages", query="sourdough").text
    assert "no message matches" in bench.run("search_messages", query="lease").text
    refused = bench.run("search_messages", query="lease", conversations=[maya])
    assert not refused.ok and "outside the scope" in refused.text
    assert not bench.run("count_messages", conversations=[maya]).ok
    assert not bench.run("conversation_stats", conversation=maya).ok
    assert "4" in bench.run("count_messages").text
    listed = bench.run("list_conversations", name="maya")
    assert "Maya" not in listed.text
    assert "Jonas Weber" in bench.run("list_conversations").text
    assert bench.run("conversation_stats", conversation=jonas).ok

    for name, arguments in bench.archive.calls:
        if name in {"search", "count"}:
            assert arguments["conversations"] == [JONAS_PHONE], (name, arguments)
        elif name in {"conversation", "stats"}:
            assert arguments["conversation_id"] == JONAS_PHONE, (name, arguments)
        elif name == "context":
            assert arguments["message_id"] in {"j1", "j2", "j3", "j4"}, arguments
        else:
            pytest.fail(f"unexpected archive call {name}")


def test_scope_blocks_read_around_of_evidence_from_another_conversation() -> None:
    registry = EvidenceRegistry()
    aliases = ConversationAliases()
    archive = tiny_archive()
    wide = ToolBox(archive, tiny_labeler(), registry, aliases, Scope(), Limits())
    wide.prepare("search_messages", json.dumps({"query": "renew"})).run(3000)
    ref = registry.ref_of("a5")
    narrow = ToolBox(archive, tiny_labeler(), registry, aliases, Scope(conversations=(JONAS_PHONE,)), Limits())
    before = len(archive.calls)
    outcome = narrow.prepare("read_around", json.dumps({"reference": ref})).run(3000)
    assert not outcome.ok and "outside the scope" in outcome.text
    assert len(archive.calls) == before


def test_date_scope_clamps_searches_and_counts() -> None:
    bench = Bench(scope=Scope(since="2023-01-01", until="2023-12-31"))
    outcome = bench.run("search_messages", query="lease", since="2019-01-01")
    assert "1 message matches" in outcome.text
    assert "the landlord wants to renew the lease" in outcome.text
    assert "2021-03-04" not in outcome.text
    assert "3" in bench.run("count_messages").text
    outside = bench.run("search_messages", query="lease", until="2020-01-01")
    assert "outside the scope" in outside.text


def test_no_tool_result_ever_shows_a_raw_handle_of_an_unsaved_person() -> None:
    bench = Bench()
    texts = [
        bench.run("search_messages", query="climbing", match="any").text,
        bench.run("search_messages", query="line").text,
        bench.run("search_messages", query="lease lisbon").text,
        bench.run("list_conversations").text,
        bench.run("list_conversations", name="0199").text,
        bench.run("count_messages", group_by="conversation").text,
        bench.run("count_messages", group_by="sender").text,
        bench.run("conversation_stats", conversation=bench.alias(UNSAVED_PHONE)).text,
        bench.run("conversation_stats", conversation=bench.alias(UNSAVED_EMAIL)).text,
        bench.run("read_around", reference=bench.registry.ref_of("u1"), radius=5).text,
        bench.status("search_messages", query="climbing", conversations=[bench.alias(UNSAVED_PHONE)]),
    ]
    assert any("other line is a number ending in 0199 too" in text for text in texts)
    for text in texts:
        for raw in RAW_HANDLES:
            assert raw not in text, (raw, text)


def test_repeated_saved_names_stay_saved() -> None:
    assert MaskingLabeler(SAVED_NAMES).sender(MAYA_PHONE) == "Maya Okafor"


def test_each_hit_gets_its_own_exchange_and_far_neighbours_are_left_out() -> None:
    rows = [
        make_message("t1", "+15550000009", "2024-01-01 10:00:00", "intro hello"),
        make_message("t2", "+15550000009", "2024-01-01 10:05:00", "alpha keyword"),
        make_message("t3", "+15550000009", "2024-03-10 09:00:00", "beta keyword"),
        make_message("t4", "+15550000009", "2024-03-10 09:01:00", "right after beta"),
        make_message("t5", "+15550000009", "2024-06-01 09:00:00", "much later"),
    ]
    outcome = Bench(archive=FakeArchive(rows)).run("search_messages", query="keyword")
    assert outcome.text.count("== c1") == 2
    assert "intro hello" in outcome.text
    assert "right after beta" in outcome.text
    assert "much later" not in outcome.text
    newest, oldest = outcome.text.split("== c1", 2)[1:]
    assert "beta keyword  <<< match" in newest and "alpha keyword" not in newest
    assert "alpha keyword  <<< match" in oldest and "intro hello" in oldest


def test_hits_of_one_exchange_share_one_block() -> None:
    outcome = Bench().run("search_messages", query="sourdough name", match="any")
    blocks = outcome.text.split("== c1")[1:]
    assert len(blocks) == 2
    assert "sourdough starter is alive  <<< match" in blocks[0]
    assert "its name is Clint Yeastwood  <<< match" in blocks[0]
    assert "sourdough again this weekend?  <<< match" in blocks[1]
