"""search_messages with match "meaning": exchanges found by meaning and by words, joined by rank."""

import logging

import pytest

from chat_engine.fake_archive import FakeArchive, make_message
from chat_engine.fake_meaning import FakeMeaning, hit
from chat_engine.render import estimate_tokens
from chat_engine.scope import Scope
from tests.chat_engine.memory_bench import RAW_HANDLES, GrowingLabeler, MemoryBench, shown_refs
from tests.chat_engine.test_tools import Bench
from tests.chat_engine.tiny import JONAS_PHONE, MAYA_PHONE, UNSAVED_EMAIL, UNSAVED_PHONE

MAYA_2021 = hit(MAYA_PHONE, "a1", "a4", 0.9)
JONAS_2024 = hit(JONAS_PHONE, "j1", "j3", 0.8)


def test_meaning_is_a_match_only_with_the_port() -> None:
    without = Bench().run("search_messages", query="a new home", match="meaning")
    assert not without.ok and "match must be one of: all, any." in without.text
    bench = MemoryBench(meaning=FakeMeaning())
    wrong = bench.run("search_messages", query="a new home", match="fuzzy")
    assert not wrong.ok and "match must be one of: all, any, meaning." in wrong.text
    assert bench.run("search_messages", query="lease signed").ok
    assert bench.box.context.memory.meaning.calls == []


def test_an_exchange_found_by_meaning_is_shown_whole_with_a_reference_on_every_line() -> None:
    port = FakeMeaning([JONAS_2024])
    bench = MemoryBench(meaning=port)
    outcome = bench.run("search_messages", query="naming bread starter", match="meaning")
    assert outcome.ok
    assert outcome.text.splitlines()[0] == (
        "Search by meaning for “naming bread starter” in all conversations: 1 exchange, best first."
    )
    assert "== c1 · Jonas Weber · direct ==" in outcome.text
    lines = [line for line in outcome.text.splitlines() if line[:1] == "m"]
    assert lines == [
        "m1 12:00 Jonas Weber: sourdough starter is alive  <<< match",
        "m2 12:01 You: you named it??",
        "m3 12:02 Jonas Weber: its name is Clint Yeastwood",
    ]
    assert [bench.registry.get(ref)["id"] for ref in ("m1", "m2", "m3")] == ["j1", "j2", "j3"]
    assert outcome.summary == "Searched by meaning for “naming bread starter”: 1 exchange shown"
    assert port.calls == [
        {"query": "naming bread starter", "conversations": None, "since": None, "until": None, "limit": 24}
    ]


def test_the_two_rankings_are_joined_by_rank() -> None:
    bench = MemoryBench(meaning=FakeMeaning([MAYA_2021, JONAS_2024]))
    outcome = bench.run("search_messages", query="sourdough landlord", match="meaning")
    text = outcome.text
    places = [
        text.index("sourdough starter is alive"),  # found by meaning (2nd) and by words (3rd)
        text.index("the lease got signed"),  # found by meaning only (1st)
        text.index("the landlord wants to renew"),  # found by words only (1st)
        text.index("sourdough again this weekend"),  # found by words only (2nd)
    ]
    assert places == sorted(places)
    assert "4 exchanges, best first" in text
    assert text.count("sourdough starter is alive") == 1  # the word hit counts for the exchange that holds it
    assert "m1 12:00 Jonas Weber: sourdough starter is alive  <<< match" in text


def test_limit_counts_exchanges_and_the_rest_is_announced() -> None:
    bench = MemoryBench(meaning=FakeMeaning([MAYA_2021, JONAS_2024]))
    outcome = bench.run("search_messages", query="sourdough landlord", match="meaning", limit=2)
    assert "sourdough starter is alive" in outcome.text and "the lease got signed" in outcome.text
    assert "landlord wants to renew" not in outcome.text
    assert "[2 more exchanges not shown." in outcome.text
    assert outcome.summary.endswith("2 exchanges shown")
    assert bench.box.context.memory.meaning.calls[0]["limit"] == 6


def test_nothing_found_is_said_plainly() -> None:
    outcome = MemoryBench(meaning=FakeMeaning()).run("search_messages", query="zanzibar zeppelin", match="meaning")
    assert outcome.ok and "nothing found." in outcome.text and shown_refs(outcome.text) == set()


def test_the_scope_of_the_user_limits_both_rankings() -> None:
    port = FakeMeaning([MAYA_2021, JONAS_2024])  # the port returns an exchange with Maya by mistake
    bench = MemoryBench(meaning=port, scope=Scope(conversations=(JONAS_PHONE,), until="2025-12-31"))
    outcome = bench.run("search_messages", query="sourdough lease", match="meaning")
    assert port.calls[0]["conversations"] == [JONAS_PHONE] and port.calls[0]["until"] == "2025-12-31"
    assert "sourdough starter is alive" in outcome.text
    for outside in ("lease", "landlord", "sourdough again this weekend", "Maya"):
        assert outside not in outcome.text.split("\n", 1)[1], outside
    for arguments in bench.read("search") + bench.read("count"):
        assert arguments["conversations"] == [JONAS_PHONE] and arguments["until"] == "2025-12-31"
    assert all(m["conversation"] == JONAS_PHONE for m in bench.registry.messages())


def test_an_exchange_that_reaches_outside_the_days_of_the_scope_shows_only_what_is_inside() -> None:
    port = FakeMeaning([hit(JONAS_PHONE, "j1", "j4"), MAYA_2021])
    bench = MemoryBench(meaning=port, scope=Scope(since="2024-01-01", until="2024-12-31"))
    outcome = bench.run("search_messages", query="bread", match="meaning")
    assert "its name is Clint Yeastwood" in outcome.text
    assert "sourdough again this weekend" not in outcome.text and "lease" not in outcome.text
    assert [m["id"] for m in bench.registry.messages()] == ["j1", "j2", "j3"]


def test_the_conversations_and_days_of_the_call_limit_both_rankings() -> None:
    port = FakeMeaning([MAYA_2021, JONAS_2024])
    bench = MemoryBench(meaning=port)
    alias = bench.alias(MAYA_PHONE)
    outcome = bench.run(
        "search_messages", query="sourdough lease", match="meaning", conversations=[alias], since="2021-01-01",
        until="2021-12-31",
    )
    assert port.calls[0] == {
        "query": "sourdough lease", "conversations": [MAYA_PHONE], "since": "2021-01-01", "until": "2021-12-31",
        "limit": 24,
    }
    assert "the lease got signed" in outcome.text and "sourdough" not in outcome.text.split("\n", 1)[1]
    outside = bench.run("search_messages", query="x", match="meaning", conversations=[alias], since="2030-01-01")
    assert outside.ok and "nothing found." in outside.text


def test_a_hit_the_archive_cannot_back_is_dropped() -> None:
    hits = [
        hit(JONAS_PHONE, "nope", "j3"),  # the archive does not hold the first message
        hit(JONAS_PHONE, "a1", "a4"),  # the messages are in another conversation
        hit(JONAS_PHONE, "j3", "j1"),  # the end is before the start
        {"conversation": JONAS_PHONE, "first_message_id": 7, "last_message_id": None, "score": 1.0},
        "not a hit",
        JONAS_2024,
        JONAS_2024,
    ]
    outcome = MemoryBench(meaning=FakeMeaning(hits)).run("search_messages", query="zzz", match="meaning")
    assert "1 exchange, best first" in outcome.text and outcome.text.count("its name is Clint Yeastwood") == 1
    assert "lease" not in outcome.text


def test_a_failure_of_the_port_falls_back_to_the_words(caplog: pytest.LogCaptureFixture) -> None:
    port = FakeMeaning(error=RuntimeError("secret text of the provider"))
    bench = MemoryBench(meaning=port)
    with caplog.at_level(logging.WARNING, logger="chat_engine"):
        outcome = bench.run("search_messages", query="sourdough starter", match="meaning")
    assert outcome.ok
    assert outcome.text.splitlines()[0].endswith(
        "The search by meaning was not available: these were found by their words only."
    )
    assert "sourdough starter is alive  <<< match" in outcome.text
    assert outcome.summary == "Searched by meaning for “sourdough starter”: 2 exchanges shown, by words only"
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "RuntimeError" in logged and "secret" not in logged and "sourdough" not in logged


def test_a_port_that_returns_no_list_is_a_failure_of_the_port() -> None:
    class Odd:
        def search(self, query: str, **_filters: object) -> object:
            return 7

    outcome = MemoryBench(meaning=Odd()).run("search_messages", query="sourdough starter", match="meaning")
    assert outcome.ok and "was not available" in outcome.text and "sourdough starter is alive" in outcome.text


def test_arguments_that_do_not_fit_a_search_by_meaning_are_refused() -> None:
    bench = MemoryBench(meaning=FakeMeaning([JONAS_2024]))
    empty = bench.run("search_messages", query="  ", match="meaning")
    assert not empty.ok and 'match "meaning" needs a short description in query' in empty.text
    one_side = bench.run("search_messages", query="bread", match="meaning", from_me=True)
    assert not one_side.ok and "from_me does not work" in one_side.text
    assert not bench.run("search_messages", query="bread", match="meaning", limit="many").ok
    assert not bench.run("search_messages", query="bread", match="meaning", conversations=["c99"]).ok
    assert bench.box.context.memory.meaning.calls == []


def test_the_status_line_is_plain_words() -> None:
    bench = MemoryBench(meaning=FakeMeaning())
    alias = bench.alias(JONAS_PHONE)
    assert bench.prepare("search_messages", query="a new pet", match="meaning").status == (
        "Searching for moments like “a new pet”"
    )
    scoped = bench.prepare("search_messages", query="a new pet", match="meaning", conversations=[alias])
    assert scoped.status == "Searching for moments like “a new pet” in Jonas Weber"


def test_no_raw_handle_goes_to_the_port_the_status_or_the_result() -> None:
    port = FakeMeaning([hit(UNSAVED_PHONE, "u1", "u3")])
    bench = MemoryBench(meaning=port)
    prepared = bench.prepare("search_messages", query="who has +1 (720) 555-0199 as a line", match="meaning")
    outcome = prepared.run(3000)
    seen = "\n".join([prepared.status, outcome.text, outcome.summary, port.calls[0]["query"]])
    assert "my other line is a number ending in 0199 too" in outcome.text
    assert port.calls[0]["query"] == "who has a number ending in 0199 as a line"
    for raw in [*RAW_HANDLES, UNSAVED_PHONE, UNSAVED_EMAIL]:
        assert raw not in seen, raw


def _long_exchange() -> FakeArchive:
    rows = [
        make_message(
            f"x{n:02d}", JONAS_PHONE, f"2024-05-01 10:{n:02d}:00", f"part {n} of the plan " + "word " * 60,
            sender=JONAS_PHONE, is_from_me=n % 2 == 0,
        )
        for n in range(30)
    ]
    return FakeArchive(rows, [{"id": JONAS_PHONE, "is_group": False, "members": [JONAS_PHONE]}])


@pytest.mark.parametrize("budget", [150, 400, 1200, 3000])
def test_a_long_exchange_is_cut_to_fit_the_budget(budget: int) -> None:
    bench = MemoryBench(meaning=FakeMeaning([hit(JONAS_PHONE, "x00", "x29")]), archive=_long_exchange())
    outcome = bench.run("search_messages", budget=budget, query="the plan", match="meaning")
    assert estimate_tokens(outcome.text) <= budget
    assert shown_refs(outcome.text) == {f"m{n}" for n in range(1, len(bench.registry) + 1)}
    if budget == 3000:  # every message is there, each cut shorter
        assert len(bench.registry) == 30 and "[Exchange cut to fit:" not in outcome.text
        assert outcome.text.count("[cut: ") == 30
    elif budget >= 400:  # the messages are as short as they get: now the exchange loses messages
        assert 1 <= len(bench.registry) < 30
        assert "[Exchange cut to fit:" in outcome.text and "of 30 messages shown" in outcome.text
    else:
        assert len(bench.registry) == 0 and "1 more exchange not shown" in outcome.text


def test_an_exchange_longer_than_the_engine_reads_is_cut_with_a_note() -> None:
    rows = [
        make_message(
            f"y{n:03d}", JONAS_PHONE, f"2024-05-01 10:{n // 60:02d}:{n % 60:02d}", f"line {n}", sender=JONAS_PHONE
        )
        for n in range(90)
    ]
    archive = FakeArchive(rows, [{"id": JONAS_PHONE, "is_group": False, "members": [JONAS_PHONE]}])
    bench = MemoryBench(meaning=FakeMeaning([hit(JONAS_PHONE, "y000", "y089")]), archive=archive)
    outcome = bench.run("search_messages", query="lines", match="meaning")
    assert "Jonas Weber: line 0" in outcome.text and "line 89" not in outcome.text
    assert "[Exchange cut to fit: 40 of more than 40 messages shown. Use read_around to read on.]" in outcome.text


def test_a_result_that_does_not_fit_after_the_masking_leaves_no_message_registered() -> None:
    port = FakeMeaning([JONAS_2024])
    bench = MemoryBench(meaning=port, labeler=GrowingLabeler())
    outcome = bench.run("search_messages", budget=200, query="sourdough starter", match="meaning")
    assert not outcome.ok and outcome.text.startswith("The result does not fit.")
    assert len(bench.registry) == 0
    assert len(port.calls) == 1  # the result was written twice; the port was asked one time


def test_an_exchange_with_no_message_inside_the_scope_is_left_out() -> None:
    port = FakeMeaning([hit(JONAS_PHONE, "j1", "j4")])  # it starts in 2024 and ends in 2026
    bench = MemoryBench(meaning=port, scope=Scope(since="2025-01-01", until="2025-12-31"))
    outcome = bench.run("search_messages", query="zzz", match="meaning")
    assert outcome.text.endswith("nothing found.") and len(bench.registry) == 0


def test_a_word_hit_next_to_an_exchange_joins_it() -> None:
    port = FakeMeaning([hit(JONAS_PHONE, "j2", "j3")])
    bench = MemoryBench(meaning=port)
    outcome = bench.run("search_messages", query="starter", match="meaning")
    lines = [line for line in outcome.text.splitlines() if line[:1] == "m"]
    assert lines == [
        "m1 12:00 Jonas Weber: sourdough starter is alive  <<< match",
        "m2 12:01 You: you named it??",
        "m3 12:02 Jonas Weber: its name is Clint Yeastwood",
    ]
    assert "1 exchange, best first." in outcome.text and "more exchange" not in outcome.text


def test_a_hit_outside_the_conversations_of_the_call_is_dropped() -> None:
    port = FakeMeaning([JONAS_2024, MAYA_2021])  # the call names Maya; the port returns Jonas by mistake
    bench = MemoryBench(meaning=port)
    outcome = bench.run("search_messages", query="zzz", match="meaning", conversations=[bench.alias(MAYA_PHONE)])
    assert "the lease got signed" in outcome.text and "Clint Yeastwood" not in outcome.text
    assert all(m["conversation"] == MAYA_PHONE for m in bench.registry.messages())
