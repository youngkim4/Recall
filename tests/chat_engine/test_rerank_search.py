"""search_messages with a reranker: which searches are reordered, and how."""

from __future__ import annotations

from chat_engine.fake_meaning import FakeMeaning, hit
from chat_engine.fake_rerank import FakeReranker, preferring
from chat_engine.rerank import RERANK_CHARS
from tests.chat_engine.helpers import RAW_HANDLES
from tests.chat_engine.memory_bench import MemoryBench
from tests.chat_engine.tiny import JONAS_PHONE, MAYA_PHONE

# the four messages that hold "lease", newest first: a5 (2023), b1 (2022), a2 (2021), g4 (2019)
RENEW, SCAN, SIGNED, LISBON = "renew the lease", "lease scan", "lease got signed", "lease for the Lisbon flat"


def _before(text: str, first: str, second: str) -> bool:
    return first in text and second in text and text.index(first) < text.index(second)


def test_the_reranker_reorders_the_hits_and_the_best_limit_are_shown() -> None:
    ranker = FakeReranker(preferring("Lisbon"))
    bench = MemoryBench(reranker=ranker)
    outcome = bench.run("search_messages", query="lease", limit=2)
    assert _before(outcome.text, LISBON, RENEW)
    assert SCAN not in outcome.text and SIGNED not in outcome.text
    assert "Best match first." in outcome.text
    [asked] = ranker.calls
    assert asked["query"] == "lease" and asked["keep"] == 2
    assert len(asked["items"]) == 4  # the pool is larger than the limit
    assert [item.id for item in asked["items"]] == ["1", "2", "3", "4"]


def test_without_a_reranker_the_order_is_the_order_of_the_call() -> None:
    outcome = MemoryBench().run("search_messages", query="lease", limit=2)
    assert _before(outcome.text, RENEW, SCAN) and LISBON not in outcome.text
    assert "Newest first." in outcome.text


def test_a_relevance_order_is_reranked() -> None:
    ranker = FakeReranker(preferring("Lisbon"))
    outcome = MemoryBench(reranker=ranker).run("search_messages", query="lease", order="relevance", limit=2)
    assert LISBON in outcome.text and len(ranker.calls) == 1


def test_a_search_for_any_word_is_reranked() -> None:
    ranker = FakeReranker(preferring("Lisbon flat"))
    outcome = MemoryBench(reranker=ranker).run("search_messages", query="sourdough lisbon", match="any", limit=1)
    assert len(ranker.calls) == 1 and len(ranker.calls[0]["items"]) == 3  # the pool of a limit of 1
    assert LISBON in outcome.text and "sourdough again this weekend" not in outcome.text


def test_a_first_or_last_time_search_is_not_reranked() -> None:
    for order, first, second in (("oldest", LISBON, SIGNED), ("newest", RENEW, SCAN)):
        ranker = FakeReranker(preferring("Clint"))
        bench = MemoryBench(reranker=ranker)
        assert bench.prepare("search_messages", query="lease", order=order, limit=2).fetch is None
        outcome = bench.run("search_messages", query="lease", order=order, limit=2)
        assert ranker.calls == [] and _before(outcome.text, first, second)


def test_reading_one_conversation_is_not_reranked() -> None:
    ranker = FakeReranker()
    bench = MemoryBench(reranker=ranker)
    alias = bench.alias(MAYA_PHONE)
    outcome = bench.run("search_messages", query="", conversations=[alias], order="oldest", limit=3)
    assert ranker.calls == [] and "are we still on for tonight" in outcome.text


def test_one_candidate_is_not_sent() -> None:
    ranker = FakeReranker()
    outcome = MemoryBench(reranker=ranker).run("search_messages", query="Yeastwood")
    assert ranker.calls == [] and "Clint Yeastwood" in outcome.text


def test_a_failing_reranker_leaves_the_order_of_the_call() -> None:
    ranker = FakeReranker(error=RuntimeError("the ranking model is down"))
    outcome = MemoryBench(reranker=ranker).run("search_messages", query="lease", limit=2)
    assert outcome.ok and _before(outcome.text, RENEW, SCAN) and LISBON not in outcome.text
    assert len(ranker.calls) == 1


def test_a_ranking_that_says_it_failed_leaves_the_order_of_the_call() -> None:
    ranker = FakeReranker(["4", "3"], failed=True)
    outcome = MemoryBench(reranker=ranker).run("search_messages", query="lease", limit=2)
    assert _before(outcome.text, RENEW, SCAN) and LISBON not in outcome.text


def test_unknown_and_repeated_ids_are_dropped_and_missing_ones_follow_in_their_order() -> None:
    ranker = FakeReranker(["4", "x", "4", "0", "", "99"])
    outcome = MemoryBench(reranker=ranker).run("search_messages", query="lease", limit=3)
    assert _before(outcome.text, LISBON, RENEW) and _before(outcome.text, RENEW, SCAN)
    assert SIGNED not in outcome.text


def test_more_ids_than_the_limit_are_cut() -> None:
    ranker = FakeReranker(["2", "1", "3", "4"])
    outcome = MemoryBench(reranker=ranker).run("search_messages", query="lease", limit=2)
    assert _before(outcome.text, SCAN, RENEW) and SIGNED not in outcome.text and LISBON not in outcome.text


def test_the_reranker_gets_masked_text_only() -> None:
    ranker = FakeReranker()
    MemoryBench(reranker=ranker).run("search_messages", query="dana line", match="any", limit=2)
    [asked] = ranker.calls
    texts = [item.text for item in asked["items"]]
    assert len(texts) == 2 and any("other line" in text for text in texts)
    for text in [asked["query"], *texts]:
        assert len(text) <= RERANK_CHARS
        assert not [raw for raw in RAW_HANDLES if raw in text], text


def test_the_query_is_masked_before_it_is_sent() -> None:
    ranker = FakeReranker()
    MemoryBench(reranker=ranker).run("search_messages", query="dana line +17205550199", match="any", limit=2)
    assert ranker.calls and "7205550199" not in ranker.calls[0]["query"]


def test_the_ranking_runs_beside_the_stream_and_says_so() -> None:
    from chat_engine.turn_fetch import run_beside

    ranker = FakeReranker(preferring("Lisbon"))
    bench = MemoryBench(reranker=ranker)
    prepared = bench.prepare("search_messages", query="lease", limit=2)
    assert prepared.fetch is not None
    events = list(run_beside(prepared.fetch, bench.cancel))
    assert events == [{"type": "status", "text": "Ranking what I found"}]
    outcome = prepared.run(3000)
    assert len(ranker.calls) == 1 and _before(outcome.text, LISBON, RENEW)


def test_a_cancel_before_the_ranking_asks_nothing() -> None:
    ranker = FakeReranker(preferring("Lisbon"))
    bench = MemoryBench(reranker=ranker)
    bench.cancel.cancel()
    outcome = bench.run("search_messages", query="lease", limit=2)
    assert ranker.calls == [] and LISBON not in outcome.text


def test_the_search_by_meaning_reranks_whole_exchanges() -> None:
    meaning = FakeMeaning([hit(JONAS_PHONE, "j1", "j3", 0.9), hit(MAYA_PHONE, "a1", "a4", 0.5)])
    plain = MemoryBench(meaning=meaning).run("search_messages", query="good news", match="meaning", limit=1)
    assert "Clint Yeastwood" in plain.text and SIGNED not in plain.text
    ranker = FakeReranker(preferring("lease got signed"))
    bench = MemoryBench(meaning=FakeMeaning([hit(JONAS_PHONE, "j1", "j3", 0.9), hit(MAYA_PHONE, "a1", "a4", 0.5)]),
                        reranker=ranker)
    outcome = bench.run("search_messages", query="good news", match="meaning", limit=1)
    assert SIGNED in outcome.text and "Clint Yeastwood" not in outcome.text
    [asked] = ranker.calls
    assert asked["query"] == "good news" and asked["keep"] == 1 and len(asked["items"]) == 2
    assert "1 more exchange not shown" in outcome.text


def test_a_failing_reranker_leaves_the_order_of_the_search_by_meaning() -> None:
    meaning = FakeMeaning([hit(JONAS_PHONE, "j1", "j3", 0.9), hit(MAYA_PHONE, "a1", "a4", 0.5)])
    ranker = FakeReranker(error=ValueError("bad answer"))
    outcome = MemoryBench(meaning=meaning, reranker=ranker).run(
        "search_messages", query="good news", match="meaning", limit=1
    )
    assert "Clint Yeastwood" in outcome.text and SIGNED not in outcome.text
