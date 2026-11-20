"""The seed narrows its first lookups to the time a question is about."""

import hashlib
import json

from chat_engine.engine import ChatEngine, EngineConfig
from chat_engine.fake_llm import FakeLLM, call, say
from chat_engine.fake_meaning import FakeMeaning
from chat_engine.ports import LLMRequest
from chat_engine.scope import Scope
from tests.chat_engine.conftest import FIXED_NOW
from tests.chat_engine.helpers import all_text, last_user, make_engine
from tests.chat_engine.tiny import tiny_archive, tiny_labeler

# taken at f1bc838, before the seed read the time of a question
NO_TIME_AT_F1BC838 = "6fdce55659f73219079b24df06c08d008a360c5c9fd1908f4092762d023e4784"

LAST_SUMMER = "The question is about 2026-06-01 to 2026-08-31 (last summer)."
SUMMER_SEARCH = 'search_messages(query="sourdough", match="any", since="2026-06-01", until="2026-08-31")'
WHOLE_SEARCH = 'search_messages(query="sourdough", match="any")'


def _request_text(request: LLMRequest) -> str:
    return "\n".join([all_text(request), str(request.allow_tools)])


def _sent(question: str, **asked) -> tuple[str, dict]:
    engine, fake, _ = make_engine([say("ok")])
    result = engine.ask(question, **asked)
    return last_user(fake.requests[0]), result


def _lookup(block: str, label: str) -> str:
    """The result of the seeded lookup with this label."""
    return block.split(label)[1].split("[lookup ")[0].split("Question:")[0]


def test_a_question_with_no_time_keeps_its_requests_byte_for_byte() -> None:
    sent: list[str] = []
    for question, scope in [
        ("What did Maya say about the lease?", None),
        ("anything about sourdough?", Scope(since="2024-01-01")),
        ("who do I text the most?", None),
    ]:
        steps = [[call("search_messages", "s1", query="lease", limit=3)], say("Maya signed the lease [m1].")]
        engine, fake, _ = make_engine(steps)
        result = engine.ask(question, scope=scope) if scope else engine.ask(question)
        result.pop("limited")
        sent.append("\n\n".join(_request_text(request) for request in fake.requests))
        sent.append(json.dumps(result, ensure_ascii=False, sort_keys=True))
    digest = hashlib.sha256("\n\n".join(sent).encode("utf-8")).hexdigest()
    assert digest == NO_TIME_AT_F1BC838


def test_a_relative_period_narrows_one_search_and_keeps_one_whole() -> None:
    block, result = _sent("sourdough last summer?")
    assert [step["tool"] for step in result["steps"]] == ["search_messages", "search_messages", "count_messages"]
    assert SUMMER_SEARCH in block and WHOLE_SEARCH in block
    summer = _lookup(block, SUMMER_SEARCH)
    assert "sourdough again this weekend" in summer
    assert "sourdough starter is alive" not in summer
    assert "sourdough starter is alive" in _lookup(block, WHOLE_SEARCH)


def test_the_time_words_are_not_search_words() -> None:
    block, _ = _sent("sourdough last summer?")
    assert 'query="sourdough last summer"' not in block
    assert 'query="sourdough"' in block


def test_a_named_period_narrows_the_only_search() -> None:
    block, result = _sent("sourdough in February 2024?")
    label = 'search_messages(query="sourdough", match="any", since="2024-02-01", until="2024-02-29")'
    assert [step["tool"] for step in result["steps"]] == ["search_messages", "count_messages"]
    assert label in block
    assert "sourdough starter is alive" in block
    assert "sourdough again this weekend" not in block
    assert "The question is about 2024-02-01 to 2024-02-29 (February 2024)." in block


def test_the_figures_are_those_of_the_period() -> None:
    block, _ = _sent("who did I talk to most last summer?")
    label = 'count_messages(group_by="conversation", since="2026-06-01", until="2026-08-31")'
    figures = _lookup(block, label)
    assert "Jonas Weber · direct: 1 (you 1, others 0)" in figures
    assert "Maya Okafor" not in figures


def test_the_question_block_names_the_days_before_the_question() -> None:
    block, _ = _sent("who did I talk to most last summer?")
    assert LAST_SUMMER in block
    assert block.index(LAST_SUMMER) < block.index("Question: who did I talk to most last summer?")


def test_the_period_stays_inside_the_scope_of_the_user() -> None:
    block, _ = _sent("who did I talk to most last summer?", scope=Scope(since="2026-07-01"))
    assert 'count_messages(group_by="conversation", since="2026-07-01", until="2026-08-31")' in block
    assert "The question is about 2026-07-01 to 2026-08-31 (last summer)." in block


def test_a_period_outside_the_scope_is_dropped() -> None:
    block, result = _sent("sourdough last summer?", scope=Scope(until="2025-12-31"))
    assert [step["tool"] for step in result["steps"]] == ["search_messages"]
    assert "The question is about" not in block
    assert "since=" not in block


def test_a_period_the_archive_does_not_hold_changes_nothing() -> None:
    block, result = _sent("what happened in march 2030?")
    assert [step["tool"] for step in result["steps"]] == ["search_messages"]
    assert "The question is about" not in block


def test_a_phone_number_in_the_question_gives_no_period() -> None:
    block, result = _sent("who is +1 (720) 555-2019?")
    assert "count_messages" not in [step["tool"] for step in result["steps"]]
    assert "The question is about" not in block


def test_the_whole_search_finds_what_the_narrowed_one_misses() -> None:
    _, result = _sent("sourdough last summer?")
    narrowed, whole = result["steps"][0]["text"], result["steps"][1]["text"]
    assert narrowed == "Searched for “sourdough”: 1 match"
    assert whole == "Searched for “sourdough”: 2 matches"


def test_the_seeded_search_by_meaning_is_narrowed_too() -> None:
    port = FakeMeaning([])
    settings = EngineConfig(model="test-model", max_output_tokens=800, timezone="Test/Zone")
    fake = FakeLLM([say("ok")])
    engine = ChatEngine(tiny_archive(), fake, tiny_labeler(), settings, meaning=port, clock=lambda: FIXED_NOW)
    engine.ask("what did we cook last summer?")
    asked = [(one["since"], one["until"], one["limit"]) for one in port.calls]
    assert asked[0][:2] == ("2026-06-01", "2026-08-31")
    assert asked[1][:2] == (None, None) and asked[1][2] < asked[0][2]
    block = last_user(fake.requests[0])
    assert 'search_messages(match="meaning", query=the question, since="2026-06-01", until="2026-08-31")' in block
    assert 'search_messages(match="meaning", query=the question)' in block
