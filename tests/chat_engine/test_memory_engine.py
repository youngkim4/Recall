"""Whole questions with the memory ports: the tool list, the prompt, the seeded search, the digest."""

import json
import math
from typing import Any

import pytest

from chat_engine.cancel import CancelToken
from chat_engine.engine import ChatEngine, EngineConfig
from chat_engine.fake_digests import FakeDigests, moment, period
from chat_engine.fake_llm import FakeLLM, call, say
from chat_engine.fake_meaning import FakeMeaning, hit
from chat_engine.render import estimate_tokens
from chat_engine.scope import Scope
from chat_engine.session import ChatSession
from chat_engine.tools import TOOL_SPECS
from tests.chat_engine.conftest import FIXED_NOW
from tests.chat_engine.helpers import (
    RAW_HANDLES,
    alias_of_line,
    all_text,
    events_of,
    last_user,
    ref_of_line,
    system_text,
    tool_results,
)
from tests.chat_engine.tiny import JONAS_PHONE, MAYA_PHONE, UNSAVED_EMAIL, UNSAVED_PHONE, tiny_archive, tiny_labeler

JONAS_2024 = hit(JONAS_PHONE, "j1", "j3", 0.8)
MEANING_RULE = 'To find a kind of moment (an apology, a plan, good news, a fight), set match to "meaning"'
DIGEST_RULE = "2. Story: call conversation_digest for the conversation that leads"
SUMMARY_RULE = "A summary from conversation_digest is not a quote"


def maya_story() -> list:
    return [
        period(
            "2021", "2021-03-04", "2021-03-04", 4, "Maya signs a lease. " * 60,
            [moment("2021-03-04", "The lease is signed", "Maya tells you.", "the lease got signed btw", ["a2", "j1"])],
        ),
        period("2023", "2023-07-11", "2023-07-11", 3, "The landlord raises the rent. " * 60),
    ]


def engine_with(script: list, *, meaning: Any = None, digests: Any = None, **config: Any) -> tuple:
    archive = tiny_archive()
    fake = FakeLLM(script)
    settings = {"model": "test-model", "max_output_tokens": 800, "timezone": "Test/Zone", **config}
    engine = ChatEngine(
        archive, fake, tiny_labeler(), EngineConfig(**settings), meaning=meaning, digests=digests,
        clock=lambda: FIXED_NOW,
    )
    return engine, fake, archive


def test_the_digest_limits_have_defaults_and_are_checked() -> None:
    config = EngineConfig(model="m", max_output_tokens=100)
    assert config.digest_tokens == 24_000 and config.digest_usd_limit == 0.5
    assert EngineConfig(model="m", max_output_tokens=100, digest_usd_limit=0).digest_usd_limit == 0
    with pytest.raises(ValueError):
        EngineConfig(model="m", max_output_tokens=100, digest_tokens=0)
    with pytest.raises(ValueError):
        EngineConfig(model="m", max_output_tokens=100, digest_usd_limit=-0.01)
    with pytest.raises(ValueError):
        EngineConfig(model="m", max_output_tokens=100, digest_usd_limit=math.nan)


@pytest.mark.parametrize(
    ("meaning", "digest"), [(False, False), (True, False), (False, True), (True, True)]
)
def test_a_tool_is_offered_and_named_in_the_prompt_only_with_its_port(meaning: bool, digest: bool) -> None:
    engine, fake, _ = engine_with(
        [say("Nothing found.")],
        meaning=FakeMeaning() if meaning else None,
        digests=FakeDigests({}) if digest else None,
    )
    engine.ask("anything about zanzibar?")
    request = fake.requests[0]
    names = [spec.name for spec in request.tools]
    search = request.tools[0]
    system = system_text(request)
    sent = system + json.dumps([[spec.name, spec.description, spec.parameters] for spec in request.tools])
    assert names == [spec.name for spec in TOOL_SPECS] + (["conversation_digest"] if digest else [])
    assert (request.tools == TOOL_SPECS) is (not meaning and not digest)
    assert ("meaning" in search.parameters["properties"]["match"]["enum"]) is meaning
    assert ('"meaning"' in search.description) is meaning
    assert (MEANING_RULE in system) is meaning
    assert (DIGEST_RULE in system) is digest and (SUMMARY_RULE in system) is digest
    assert ("conversation_digest" in sent) is digest
    assert ("meaning" in sent) is meaning
    for spec in request.tools:
        schema = spec.parameters
        assert schema["additionalProperties"] is False and sorted(schema["required"]) == sorted(schema["properties"])


def test_the_added_rules_stand_after_the_rules_they_belong_to() -> None:
    engine, fake, _ = engine_with([say("Nothing.")], meaning=FakeMeaning(), digests=FakeDigests({}))
    engine.ask("anything about zanzibar?")
    rules = system_text(fake.requests[0]).splitlines()
    where = {name: next(i for i, rule in enumerate(rules) if text in rule) for name, text in {
        "search": "the words people really write",
        "meaning": MEANING_RULE,
        "quote": "Never reword a quote.",
        "summary": SUMMARY_RULE,
    }.items()}
    assert where["meaning"] == where["search"] + 1
    assert where["summary"] == where["quote"] + 1
    assert "only a message reference such as [m3] is a citation" in rules[where["summary"]]


def test_the_seeded_search_joins_meaning_and_words_with_the_question_as_the_query() -> None:
    port = FakeMeaning([JONAS_2024])
    engine, fake, _ = engine_with([say("He called it Clint Yeastwood.")], meaning=port)
    result = engine.ask("What did Jonas call his bread experiment?")
    assert port.calls == [
        {
            "query": "What did Jonas call his bread experiment?", "conversations": None, "since": None,
            "until": None, "limit": 24,
        }
    ]
    block = last_user(fake.requests[0])
    assert '[lookup 1] search_messages(match="meaning", query=the question)' in block
    assert "Search by meaning for the question in all conversations: 2 exchanges, best first." in block
    found_by_meaning = block.index("Jonas Weber: its name is Clint Yeastwood")
    found_by_words = block.index("You: I call the aisle  <<< match")  # with 3 messages on each side
    assert found_by_meaning < found_by_words and "Maya Okafor: flights to Lisbon are booked" in block
    assert result["steps"][0] == {
        "tool": "search_messages", "text": "Searched by meaning for the question: 2 exchanges shown", "seeded": True,
    }


def test_the_seeded_search_by_meaning_keeps_the_scope_and_masks_the_question() -> None:
    port = FakeMeaning([hit(MAYA_PHONE, "a1", "a4"), JONAS_2024])
    engine, fake, archive = engine_with([say("Nothing in this scope.")], meaning=port)
    scope = Scope(conversations=(JONAS_PHONE,), since="2024-01-01", until="2024-12-31")
    engine.ask("did +1 (720) 555-0199 or Maya talk about the lease?", scope=scope)
    assert port.calls[0] == {
        "query": "did a number ending in 0199 or Maya talk about the lease?", "conversations": [JONAS_PHONE],
        "since": "2024-01-01", "until": "2024-12-31", "limit": 24,
    }
    sent = all_text(fake.requests[0])
    assert "its name is Clint Yeastwood" in sent and "the lease got signed" not in sent
    for raw in [*RAW_HANDLES, UNSAVED_PHONE, UNSAVED_EMAIL]:
        assert raw not in sent, raw
    for name, arguments in archive.calls:
        if name in {"search", "count"}:
            assert arguments["conversations"] == [JONAS_PHONE], (name, arguments)


def test_a_failure_of_the_port_in_the_seeded_search_does_not_fail_the_question() -> None:
    engine, fake, _ = engine_with([say("Sourdough.")], meaning=FakeMeaning(error=RuntimeError("down")))
    result = engine.ask("when did the sourdough starter begin?")
    block = last_user(fake.requests[0])
    assert "The search by meaning was not available: these were found by their words only." in block
    assert "sourdough starter is alive  <<< match" in block
    assert result["answer"] == "Sourdough."


def test_a_question_with_no_word_to_search_runs_no_seeded_search_by_meaning() -> None:
    port = FakeMeaning([JONAS_2024])
    engine, _, _ = engine_with([say("Tell me more.")], meaning=port)
    result = engine.ask("so?")
    assert port.calls == [] and result["steps"] == []


def test_the_model_can_search_by_meaning_and_cite_a_line_of_an_exchange() -> None:
    port = FakeMeaning(by_query={"naming a pet or a project": [JONAS_2024]})

    def cite(request: Any) -> list:
        ref = ref_of_line(tool_results(request)[0], "its name is Clint Yeastwood")
        return say(f'Jonas wrote "its name is Clint Yeastwood" [{ref}].')

    engine, fake, _ = engine_with(
        [[call("search_messages", "s1", query="naming a pet or a project", match="meaning", limit=3)], cite],
        meaning=port,
    )
    events: list[dict] = []
    result = engine.ask("zzz?", on_event=events.append)
    assert result["answer"] == 'Jonas wrote "its name is Clint Yeastwood" [1].'
    assert [c["id"] for c in result["citations"]] == ["j3"] and result["unverified_quotes"] == []
    statuses = [event["text"] for event in events_of("status", events)]
    assert "Searching for moments like “naming a pet or a project”" in statuses
    assert port.calls[-1]["limit"] == 9


def test_the_model_reads_a_digest_and_cites_a_message_of_a_moment() -> None:
    digests = FakeDigests({MAYA_PHONE: maya_story()}, usd=0.4)
    session = ChatSession()

    def ask_digest(request: Any) -> list:
        alias = alias_of_line(last_user(request), "Maya Okafor · direct, phone")
        return [call("conversation_digest", "d1", conversation=alias)]

    def cite(request: Any) -> list:
        digest = tool_results(request)[0]
        assert "== 2021 · 2021-03-04 to 2021-03-04 · 4 messages ==" in digest
        ref = ref_of_line(digest, "the lease got signed btw")
        return say(f'In 2021 Maya wrote "the lease got signed btw" [{ref}]. Later the rent went up [m99].')

    engine, fake, archive = engine_with([ask_digest, cite], digests=digests)
    events: list[dict] = []
    result = engine.ask("What is my friendship with Maya like?", session=session, on_event=events.append)
    assert result["answer"] == 'In 2021 Maya wrote "the lease got signed btw" [1]. Later the rent went up.'
    assert [c["id"] for c in result["citations"]] == ["a2"]
    assert result["invalid_references"] == ["m99"] and result["unverified_quotes"] == []
    assert "Reading your whole conversation with Maya Okafor" in [e["text"] for e in events_of("status", events)]
    assert result["steps"][-1] == {
        "tool": "conversation_digest",
        "text": "Read the story of your conversation with Maya Okafor: 2 periods",
        "seeded": False,
    }
    assert session.registry.ref_of("a2") is not None
    assert digests.calls[1][1] == {"conversation_id": MAYA_PHONE, "since": None, "until": None}


def test_an_answer_cannot_cite_a_message_the_digest_named_outside_the_scope() -> None:
    story = [
        period(
            "2021", "2021-03-04", "2021-03-04", 4, "Maya signs a lease.",
            [
                moment("2021-03-04", "The lease is signed", "Maya tells you.", "", ["a2"]),
                moment("2021-03-04", "Bread", "Jonas bakes sourdough.", "", ["j1"]),
            ],
        )
    ]
    digests = FakeDigests({MAYA_PHONE: story})
    session = ChatSession()

    def ask_digest(request: Any) -> list:
        alias = alias_of_line(last_user(request), "Maya Okafor · direct")
        return [call("conversation_digest", "d1", conversation=alias)]

    def cite(request: Any) -> list:
        digest = tool_results(request)[0]
        ref = ref_of_line(digest, "the lease got signed btw")
        return say(f"Signed [{ref}]. Bread [m{len(session.registry) + 1}].")

    engine, fake, _ = engine_with([ask_digest, cite], digests=digests)
    result = engine.ask("how are things with her?", scope=Scope(conversations=(MAYA_PHONE,)), session=session)
    assert [c["id"] for c in result["citations"]] == ["a2"]
    assert len(result["invalid_references"]) == 1
    assert session.registry.ref_of("j1") is None  # the digest named it; the scope refuses it
    sent = "\n".join(all_text(request) for request in fake.requests)
    assert "sourdough" not in sent and "Bread" not in sent
    assert "[Outside the scope and not shown: 1 moment.]" in sent


def test_a_digest_has_a_budget_of_its_own() -> None:
    digests = FakeDigests({MAYA_PHONE: maya_story()})

    def ask_digest(request: Any) -> list:
        alias = alias_of_line(last_user(request), "Maya Okafor · direct, phone")
        return [call("conversation_digest", "d1", conversation=alias), call("count_messages", "d2", group_by="year")]

    engine, fake, _ = engine_with(
        [ask_digest, say("Close friends.")], digests=digests, total_tool_tokens=1500, result_tokens=400
    )
    engine.ask("What is my friendship with Maya like?")
    digest, count = tool_results(fake.requests[1])
    assert estimate_tokens(digest) > 400  # more than one result of the question may hold
    assert digest.count("Maya signs a lease. ") == 59 and "[cut]" not in digest
    assert "by year. Total 27" in count  # the lookup after the digest still had its tokens
    assert fake.requests[1].allow_tools is True


def test_a_digest_is_cut_to_the_digest_budget_of_the_engine() -> None:
    digests = FakeDigests({MAYA_PHONE: maya_story()})

    def ask_digest(request: Any) -> list:
        alias = alias_of_line(last_user(request), "Maya Okafor · direct, phone")
        return [call("conversation_digest", "d1", conversation=alias)]

    engine, fake, _ = engine_with([ask_digest, say("Close.")], digests=digests, digest_tokens=600)
    engine.ask("What is my friendship with Maya like?")
    digest = tool_results(fake.requests[1])[0]
    assert estimate_tokens(digest) <= 600 and digest.count(" [cut]") == 2
    assert "[Cut to fit: each summary keeps about " in digest


def test_the_same_digest_twice_is_answered_as_a_repeat() -> None:
    digests = FakeDigests({MAYA_PHONE: maya_story()})

    def first(request: Any) -> list:
        alias = alias_of_line(last_user(request), "Maya Okafor · direct, phone")
        return [call("conversation_digest", "d1", conversation=alias)]

    def second(request: Any) -> list:
        alias = alias_of_line(last_user(request), "Maya Okafor · direct, phone")
        return [call("conversation_digest", "d2", conversation=alias, since=None, until=None)]

    engine, fake, _ = engine_with([first, second, say("Close.")], digests=digests)
    result = engine.ask("What is my friendship with Maya like?")
    assert tool_results(fake.requests[2])[1] == (
        "Same lookup as before: its result is above. Try a different lookup, or answer."
    )
    assert [name for name, _ in digests.calls] == ["estimate", "digest"]
    assert result["usage"]["tool_calls"] == 1
    assert [step["tool"] for step in result["steps"] if not step["seeded"]] == ["conversation_digest"]


def test_a_cancel_during_a_digest_ends_the_question_as_any_cancel_does() -> None:
    cancel = CancelToken()
    digests = FakeDigests({MAYA_PHONE: maya_story()}, during=cancel.cancel)
    session = ChatSession()

    def ask_digest(request: Any) -> list:
        alias = alias_of_line(last_user(request), "Maya Okafor · direct, phone")
        return [call("conversation_digest", "d1", conversation=alias), call("search_messages", "d2", query="lease")]

    engine, fake, archive = engine_with([ask_digest, say("never")], digests=digests)
    events: list[dict] = []
    result = engine.ask("What is my friendship with Maya like?", cancel=cancel, session=session, on_event=events.append)
    assert result["cancelled"] is True and result["answer"] == ""
    assert events[-1]["type"] == "done" and events[-1]["payload"]["cancelled"] is True
    assert len(fake.requests) == 1
    assert [name for name, _ in archive.calls if name == "message"] == []
    assert not [a for name, a in archive.calls if name == "search" and a["query"] == "lease"]
    assert session.turns == () and len(session.registry) == 0


def test_a_failure_of_the_digest_is_one_line_and_the_question_goes_on() -> None:
    digests = FakeDigests({MAYA_PHONE: maya_story()}, error=RuntimeError("the provider said no"))

    def ask_digest(request: Any) -> list:
        alias = alias_of_line(last_user(request), "Maya Okafor · direct, phone")
        return [call("conversation_digest", "d1", conversation=alias)]

    def go_on(request: Any) -> list:
        failed = tool_results(request)[0]
        assert failed.endswith("could not be built now. Go on with the other tools.") and "\n" not in failed
        assert [m.tool_call_id for m in request.messages if m.role == "tool"] == ["d1"]
        return [call("search_messages", "s1", query="lease signed")]

    def answer(request: Any) -> list:
        ref = ref_of_line(tool_results(request)[1], "the lease got signed btw")
        return say(f"She signed a lease [{ref}].")

    engine, fake, _ = engine_with([ask_digest, go_on, answer], digests=digests)
    result = engine.ask("What is my friendship with Maya like?")
    assert result["answer"] == "She signed a lease [1]."
    assert [step["text"] for step in result["steps"] if not step["seeded"]] == [
        "Could not read the story of your conversation with Maya Okafor",
        "Searched for “lease signed”: 1 match",
    ]
    assert "provider said no" not in "\n".join(all_text(request) for request in fake.requests)


def test_no_raw_handle_of_a_digest_reaches_the_model_or_a_status_line() -> None:
    story = [
        period(
            "2025", "2025-01-15", "2025-01-16", 3,
            "Dana writes from +1 720 555 0199 and from rk.unsaved@example.org.",
            [moment("2025-01-16", "Line 720.555.0199", "Mail: rk.unsaved@example.org", "", ["u3"])],
        )
    ]
    digests = FakeDigests({UNSAVED_PHONE: story}, usd=0.1)

    def ask_digest(request: Any) -> list:
        alias = alias_of_line(last_user(request), "a number ending in 0199 · direct")
        return [call("conversation_digest", "d1", conversation=alias)]

    engine, fake, _ = engine_with([ask_digest, say("That is Dana.")], digests=digests)
    events: list[dict] = []
    result = engine.ask("who is the 0199 number?", on_event=events.append)
    seen = "\n".join(all_text(request) for request in fake.requests)
    seen += "\n".join(event.get("text", "") for event in events if event["type"] in {"status", "delta"})
    seen += "\n".join(step["text"] for step in result["steps"])
    assert "Dana writes from a number ending in 0199 and from an unsaved email contact." in seen
    assert "Reading your whole conversation with a number ending in 0199" in seen
    for raw in [*RAW_HANDLES, UNSAVED_PHONE, UNSAVED_EMAIL, "720 555", "720.555"]:
        assert raw not in seen, raw
