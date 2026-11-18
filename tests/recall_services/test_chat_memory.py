"""chat.py gives the engine the two ports of `memory.ports`. The model is a script."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from archive import Archive
from chat_engine.fake_digests import FakeDigests, period
from chat_engine.fake_llm import FakeLLM, call, say
from chat_engine.fake_meaning import FakeMeaning
from chat_engine.ports import LLMRequest
from recall_services import chat, memory
from recall_services.names import Names
from tests.recall_services import world

NOW = datetime(2026, 11, 17, 9, 30)
QUESTION = "Who said the wheelbarrow has a flat tire?"


def ask(archive: Archive, saved: Names, llm: FakeLLM, **more: Any) -> dict[str, Any]:
    settings = {"model": "gpt-6.1-sol", "llm_factory": lambda _config: llm, "clock": lambda: NOW, **more}
    events = list(chat.stream(QUESTION, archive=archive, names=saved, **settings))
    assert events[-1]["type"] == "done", events[-1]
    return events[-1]["payload"]


def tools_of(request: LLMRequest) -> dict[str, Any]:
    return {spec.name: spec for spec in request.tools}


def matches_of(request: LLMRequest) -> list[Any]:
    return list(tools_of(request)["search_messages"].parameters["properties"]["match"]["enum"])


def given(monkeypatch: pytest.MonkeyPatch, started: list[Any] | None = None, **ports: Any) -> list[tuple[Any, Any]]:
    """Put these ports in the place of the ones of the app. Gives what `memory.ports` was asked with."""
    asked: list[tuple[Any, Any]] = []

    def stand_in(archive: Any, names: Any, *, start_digest: Any = None) -> dict[str, Any]:
        asked.append((archive, names))
        if started is not None:
            started.append(start_digest)
        return {"meaning": None, "digests": None, **ports}

    monkeypatch.setattr(memory, "ports", stand_in)
    return asked


def test_the_engine_gets_the_ports_of_the_archive_and_the_names_of_the_question(
    archive: Archive, saved: Names, monkeypatch: pytest.MonkeyPatch
) -> None:
    meaning, digests = FakeMeaning(), FakeDigests({})
    asked = given(monkeypatch, meaning=meaning, digests=digests)
    llm = FakeLLM([say("Nothing says so.")])
    ask(archive, saved, llm)
    assert asked == [(archive, saved)]
    assert "meaning" in matches_of(llm.requests[0]) and "conversation_digest" in tools_of(llm.requests[0])
    # the seeded search asked the port, with the question as the description
    assert [made["query"] for made in meaning.calls] == [QUESTION]


def test_with_no_port_neither_tool_is_offered(archive: Archive, saved: Names, monkeypatch: pytest.MonkeyPatch) -> None:
    given(monkeypatch)
    llm = FakeLLM([say("Nothing says so.")])
    ask(archive, saved, llm)
    assert "meaning" not in matches_of(llm.requests[0]) and "conversation_digest" not in tools_of(llm.requests[0])


def test_a_call_of_the_digest_tool_reaches_the_port_with_the_conversation_of_the_scope(
    archive: Archive, saved: Names, monkeypatch: pytest.MonkeyPatch
) -> None:
    story = [period("2024", "2024-04-01", "2024-04-02", 5, "They mend a wheelbarrow.")]
    digests = FakeDigests({world.GARDEN: story}, cached=True)
    given(monkeypatch, digests=digests)
    llm = FakeLLM([[call("conversation_digest", "d1", conversation=world.GARDEN_TITLE)], say("They mended it.")])
    payload = ask(archive, saved, llm, contact=world.GARDEN)
    assert payload["answer"] == "They mended it."
    assert [(kind, made["conversation_id"]) for kind, made in digests.calls] == [
        ("estimate", world.GARDEN),
        ("digest", world.GARDEN),
    ]
    shown = "\n".join(message.content for message in llm.requests[1].messages if message.role == "tool")
    assert "They mend a wheelbarrow." in shown


def test_the_local_answer_asks_for_no_port(archive: Archive, saved: Names, monkeypatch: pytest.MonkeyPatch) -> None:
    asked = given(monkeypatch)
    events = list(chat.stream(QUESTION, archive=archive, names=saved, use_ai=False))
    assert events[-1]["type"] == "done" and asked == []


# --- a long conversation is read outside the question ------------------------------


def test_the_function_that_starts_a_digest_goes_from_the_route_to_the_ports(
    archive: Archive, saved: Names, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[Any] = []
    given(monkeypatch, started)

    def start(_conversation: str, _since: Any, _until: Any) -> bool:
        return True

    ask(archive, saved, FakeLLM([say("Nothing says so.")]), start_digest=start)
    ask(archive, saved, FakeLLM([say("Nothing says so.")]))
    assert started == [start, None]


def test_the_engine_gets_the_limit_of_parts_of_memory(
    archive: Archive, saved: Names, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine and the digest port read one number, so the two can not differ."""
    made: list[Any] = []
    real = chat.ChatEngine

    def noted(adapter: Any, llm: Any, labeler: Any, settings: Any, **more: Any) -> Any:
        made.append(settings)
        return real(adapter, llm, labeler, settings, **more)

    given(monkeypatch)
    monkeypatch.setattr(chat, "ChatEngine", noted)
    ask(archive, saved, FakeLLM([say("Nothing says so.")]))
    monkeypatch.setattr(memory, "DIGEST_PARTS_LIMIT", 1)
    ask(archive, saved, FakeLLM([say("Nothing says so.")]))
    assert [settings.digest_parts_limit for settings in made] == [2, 1]
    assert not hasattr(chat, "_digest_limits")  # no look at whether the engine has the field
