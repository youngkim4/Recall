"""Whole questions with a digest that is bounded: a long conversation, two questions at once, a provider that fails.

The pieces are those of `test_chat_memory.py`. The long conversation is the
one of the report tests: four parts, two more than a question reads.

Some lines hold only when the chat engine has the limit of parts
(`EngineConfig.digest_parts_limit`) and calls `prepare`. `ENGINE_PREPARES`
marks them. Without it the engine asks the port, and the port refuses.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from chat_engine.engine import EngineConfig
from chat_engine.fake_llm import FakeLLM, say
from recall_services import chat, memory
from report_engine import LLMError
from report_engine.fake_llm import FakeCall
from report_engine.fake_llm import FakeLLM as ReportLLM
from tests.integration.test_chat_memory import PORTS, Harbor, ask, digest_of, done, harbor, tool_results, wire  # noqa: F401
from tests.recall_services import report_world
from tests.recall_services.report_world import MODEL
from tests.report_engine.synth import MINA

NOW = datetime(2026, 11, 17, 9, 30)
ENGINE_PREPARES = "digest_parts_limit" in {field.name for field in dataclasses.fields(EngineConfig)}


@pytest.fixture
def desk(tmp_path: Path) -> Iterator[report_world.Desk]:
    made = report_world.build_desk(tmp_path)
    memory.forget()
    try:
        yield made
    finally:
        memory.forget()
        report_world.close_desk(made)


def test_a_long_conversation_is_not_read_inside_the_question(
    desk: report_world.Desk, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four parts are not read yet, and a question reads two: no part is read, and the question still ends."""
    digest_llm = ReportLLM()
    started: list[tuple[Any, ...]] = []
    more = desk.more(digest_llm)
    settings = memory.DigestSettings(
        model=MODEL, places=more["places"], options=more["options"], counter=more["counter"],
        llm_factory=more["llm_factory"], runtime=more["runtime"],
    )  # fmt: skip
    monkeypatch.setattr(memory, "ports", partial(PORTS, digest=settings))

    def start(conversation: str, since: Any, until: Any) -> bool:
        started.append((conversation, since, until))
        return True

    llm = FakeLLM([[digest_of()], say("A fuller answer is possible in a few minutes.")])
    events = list(
        chat.stream(
            "What is it like with Mina?", archive=desk.archive, names=desk.names, model=MODEL,
            llm_factory=lambda _config: llm, clock=lambda: NOW, start_digest=start,
        )  # fmt: skip
    )
    payload = done(events)
    assert digest_llm.calls == [] and payload["cancelled"] is False
    (told,) = tool_results(llm.requests[1])
    assert "Jan 2023" not in told
    if ENGINE_PREPARES:  # the engine's side: it asks the server to read the conversation outside the question
        assert started == [(MINA, None, None)] and "background" in told


def slow(seconds: float) -> ReportLLM:
    def respond(_call: FakeCall) -> Any:
        time.sleep(seconds)
        return None

    return ReportLLM(respond)


def test_two_questions_at_once_pay_for_the_parts_of_a_digest_one_time(
    harbor: Harbor, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    digest_llm = slow(0.2)
    harbor.settings = dataclasses.replace(harbor.settings, llm_factory=lambda _info: digest_llm)
    wire(monkeypatch, harbor)
    answers: list[str] = []

    def one() -> None:
        events, _llm = ask(harbor, [[digest_of()], say("Read.")])
        answers.append(done(events)["answer"])

    askers = [threading.Thread(target=one) for _ in range(2)]
    for asker in askers:
        asker.start()
    for asker in askers:
        asker.join(30)
    assert answers == ["Read.", "Read."] and len(digest_llm.map_calls) == 2


def test_a_provider_that_fails_each_call_is_asked_for_two_parts_at_most(
    harbor: Harbor, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    def offline(_call: FakeCall) -> Any:
        raise LLMError("the model service failed (APIConnectionError)")

    failing = ReportLLM(offline)
    harbor.settings = dataclasses.replace(harbor.settings, llm_factory=lambda _info: failing)
    wire(monkeypatch, harbor)
    events, llm = ask(harbor, [[digest_of()], say("No digest.")])
    assert done(events)["answer"] == "No digest."
    assert len(failing.calls) <= memory.DIGEST_PARTS_LIMIT
    assert "could not be built now" in tool_results(llm.requests[1])[0]


def test_the_cost_of_a_digest_is_in_the_usage_of_the_answer(harbor: Harbor) -> None:  # noqa: F811
    events, _llm = ask(harbor, [[digest_of()], say("Read.")])
    usage = done(events)["usage"]
    if ENGINE_PREPARES:  # the engine's side: it adds the cost that the port returns
        assert usage["digest_usd"] > 0
    else:
        assert "digest_usd" not in usage
