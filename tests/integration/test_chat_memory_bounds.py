"""Whole questions with a digest that is bounded: a long conversation, two questions at once, a provider that fails.

The pieces are those of `test_chat_memory.py`. The long conversation is the
one of the report tests: four parts, two more than a question reads.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from chat_engine.cancel import CancelToken
from chat_engine.fake_llm import FakeLLM, say
from recall_services import chat, memory
from report_engine import LLMError
from report_engine.fake_llm import FakeCall
from report_engine.fake_llm import FakeLLM as ReportLLM
from report_engine.fake_llm import completed
from tests.integration.test_chat_memory import PORTS, Harbor, ask, digest_of, done, harbor, tool_results, wire  # noqa: F401
from tests.recall_services import report_world
from tests.recall_services.report_world import MODEL
from tests.report_engine.synth import MINA

NOW = datetime(2026, 11, 17, 9, 30)
QUESTION = "What is it like with Mina?"


def settings_of(desk: report_world.Desk, llm: ReportLLM) -> memory.DigestSettings:
    more = desk.more(llm)
    return memory.DigestSettings(
        model=MODEL, places=more["places"], options=more["options"], counter=more["counter"],
        llm_factory=more["llm_factory"], runtime=more["runtime"],
    )  # fmt: skip


def ask_desk(desk: report_world.Desk, llm: FakeLLM, **more: Any) -> dict[str, Any]:
    events = list(
        chat.stream(
            QUESTION, archive=desk.archive, names=desk.names, model=MODEL, llm_factory=lambda _config: llm,
            clock=lambda: NOW, **more,
        )  # fmt: skip
    )
    return done(events)


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
    monkeypatch.setattr(memory, "ports", partial(PORTS, digest=settings_of(desk, digest_llm)))

    def start(conversation: str, since: Any, until: Any) -> bool:
        started.append((conversation, since, until))
        return True

    llm = FakeLLM([[digest_of()], say("A fuller answer is possible in a few minutes.")])
    payload = ask_desk(desk, llm, start_digest=start)
    assert digest_llm.calls == [] and payload["cancelled"] is False
    (told,) = tool_results(llm.requests[1])
    assert "Jan 2023" not in told
    # the engine asks the server to read the conversation outside the question
    assert started == [(MINA, None, None)] and "in the background" in told


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


def offline(_call: FakeCall) -> Any:
    raise LLMError("the model service failed (APIConnectionError)")


def test_a_provider_that_fails_each_call_is_asked_for_two_parts_at_most(
    desk: report_world.Desk, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conversation of four parts. Two are in the notes, so the other two are within the limit of a question."""
    reading = memory.DigestPort(desk.archive, desk.names, settings_of(desk, ReportLLM()))
    reading.digest(MINA, since="2023-01-01", until="2023-02-28", cancel=CancelToken())
    failing = ReportLLM(offline)
    monkeypatch.setattr(memory, "ports", partial(PORTS, digest=settings_of(desk, failing)))
    llm = FakeLLM([[digest_of()], say("Two parts are missing.")])
    payload = ask_desk(desk, llm)
    (told,) = tool_results(llm.requests[1])
    # each part that is not read was tried one time, and no more than a question reads
    assert len(failing.calls) == 2 == memory.DIGEST_PARTS_LIMIT
    assert "== Jan 2023" in told and "== Feb 2023" in told and "== Mar 2023" not in told
    assert payload["answer"] == "Two parts are missing."


def test_a_provider_that_fails_when_nothing_is_read_yet_ends_the_question_with_one_line(
    harbor: Harbor, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    failing = ReportLLM(offline)
    harbor.settings = dataclasses.replace(harbor.settings, llm_factory=lambda _info: failing)
    wire(monkeypatch, harbor)
    events, llm = ask(harbor, [[digest_of()], say("No digest.")])
    assert done(events)["answer"] == "No digest."
    assert len(failing.calls) == 2 and "could not be built now" in tool_results(llm.requests[1])[0]


def test_the_cost_of_a_digest_is_in_the_usage_of_the_answer(harbor: Harbor) -> None:  # noqa: F811
    events, _llm = ask(harbor, [[digest_of()], say("Read.")])
    assert done(events)["usage"]["digest_usd"] > 0


def test_the_cost_of_a_digest_that_was_paid_for_and_failed_is_in_the_usage_too(
    harbor: Harbor, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture  # noqa: F811
) -> None:
    caplog.set_level(logging.INFO, logger=memory.SPEND_LOG)
    paid = ReportLLM(lambda call: completed(call, None, text="", status="refused"))
    harbor.settings = dataclasses.replace(harbor.settings, llm_factory=lambda _info: paid)
    wire(monkeypatch, harbor)
    events, _llm = ask(harbor, [[digest_of()], say("No digest.")])
    usage = done(events)["usage"]
    (line,) = [record.getMessage() for record in caplog.records if record.name == memory.SPEND_LOG]
    assert len(paid.calls) == 2 and line.startswith("a digest inside a question gave nothing (model_error): 2 model calls, ")
    spent = float(line.split(", ")[-1].removesuffix(" USD"))
    assert spent > 0
    # the engine adds the `usd` of the failure. An engine that does not do so yet says 0.0.
    assert usage["digest_usd"] in (pytest.approx(spent, abs=1e-4), 0.0)
