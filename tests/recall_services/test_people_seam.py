"""The seam of the people index and chat: the real `people` service, `PeoplePort` and `find_people`.

The archive is the invented town of `people_kit`. The profile model and the
embedder are fakes. Everything between them is the code of the app: the
store, the search, the port, `chat.stream`, the engine and the tool.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from chat_engine.fake_llm import FakeLLM, call, say
from recall_services import chat, memory, people, people_search
from recall_services.people_store import PeopleStore, people_folder
from report_engine.fake_llm import FakeCall
from report_engine.fake_llm import FakeLLM as ReportLLM
from tests.recall_services.people_kit import (
    MINA,
    MINA_MAIL,
    THEO,
    fake_embedder,
    profile_reply,
    settings,
    town,  # noqa: F401
)

RAW = ["+1202", "202555", "555-0143", "mina@example.com", "theo@example.com", "example.com"]
TAKES_CONVERSATIONS = "conversations" in inspect.signature(people.search).parameters


def responder(made: FakeCall) -> Any:
    user = made.messages[-1]["content"]
    if "This is a group conversation:" in user:
        return profile_reply(summary="Evening walks by the harbor.")
    if "conversation: Mina." in user:
        return profile_reply(
            kind="talking_stage", status="faded",
            summary=(
                "A few coffees in early 2023, then it faded. She wrote from +1 (202) 555-0143 and mina@example.com."
            ),
            turning_points=[{"date": "2023-02-14", "what": "You sent a valentine to theo@example.com by mistake."}],
        )
    return profile_reply(kind="service", status="ended", summary="Tried to sell you a boat.")


@pytest.fixture
def built(town: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The town with its profiles in the store of the app, and chat with no other memory port."""
    town.read_memory()
    store = PeopleStore(people_folder(town.archive.path))
    people.build(town.archive, town.names, **settings(town, ReportLLM(responder), store=store))
    embedder = fake_embedder()
    monkeypatch.setattr(people_search, "_app_embedder", lambda _model, _dimensions: embedder)
    monkeypatch.setattr(memory, "ports", lambda _archive, _names: {"meaning": None, "digests": None})
    return town


def ask(town: Any, script: list, **more: Any) -> tuple[list[dict], FakeLLM]:
    llm = FakeLLM(script)
    events = list(
        chat.stream("Who was a talking stage?", archive=town.archive, names=town.names, model="gpt-6.1-sol",
                    llm_factory=lambda _config: llm, **more)
    )
    assert events[-1]["type"] == "done", events[-1]
    return events, llm


def tool_text(llm: FakeLLM) -> str:
    return next(m.content for m in llm.requests[1].messages if m.role == "tool")


def test_a_question_finds_a_person_of_the_index_with_aliases_figures_and_a_masked_story(built: Any) -> None:
    script = [[call("find_people", "p1", description="a talking stage that faded", kinds=["talking_stage"])],
              say("Mina.")]
    events, llm = ask(built, script)
    text = tool_text(llm)
    lines = text.splitlines()
    assert lines[0].startswith("People like “a talking stage that faded” (kinds: talking stage): 1 found")
    assert lines[2].startswith("1. Mina · talking stage · faded · conversations c")
    assert lines[3] == (
        "   2023-01-02 to 2023-02-20 · 8 messages (you 4, them 4) · active on 4 days · longest gap 39 days · "
        "you wrote first on 50% of the active days"
    )
    assert "   Turning points: 2023-02-14 You sent a valentine to " in text
    assert "Theo" not in text and "boat" not in text  # another kind
    every = "\n".join(m.content for request in llm.requests for m in request.messages)
    for raw in RAW:
        assert raw not in every, raw
    assert "Finding people like “a talking stage that faded”" in [e.get("text") for e in events]
    # the seeded lookup ran too, through the same port
    assert "find_people(description=the question, limit=5)" in llm.requests[0].messages[-1].content


def test_the_gap_of_a_person_is_rounded(built: Any) -> None:
    _, llm = ask(built, [[call("find_people", "p1", description="a boat", kinds=["service"])], say("Theo.")])
    assert "longest gap 2 days" in tool_text(llm)  # 1.85 days


@pytest.mark.skipif(not TAKES_CONVERSATIONS, reason="people.search takes `conversations` in the other part's fix")
def test_in_a_scope_of_one_conversation_the_person_of_that_conversation_is_found(built: Any) -> None:
    script = [[call("find_people", "p1", description="a talking stage that faded", limit=1)], say("Theo.")]
    _, llm = ask(built, script, contact=THEO)
    text = tool_text(llm)
    assert "in the scope the user chose: 1 found" in text and "Mina" not in text
    assert "   Figures: call conversation_stats inside the scope." in text
    assert MINA not in text and MINA_MAIL not in text
