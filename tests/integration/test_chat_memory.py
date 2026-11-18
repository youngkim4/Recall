"""Whole questions through chat with the search by meaning and the digest, on a real archive.

Every piece is the real one but the three that would leave this machine:

- the archive is `archive.Archive`, pulled from an invented chat.db;
- the index is built by `recall_services.meaning.build` with `FakeEmbedder`;
- the digest is built by `recall_services.digests` with the fake model of the report engine;
- the question goes through `recall_services.chat` with the ports of `memory.ports`,
  and the chat model is a script that calls the two tools.

Mina's conversation holds two months of invented chatter and one evening about
a ferry and a supper. Theo, who is not saved, has ten days and a ferry of his
own. No line holds a word of `BOAT_QUERY`: only a search by meaning finds them.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from archive import Archive
from chat_engine.cancel import CancelToken
from chat_engine.fake_llm import FakeLLM, Step, call, say
from chat_engine.ports import LLMRequest
from meaning_index import EmbeddingNetworkError, FakeEmbedder, IndexStore
from recall_services import chat, meaning, memory, reports
from recall_services.names import Names
from report_engine.fake_llm import FakeCall
from report_engine.fake_llm import FakeLLM as ReportLLM
from tests.integration.report_data import MINA_CONTACT, Pulled
from tests.recall_services import memory_kit as kit
from tests.recall_services.report_world import MODEL, OPTIONS
from tests.report_engine.support import runtime
from tests.report_engine.synth import MINA, THEO, WordCounter, conversation

NOW = datetime(2026, 11, 17, 9, 30)
QUESTION = "What did we do on the water?"
EVENING = "2023-01-20"
NOT_AVAILABLE = "The search by meaning was not available: these were found by their words only."
HANDLE_PARTS = (MINA, THEO, "2025550143", "theo@", "example.com")
LINES = (
    (MINA, f"{EVENING} 20:00:00", kit.FERRY_LINE, False),
    (MINA, f"{EVENING} 20:02:00", kit.FERRY_REPLY, True),
    (THEO, "2024-06-05 20:00:00", kit.LATE_FERRY_LINE, False),
)


@dataclass
class Harbor:
    """The archive, the index folder, and the stand-ins that count what a question asked them for."""

    pulled: Pulled
    archive: Archive
    names: Names
    store: IndexStore
    embedder: FakeEmbedder  # embeds the descriptions of chat
    digest_llm: ReportLLM  # reads the parts of a digest
    settings: memory.DigestSettings

    def report_more(self, llm: ReportLLM) -> dict[str, Any]:
        """What a report run gets, so that it shares the notes of a digest."""
        chosen = self.settings
        return {
            "archive": self.archive, "names": self.names, "places": chosen.places, "options": chosen.options,
            "counter": chosen.counter, "runtime": chosen.runtime, "llm_factory": lambda _info: llm,
        }  # fmt: skip


def wire(monkeypatch: pytest.MonkeyPatch, harbor: Harbor, embedder: Any = None, store: IndexStore | None = None) -> None:
    """Make `memory.ports` give chat the ports of this harbor. Every other piece is the one of the app."""
    made = embedder or harbor.embedder
    given = partial(
        PORTS, store=store or harbor.store, embedder_factory=lambda _model, _dimensions: made, digest=harbor.settings
    )
    monkeypatch.setattr(memory, "ports", given)


PORTS = memory.ports


@pytest.fixture
def harbor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harbor]:
    pulled = Pulled(tmp_path / "harbor")
    pulled.write(MINA, conversation("2023-01-01", days=59, per_day=6, words=8))
    pulled.write(THEO, conversation("2024-06-01", days=10, per_day=4, others=(THEO,)))
    kit.write(pulled, LINES)
    pulled.pull([MINA_CONTACT])
    saves = tmp_path / "saves"
    digest_llm = ReportLLM()
    settings = memory.DigestSettings(
        model=MODEL,
        places=reports.ReportPlaces(tmp_path / "out", saves / "reports", saves / "report-cache.sqlite3"),
        options=OPTIONS,
        counter=WordCounter(),
        llm_factory=lambda _info: digest_llm,
        runtime=runtime(),
    )
    made = Harbor(
        pulled, pulled.open(), Names.from_records([MINA_CONTACT]), IndexStore(saves / "meaning"), FakeEmbedder(),
        digest_llm, settings,
    )  # fmt: skip
    meaning.build(
        made.archive, store=made.store, facts=kit.FAKE_FACTS, count_tokens=kit.words,
        embedder_factory=lambda _model, _dimensions: FakeEmbedder(),
    )  # fmt: skip
    memory.forget()
    wire(monkeypatch, made)
    try:
        yield made
    finally:
        memory.forget()
        made.archive.close()
        pulled.close()


def ask(harbor: Harbor, script: Sequence[Step], question: str = QUESTION, **more: Any) -> tuple[list[dict[str, Any]], FakeLLM]:
    llm = FakeLLM(script)
    events = list(
        chat.stream(
            question, archive=harbor.archive, names=harbor.names, model=MODEL, llm_factory=lambda _config: llm,
            clock=lambda: NOW, **more,
        )  # fmt: skip
    )
    return events, llm


def done(events: list[dict[str, Any]]) -> dict[str, Any]:
    assert events[-1]["type"] == "done", events[-1]
    return events[-1]["payload"]


def tool_results(request: LLMRequest) -> list[str]:
    return [message.content for message in request.messages if message.role == "tool"]


def sent(llm: FakeLLM) -> str:
    """Everything the chat model was sent, in every call."""
    return "\n".join(message.content for request in llm.requests for message in request.messages)


def ref_of(text: str, words: str) -> str:
    """The reference of the message line that holds these words."""
    for line in text.splitlines():
        found = re.match(r"(m\d+) \d{2}:\d{2} ", line)
        if found and words in line:
            return found.group(1)
    raise AssertionError(f"no message line holds {words!r}")


def tools_of(request: LLMRequest) -> dict[str, Any]:
    return {spec.name: spec for spec in request.tools}


def matches_of(request: LLMRequest) -> list[Any]:
    return list(tools_of(request)["search_messages"].parameters["properties"]["match"]["enum"])


def by_meaning(query: str = kit.BOAT_QUERY, **more: Any) -> Any:
    return call("search_messages", "s1", match="meaning", query=query, **more)


def digest_of(conversation: str = "Mina", call_id: str = "d1", **more: Any) -> Any:
    return call("conversation_digest", call_id, conversation=conversation, **more)


def cite_both(request: LLMRequest) -> list[Any]:
    """An answer from the two tool results: the ferry line, and the quote of the first moment of the digest."""
    found, story = tool_results(request)
    quote = re.search(r'Quote: "([^"]+)"', story)
    assert quote is not None
    words = quote.group(1)
    text = (
        f'You took a ferry: Mina wrote "{kit.FERRY_LINE}" [{ref_of(found, kit.FERRY_LINE)}]. '
        f'In January one of you wrote "{words}" [{ref_of(story, words)}].'
    )
    return say(text, chunk=24)


def resolved(harbor: Harbor, citation: dict[str, Any]) -> dict[str, Any]:
    """The message of the archive that a citation of the payload names."""
    stored = harbor.archive.message(citation["messageId"])
    assert stored is not None and stored["text"] == citation["text"] and stored["conversation"] == citation["chatId"]
    return dict(stored)


# --- a whole question --------------------------------------------------------------


def test_a_whole_question_finds_the_exchange_by_meaning_reads_the_digest_and_cites_real_messages(
    harbor: Harbor,
) -> None:
    events, llm = ask(harbor, [[by_meaning(), digest_of()], cite_both])
    payload = done(events)
    found, story = tool_results(llm.requests[1])

    # the exchange that holds no word of the description is the first one, each line with a reference
    first = found.split("\n\n")[1]
    assert re.search(rf"^m\d+ 20:00 Mina: {kit.FERRY_LINE}$", first, re.MULTILINE)
    assert re.search(rf"^m\d+ 20:02 You: {kit.FERRY_REPLY}$", first, re.MULTILINE)
    assert NOT_AVAILABLE not in found and kit.LATE_FERRY_LINE in found

    # the digest gives each part of the conversation, and the message of each moment with a reference
    assert "== Jan 2023 · 2023-01-01 to 2023-01-31" in story and "== Feb 2023 · 2023-02-01 to 2023-02-28" in story
    quotes = re.findall(r'Quote: "([^"]+)"', story)
    assert len(quotes) == 2 and all(ref_of(story, quote) for quote in quotes)
    assert len(harbor.digest_llm.map_calls) == 2 and harbor.digest_llm.reduce_calls == []

    # the answer cites two messages, and each one is a message of the archive
    assert payload["unverifiedQuotes"] == [] and payload["cancelled"] is False
    ferry, moment = (resolved(harbor, citation) for citation in payload["citations"])
    assert (ferry["text"], ferry["conversation"], ferry["timestamp"]) == (kit.FERRY_LINE, MINA, f"{EVENING} 20:00:00")
    assert (moment["text"], moment["conversation"]) == (quotes[0], MINA)
    assert [citation["displayName"] for citation in payload["citations"]] == ["Mina", "Mina"]
    statuses = [event["text"] for event in events if event["type"] == "status"]
    assert "Reading your whole conversation with Mina" in statuses

    # only the question and the description went to the embedder, and no handle went anywhere
    assert harbor.embedder.texts == [QUESTION, kit.BOAT_QUERY]
    told = "\n".join([sent(llm), *harbor.embedder.texts, *(made.text for made in harbor.digest_llm.calls)])
    assert not [part for part in HANDLE_PARTS if part in told]


def test_the_same_question_again_embeds_nothing_and_reads_no_part_again(harbor: Harbor) -> None:
    first = done(ask(harbor, [[by_meaning(), digest_of()], cite_both])[0])
    events, _llm = ask(harbor, [[by_meaning(), digest_of()], cite_both])
    assert done(events)["answer"] == first["answer"]
    assert harbor.embedder.texts == [QUESTION, kit.BOAT_QUERY]
    assert len(harbor.digest_llm.calls) == 2
    assert "Recalling your conversation with Mina" in [event["text"] for event in events if event["type"] == "status"]


# --- scope -------------------------------------------------------------------------


class Noted:
    """Notes what the two ports of every question were asked for."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.searches: list[Any] = []
        self.digests: list[str] = []
        search, estimate, digest = memory.MeaningPort.search, memory.DigestPort.estimate, memory.DigestPort.digest

        def noted_search(port: Any, query: str, **more: Any) -> Any:
            self.searches.append(more["conversations"])
            return search(port, query, **more)

        def noted(real: Callable[..., Any]) -> Callable[..., Any]:
            def run(port: Any, conversation_id: str, **more: Any) -> Any:
                self.digests.append(conversation_id)
                return real(port, conversation_id, **more)

            return run

        monkeypatch.setattr(memory.MeaningPort, "search", noted_search)
        monkeypatch.setattr(memory.DigestPort, "estimate", noted(estimate))
        monkeypatch.setattr(memory.DigestPort, "digest", noted(digest))


def test_a_scope_of_one_conversation_limits_both_tools(harbor: Harbor, monkeypatch: pytest.MonkeyPatch) -> None:
    noted = Noted(monkeypatch)
    script = [[by_meaning(), digest_of("Mina"), digest_of("c2", "d2")], say("Only Mina is in this chat.")]
    events, llm = ask(harbor, script, contact=MINA)
    assert done(events)["contact"] == MINA
    found, story, refused = tool_results(llm.requests[1])
    assert kit.FERRY_LINE in found and "== Jan 2023" in story
    assert "Cannot run conversation_digest" in refused
    # nothing of the other conversation reached a port, the digest model or the chat model
    assert noted.searches == [[MINA], [MINA]] and set(noted.digests) == {MINA}
    assert kit.LATE_FERRY_LINE not in sent(llm) and "unsaved email contact:" not in sent(llm)
    assert len(harbor.digest_llm.map_calls) == 2


def test_with_no_scope_the_ports_are_asked_for_every_conversation(harbor: Harbor, monkeypatch: pytest.MonkeyPatch) -> None:
    noted = Noted(monkeypatch)
    events, llm = ask(harbor, [[by_meaning(conversations=["Mina"])], say("Found.")])
    done(events)
    assert noted.searches == [None, [MINA]]
    assert kit.LATE_FERRY_LINE not in tool_results(llm.requests[1])[0]


def test_the_days_of_a_call_reach_both_ports_in_the_form_that_each_one_reads(harbor: Harbor) -> None:
    script = [
        [by_meaning(since="2023-01-01", until="2023-12-31"), digest_of(since="2023-02-01", until="2023-02-28")],
        say("Read."),
    ]
    events, llm = ask(harbor, script)
    done(events)
    found, story = tool_results(llm.requests[1])
    assert kit.FERRY_LINE in found and kit.LATE_FERRY_LINE not in found
    assert "== Feb 2023" in story and "== Jan 2023" not in story
    assert len(harbor.digest_llm.map_calls) == 1


# --- when a piece is missing or fails ----------------------------------------------


class Offline:
    """An embedder with no connection."""

    model, dimensions = kit.FAKE_FACTS.model, kit.FAKE_FACTS.dimensions

    def embed(self, _texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise EmbeddingNetworkError("the embedding service failed (APIConnectionError)")


def test_when_the_embedder_fails_the_search_falls_back_to_words_and_the_question_ends(
    harbor: Harbor, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire(monkeypatch, harbor, embedder=Offline())

    def answer(request: LLMRequest) -> list[Any]:
        (found,) = tool_results(request)
        return say(f"Mina wrote it [{ref_of(found, kit.FERRY_LINE)}].")

    events, llm = ask(harbor, [[by_meaning("ferry supper")], answer])
    payload = done(events)
    (found,) = tool_results(llm.requests[1])
    assert found.splitlines()[0].endswith(NOT_AVAILABLE) and kit.FERRY_LINE in found
    assert [resolved(harbor, citation)["text"] for citation in payload["citations"]] == [kit.FERRY_LINE]
    assert not [event for event in events if event["type"] == "error"]


def test_with_no_index_the_search_by_meaning_is_not_offered(harbor: Harbor, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wire(monkeypatch, harbor, store=IndexStore(tmp_path / "no-index" / "meaning"))
    events, llm = ask(harbor, [say("Nothing says so.")])
    done(events)
    assert "meaning" not in matches_of(llm.requests[0]) and "conversation_digest" in tools_of(llm.requests[0])
    assert 'match="meaning"' not in sent(llm) and harbor.embedder.calls == []


def test_with_no_key_neither_tool_is_offered(harbor: Harbor, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    events, llm = ask(harbor, [say("Nothing says so.")])
    done(events)
    assert "meaning" not in matches_of(llm.requests[0]) and "conversation_digest" not in tools_of(llm.requests[0])
    assert harbor.embedder.calls == [] and harbor.digest_llm.calls == []


def test_with_both_ports_both_tools_are_offered(harbor: Harbor) -> None:
    _events, llm = ask(harbor, [say("Nothing says so.")])
    assert "meaning" in matches_of(llm.requests[0]) and "conversation_digest" in tools_of(llm.requests[0])


# --- the digest and a report -------------------------------------------------------


def test_a_digest_after_a_report_on_the_same_conversation_makes_no_call(harbor: Harbor) -> None:
    written = ReportLLM()
    report = reports.run(MINA, MODEL, **harbor.report_more(written)).report
    assert len(written.map_calls) == 2

    def answer(request: LLMRequest) -> list[Any]:
        (story,) = tool_results(request)
        return say(f"The summary of January: {report.periods[0].summary} It is a summary, with no quote.")

    events, llm = ask(harbor, [[digest_of()], answer])
    done(events)
    (story,) = tool_results(llm.requests[1])
    assert harbor.digest_llm.calls == []
    assert all(period.summary in story for period in report.periods)
    assert "Recalling your conversation with Mina" in [event["text"] for event in events if event["type"] == "status"]


def test_a_report_after_a_digest_reads_no_part_again(harbor: Harbor) -> None:
    done(ask(harbor, [[digest_of()], say("Read.")])[0])
    written = ReportLLM()
    reports.run(MINA, MODEL, **harbor.report_more(written))
    assert written.map_calls == [] and len(written.reduce_calls) == 1


def test_a_cancel_while_the_digest_is_built_ends_the_question_as_cancelled(
    harbor: Harbor, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = CancelToken()

    def stops(made: FakeCall) -> Any:
        token.cancel()
        return None

    slow = ReportLLM(stops)
    harbor.settings = replace(harbor.settings, llm_factory=lambda _info: slow)
    wire(monkeypatch, harbor)
    events, llm = ask(harbor, [[digest_of()]], cancel=token)
    payload = done(events)
    assert payload["cancelled"] is True and len(slow.calls) == 1 and len(llm.requests) == 1
