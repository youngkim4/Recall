"""Whole questions through the chat engine on the real archive.

The model is a script (`chat_engine.fake_llm.FakeLLM`): no request leaves this
machine. The engine reads the real archive through
`chat_engine.archive_adapter.open_archive`. The archive was loaded from an
invented export through the legacy bridge, so a message id is `legacy:<ROWID>`.

The first tests use a few people and lines that a person can read. The last
ones run the scripted question and the tool calls of the volume run on the
invented export, in the real archive and in `FrameArchive`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from archive import Archive
from chat_engine.archive_adapter import ArchiveAdapter, open_archive
from chat_engine.engine import ChatEngine, EngineConfig
from chat_engine.fake_llm import FakeLLM, Step, call, say
from chat_engine.labeler import MaskingLabeler
from chat_engine.ports import LLMRequest
from chat_engine.scope import Scope
from chat_engine.session import ChatSession
from tests.integration import engine_data as data
from tests.integration.engine_data import Seam

CONFIG = EngineConfig(model="scripted", max_output_tokens=800)
NOW = datetime(2026, 11, 17, 9, 30)
PAYLOAD_KEYS = {
    "answer",
    "limited",
    "citations",
    "unverified_quotes",
    "invalid_references",
    "steps",
    "usage",
    "cancelled",
    "truncated",
    "mode",
}
CITATION_KEYS = {"number", "reference", "conversation_label", "sender_label", "text_truncated"}
SAME_IN_BOTH = ("timestamp", "conversation", "is_from_me", "sender", "text", "number", "reference", "sender_label")
STARTER_QUESTION = "What did Mina name her sourdough starter?"
MADE_UP_QUOTE = "this was never written anywhere"
NUMBER_RUN = re.compile(r"\+?\d[\d ().\-]{5,}\d")
RAW_HANDLES = (*data.UNSAVED_HANDLES, data.JONAS_IN_TEXT, data.JONAS_LOCAL)
MASKS = ("a number ending in 0199", "a number ending in 0177", "an unsaved email contact")
# figures of the clock: an hour that the clocks skip, a gap over a clock change. See test_seam_totals.py.
ON_THE_CLOCK = ("count by weekday and hour", "figures of")


@dataclass(frozen=True)
class Cast:
    adapter: ArchiveAdapter  # what the engine reads
    archive: Archive  # the same file, for the checks


@pytest.fixture(scope="module")
def cast_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return data.build_cast_archive(tmp_path_factory.mktemp("cast"))


@pytest.fixture
def cast(cast_path: Path) -> Iterator[Cast]:
    with open_archive(cast_path) as adapter, Archive.open(cast_path) as archive:
        yield Cast(adapter, archive)


@pytest.fixture(scope="module")
def seam(request: pytest.FixtureRequest) -> Seam:
    return data.shared_seam(request)


def _engine(archive: Any, script: Sequence[Step], names: Mapping[str, str]) -> tuple[ChatEngine, FakeLLM]:
    model = FakeLLM(script)
    return ChatEngine(archive, model, MaskingLabeler(names), CONFIG, clock=lambda: NOW), model


def _shown_ref(request: LLMRequest, words: str) -> str:
    """The reference of the line with these words, in what the model was shown."""
    for message in request.messages:
        for line in message.content.splitlines():
            found = re.match(r"(m\d+) ", line)
            if found and words in line:
                return found.group(1)
    raise AssertionError(f"no line that the model was shown holds {words!r}")


def _all_text(model: FakeLLM) -> str:
    """Everything the model was sent, in every call."""
    return "\n".join(message.content for request in model.requests for message in request.messages)


def _assert_resolves(archive: Archive, citation: Mapping[str, Any]) -> None:
    """The citation names a message of the archive, and holds that message as the archive has it."""
    stored = archive.message(citation["id"])
    assert stored is not None
    assert {key: citation[key] for key in stored} == stored
    assert set(citation) == set(stored) | CITATION_KEYS
    assert citation["text_truncated"] is False


def _starter_script() -> list[Step]:
    """Two lookups, one read around the first hit, then an answer with one true quote and one made-up quote."""

    def look(_request: LLMRequest) -> list[Any]:
        return [
            call("search_messages", "s1", query="clementine"),
            call("count_messages", "s2", text="starter", group_by="year"),
        ]

    def read(request: LLMRequest) -> list[Any]:
        return [call("read_around", "s3", reference=_shown_ref(request, data.STARTER_LINE), radius=3)]

    def answer(request: LLMRequest) -> list[Any]:
        reference = _shown_ref(request, data.STARTER_LINE)
        text = f'Mina named it Clementine [{reference}]. She wrote "{data.STARTER_LINE}" and "{MADE_UP_QUOTE}".'
        return say(text, chunk=16)

    return [look, read, answer]


def _ask_about_the_starter(cast: Cast, session: ChatSession) -> tuple[dict[str, Any], FakeLLM]:
    engine, model = _engine(cast.adapter, _starter_script(), data.SAVED_NAMES)
    return engine.ask(STARTER_QUESTION, session=session), model


# --- a few people, lines that a person can read ---------------------------------


def test_a_whole_question_is_answered_from_the_archive(cast: Cast) -> None:
    result, model = _ask_about_the_starter(cast, ChatSession())
    assert set(result) == PAYLOAD_KEYS
    assert (result["mode"], result["cancelled"], result["truncated"]) == ("ai", False, False)
    assert result["answer"] == f'Mina named it Clementine [1]. She wrote "{data.STARTER_LINE}" and "{MADE_UP_QUOTE}".'
    assert result["unverified_quotes"] == [MADE_UP_QUOTE]
    assert result["invalid_references"] == []
    assert [(step["tool"], step["seeded"]) for step in result["steps"]] == [
        ("search_messages", True),
        ("list_conversations", True),
        ("search_messages", True),  # the first messages with the person the question names
        ("search_messages", False),
        ("count_messages", False),
        ("read_around", False),
    ]
    assert all(step["text"] for step in result["steps"])
    assert (result["usage"]["model_calls"], result["usage"]["tool_calls"]) == (3, 3)
    assert len(model.requests) == 3
    (citation,) = result["citations"]
    _assert_resolves(cast.archive, citation)
    assert citation["id"] == data.cast_id(data.STARTER_LINE)
    assert (citation["number"], citation["conversation"], citation["sender"]) == (1, data.MINA, data.MINA)
    assert (citation["conversation_label"], citation["sender_label"]) == (data.MINA_NAME, data.MINA_NAME)
    # the count came from the archive: three messages of 2024 hold the word
    counted = [message.content for message in model.requests[-1].messages if message.role == "tool"][1]
    assert "2024: 3 (you 2, others 1)" in counted


def test_a_follow_up_cites_the_evidence_of_the_first_turn(cast: Cast) -> None:
    session = ChatSession()
    first, _ = _ask_about_the_starter(cast, session)

    def answer(request: LLMRequest) -> list[Any]:
        return say(f"That was on 1 March 2024 [{_shown_ref(request, data.STARTER_LINE)}].")

    engine, model = _engine(cast.adapter, [answer], data.SAVED_NAMES)
    second = engine.ask("When was that?", session=session)
    earlier = [(message.role, message.content) for message in model.requests[0].messages[1:3]]
    assert earlier[0] == ("user", STARTER_QUESTION)
    assert earlier[1][0] == "assistant" and "Clementine" in earlier[1][1]
    assert "Evidence from earlier in this chat" in model.requests[0].messages[-1].content
    assert second["answer"] == "That was on 1 March 2024 [1]."
    assert (second["usage"]["model_calls"], second["usage"]["tool_calls"]) == (1, 0)
    assert all(step["seeded"] for step in second["steps"])
    assert [citation["id"] for citation in second["citations"]] == [first["citations"][0]["id"]]
    _assert_resolves(cast.archive, second["citations"][0])
    assert second["citations"][0]["timestamp"] == "2024-03-01 09:00:00"
    assert [turn.cited for turn in session.turns] == [("m1",), ("m1",)]


def test_a_scoped_question_reads_and_cites_only_its_scope(cast: Cast) -> None:
    def look(_request: LLMRequest) -> list[Any]:
        return [
            call("search_messages", "t1", query="ferry"),
            call("search_messages", "t2", query="ferry", conversations=[data.MINA_NAME]),
            call("search_messages", "t3", query="ferry", conversations=["c2"]),
        ]

    def answer(request: LLMRequest) -> list[Any]:
        asked = _shown_ref(request, "who has the ferry tickets")
        said = _shown_ref(request, data.TICKETS_LINE)
        return say(f'Jonas wrote "{data.TICKETS_LINE}" [{said}] after the question [{asked}]. See also [m99].')

    engine, model = _engine(cast.adapter, [look, answer], data.SAVED_NAMES)
    scope = Scope(conversations=(data.GROUP,))
    result = engine.ask("Who has the ferry tickets?", scope=scope, session=ChatSession())
    assert [citation["id"] for citation in result["citations"]] == [
        data.cast_id(data.TICKETS_LINE),
        data.cast_id("who has the ferry tickets"),
    ]
    for citation in result["citations"]:
        _assert_resolves(cast.archive, citation)
        assert citation["conversation"] == data.GROUP
        assert citation["conversation_label"] == data.GROUP_TITLE
    assert result["unverified_quotes"] == []
    assert result["invalid_references"] == ["m99"]
    assert "m99" not in result["answer"]
    # two lookups named a conversation outside the scope: they were not run
    refused = [message.content for message in model.requests[-1].messages if message.role == "tool"][1:]
    assert len(refused) == 2
    assert all(text.startswith("Cannot run search_messages:") for text in refused)
    assert [step["text"] for step in result["steps"]].count("Could not run search_messages") == 2
    shown = _all_text(model)
    assert f"Scope: only c1 ({data.GROUP_TITLE})" in shown
    # the other conversations also talk about the ferry: none of their lines was shown
    assert "the ferry leaves at seven" not in shown
    assert "harbor office" not in shown
    assert data.STARTER_LINE not in shown


def test_the_stream_gives_status_delta_and_done(cast: Cast) -> None:
    engine, _ = _engine(cast.adapter, _starter_script(), data.SAVED_NAMES)
    events = list(engine.stream(STARTER_QUESTION, session=ChatSession()))
    kinds = [event["type"] for event in events]
    assert set(kinds) == {"status", "delta", "done"}
    assert kinds[0] == "status" and kinds[-1] == "done" and kinds.count("done") == 1
    assert events[0] == {"type": "status", "text": "Reading your archive"}
    assert not any(event.get("reset") for event in events)
    payload = events[-1]["payload"]
    assert "".join(event["text"] for event in events if event["type"] == "delta") == payload["answer"]
    assert kinds.count("delta") > 1
    asked, _ = _ask_about_the_starter(cast, ChatSession())
    assert payload == asked
    assert json.loads(json.dumps(events)) == events  # the server writes each event as one JSON frame


def test_the_model_reads_no_handle_of_a_person_who_is_not_saved(cast: Cast) -> None:
    """The masking seam. The archive gives raw handles as ids, senders and members, and some
    messages hold a number or an address in their text. The model reads a mask for each one.
    """

    def look(_request: LLMRequest) -> list[Any]:
        return [
            call("search_messages", "u1", query="ferry", limit=20),
            call("search_messages", "u2", query="address"),
            call("list_conversations", "u3", name=None, limit=10),
            call("count_messages", "u4", group_by="sender"),
        ]

    def answer(request: LLMRequest) -> list[Any]:
        return say(f"It leaves at seven [{_shown_ref(request, 'the ferry leaves at seven')}].")

    engine, model = _engine(cast.adapter, [look, answer], data.SAVED_NAMES)
    events: list[dict[str, Any]] = []
    question = f"When does the ferry leave, and is {data.JONAS_IN_TEXT} the number of Jonas?"
    result = engine.ask(question, session=ChatSession(), on_event=events.append)
    tools = [message.content for message in model.requests[-1].messages if message.role == "tool"]
    assert len(tools) == 4
    spoken = [str(event.get("text", "")) for event in events if event["type"] != "done"]
    steps = [step["text"] for step in result["steps"]]
    for text in (_all_text(model), *tools, *spoken, *steps, result["answer"]):
        for handle in RAW_HANDLES:
            assert handle not in text
        for run in NUMBER_RUN.findall(text):
            digits = re.sub(r"\D", "", run)
            assert not digits.endswith(("5550100199", "5550100177", "0100199")), run
    shown = _all_text(model)
    for mask in MASKS:
        assert mask in shown
    assert data.MINA_NAME in shown
    assert "call me at a number ending in 0199 if you are late" in shown
    assert "is an unsaved email contact still the address of Rowan?" in shown
    assert "is a number ending in 0199 the number of Jonas?" in shown
    # the screen gets the message as it is stored: the server maps the names
    (citation,) = result["citations"]
    assert (citation["id"], citation["text"], citation["sender"]) == (
        data.cast_id(data.FERRY_LINE),
        data.FERRY_LINE,
        data.JONAS,
    )
    assert citation["sender_label"] == "a number ending in 0199"


# --- the invented export: the real archive beside FrameArchive ------------------


def test_the_same_question_cites_the_same_messages_as_on_the_frame_archive(seam: Seam) -> None:
    """The label of a group is not compared: it names the members in the order of each archive."""
    picks = seam.picks
    question = f"{picks.medium} {picks.rare} {picks.common}?"
    chats = (ChatSession(), ChatSession())
    plans = [
        (Scope(), chats),
        (Scope(), chats),  # a follow-up in the same two chats
        (Scope(conversations=(picks.largest,)), (ChatSession(), ChatSession())),
        (Scope(conversations=(picks.group,)), (ChatSession(), ChatSession())),
    ]
    cited = 0
    for scope, (frame_chat, real_chat) in plans:
        frame_engine, _ = _engine(seam.frame_archive, data.volume_script(picks), {})
        real_engine, _ = _engine(seam.adapter, data.volume_script(picks), {})
        frame = frame_engine.ask(question, scope=scope, session=frame_chat)
        real = real_engine.ask(question, scope=scope, session=real_chat)
        for key in ("answer", "unverified_quotes", "invalid_references", "steps", "usage"):
            assert real[key] == frame[key], key
        assert len(real["citations"]) == len(frame["citations"])
        for mine, theirs in zip(real["citations"], frame["citations"], strict=True):
            _assert_resolves(seam.archive, mine)
            assert data.plain_id(mine["id"]) == theirs["id"]
            assert {key: mine[key] for key in SAME_IN_BOTH} == {key: theirs[key] for key in SAME_IN_BOTH}
            assert scope.allows(mine["conversation"])
            cited += 1
    assert cited >= 3


def test_every_tool_of_the_volume_run_works_on_the_real_archive(seam: Seam) -> None:
    """Nobody is saved in this export, so no result may hold a handle of the archive."""
    shown = data.tool_results(seam.adapter, seam.picks)
    assert len(shown) == 37
    handles = {handle for conversation in seam.adapter.conversations() for handle in conversation["members"]}
    assert len(handles) > 50
    for label, result in shown.items():
        assert result.ok, label
        assert result.tokens <= CONFIG.result_tokens, label
        for handle in handles:
            assert handle not in result.text, label
            assert handle.removeprefix("+1") not in result.text, label
    assert "a number ending in" in shown["list the largest conversations"].text
    assert "an unsaved email contact" in shown["count by conversation"].text


def test_tool_results_are_the_same_text_once_both_archives_have_one_order(seam: Seam) -> None:
    """Member order and clock changes are the only differences of the two archives in a tool result.

    `OneOrder` gives both the same order of members. The results that are
    left out hold a figure of the clock.
    """
    group = seam.picks.group
    mine_in_order, theirs_in_order = data.OneOrder(seam.adapter), data.OneOrder(seam.frame_archive)
    real = data.tool_results(mine_in_order, seam.picks)
    frame = data.tool_results(theirs_in_order, seam.picks)
    assert list(real) == list(frame)
    compared = [label for label in real if not label.startswith(ON_THE_CLOCK)]
    assert len(compared) == len(real) - 4
    for label in compared:
        assert real[label] == frame[label], label
    assert data.group_label(mine_in_order, group) == data.group_label(theirs_in_order, group)
    # without one order, the label of a group names its members in the order of each archive
    mine = data.group_label(seam.adapter, group)
    assert mine != data.group_label(seam.frame_archive, group)
    assert "a number ending in" in mine
