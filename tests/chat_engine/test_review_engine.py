"""Regression tests for the findings of the review of the engine."""

import time

import pytest

from chat_engine.citations import CitationStream
from chat_engine.errors import ModelError
from chat_engine.fake_llm import call, done, say
from chat_engine.local_answer import local_answer
from chat_engine.ports import Completed
from chat_engine.scope import Scope
from chat_engine.session import ChatSession, Turn
from chat_engine.verify import quoted_spans, unverified_quotes
from tests.chat_engine.helpers import all_text, last_user, make_engine, ref_of_line
from tests.chat_engine.tiny import JONAS_PHONE, MAYA_PHONE, tiny_archive, tiny_labeler


def _replay(events: list[dict]) -> str:
    """The text a client shows: deltas joined, cleared by a status with reset."""
    shown = ""
    for event in events:
        if event["type"] == "status" and event.get("reset"):
            shown = ""
        elif event["type"] == "delta":
            shown += event["text"]
    return shown


# --- the stream ---------------------------------------------------------------


def test_text_streamed_before_a_lookup_is_reset_even_when_no_lookup_runs() -> None:
    first = [*say("Let me look."), call("search_messages", "s1", query="lease")]
    again = [*say("Looking again."), call("search_messages", "s2", query="lease")]
    engine, _, _ = make_engine([first, again, say("The lease was signed.")])
    events: list[dict] = []
    result = engine.ask("lease?", on_event=events.append)
    assert _replay(events) == result["answer"] == "The lease was signed."


def test_an_empty_question_is_refused_at_the_call_not_at_the_first_event() -> None:
    engine, fake, _ = make_engine([])
    with pytest.raises(ValueError):
        engine.stream("   ")
    with pytest.raises(ValueError):
        engine.ask("")
    assert fake.requests == []


@pytest.mark.parametrize(
    ("stop_reason", "word"), [("length", "output"), ("filtered", "blocked"), ("stop", "lookup")]
)
def test_no_text_in_the_forced_round_says_why(stop_reason: str, word: str) -> None:
    script = [[call("search_messages", "s1", query="lease")], [done(stop_reason=stop_reason)]]
    engine, _, _ = make_engine(script, max_steps=1)
    with pytest.raises(ModelError) as caught:
        engine.ask("lease?")
    assert word in caught.value.message


# --- tool calls ---------------------------------------------------------------


def test_two_calls_with_one_id_get_two_ids_and_two_results() -> None:
    carried = ({"type": "function_call", "call_id": "same", "name": "search_messages", "arguments": "{}"},)
    first = [
        call("search_messages", "same", query="lease"),
        call("count_messages", "same", text="lease"),
        Completed(provider_items=carried),
    ]
    second = [call("count_messages", "same", text="sourdough")]
    engine, fake, _ = make_engine([first, second, say("done")])
    engine.ask("zzz?")
    last = fake.requests[2]
    called = [c.id for m in last.messages if m.role == "assistant" for c in m.tool_calls]
    answered = [m.tool_call_id for m in last.messages if m.role == "tool"]
    assert len(set(called)) == 3 and called == answered
    assert all(m.provider_items == () for m in last.messages if m.role == "assistant")


def test_provider_items_stay_when_every_id_is_new() -> None:
    carried = ({"type": "function_call", "call_id": "s1", "name": "search_messages", "arguments": "{}"},)
    first = [call("search_messages", "s1", query="lease"), Completed(provider_items=carried)]
    engine, fake, _ = make_engine([first, say("done")])
    engine.ask("zzz?")
    assistant = next(m for m in fake.requests[1].messages if m.role == "assistant")
    assert assistant.provider_items == carried and [c.id for c in assistant.tool_calls] == ["s1"]


# --- the scope and the chat ---------------------------------------------------


def _lease_chat() -> tuple[ChatSession, str, list]:
    """A chat whose first answer cites a message of Maya."""
    session = ChatSession()
    found: dict[str, str] = {}

    def cite(request):
        found["ref"] = ref_of_line(last_user(request), "the lease got signed")
        return say(f'Maya wrote "the lease got signed btw" [{found["ref"]}].')

    def repeat(_request):
        return say(f'As before: "the lease got signed btw" [{found["ref"]}].')

    engine, fake, _ = make_engine([cite, repeat, say("Nothing new.")])
    first = engine.ask("when was the lease signed?", session=session)
    assert [c["id"] for c in first["citations"]] == ["a2"]
    return session, found["ref"], [engine, fake]


def test_a_message_outside_the_scope_is_no_citation_and_no_quote_source() -> None:
    session, ref, (engine, _fake) = _lease_chat()
    result = engine.ask("and with Jonas?", scope=Scope(conversations=(JONAS_PHONE,)), session=session)
    assert result["citations"] == []
    assert result["invalid_references"] == [ref]
    assert result["unverified_quotes"] == ["the lease got signed btw"]
    assert "[1]" not in result["answer"]


def test_an_answer_from_a_wider_scope_is_left_out_of_a_narrower_question() -> None:
    session, _ref, (engine, fake) = _lease_chat()
    engine.ask("and with Jonas?", scope=Scope(conversations=(JONAS_PHONE,)), session=session)
    narrow = fake.requests[1]
    assert "the lease got signed" not in all_text(narrow)
    assert "when was the lease signed?" in all_text(narrow)
    assert "left out" in all_text(narrow)
    engine.ask("and overall?", session=session)
    wide = all_text(fake.requests[2])
    assert 'Maya wrote "the lease got signed btw"' in wide and "As before" in wide


def test_an_answer_from_a_narrower_scope_stays_in_a_wider_question() -> None:
    session = ChatSession()
    engine, fake, _ = make_engine([say("Maya only."), say("Both.")])
    engine.ask("lease?", scope=Scope(conversations=(MAYA_PHONE,), since="2021-01-01"), session=session)
    engine.ask("more?", scope=Scope(conversations=(MAYA_PHONE, JONAS_PHONE)), session=session)
    assert "Maya only." in all_text(fake.requests[1])


def test_history_tokens_zero_sends_no_earlier_turn() -> None:
    session = ChatSession()
    engine, fake, _ = make_engine([say("one"), say("two")], history_tokens=0)
    engine.ask("first question about the lease?", session=session)
    engine.ask("second?", session=session)
    assert [m.role for m in fake.requests[1].messages] == ["system", "user"]


def test_scope_within() -> None:
    everything = Scope()
    maya = Scope(conversations=(MAYA_PHONE,))
    both = Scope(conversations=(MAYA_PHONE, JONAS_PHONE))
    year = Scope(since="2024-01-01", until="2024-12-31")
    spring = Scope(since="2024-03-01", until="2024-05-31")
    assert maya.within(both) and maya.within(everything) and both.within(both)
    assert not both.within(maya) and not everything.within(maya)
    assert spring.within(year) and not year.within(spring) and not everything.within(year)
    assert Scope(conversations=(MAYA_PHONE,), since="2024-03-01", until="2024-03-02").within(year)
    assert not Scope(since="2024-03-01").within(year)


# --- the session --------------------------------------------------------------


def test_a_session_keeps_the_scope_of_each_turn() -> None:
    session = ChatSession()
    session.add_turn(Turn("q", "a", ("m1",), Scope(conversations=(MAYA_PHONE,), until="2024-12-31")))
    again = ChatSession.from_dict(session.to_dict())
    assert again.turns == session.turns
    assert ChatSession.from_dict({"turns": [{"question": "q", "answer": "a"}]}).turns[0].scope == Scope()


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        "text",
        {"turns": "x"},
        {"turns": [1]},
        {"turns": [{"question": "q", "answer": "a", "cited": "m1"}]},
        {"turns": [{"question": 1, "answer": "a"}]},
        {"turns": [{"question": "q", "answer": "a", "scope": {"since": "not a day"}}]},
        {"turns": [{"question": "q", "answer": "a", "scope": "all"}]},
        {"evidence": "x"},
        {"evidence": [{"ref": "m1"}]},
        {"evidence": [{"ref": "m1", "message": {"text": "no id"}}]},
        {"evidence": [{"ref": "m2", "message": {"id": "a", "conversation": "c", "timestamp": "", "text": ""}}]},
        {"evidence": [{"ref": "m1", "message": {"id": "a", "conversation": "c", "text": 5}}]},
        {"conversations": "x"},
        {"conversations": [1, {"a": 1}]},
    ],
)
def test_from_dict_refuses_malformed_data_with_a_value_error(data: object) -> None:
    with pytest.raises(ValueError):
        ChatSession.from_dict(data)  # type: ignore[arg-type]


def test_from_dict_takes_what_to_dict_gives() -> None:
    session, ref, _ = _lease_chat()
    again = ChatSession.from_dict(session.to_dict())
    assert again.to_dict() == session.to_dict()
    assert again.registry.get(ref)["id"] == "a2"
    assert ChatSession.from_dict({}).turns == ()


def test_from_history_takes_only_user_rows_as_questions() -> None:
    session = ChatSession.from_history(
        [
            {"role": "system", "content": "be nice"},
            {"role": "assistant", "content": "orphan"},
            {"role": "user", "content": "q1"},
            {"role": "tool", "content": "x"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
    )
    assert [(turn.question, turn.answer) for turn in session.turns] == [("q1", "a1")]
    with pytest.raises(ValueError):
        ChatSession.from_history(["not a row"])  # type: ignore[list-item]


# --- quotes and references ----------------------------------------------------


def test_an_inch_mark_or_a_lone_quote_mark_does_not_shift_the_pairs() -> None:
    assert quoted_spans('the 3.5" disk and "floppy drive is here" were') == ["floppy drive is here"]
    assert quoted_spans('He is 5\'10" and wrote "see you at the station"') == ["see you at the station"]
    assert quoted_spans('the " mark and the " mark') == []
    assert quoted_spans('"Hello there," she said, "how are you today?"') == ["Hello there,", "how are you today?"]


def test_an_apostrophe_inside_a_single_quoted_span_must_sit_inside_a_word() -> None:
    assert quoted_spans("she wrote 'I don't know yet' and left") == ["I don't know yet"]
    assert quoted_spans("the Smiths' house and Maya's 'big news today'") == ["big news today"]
    assert quoted_spans("'tis the season and 'twas the night' before") == ["twas the night"]


def test_many_quotes_are_checked_in_linear_time() -> None:
    answer = '"aaaaaaaaaaaaaaaa" and ' * 20_000
    started = time.perf_counter()
    assert len(quoted_spans(answer)) == 20_000
    assert unverified_quotes(answer, ["nothing here"]) == ["aaaaaaaaaaaaaaaa"]
    assert time.perf_counter() - started < 5.0


def test_a_bare_number_in_brackets_is_read_as_a_reference() -> None:
    stream = CitationStream(lambda ref: ref in {"m3", "m7"})
    shown = stream.feed("Signed in March [3], renewed later [7][3]. Not known [9]. Kept [sic].")
    shown += stream.finish()
    assert shown == "Signed in March [1], renewed later [2][1]. Not known. Kept [sic]."
    assert stream.order == ("m3", "m7") and stream.unknown == ("m9",)


# --- the answer without a model -----------------------------------------------


def test_local_answer_never_searches_for_the_digits_of_an_unsaved_number() -> None:
    result = local_answer(tiny_archive(), "who is +1 (720) 555-0199 from the climbing gym?", labeler=tiny_labeler())
    step = result["steps"][0]["text"]
    assert "climbing" in step
    for digits in ("720", "555", "0199"):
        assert digits not in step
