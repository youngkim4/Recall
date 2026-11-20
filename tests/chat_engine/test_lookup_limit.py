"""The lookup limit of a question: the answer says it stopped, and the next question knows it."""

from chat_engine.engine import EngineConfig
from chat_engine.fake_llm import call, say
from chat_engine.prompt import LIMIT_NOTE
from chat_engine.session import ChatSession, Turn
from tests.chat_engine.helpers import last_user, make_engine

LOOPING = [call("search_messages", query="sourdough")]


def _earlier_answers(request) -> list[str]:
    return [m.content for m in request.messages if m.role == "assistant" and not m.tool_calls]


def test_an_answer_at_the_limit_is_marked_limited() -> None:
    engine, _, _ = make_engine([LOOPING, say("From what I have: it started in 2024.")], max_steps=1)
    assert engine.ask("bread?")["limited"] is True


def test_an_answer_before_the_limit_is_not_limited() -> None:
    engine, _, _ = make_engine([say("It started in 2024.")])
    assert engine.ask("bread?")["limited"] is False


def test_the_limit_note_asks_for_a_plain_stop_not_an_inability() -> None:
    note = LIMIT_NOTE.lower()
    assert "keep looking" in note
    assert "lookup limit" in note
    assert "say what you could not check" not in note


def test_the_session_remembers_that_an_answer_stopped_at_the_limit() -> None:
    session = ChatSession()
    engine, _, _ = make_engine([LOOPING, say("It started in 2024.")], max_steps=1)
    engine.ask("bread?", session=session)
    assert session.turns[-1].limited is True


def test_the_next_question_is_told_that_the_earlier_answer_stopped_at_the_limit() -> None:
    session = ChatSession()
    first, _, _ = make_engine([LOOPING, say("It started in 2024.")], max_steps=1)
    first.ask("bread?", session=session)
    second, fake, _ = make_engine([say("Yes.")])
    second.ask("why could you not check?", session=session)
    earlier = _earlier_answers(fake.requests[0])
    assert earlier and "lookup limit" in earlier[-1]


def test_an_earlier_answer_that_was_not_limited_gets_no_note() -> None:
    session = ChatSession()
    first, _, _ = make_engine([say("It started in 2024.")])
    first.ask("bread?", session=session)
    second, fake, _ = make_engine([say("Yes.")])
    second.ask("and then?", session=session)
    assert all("lookup limit" not in text for text in _earlier_answers(fake.requests[0]))
    assert "lookup limit" not in last_user(fake.requests[0])


def test_limited_survives_a_stored_session() -> None:
    session = ChatSession()
    session.add_turn(Turn("bread?", "It started in 2024.", (), limited=True))
    assert ChatSession.from_dict(session.to_dict()).turns[-1].limited is True


def test_a_stored_session_of_an_older_build_is_not_limited() -> None:
    row = {"turns": [{"question": "bread?", "answer": "It started in 2024.", "cited": []}]}
    assert ChatSession.from_dict(row).turns[-1].limited is False


def test_the_budgets_leave_room_for_a_question_that_needs_many_lookups() -> None:
    settings = EngineConfig(model="m", max_output_tokens=800)
    assert settings.max_steps >= 16
    assert settings.total_tool_tokens >= 120_000
    assert settings.result_tokens >= 6_000
    assert settings.history_tokens >= 24_000
