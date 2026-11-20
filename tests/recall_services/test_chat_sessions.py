"""chat_sessions.py: a follow-up question keeps what the chat read before. The model is a script."""

from __future__ import annotations

from typing import Any

import pytest

from archive import Archive
from chat_engine.fake_llm import call, say
from chat_engine.session import ChatSession, Turn
from recall_services import chat, chat_sessions
from recall_services.names import Names
from tests.recall_services import world
from tests.recall_services.test_chat import FLAT_TIRE, QUESTION, Factory, cite, done, run, shown_ref


@pytest.fixture(autouse=True)
def _empty_store() -> None:
    chat_sessions.clear()


def _history(question: str, answer: str, contact: str = "") -> list[dict[str, Any]]:
    return [{"role": "user", "content": question, "contact": contact}, {"role": "assistant", "content": answer}]


def _session_with(question: str, answer: str) -> ChatSession:
    session = ChatSession()
    session.add_turn(Turn(question, answer, ()))
    return session


# --- the store ---


def test_a_kept_session_is_found_by_its_last_question_and_answer(tmp_path) -> None:
    session = _session_with("bread?", "It started in 2024.")
    chat_sessions.keep(tmp_path / "a.sqlite3", "bread?", "It started in 2024.", session)
    assert chat_sessions.take(tmp_path / "a.sqlite3", _history("bread?", "It started in 2024.")) == (
        session,
        ("bread?", "It started in 2024."),
    )


def test_a_session_is_taken_once(tmp_path) -> None:
    session = _session_with("bread?", "It started in 2024.")
    chat_sessions.keep(tmp_path / "a.sqlite3", "bread?", "It started in 2024.", session)
    history = _history("bread?", "It started in 2024.")
    assert chat_sessions.take(tmp_path / "a.sqlite3", history)[0] is session
    assert chat_sessions.take(tmp_path / "a.sqlite3", history) is None


def test_another_archive_does_not_get_the_session(tmp_path) -> None:
    chat_sessions.keep(tmp_path / "a.sqlite3", "bread?", "Yes.", _session_with("bread?", "Yes."))
    assert chat_sessions.take(tmp_path / "b.sqlite3", _history("bread?", "Yes.")) is None


def test_a_changed_answer_does_not_match(tmp_path) -> None:
    chat_sessions.keep(tmp_path / "a.sqlite3", "bread?", "Yes.", _session_with("bread?", "Yes."))
    assert chat_sessions.take(tmp_path / "a.sqlite3", _history("bread?", "No.")) is None


def test_no_history_takes_nothing(tmp_path) -> None:
    chat_sessions.keep(tmp_path / "a.sqlite3", "bread?", "Yes.", _session_with("bread?", "Yes."))
    assert chat_sessions.take(tmp_path / "a.sqlite3", None) is None
    assert chat_sessions.take(tmp_path / "a.sqlite3", []) is None


def test_the_store_keeps_only_the_newest_sessions(tmp_path) -> None:
    path = tmp_path / "a.sqlite3"
    for number in range(chat_sessions.KEPT + 1):
        chat_sessions.keep(path, f"q{number}?", "a.", _session_with(f"q{number}?", "a."))
    assert chat_sessions.take(path, _history("q0?", "a.")) is None
    assert chat_sessions.take(path, _history(f"q{chat_sessions.KEPT}?", "a.")) is not None


def test_a_failed_question_after_the_answer_does_not_hide_its_session(tmp_path) -> None:
    session = _session_with("bread?", "Yes.")
    chat_sessions.keep(tmp_path / "a.sqlite3", "bread?", "Yes.", session)
    history = [*_history("bread?", "Yes."), {"role": "user", "content": "and then?"}]
    assert chat_sessions.take(tmp_path / "a.sqlite3", history)[0] is session


def test_a_stopped_answer_after_the_answer_does_not_hide_its_session(tmp_path) -> None:
    session = _session_with("bread?", "Yes.")
    chat_sessions.keep(tmp_path / "a.sqlite3", "bread?", "Yes.", session)
    history = [*_history("bread?", "Yes."), *_history("and then?", "It was in")]
    assert chat_sessions.take(tmp_path / "a.sqlite3", history)[0] is session


def test_a_session_is_put_back_under_the_pair_it_was_taken_by(tmp_path) -> None:
    session = _session_with("bread?", "Yes.")
    chat_sessions.keep(tmp_path / "a.sqlite3", "bread?", "Yes.", session)
    history = [*_history("bread?", "Yes."), {"role": "user", "content": "and then?"}]
    _, pair = chat_sessions.take(tmp_path / "a.sqlite3", history)
    chat_sessions.put_back(tmp_path / "a.sqlite3", pair, session)
    assert chat_sessions.take(tmp_path / "a.sqlite3", _history("bread?", "Yes."))[0] is session


# --- through chat.stream ---


def test_a_follow_up_can_read_around_a_message_the_first_answer_cited(archive: Archive, saved: Names) -> None:
    first = done(run(archive, saved, Factory([cite(FLAT_TIRE)])))

    def read_on(request):
        ref = shown_ref(request, FLAT_TIRE)
        return [call("read_around", reference=ref)]

    factory = Factory([read_on, say("I read on from it.")])
    history = _history(QUESTION, first["answer"])
    payload = done(run(archive, saved, factory, "read on from that message", history=history))
    assert payload["answer"] == "I read on from it."
    tool_text = [m.content for m in factory.llm.requests[1].messages if m.role == "tool"]
    assert tool_text and FLAT_TIRE in tool_text[0]


def test_a_follow_up_is_told_that_the_first_answer_stopped_at_the_limit(
    archive: Archive, saved: Names, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat, "LIMITS", {"max_steps": 1})
    looping = [call("search_messages", query="wheelbarrow")]
    first = done(run(archive, saved, Factory([looping, say("Someone in the group wrote it.")])))
    assert first["limited"] is True
    factory = Factory([say("Yes.")])
    done(run(archive, saved, factory, "why could you not check?", history=_history(QUESTION, first["answer"])))
    assert "lookup limit" in factory.sent()


def test_a_follow_up_with_a_narrower_scope_does_not_carry_a_message_outside_it(
    archive: Archive, saved: Names
) -> None:
    first = done(run(archive, saved, Factory([cite(FLAT_TIRE)])))
    assert first["citations"][0]["chatId"] != world.INES_PHONE
    narrow = world.INES_PHONE
    factory = Factory([say("Nothing in this one.")])
    history = _history(QUESTION, first["answer"])
    payload = done(run(archive, saved, factory, "what about here?", contact=narrow, history=history))
    assert FLAT_TIRE not in factory.sent()
    assert payload["citations"] == []


def test_an_import_that_changed_the_archive_drops_the_kept_chats(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ui_server

    chat_sessions.keep(tmp_path / "a.sqlite3", "bread?", "Yes.", _session_with("bread?", "Yes."))
    monkeypatch.setattr(ui_server, "changed_the_archive", lambda job_id, action: True)
    ui_server.forget_chat_sessions("job", "setup_import")
    assert chat_sessions.take(tmp_path / "a.sqlite3", _history("bread?", "Yes.")) is None


def test_a_job_that_did_not_change_the_archive_keeps_the_chats(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ui_server

    chat_sessions.keep(tmp_path / "a.sqlite3", "bread?", "Yes.", _session_with("bread?", "Yes."))
    monkeypatch.setattr(ui_server, "changed_the_archive", lambda job_id, action: False)
    ui_server.forget_chat_sessions("job", "meaning")
    assert chat_sessions.take(tmp_path / "a.sqlite3", _history("bread?", "Yes.")) is not None


def test_an_unknown_history_still_works_as_before(archive: Archive, saved: Names) -> None:
    factory = Factory([say("Yes.")])
    payload = done(run(archive, saved, factory, "and then?", history=_history("bread?", "It started.")))
    assert payload["answer"] == "Yes."
    assert "It started." in factory.sent()
