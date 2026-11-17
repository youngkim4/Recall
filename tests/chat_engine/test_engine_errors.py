"""Failures are typed, reported and raised. Nothing falls back silently."""

import pytest

import chat_engine.local_answer as local_answer_module
from chat_engine.errors import (
    MODEL_ERRORS,
    ArchiveError,
    ChatEngineError,
    ModelError,
    RateLimitedError,
)
from chat_engine.fake_llm import call, say
from chat_engine.session import ChatSession
from tests.chat_engine.helpers import events_of, make_engine
from tests.chat_engine.tiny import tiny_archive


@pytest.fixture(autouse=True)
def local_answer_must_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the engine must never switch to the local answer by itself")

    monkeypatch.setattr(local_answer_module, "local_answer", forbidden)


def test_there_is_one_class_per_failure() -> None:
    codes = [error.code for error in MODEL_ERRORS]
    assert codes == ["no_key", "key_rejected", "unknown_model", "rate_limit", "offline", "timeout", "model_error"]
    assert all(issubclass(error, ChatEngineError) for error in MODEL_ERRORS)


@pytest.mark.parametrize("error_class", MODEL_ERRORS, ids=lambda error: error.code)
def test_each_error_type_is_raised_and_reported(error_class: type[ChatEngineError]) -> None:
    engine, fake, _ = make_engine([[error_class()]])
    events: list[dict] = []
    session = ChatSession()
    with pytest.raises(error_class) as caught:
        engine.ask("what about the lease?", on_event=events.append, session=session)
    assert caught.value.code == error_class.code
    assert events[-1] == {"type": "error", "code": error_class.code, "error": error_class.default_message}
    assert events_of("done", events) == []
    assert events_of("delta", events) == []
    assert len(fake.requests) == 1
    assert session.turns == ()


def test_error_after_some_text_is_still_raised() -> None:
    engine, _, _ = make_engine([[*say("The lease was sig"), RateLimitedError(retry_after=12.0)]])
    events: list[dict] = []
    with pytest.raises(RateLimitedError):
        engine.ask("lease?", on_event=events.append)
    assert events_of("delta", events)
    assert events[-1]["type"] == "error" and events[-1]["retry_after"] == 12.0
    assert events_of("done", events) == []


def test_error_in_a_later_step_is_raised() -> None:
    engine, fake, _ = make_engine([[call("search_messages", query="lease")], [ModelError()]])
    with pytest.raises(ModelError):
        engine.ask("lease?")
    assert len(fake.requests) == 2


def test_unexpected_failure_of_the_model_port_becomes_a_model_error() -> None:
    engine, _, _ = make_engine([[RuntimeError("socket closed by peer: sk-secret")]])
    events: list[dict] = []
    with pytest.raises(ModelError) as caught:
        engine.ask("lease?", on_event=events.append)
    assert "sk-secret" not in events[-1]["error"]
    assert "RuntimeError" in caught.value.detail


def test_answer_with_no_text_is_a_model_error() -> None:
    engine, _, _ = make_engine([[]])
    with pytest.raises(ModelError):
        engine.ask("lease?")


def test_stream_api_reports_the_error_as_its_last_event() -> None:
    engine, _, _ = make_engine([[RateLimitedError()]])
    events = list(engine.stream("lease?"))
    assert events[-1] == {
        "type": "error",
        "code": "rate_limit",
        "error": RateLimitedError.default_message,
    }


def test_archive_failure_is_typed() -> None:
    class BrokenArchive:
        def __init__(self) -> None:
            self._inner = tiny_archive()

        def summary(self) -> dict:
            raise OSError("database is locked")

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    engine, fake, _ = make_engine([say("unused")], archive=BrokenArchive())
    events: list[dict] = []
    with pytest.raises(ArchiveError) as caught:
        engine.ask("lease?", on_event=events.append)
    assert events[-1]["code"] == "archive_error"
    assert "database is locked" in caught.value.detail
    assert "database is locked" not in events[-1]["error"]
    assert fake.requests == []


def test_archive_failure_inside_a_tool_is_typed() -> None:
    class FailingSearch:
        def __init__(self) -> None:
            self._inner = tiny_archive()
            self.armed = False

        def search(self, query: str, **filters: object) -> dict:
            if self.armed and query == "boom":
                raise OSError("disk I/O error")
            return self._inner.search(query, **filters)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    archive = FailingSearch()
    engine, _, _ = make_engine([[call("search_messages", query="boom")], say("unused")], archive=archive)
    archive.armed = True
    with pytest.raises(ArchiveError):
        engine.ask("lease?")
