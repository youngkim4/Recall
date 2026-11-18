"""memory.py: the digest port of chat over `recall_services.digests`. The model is the fake of the report engine."""

from __future__ import annotations

from typing import Any

import pytest

from chat_engine.cancel import CancelToken
from recall_services import digests, memory, models, reports
from report_engine import LLMError
from report_engine.fake_llm import FakeCall, FakeLLM
from tests.recall_services.report_world import MODEL, PERIOD_LABELS, Desk
from tests.report_engine.synth import MINA

CHEAP = "gpt-6-luna"
# room for each part of the long conversation of the desk, for the tests that read all of it
WHOLE = 4
COST_KEYS = {"periods", "periods_cached", "usd"}
PERIOD_KEYS = {"label", "first_day", "last_day", "messages", "summary", "moments"}
MOMENT_KEYS = {"date", "title", "detail", "quote", "message_ids"}


def settings(desk: Desk, llm: Any = None, parts_limit: int | None = WHOLE) -> memory.DigestSettings:
    more = desk.more(llm)
    return memory.DigestSettings(
        model=MODEL,
        parts_limit=parts_limit,
        places=more["places"],
        options=more["options"],
        counter=more["counter"],
        llm_factory=more["llm_factory"],
        runtime=more["runtime"],
    )


def port(desk: Desk, llm: Any = None) -> memory.DigestPort:
    found = memory.ports(desk.archive, desk.names, digest=settings(desk, llm))["digests"]
    assert isinstance(found, memory.DigestPort)
    return found


def test_the_estimate_is_the_cost_of_the_parts_that_are_not_read_yet(desk: Desk) -> None:
    found = port(desk).estimate(MINA, since=None, until=None)
    told = digests.estimate(MINA, model=MODEL, **desk.estimate_more())
    assert set(found) == COST_KEYS
    assert (found["periods"], found["periods_cached"], found["usd"]) == (4, 0, told.usd) and found["usd"] > 0
    assert not desk.places.cache_path.exists()


def test_the_digest_has_the_shape_of_the_chat_port_and_ids_of_the_archive(desk: Desk) -> None:
    model = FakeLLM()
    found = port(desk, model).digest(MINA, since=None, until=None, cancel=CancelToken())
    assert set(found) == {"periods", "periods_failed", "usd"} and found["periods_failed"] == 0
    assert [period["label"] for period in found["periods"]] == PERIOD_LABELS
    for period in found["periods"]:
        assert set(period) == PERIOD_KEYS and isinstance(period["moments"], list)
        assert isinstance(period["messages"], int) and period["summary"]
        for moment in period["moments"]:
            assert set(moment) == MOMENT_KEYS and isinstance(moment["message_ids"], list)
            held = [desk.archive.message(message_id) for message_id in moment["message_ids"]]
            assert held and all(message is not None and message["conversation"] == MINA for message in held)
    assert len(model.map_calls) == 4 and model.reduce_calls == []
    after = port(desk).estimate(MINA, since=None, until=None)
    assert (after["periods"], after["periods_cached"], after["usd"]) == (4, 4, 0.0)


def test_the_days_go_through_to_the_estimate_and_to_the_digest(desk: Desk) -> None:
    model = FakeLLM()
    asked = port(desk, model)
    assert asked.estimate(MINA, since="2023-02-01", until="2023-02-28")["periods"] == 1
    found = asked.digest(MINA, since="2023-02-01", until="2023-02-28", cancel=CancelToken())
    assert [period["label"] for period in found["periods"]] == ["Feb 2023"]
    assert len(model.calls) == 1


def test_a_part_that_a_report_read_costs_the_digest_no_call(desk: Desk) -> None:
    reports.run(MINA, MODEL, **desk.more())
    model = FakeLLM()
    asked = port(desk, model)
    assert asked.estimate(MINA, since=None, until=None)["usd"] == 0.0
    assert len(asked.digest(MINA, since=None, until=None, cancel=CancelToken())["periods"]) == 4
    assert model.calls == []


def test_a_part_that_fails_is_counted(desk: Desk) -> None:
    def responder(call: FakeCall) -> Any:
        if call.first_day.startswith("2023-02"):
            raise LLMError("x (HTTP 500)", status_code=500)
        return None

    found = port(desk, FakeLLM(responder)).digest(MINA, since=None, until=None, cancel=CancelToken())
    assert found["periods_failed"] == 1 and len(found["periods"]) == 3


def test_the_cancel_of_chat_stops_the_digest_between_two_parts(desk: Desk) -> None:
    token = CancelToken()

    def responder(call: FakeCall) -> Any:
        if call.first_day.startswith("2023-02"):
            token.cancel()
        return None

    model = FakeLLM(responder)
    with pytest.raises(digests.ReportFailure) as caught:
        port(desk, model).digest(MINA, since=None, until=None, cancel=token)
    assert caught.value.code == "cancelled" and len(model.calls) == 2


def test_a_token_that_is_cancelled_before_the_digest_makes_no_call(desk: Desk) -> None:
    token = CancelToken()
    token.cancel()
    model = FakeLLM()
    with pytest.raises(digests.ReportFailure) as caught:
        port(desk, model).digest(MINA, since=None, until=None, cancel=token)
    assert caught.value.code == "cancelled" and model.calls == []


def test_the_token_of_chat_is_let_go_after_the_digest(desk: Desk) -> None:
    token = CancelToken()
    heard: list[str] = []
    on_cancel = token.on_cancel

    def noted(callback: Any) -> Any:
        forget = on_cancel(callback)
        return lambda: (heard.append("forgotten"), forget())[1]

    token.on_cancel = noted  # type: ignore[method-assign]
    port(desk).digest(MINA, since="2023-01-01", until="2023-01-31", cancel=token)
    assert heard == ["forgotten"]


def test_a_failure_of_the_digest_is_raised_for_the_line_of_the_engine(desk: Desk) -> None:
    def responder(call: FakeCall) -> Any:
        raise LLMError("x (HTTP 401)", status_code=401)

    asked = port(desk, FakeLLM(responder))
    with pytest.raises(digests.ReportFailure) as caught:
        asked.digest(MINA, since=None, until=None, cancel=CancelToken())
    assert caught.value.code == "key_rejected"
    with pytest.raises(ValueError):
        asked.estimate("", since=None, until=None)
    with pytest.raises(ValueError):
        asked.estimate(MINA, since="yesterday", until=None)


def test_the_digest_of_the_app_is_read_by_the_cheap_model_and_an_empty_name_is_the_model_of_the_reports(
    desk: Desk, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert memory.DIGEST_MODEL == CHEAP and CHEAP in models.offered_ids() and CHEAP != models.default_id()
    seen: list[str] = []
    model = FakeLLM()

    def factory(info: Any) -> FakeLLM:
        seen.append(info.id)
        return model

    more = desk.more()
    plain = memory.DigestSettings(
        places=more["places"], options=more["options"], counter=more["counter"], llm_factory=factory,
        runtime=more["runtime"], parts_limit=WHOLE,
    )  # fmt: skip
    asked = memory.ports(desk.archive, desk.names, digest=plain)["digests"]
    asked.digest(MINA, since="2023-01-01", until="2023-01-31", cancel=CancelToken())
    monkeypatch.setattr(memory, "DIGEST_MODEL", "")
    memory.ports(desk.archive, desk.names, digest=plain)["digests"].digest(
        MINA, since="2023-01-01", until="2023-01-31", cancel=CancelToken()
    )
    assert seen == [CHEAP, models.default_id()]


def test_the_digest_of_the_app_shares_the_notes_only_with_a_report_of_the_same_model(desk: Desk) -> None:
    """The notes of a part belong to one model: a report by another model saves a digest nothing."""
    more = desk.more()
    plain = memory.DigestSettings(places=more["places"], options=more["options"], counter=more["counter"])
    asked = memory.ports(desk.archive, desk.names, digest=plain)["digests"]
    reports.run(MINA, models.default_id(), **desk.more())
    after_another = asked.estimate(MINA, since=None, until=None)
    assert after_another["periods_cached"] == 0 and after_another["usd"] > 0
    reports.run(MINA, CHEAP, **desk.more())
    after_the_same = asked.estimate(MINA, since=None, until=None)
    assert (after_the_same["periods_cached"], after_the_same["usd"]) == (after_the_same["periods"], 0.0)


def test_notes_of_earlier_runs_that_can_not_be_opened_give_no_digest_port(
    desk: Desk, caplog: pytest.LogCaptureFixture
) -> None:
    """A digest would fail at its first call: the tool is not offered. A check makes no file."""
    assert isinstance(memory.ports(desk.archive, desk.names, digest=settings(desk))["digests"], memory.DigestPort)
    assert not desk.places.cache_path.exists()
    desk.places.cache_path.parent.mkdir(parents=True, exist_ok=True)
    desk.places.cache_path.write_bytes(b"not a database " * 64)
    assert memory.ports(desk.archive, desk.names, digest=settings(desk))["digests"] is None
    assert "StoreError" in caplog.text
