"""reports.estimate: what a run would send and cost, with the keys that the app reads."""

from __future__ import annotations

import json

import pytest

from recall_services import models, reports
from report_engine import PeriodCache, estimate
from report_engine.archive_source import ArchiveSource
from report_engine.model_info import LongContextRule
from tests.recall_services.report_world import MINA_MESSAGES, MODEL, OPTIONS, THEO_MESSAGES, Desk
from tests.report_engine.synth import MINA, THEO, WordCounter

# Estimate in app/src/types.ts at 89ab504
ESTIMATE_KEYS = {
    "msg_count",
    "input_tokens",
    "output_tokens",
    "estimated_cost",
    "needs_chunking",
    "token_budget",
    "long_context_pricing",
    "years",
}
MORE_KEYS = {"periods", "periods_cached"}


def test_the_estimate_has_the_keys_the_app_reads(desk: Desk) -> None:
    found = reports.estimate(MINA, MODEL, **desk.estimate_more())
    assert set(found) == ESTIMATE_KEYS | MORE_KEYS
    json.dumps(found)


def test_the_numbers_are_the_ones_of_the_report_engine(desk: Desk) -> None:
    found = reports.estimate(MINA, MODEL, **desk.estimate_more())
    engine = estimate(
        ArchiveSource(desk.archive),
        MINA,
        None,
        models.model_info(MODEL),
        PeriodCache(desk.places.cache_path),
        labeler=desk.names.report_labeler(desk.archive.conversation(MINA)),
        options=OPTIONS,
        counter=WordCounter(),
    )
    assert found["input_tokens"] == engine["input_tokens"] > 0
    assert found["output_tokens"] == engine["output_tokens"] > 0
    assert found["estimated_cost"] == pytest.approx(engine["cost_usd"])
    assert (found["periods"], found["periods_cached"]) == (engine["periods"], engine["periods_cached"]) == (4, 0)


def test_the_estimate_keeps_the_safety_margin_of_a_tokenizer_that_is_not_confirmed(desk: Desk) -> None:
    """The family of the 6 series is an assumption, so every count is 1.35 times the count of the tokenizer."""
    found = reports.estimate(THEO, MODEL, **desk.estimate_more())
    sure = models.model_info(MODEL).model_copy(update={"tokenizer_family": WordCounter.family})
    exact = estimate(
        ArchiveSource(desk.archive),
        THEO,
        None,
        sure,
        None,
        labeler=desk.names.report_labeler(desk.archive.conversation(THEO)),
        options=OPTIONS,
        counter=WordCounter(),
    )
    assert found["input_tokens"] > exact["input_tokens"] * 1.2


def test_the_figures_beside_the_numbers(desk: Desk) -> None:
    found = reports.estimate(MINA, MODEL, **desk.estimate_more())
    assert found["msg_count"] == MINA_MESSAGES
    # the conversation is read in more than one call
    assert found["needs_chunking"] is True
    assert found["years"] == [2023]
    assert found["token_budget"] == OPTIONS.period_token_budget
    assert found["long_context_pricing"] is False


def test_a_conversation_that_one_call_reads(desk: Desk) -> None:
    found = reports.estimate(THEO, MODEL, **desk.estimate_more())
    assert (found["msg_count"], found["periods"], found["needs_chunking"], found["years"]) == (THEO_MESSAGES, 1, False, [])


def test_the_price_is_the_price_of_the_model_that_was_named(desk: Desk) -> None:
    low = reports.estimate(MINA, "gpt-6-luna", **desk.estimate_more())
    high = reports.estimate(MINA, "gpt-6.1-sol", **desk.estimate_more())
    assert (low["input_tokens"], low["output_tokens"]) == (high["input_tokens"], high["output_tokens"])
    assert high["estimated_cost"] == pytest.approx(low["estimated_cost"] * 20)
    expected = (high["input_tokens"] * 2.00 + high["output_tokens"] * 10.00) / 1_000_000
    assert high["estimated_cost"] == pytest.approx(expected)


@pytest.mark.parametrize("nothing", ["", None])
def test_no_model_named_is_the_default_model(desk: Desk, nothing: object) -> None:
    named = reports.estimate(MINA, models.default_id(), **desk.estimate_more())
    assert reports.estimate(MINA, nothing, **desk.estimate_more()) == named  # type: ignore[arg-type]


def test_a_model_that_can_not_be_run_is_refused_and_not_priced(desk: Desk) -> None:
    with pytest.raises(models.UnknownModelIdError):
        reports.estimate(MINA, "gpt-7", **desk.estimate_more())
    with pytest.raises(models.ModelNotOfferedError):
        reports.estimate(MINA, "gpt-5.5", **desk.estimate_more())


def test_the_dates_narrow_the_estimate(desk: Desk) -> None:
    whole = reports.estimate(MINA, MODEL, **desk.estimate_more())
    january = reports.estimate(MINA, MODEL, "2023-01-01", "2023-01-31", **desk.estimate_more())
    assert january["periods"] == 1 and january["needs_chunking"] is False
    assert january["msg_count"] == desk.archive.count(conversations=[MINA], since="2023-01-01", until="2023-01-31")
    assert 0 < january["input_tokens"] < whole["input_tokens"]
    from_march = reports.estimate(MINA, MODEL, "2023-03-01", "", **desk.estimate_more())
    assert from_march["periods"] == 2


def test_days_with_no_message_cost_nothing(desk: Desk) -> None:
    found = reports.estimate(MINA, MODEL, "2020-01-01", "2020-12-31", **desk.estimate_more())
    assert found == {
        "msg_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
        "needs_chunking": False,
        "token_budget": OPTIONS.period_token_budget,
        "long_context_pricing": False,
        "years": [],
        "periods": 0,
        "periods_cached": 0,
    }


def test_an_unknown_conversation_costs_nothing(desk: Desk) -> None:
    found = reports.estimate("+15550000000", MODEL, **desk.estimate_more())
    assert (found["msg_count"], found["input_tokens"], found["periods"]) == (0, 0, 0)


@pytest.mark.parametrize("contact", ["", None, "  "])
def test_a_conversation_is_needed(desk: Desk, contact: object) -> None:
    with pytest.raises(ValueError, match="A conversation is needed"):
        reports.estimate(contact, MODEL, **desk.estimate_more())  # type: ignore[arg-type]


@pytest.mark.parametrize(("since", "until"), [("2023-1-1", None), (None, "tomorrow"), ("2023-02-30", None), ("2023-03-01", "2023-01-01")])
def test_dates_that_are_no_dates_are_refused(desk: Desk, since: str | None, until: str | None) -> None:
    with pytest.raises(ValueError):
        reports.estimate(MINA, MODEL, since, until, **desk.estimate_more())


def test_after_a_run_the_periods_are_not_paid_for_again(desk: Desk) -> None:
    before = reports.estimate(MINA, MODEL, **desk.estimate_more())
    reports.run(MINA, MODEL, **desk.more())
    after = reports.estimate(MINA, MODEL, **desk.estimate_more())
    assert (after["periods"], after["periods_cached"]) == (4, 4)
    assert 0 < after["input_tokens"] < before["input_tokens"]
    assert after["estimated_cost"] < before["estimated_cost"]


def test_a_call_above_the_long_context_size_is_said(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    info = models.model_info(MODEL)
    low_rule = LongContextRule(threshold_tokens=1_000, input_multiplier=2.0, output_multiplier=1.5)
    monkeypatch.setattr(reports.models, "model_info", lambda _id: info.model_copy(update={"long_context": low_rule}))
    found = reports.estimate(MINA, MODEL, **desk.estimate_more())
    assert found["long_context_pricing"] is True
    monkeypatch.setattr(reports.models, "model_info", lambda _id: info.model_copy(update={"long_context": None}))
    assert reports.estimate(MINA, MODEL, **desk.estimate_more())["long_context_pricing"] is False


def test_the_default_places_are_in_the_data_folder() -> None:
    import recall_paths

    places = reports.default_places()
    assert places.out_dir == recall_paths.OUT_DIR
    assert places.store_dir == recall_paths.SAVES_DIR / "reports"
    assert places.cache_path == recall_paths.SAVES_DIR / "report-cache.sqlite3"
