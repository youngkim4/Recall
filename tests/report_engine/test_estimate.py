"""The estimate counts only what a run would send, and never calls a model."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from report_engine.cache import PeriodCache
from report_engine.estimate import estimate, planned_calls
from report_engine.fake_llm import FakeLLM
from report_engine.model_info import LongContextRule
from report_engine.models import Scope
from report_engine.ports import LLMUsage
from report_engine.pricing import call_cost_usd

from .support import LARGE, SMALL, SMALL_OPTIONS, run
from .synth import CONVERSATION, ListSource, WordCounter, conversation, labeler, model_info

KEYS = {"input_tokens", "cached_tokens", "output_tokens", "cost_usd", "periods", "periods_cached"}


def guess(messages: list, cache: Any = None, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "labeler": labeler,
        "options": SMALL_OPTIONS,
        "counter": WordCounter(),
    }
    values.update(overrides)
    info = values.pop("info", None) or model_info()
    scope = values.pop("scope", None)
    return estimate(ListSource(messages), CONVERSATION, scope, info, cache, **values)


def test_the_estimate_has_exactly_the_six_keys() -> None:
    result = guess(SMALL)
    assert set(result) == KEYS
    assert all(isinstance(result[key], int) for key in KEYS - {"cost_usd"})
    assert isinstance(result["cost_usd"], float)


def test_with_no_cache_every_period_is_counted() -> None:
    result = guess(LARGE)
    assert (result["periods"], result["periods_cached"], result["cached_tokens"]) == (4, 0, 0)
    assert result["output_tokens"] == 4 * 500 + 800


def test_the_estimate_is_close_to_what_a_run_really_sends() -> None:
    result = guess(LARGE)
    report = run(LARGE, llm=FakeLLM())
    assert report.usage.input_tokens <= result["input_tokens"] <= report.usage.input_tokens * 1.4


def test_cached_periods_are_not_counted(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    before = guess(LARGE, cache)
    run(LARGE, cache=cache)
    after = guess(LARGE, cache)
    assert (after["periods"], after["periods_cached"]) == (4, 4)
    assert after["cached_tokens"] > 8_000
    assert after["input_tokens"] + after["cached_tokens"] == before["input_tokens"]
    assert after["output_tokens"] == 800
    assert after["cost_usd"] < before["cost_usd"] / 3


def test_after_new_messages_only_the_changed_period_is_counted(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    run(LARGE, cache=cache)
    grown = [*LARGE, *conversation("2023-05-01", days=2, per_day=3, first_id=50_000)]
    result = guess(grown, cache)
    assert (result["periods"], result["periods_cached"]) == (4, 3)
    assert result["output_tokens"] == 500 + 800
    fresh = [
        call for call in planned_calls_for(grown, cache) if call.kind == "map" and not call.cached
    ]
    assert [call.label for call in fresh] == ["Apr 2023 – May 2023"]


def planned_calls_for(messages: list, cache: Any) -> Any:
    return planned_calls(
        ListSource(messages),
        CONVERSATION,
        None,
        model_info(),
        cache,
        labeler=labeler,
        options=SMALL_OPTIONS,
        counter=WordCounter(),
    )


def test_the_cost_is_the_sum_of_the_calls_at_model_prices(cache_path: Path) -> None:
    info = model_info()
    calls = [call for call in planned_calls_for(LARGE, None) if not call.cached]
    expected = sum(
        call_cost_usd(
            info, LLMUsage(input_tokens=call.input_tokens, output_tokens=call.output_tokens)
        )
        for call in calls
    )
    assert [call.kind for call in calls] == ["map", "map", "map", "map", "reduce"]
    assert guess(LARGE)["cost_usd"] == pytest.approx(expected)
    assert guess(LARGE)["input_tokens"] == sum(call.input_tokens for call in calls)


def test_the_long_context_rule_is_judged_per_call() -> None:
    rule = LongContextRule(threshold_tokens=2_450, input_multiplier=2.0, output_multiplier=1.5)
    flat = guess(LARGE, info=model_info(long_context=None))["cost_usd"]
    ruled = guess(LARGE, info=model_info(long_context=rule))["cost_usd"]
    assert ruled > flat
    info = model_info(long_context=rule)
    calls = planned_calls_for(LARGE, None)
    over = [call for call in calls if call.input_tokens > 2_450]
    assert 0 < len(over) < len(calls)
    expected = sum(
        call_cost_usd(
            info, LLMUsage(input_tokens=call.input_tokens, output_tokens=call.output_tokens)
        )
        for call in calls
    )
    assert ruled == pytest.approx(expected)


def test_a_margin_is_added_when_the_tokenizer_family_is_not_the_models() -> None:
    exact = guess(SMALL, info=model_info(tokenizer_family="words"))
    padded = guess(SMALL, info=model_info(tokenizer_family="some-other-family"))
    assert padded["periods"] == exact["periods"] == 1
    ratio = padded["input_tokens"] / exact["input_tokens"]
    assert 1.3 < ratio < 1.4


def test_the_margin_can_be_set() -> None:
    options = replace(SMALL_OPTIONS, tokenizer_mismatch_margin=2.0)
    exact = guess(SMALL)
    padded = guess(SMALL, info=model_info(tokenizer_family="other"), options=options)
    assert padded["input_tokens"] / exact["input_tokens"] == pytest.approx(2.0, rel=0.02)


def test_an_empty_scope_costs_nothing() -> None:
    assert guess(SMALL, scope=Scope(since="2031-01-01")) == {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "periods": 0,
        "periods_cached": 0,
    }


def test_the_scope_limits_what_is_counted() -> None:
    whole = guess(SMALL)
    part = guess(SMALL, scope={"since": "2024-06-03", "until": "2024-06-05"})
    assert part["input_tokens"] < whole["input_tokens"]
    assert part["periods"] == 1


def test_an_estimate_does_not_mark_cache_entries_as_used(cache_path: Path) -> None:
    now = [datetime(2024, 1, 1, tzinfo=UTC)]
    cache = PeriodCache(cache_path, now=lambda: now[0])
    run(LARGE, cache=cache)
    now[0] += timedelta(days=100)
    assert guess(LARGE, cache)["periods_cached"] == 4
    assert cache.prune(unused_for_days=30) == 4


def test_other_options_mean_other_periods_and_no_cache_hits(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    run(LARGE, cache=cache)
    result = guess(LARGE, cache, options=replace(SMALL_OPTIONS, period_token_budget=5_000))
    assert result["periods"] == 2
    assert result["periods_cached"] == 0


def test_a_cache_that_can_not_be_read_counts_as_empty(caplog: pytest.LogCaptureFixture) -> None:
    from report_engine.errors import StoreError

    class Broken:
        def get_many(self, keys: Any, touch: bool = False) -> dict:
            raise StoreError("disk is not there")

    with caplog.at_level("WARNING"):
        result = guess(LARGE, Broken())
    assert result["periods_cached"] == 0
    assert "cache" in caplog.text


def test_the_labeler_is_required() -> None:
    with pytest.raises(TypeError):
        estimate(ListSource(SMALL), CONVERSATION, None, model_info(), None)  # type: ignore[call-arg]
