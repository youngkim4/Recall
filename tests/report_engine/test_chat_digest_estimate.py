"""The estimate of a digest: the map calls of a report, and never a model call."""

from __future__ import annotations

import dataclasses
from dataclasses import replace
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from report_engine.cache import PeriodCache
from report_engine.chat_digest import DigestEstimate
from report_engine.errors import ConfigError
from report_engine.estimate import NOTE_TOKENS, planned_calls
from report_engine.fake_llm import FakeLLM
from report_engine.options import ReportOptions
from report_engine.ports import LLMUsage
from report_engine.pricing import call_cost_usd

from .chat_digest_support import digest, guess
from .support import LARGE, SMALL_OPTIONS, run
from .synth import CONVERSATION, GROUP, ListSource, WordCounter, conversation, labeler, model_info

NOTHING = DigestEstimate(periods=0, periods_cached=0, input_tokens=0, output_tokens=0, usd=0.0)


def map_calls_of_a_report(messages: list, cache: PeriodCache | None = None) -> list:
    calls = planned_calls(
        ListSource(messages),
        CONVERSATION,
        None,
        model_info(),
        cache,
        labeler=labeler,
        options=SMALL_OPTIONS,
        counter=WordCounter(),
    )
    return [call for call in calls if call.kind == "map"]


def cost(calls: list) -> float:
    return sum(
        call_cost_usd(
            model_info(), LLMUsage(input_tokens=call.input_tokens, output_tokens=call.output_tokens)
        )
        for call in calls
    )


def test_the_estimate_is_the_map_calls_of_a_report() -> None:
    found = guess(LARGE)
    calls = map_calls_of_a_report(LARGE)
    assert isinstance(found, DigestEstimate)
    assert (found.periods, found.periods_cached) == (4, 0)
    assert found.input_tokens == sum(call.input_tokens for call in calls) > 0
    assert found.output_tokens == sum(call.output_tokens for call in calls) == 4 * 500
    assert found.usd == pytest.approx(cost(calls))
    assert isinstance(found.usd, float) and found.usd > 0


def test_the_estimate_can_not_be_changed() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        guess(LARGE).periods = 1  # type: ignore[misc]


def test_the_estimate_is_close_to_what_a_digest_really_sends() -> None:
    found = guess(LARGE)
    built = digest(LARGE)
    assert built.usage.input_tokens <= found.input_tokens <= built.usage.input_tokens * 1.4


def test_cached_periods_are_not_counted(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    run(LARGE[: 59 * 6], cache=cache)
    found = guess(LARGE, cache)
    calls = map_calls_of_a_report(LARGE, cache)
    assert (found.periods, found.periods_cached) == (4, 2)
    assert found.input_tokens == sum(call.input_tokens for call in calls if not call.cached)
    assert found.usd == pytest.approx(cost([call for call in calls if not call.cached]))
    digest(LARGE, cache=cache)
    assert guess(LARGE, cache) == dataclasses.replace(NOTHING, periods=4, periods_cached=4)


@pytest.mark.parametrize(
    ("span", "cached"),
    [
        ({}, 2),
        ({"since": "2023-02-15"}, 1),
        ({"until": "2023-03-15"}, 2),
        ({"since": "2023-04-01"}, 0),
    ],
)
def test_the_estimate_says_how_many_calls_the_digest_makes(
    cache_path: Path, span: dict[str, str], cached: int
) -> None:
    """January and February are in the cache. A span that starts in April has
    March before it, which is not: it is read under keys of its own."""
    cache = PeriodCache(cache_path)
    run(LARGE[: 59 * 6], cache=cache)
    found = guess(LARGE, cache, **span)
    llm = FakeLLM()
    built = digest(LARGE, llm=llm, cache=cache, **span)
    assert found.periods - found.periods_cached == built.periods_built == len(llm.calls)
    assert found.periods_cached == built.periods_cached == cached


def test_a_span_counts_the_periods_that_overlap_it(cache_path: Path) -> None:
    calls = map_calls_of_a_report(LARGE)[1:3]
    # with nothing in the cache the span sends no notes from before it:
    # none with its first period, and the notes of one period with its second
    cold = [replace(call, input_tokens=call.input_tokens - NOTE_TOKENS) for call in calls]
    found = guess(LARGE, since="2023-02-10", until="2023-03-05")
    assert (found.periods, found.periods_cached) == (2, 0)
    assert found.input_tokens == sum(call.input_tokens for call in cold)
    assert found.output_tokens == 2 * 500
    assert found.usd == pytest.approx(cost(cold))
    # with January in the cache the span is read as a report reads it
    cache = PeriodCache(cache_path)
    run(LARGE[: 31 * 6], cache=cache)
    warm = guess(LARGE, cache, since="2023-02-10", until="2023-03-05")
    assert warm.input_tokens == sum(call.input_tokens for call in calls)
    assert warm.usd == pytest.approx(cost(calls))


def test_nothing_to_read_is_an_estimate_of_zero() -> None:
    assert guess([]) == NOTHING
    assert guess(LARGE, since="2024-01-01") == NOTHING
    assert guess((), conversation_id=GROUP, source=ListSource(LARGE)) == NOTHING


@pytest.mark.parametrize(("since", "until"), [("2023-03-01", "2023-02-01"), ("March", None)])
def test_a_span_that_is_not_understood_is_refused(since: str | None, until: str | None) -> None:
    with pytest.raises(ConfigError):
        guess(LARGE, since=since, until=until)


def test_the_estimate_raises_the_config_error_that_the_digest_raises() -> None:
    tiny = dataclasses.replace(SMALL_OPTIONS, period_token_budget=1)
    with pytest.raises(ConfigError):
        guess(LARGE, options=tiny)
    with pytest.raises(ConfigError):
        digest(LARGE, options=tiny)


def test_the_estimate_writes_nothing_and_calls_no_model(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    run(LARGE, cache=cache)
    with closing(sqlite3.connect(cache_path)) as conn:
        before = list(conn.execute("SELECT * FROM period_results ORDER BY key"))
    guess(LARGE, cache)
    guess(LARGE, cache, since="2023-02-01")
    with closing(sqlite3.connect(cache_path)) as conn:
        assert list(conn.execute("SELECT * FROM period_results ORDER BY key")) == before


def test_a_cache_entry_that_does_not_parse_is_not_counted_as_cached(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    run(LARGE, cache=cache)
    with closing(sqlite3.connect(cache_path)) as conn, conn:
        conn.execute("UPDATE period_results SET payload = '{}' WHERE label = 'Feb 2023'")
    assert guess(LARGE, cache).periods_cached == 3


def test_the_cost_of_a_long_conversation_and_how_a_span_bounds_it() -> None:
    """Three years of dense chatter, 65,700 messages, with the default options:
    one call for each period of 80,000 tokens at most. The prices are the
    invented ones of `model_info`: 2 USD and 10 USD for one million tokens.
    Nothing is in the cache, so the two spans send no notes from before them."""
    long = conversation("2021-01-01", days=1_095, per_day=60, words=12)
    whole = guess(long, options=ReportOptions())
    year = guess(long, options=ReportOptions(), since="2023-01-01", until="2023-12-31")
    month = guess(long, options=ReportOptions(), since="2023-06-01", until="2023-06-30")
    assert len(long) == 65_700
    assert (whole.periods, whole.input_tokens, whole.output_tokens) == (18, 1_012_866, 72_000)
    assert (year.periods, year.input_tokens, year.output_tokens) == (6, 333_662, 24_000)
    assert (month.periods, month.input_tokens, month.output_tokens) == (1, 55_486, 4_000)
    assert [round(found.usd, 2) for found in (whole, year, month)] == [2.75, 0.91, 0.15]
    # a period holds 80,000 tokens at most, and the notes of earlier periods 4,000 at most
    assert all(found.input_tokens <= found.periods * 86_000 for found in (whole, year, month))
