"""Edges: bad options, bad rows, full disks, odd headers."""

import email.utils
import re
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import openai
import pandas as pd
import pytest

from report_engine.cache import CacheEntry, PeriodCache
from report_engine.control import Runtime, new_report_id
from report_engine.dates import parse_stamp
from report_engine.digest import parse_digest
from report_engine.errors import ConfigError, SourceError, StoreError
from report_engine.fake_llm import FakeLLM, default_reply
from report_engine.frame_source import FrameSource
from report_engine.jsonsafe import jsonable
from report_engine.openai_llm import OpenAIResponsesLLM
from report_engine.options import ReportOptions
from report_engine.ports import LLMUsage
from report_engine.prompts import MAP_SCHEMA, map_messages
from report_engine.ratelimit import TokenBudget
from report_engine.selection import select_moments
from report_engine.store import ReportStore, write_text_atomic
from report_engine.tokens import TiktokenCounter
from report_engine.transcript import build_lines
from report_engine.verify import MessageIndex

from .support import SMALL, SMALL_OPTIONS, run
from .synth import CONVERSATION, labeler, message, model_info
from .test_models import report as make_report
from .test_openai_llm import MESSAGES, REQUEST, Client, adapter, reply
from .test_selection import moment


@pytest.mark.parametrize(
    "change",
    [
        {"period_token_budget": 0},
        {"map_max_output_tokens": -1},
        {"reduce_max_output_tokens": 0},
        {"max_message_chars": 0},
        {"max_moments": -1},
        {"carry_over_token_budget": -1},
        {"map_output_estimate_tokens": -1},
        {"reduce_output_estimate_tokens": -1},
        {"score_floor": 1.5},
        {"score_floor": -0.1},
        {"tokenizer_mismatch_margin": 0.5},
        {"tokenizer_mismatch_margin": float("nan")},
    ],
)
def test_options_refuse_values_that_make_no_sense(change: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        replace(ReportOptions(), **change)


def test_the_default_period_budget_is_chosen_for_reading_quality() -> None:
    assert ReportOptions().period_token_budget == 80_000


def test_the_default_runtime_gives_utc_time_and_ids_that_sort_by_time() -> None:
    runtime = Runtime()
    now = runtime.now()
    assert now.tzinfo is not None and now.utcoffset().total_seconds() == 0  # type: ignore[union-attr]
    early = new_report_id(datetime(2024, 7, 1, 10, 15, 0, tzinfo=UTC))
    late = new_report_id(datetime(2024, 7, 1, 10, 15, 1, tzinfo=UTC))
    assert re.fullmatch(r"20240701T101500Z-[0-9a-f]{6}", early)
    assert early < late
    assert new_report_id(now) != new_report_id(now)
    assert runtime.clock() <= runtime.clock()


def test_a_message_that_lacks_a_field_is_a_source_error() -> None:
    broken = dict(message(1, "2024-06-01 09:00:00", "hi"))
    del broken["is_from_me"]
    with pytest.raises(SourceError) as caught:
        build_lines([broken], labeler)
    assert "is_from_me" in str(caught.value)


def test_a_period_too_large_for_the_model_is_not_sent() -> None:
    messages = [
        message(1, "2024-06-01 09:00:00", "small start"),
        message(2, "2024-06-01 09:01:00", "word " * 600),
        message(3, "2024-06-01 09:02:00", "small end"),
    ]
    llm = FakeLLM()
    report = run(
        messages,
        llm=llm,
        info=model_info(max_input_tokens=700, context_window=10_000, max_output_tokens=2_000),
    )
    assert [period.label for period in report.periods] == [
        "2024-06-01 part 1",
        "2024-06-01 part 2",
        "2024-06-01 part 3",
    ]
    assert len(llm.map_calls) == 2
    assert all("word word word" not in call.text for call in llm.calls)
    assert report.status == "partial"
    (warning,) = report.warnings
    assert warning.startswith(
        'Period "2024-06-01 part 2" was not analysed: the period is too large for the model'
    )
    assert report.periods[1].usage.calls == 0


def test_retry_after_can_be_a_date() -> None:
    later = email.utils.formatdate(time.time() + 20, usegmt=True)
    limited = openai.RateLimitError(
        "slow down",
        response=httpx.Response(429, headers={"retry-after": later}, request=REQUEST),
        body=None,
    )
    llm, slept = adapter(Client(limited, reply()))
    llm.generate(MESSAGES, max_output_tokens=10)
    assert len(slept) == 1 and 17.0 < slept[0] <= 20.0


def test_a_retry_after_that_can_not_be_read_falls_back_to_backoff() -> None:
    limited = openai.RateLimitError(
        "slow down",
        response=httpx.Response(429, headers={"retry-after": "soon"}, request=REQUEST),
        body=None,
    )
    llm, slept = adapter(Client(limited, reply()))
    llm.generate(MESSAGES, max_output_tokens=10)
    assert slept == [2.0]


def test_the_adapter_refuses_a_negative_retry_count() -> None:
    with pytest.raises(ValueError):
        OpenAIResponsesLLM(Client(), "test-model-1", max_retries=-1)


def test_zero_retries_means_one_attempt() -> None:
    client = Client(openai.APITimeoutError(REQUEST), reply())
    llm, slept = adapter(client, max_retries=0)
    with pytest.raises(Exception) as caught:
        llm.generate(MESSAGES, max_output_tokens=10)
    assert getattr(caught.value, "attempts", None) == 1
    assert slept == []


def test_the_cache_refuses_a_result_that_is_not_json(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    entry = CacheEntry(key="k", payload={"summary": object()}, usage=LLMUsage(), cost_usd=0.0)
    with pytest.raises(StoreError):
        cache.put(entry, conversation="c", label="l", model="m", prompt_version="v")
    assert cache.count() == 0


def test_a_cache_or_store_under_a_file_is_a_store_error(tmp_path: Path) -> None:
    blocker = tmp_path / "a-file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(StoreError):
        PeriodCache(blocker / "sub" / "cache.sqlite")
    with pytest.raises(StoreError):
        ReportStore(blocker / "reports").save(make_report())
    with pytest.raises(StoreError):
        write_text_atomic(blocker / "out" / "analysis.md", "text")


def test_frame_rows_with_odd_ids_and_flags_still_come_through() -> None:
    frame = pd.DataFrame(
        {
            "message_id": [None, "guid-2", 3],
            "timestamp": ["2024-06-01 09:00:00", "2024-06-01 09:01:00", "2024-06-01 09:02:00"],
            "sender": ["a", "a", "a"],
            "text": ["x", "y", "z"],
            "chat_id": [CONVERSATION] * 3,
            "is_from_me": [None, object(), "YES"],
        }
    )
    found = list(FrameSource(frame).messages(CONVERSATION, None, None))
    assert [(row["id"], row["is_from_me"]) for row in found] == [
        ("row-0", False),
        ("guid-2", False),
        ("3", True),
    ]


def test_jsonable_falls_back_to_text_for_a_value_it_can_not_unwrap() -> None:
    class Odd:
        def item(self) -> Any:
            raise TypeError("no single value")

        def __str__(self) -> str:
            return "odd"

    assert jsonable({"x": Odd()}) == {"x": "odd"}


def test_small_helpers_at_their_edges() -> None:
    with pytest.raises(ValueError):
        parse_stamp("2024-06-01")
    assert parse_digest({"summary": "s", "moments": [{"refs": 7}]}).moments[0].refs == ()
    assert TiktokenCounter(object(), "x").count_many([]) == []
    with pytest.raises(ValueError):
        TokenBudget(100, lambda: 0.0).delay(-1)
    index = MessageIndex(build_lines([message(1, "2024-06-01 09:00:00", "hi")], labeler))
    assert index.has_day("2024-06-01") and not index.has_day("2024-06-02")
    assert list(index.near("not a day", 3)) == []


def test_moments_with_empty_titles_are_never_taken_for_duplicates() -> None:
    first = moment("2024-01-10", "!!!", 0.9)
    second = moment("2024-01-10", "???", 0.8)
    assert len(select_moments([[first], [second]], score_floor=0.4, max_moments=10)) == 2


def test_the_fake_answers_an_empty_transcript_with_no_moment() -> None:
    llm = FakeLLM()
    result = llm.generate(map_messages("", (), ()), schema=MAP_SCHEMA, max_output_tokens=10)
    assert result.parsed == {
        "summary": "Fake summary of an empty period.",
        "carry_over": "",
        "moments": [],
    }
    assert default_reply(llm.calls[0]) == result.parsed


def test_a_run_with_default_options_and_a_plain_counter_works() -> None:
    report = run(SMALL, options=ReportOptions())
    assert report.status == "complete"
    assert len(report.periods) == 1
    assert ReportOptions() != SMALL_OPTIONS
