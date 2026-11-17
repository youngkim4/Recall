"""Helpers that run the pipeline with a fake model, a fake clock and fixed ids."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from report_engine.control import Runtime
from report_engine.fake_llm import FakeLLM
from report_engine.model_info import ModelInfo
from report_engine.models import Report
from report_engine.options import ReportOptions
from report_engine.pipeline import run_report
from report_engine.ports import SourceMessage

from .synth import CONVERSATION, ListSource, WordCounter, conversation, labeler, model_info

CREATED = datetime(2024, 7, 1, 10, 15, 0, tzinfo=UTC)

SMALL_OPTIONS = ReportOptions(
    period_token_budget=2_500,
    map_max_output_tokens=500,
    reduce_max_output_tokens=800,
    carry_over_token_budget=300,
)

# ten days in one month: one period
SMALL = conversation("2024-06-01", days=10, per_day=4)
# four full months of about 2,100 word-tokens each: four periods under SMALL_OPTIONS
LARGE = conversation("2023-01-01", days=120, per_day=6, words=8)
LARGE_LABELS = ["Jan 2023", "Feb 2023", "Mar 2023", "Apr 2023"]


class FakeTime:
    """A clock that moves only when something sleeps or a test moves it."""

    def __init__(self, start: float = 5_000.0) -> None:
        self.monotonic = start
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.monotonic

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.monotonic += seconds


def runtime(time: FakeTime | None = None) -> Runtime:
    ticks = time or FakeTime()
    numbers = itertools.count(1)
    return Runtime(
        clock=ticks.clock,
        sleep=ticks.sleep,
        now=lambda: CREATED,
        new_id=lambda created: f"{created:%Y%m%dT%H%M%SZ}-{next(numbers):06x}",
    )


def run(
    messages: Sequence[SourceMessage],
    *,
    llm: Any = None,
    info: ModelInfo | None = None,
    options: ReportOptions = SMALL_OPTIONS,
    time: FakeTime | None = None,
    conversation_id: str = CONVERSATION,
    **kwargs: Any,
) -> Report:
    return run_report(
        source=kwargs.pop("source", None) or ListSource(messages, conversation_id),
        conversation=conversation_id,
        llm=llm if llm is not None else FakeLLM(),
        model_info=info or model_info(),
        labeler=kwargs.pop("labeler", labeler),
        options=options,
        counter=WordCounter(),
        runtime=runtime(time),
        **kwargs,
    )
