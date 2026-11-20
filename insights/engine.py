"""The insight engine: every insight of every thread, and the digest.

`run` counts the rhythm and date insights of each thread with no model. With
a model, it also reads each thread that is worth a call (`reading.worth_reading`),
`parallel` at a time. A call that fails is counted, and the other threads go on.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

from insights import dates, digest, reading, rhythm
from insights.thread import Insight, Thread
from report_engine.ports import LLM, LLMUsage

logger = logging.getLogger(__name__)

PARALLEL = 4
COUNTED: tuple[Callable[[Thread, datetime], Insight | None], ...] = (
    rhythm.drift,
    rhythm.balance,
    dates.unanswered,
    dates.birthday,
)


@dataclass(frozen=True)
class Report:
    """Every insight, the digest, and what the model calls cost."""

    insights: tuple[Insight, ...]
    picks: tuple[digest.Pick, ...]
    calls: int = 0
    failed: int = 0
    dropped: int = 0
    usage: tuple[LLMUsage, ...] = field(default=())


def counted(thread: Thread, now: datetime) -> tuple[Insight, ...]:
    """The insights of one thread that need no model."""
    return tuple(found for finder in COUNTED if (found := finder(thread, now)) is not None)


def _read(llm: LLM, thread: Thread, now: datetime, reasoning: str | None) -> reading.Reading | None:
    try:
        return reading.read(llm, thread, now, reasoning=reasoning)
    except Exception as error:  # noqa: BLE001 - one thread that fails leaves the others
        logger.warning("an insight reading failed (%s)", type(error).__name__)
        return None


def run(
    threads: Sequence[Thread],
    now: datetime,
    llm: LLM | None = None,
    *,
    reasoning: str | None = None,
    parallel: int = PARALLEL,
) -> Report:
    found: list[Insight] = [item for thread in threads for item in counted(thread, now)]
    to_read = [thread for thread in threads if llm is not None and reading.worth_reading(thread, now)]
    readings: list[reading.Reading | None] = []
    if llm is not None and to_read:
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            readings = list(pool.map(lambda thread: _read(llm, thread, now, reasoning), to_read))
    done = [item for item in readings if item is not None]
    for item in done:
        found.extend(item.insights)
    ordered = tuple(sorted(found, key=lambda item: (item.thread, item.kind, item.when)))
    return Report(
        insights=ordered,
        picks=digest.picks(ordered, now),
        calls=len(to_read),
        failed=len(readings) - len(done),
        dropped=sum(item.dropped for item in done),
        usage=tuple(item.usage for item in done),
    )
