"""The calls of the profile model in a build: one profile, one reply, asked once more when its shape is wrong.

The calls run `PEOPLE_PARALLEL` at a time (`report_engine.run_in_parallel`).
A failure that each profile would have (`FATAL` of `people_words.py`) stops
the calls: no new one starts, and the calls in flight end and are kept. Any
other failure counts for its profile only. A cancel stops the calls too.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from recall_services import people_prompt
from recall_services.people_inputs import Planned
from recall_services.people_reply import ProfileText, parse_reply
from recall_services.people_words import FATAL, code_of
from report_engine import LLMError, Usage, run_in_parallel
from report_engine.model_info import ModelInfo
from report_engine.ports import LLM, LLMResult
from report_engine.pricing import call_cost_usd

TRIES = 2
BAD_REPLY = "bad_reply"
RUN_FAILED = "run_failed"


@dataclass(frozen=True)
class Written:
    """What one profile got: its text, or the code of its failure, and what its calls spent."""

    planned: Planned
    text: ProfileText | None
    usage: Usage
    code: str | None = None


def _usage(info: ModelInfo, result: LLMResult) -> Usage:
    spent = result.usage
    return Usage(
        input_tokens=spent.input_tokens,
        cached_input_tokens=spent.cached_input_tokens,
        output_tokens=spent.output_tokens,
        cost_usd=call_cost_usd(info, spent),
        calls=1,
    )


def ask(llm: LLM, info: ModelInfo, planned: Planned, redact: Callable[[str], str]) -> Written:
    """Ask for one profile. A reply in the wrong shape is asked once more, then the profile fails."""
    figures = planned.figures
    assert figures is not None and figures.first_day and figures.last_day
    usage = Usage()
    for _ in range(TRIES):
        try:
            result = llm.generate(
                list(planned.messages),
                schema=people_prompt.PROFILE_SCHEMA,
                max_output_tokens=people_prompt.MAX_OUTPUT_TOKENS,
                reasoning=people_prompt.PROFILE_REASONING,
            )
        except LLMError as error:
            return Written(planned, None, usage, code_of(error))
        usage = usage.plus(_usage(info, result))
        shaped = result.parsed if result.status == "completed" else None
        text = parse_reply(
            shaped, span=(figures.first_day, figures.last_day), is_group=planned.subject.is_group, redact=redact
        )
        if text is not None:
            return Written(planned, text, usage)
    return Written(planned, None, usage, BAD_REPLY)


class Calls:
    """The calls of one build. `stop_code` is the failure that stopped them, or "cancelled", or None."""

    def __init__(self, llm: LLM, info: ModelInfo, redact: Callable[[str], str], cancel: Any, heard: Callable[[], None]) -> None:
        self._llm = llm
        self._info = info
        self._redact = redact
        self._cancel = cancel
        self._heard = heard
        self._lock = threading.Lock()
        self._fatal: str | None = None

    def cancelled(self) -> bool:
        if callable(self._cancel):
            return bool(self._cancel())
        return bool(getattr(self._cancel, "cancelled", False))

    def stopped(self) -> bool:
        with self._lock:
            return self._fatal is not None or self.cancelled()

    @property
    def stop_code(self) -> str | None:
        with self._lock:
            if self._fatal is not None:
                return self._fatal
        return "cancelled" if self.cancelled() else None

    def _task(self, planned: Planned) -> Callable[[], Written]:
        def work() -> Written:
            try:
                written = ask(self._llm, self._info, planned, self._redact)
            except Exception:
                with self._lock:
                    self._fatal = self._fatal or RUN_FAILED
                raise
            if written.code in FATAL:
                with self._lock:
                    self._fatal = self._fatal or written.code
            self._heard()
            return written

        return work

    def run(self, planned: Sequence[Planned], parallel: int) -> tuple[tuple[Written, ...], Exception | None]:
        """Ask for each profile. Gives what came back, and the first error of no known kind, or None.

        A profile that a stop kept from starting is not in the result.
        """
        if not planned or self.cancelled():
            return (), None
        outcomes = run_in_parallel([self._task(one) for one in planned], max_parallel=parallel, stop=self.stopped)
        unplanned = next((outcome.error for outcome in outcomes if outcome.error is not None), None)
        return tuple(outcome.value for outcome in outcomes if outcome.value is not None), unplanned
