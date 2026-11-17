"""The one place a run talks to the model.

Every request passes here: the cancel token is checked, the tokens-per-minute
budget is honoured, and the real usage of the reply is counted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from report_engine.control import CancelToken, ProgressEvent, Runtime
from report_engine.errors import LLMError, ReportCancelled
from report_engine.model_info import ModelInfo
from report_engine.models import Usage
from report_engine.ports import LLM, ChatMessage, LLMResult, LLMUsage
from report_engine.pricing import call_cost_usd
from report_engine.ratelimit import TokenBudget

MAX_SLEEP_SLICE_SECONDS = 1.0

Emit = Callable[[ProgressEvent], None]


def usage_of(model_info: ModelInfo, usage: LLMUsage) -> Usage:
    """One call's real usage with its real cost."""
    input_tokens = max(0, int(usage.input_tokens))
    return Usage(
        input_tokens=input_tokens,
        cached_input_tokens=min(max(0, int(usage.cached_input_tokens)), input_tokens),
        output_tokens=max(0, int(usage.output_tokens)),
        cost_usd=call_cost_usd(model_info, usage),
        calls=1,
    )


class Caller:
    """Sends the requests of one run and keeps the run's usage."""

    def __init__(
        self,
        *,
        llm: LLM,
        model_info: ModelInfo,
        runtime: Runtime,
        cancel: CancelToken | None,
        emit: Emit,
        reasoning: str | None,
    ) -> None:
        self._llm = llm
        self._model_info = model_info
        self._runtime = runtime
        self._cancel = cancel
        self._emit = emit
        self._reasoning = reasoning
        self._budget = TokenBudget(model_info.tokens_per_minute, runtime.clock)
        self._usage = Usage()

    @property
    def usage(self) -> Usage:
        return self._usage

    def check_cancel(self) -> None:
        if self._cancel is not None and self._cancelled():
            raise ReportCancelled("the run was cancelled", usage=self._usage)

    def _cancelled(self) -> bool:
        return self._cancel is not None and self._cancel.cancelled

    def _wait_for_budget(self, tokens: int, waiting: ProgressEvent) -> None:
        announced = False
        while True:
            delay = self._budget.delay(tokens)
            if delay <= 0:
                return
            if not announced:
                self._emit(waiting)
                announced = True
            self._runtime.sleep(min(delay, MAX_SLEEP_SLICE_SECONDS))
            self.check_cancel()

    def send(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: Mapping[str, Any],
        max_output_tokens: int,
        input_estimate: int,
        waiting: ProgressEvent,
        starting: ProgressEvent,
    ) -> LLMResult:
        """One request. Raises ReportCancelled before sending, LLMError when the port fails."""
        self.check_cancel()
        reserved = input_estimate + max_output_tokens
        self._wait_for_budget(reserved, waiting)
        self._emit(starting)
        ticket = self._budget.spend(reserved)
        try:
            result = self._llm.generate(
                messages,
                schema=schema,
                max_output_tokens=max_output_tokens,
                reasoning=self._reasoning,
            )
        except LLMError as exc:
            # the request went out, maybe more than once; keep its room taken
            self._budget.settle(ticket, reserved * max(1, exc.attempts))
            raise
        spent = usage_of(self._model_info, result.usage)
        extra_attempts = max(1, result.attempts) - 1
        self._budget.settle(
            ticket, spent.input_tokens + spent.output_tokens + reserved * extra_attempts
        )
        self._usage = self._usage.plus(spent)
        return result
