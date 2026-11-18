"""The one place a run talks to the model.

Every request passes here: the cancel token is checked, the tokens-per-minute
budget is honoured, and the real usage of the reply is counted.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from report_engine.call_pool import CallPool
from report_engine.control import CancelToken, ProgressEvent, Runtime
from report_engine.errors import CallLimitReached, LLMError, ReportCancelled
from report_engine.model_info import ModelInfo
from report_engine.models import Usage
from report_engine.ports import LLM, ChatMessage, LLMResult, LLMUsage
from report_engine.pricing import call_cost_usd

logger = logging.getLogger(__name__)

MAX_SLEEP_SLICE_SECONDS = 1.0

Emit = Callable[[ProgressEvent], None]


def usage_of(model_info: ModelInfo, usage: LLMUsage) -> Usage:
    """One call's real usage with its real cost."""
    input_tokens = max(0, int(usage.input_tokens))
    if min(usage.input_tokens, usage.cached_input_tokens, usage.output_tokens) < 0 or (
        usage.cached_input_tokens > usage.input_tokens
    ):
        logger.warning(
            "the model port reported usage that can not be true (input %s, cached %s, output %s); "
            "it is clamped",
            usage.input_tokens,
            usage.cached_input_tokens,
            usage.output_tokens,
        )
    return Usage(
        input_tokens=input_tokens,
        cached_input_tokens=min(max(0, int(usage.cached_input_tokens)), input_tokens),
        output_tokens=max(0, int(usage.output_tokens)),
        cost_usd=call_cost_usd(model_info, usage),
        calls=1,
    )


class Caller:
    """Sends the requests of one run and keeps the run's usage.

    Its requests go through a `CallPool`: the pool of a caller that reads
    several conversations at once, or a pool of its own with one slot. The
    calls of the run are counted under a lock, so its limit holds also when
    threads share it: a call counts when it is sent, and the call that would
    pass the limit is refused before it is sent."""

    def __init__(
        self,
        *,
        llm: LLM,
        model_info: ModelInfo,
        runtime: Runtime,
        cancel: CancelToken | None,
        emit: Emit,
        reasoning: str | None,
        max_calls: int | None = None,
        pool: CallPool | None = None,
    ) -> None:
        self._llm = llm
        self._model_info = model_info
        self._runtime = runtime
        self._cancel = cancel
        self._emit = emit
        self._reasoning = reasoning
        # a request that is sent counts, also when the port fails it: nobody
        # knows whether it was paid
        self._max_calls = max_calls
        self._sent = 0
        self._lock = threading.Lock()
        self._pool = pool or CallPool(
            max_parallel=1, tokens_per_minute=model_info.tokens_per_minute, clock=runtime.clock
        )
        self._usage = Usage()

    @property
    def usage(self) -> Usage:
        return self._usage

    def check_cancel(self) -> None:
        if self._cancelled():
            raise ReportCancelled("the run was cancelled", usage=self._usage)

    def _cancelled(self) -> bool:
        return (self._cancel is not None and self._cancel.cancelled) or self._pool.stopped

    def _past_limit(self) -> bool:
        return self._max_calls is not None and self._sent >= self._max_calls

    def _refuse_past_limit(self) -> None:
        with self._lock:
            past = self._past_limit()
        if past:
            raise CallLimitReached(self._max_calls or 0, usage=self._usage)

    def _take_room(self, tokens: int, waiting: ProgressEvent) -> int:
        """Wait until the budget of the pool has room for `tokens`, and take it."""
        announced = False
        while True:
            delay, ticket = self._pool.reserve(tokens)
            if ticket is not None:
                return ticket
            if not announced:
                self._emit(waiting)
                announced = True
            self._runtime.sleep(min(delay, MAX_SLEEP_SLICE_SECONDS))
            self.check_cancel()

    def _claim(self, ticket: int) -> None:
        """Count the call that goes out now, or refuse the one that would pass the limit."""
        with self._lock:
            refused = self._past_limit()
            if not refused:
                self._sent += 1
        if refused:
            self._pool.settle(ticket, 0)
            raise CallLimitReached(self._max_calls or 0, usage=self._usage)

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
        """One request. Raises ReportCancelled before sending, LLMError when the port
        fails in any way, and CallLimitReached for the request that would pass
        the limit of calls of this run."""
        self.check_cancel()
        self._refuse_past_limit()
        reserved = input_estimate + max_output_tokens
        with self._pool.slot(self.check_cancel):
            self.check_cancel()
            ticket = self._take_room(reserved, waiting)
            self._claim(ticket)
            self._emit(starting)
            result = self._generate(messages, schema, max_output_tokens, ticket, reserved)
        spent = usage_of(self._model_info, result.usage)
        extra_attempts = max(1, result.attempts) - 1
        self._pool.settle(
            ticket,
            spent.input_tokens + spent.output_tokens + reserved * extra_attempts,
            moved_to_now=extra_attempts > 0,
        )
        with self._lock:
            self._usage = self._usage.plus(spent)
        return result

    def _generate(
        self,
        messages: Sequence[ChatMessage],
        schema: Mapping[str, Any],
        max_output_tokens: int,
        ticket: int,
        reserved: int,
    ) -> LLMResult:
        """The request itself, counted in flight in the pool while it runs."""
        self._pool.sending()
        try:
            return self._llm.generate(
                messages,
                schema=schema,
                max_output_tokens=max_output_tokens,
                reasoning=self._reasoning,
            )
        except LLMError as exc:
            # the request went out, maybe more than once; keep its room taken
            attempts = max(1, exc.attempts)
            self._pool.settle(ticket, reserved * attempts, moved_to_now=attempts > 1)
            raise
        except Exception as exc:
            # the port is foreign code: its error fails this call, not the run.
            # Only the type is named; a message could hold prompt or reply text.
            logger.warning("the model port raised %s", type(exc).__name__)
            raise LLMError(f"the model port raised {type(exc).__name__}") from exc
        finally:
            self._pool.answered()
