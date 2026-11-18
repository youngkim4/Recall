"""What runs that send requests at the same time share.

The parts of one conversation are read one at a time: the prompt of a part
holds the notes of the parts before it (`map_step.py`). Runs of different
conversations do not wait for each other, so a caller reads several
conversations at once, one thread each, and gives every run one `CallPool`:

- at most `max_parallel` requests are in flight at one time;
- one tokens-per-minute budget for all the runs, so N runs do not take N
  times the rate of the model;
- `stop()`: no run of the pool sends a new request; the requests in flight end.

A run with no pool gets a pool of its own with one slot (`Caller`). The
limit of calls of a run stays with the run (`Caller.max_calls`).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from report_engine.errors import ConfigError
from report_engine.ratelimit import TokenBudget

SLOT_POLL_SECONDS = 0.05


def _checked_slots(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError("max_parallel must be a whole number, one or more")
    return value


class CallPool:
    """Slots for requests and a budget of tokens, shared by threads."""

    def __init__(
        self,
        *,
        max_parallel: int,
        tokens_per_minute: int | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_parallel = _checked_slots(max_parallel)
        self._slots = threading.BoundedSemaphore(self._max_parallel)
        self._lock = threading.Lock()
        self._budget = TokenBudget(tokens_per_minute, clock)
        self._stop = threading.Event()
        self._sent = 0
        self._in_flight = 0
        self._most_in_flight = 0

    @property
    def max_parallel(self) -> int:
        return self._max_parallel

    @property
    def sent(self) -> int:
        """Requests sent through this pool so far."""
        with self._lock:
            return self._sent

    @property
    def most_in_flight(self) -> int:
        """The most requests that were in flight at one time."""
        with self._lock:
            return self._most_in_flight

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        """No run of this pool sends a new request. Requests in flight end."""
        self._stop.set()

    @contextmanager
    def slot(self, check: Callable[[], None]) -> Iterator[None]:
        """Hold one slot. `check` runs while the slot is waited for: it raises to stop the wait."""
        while not self._slots.acquire(timeout=SLOT_POLL_SECONDS):
            check()
        try:
            yield
        finally:
            self._slots.release()

    def reserve(self, tokens: int) -> tuple[float, int | None]:
        """Take room for a request, or say how long to wait: `(0.0, ticket)` or `(delay, None)`.

        Raises ConfigError for one request that is larger than the limit."""
        with self._lock:
            delay = self._budget.delay(tokens)
            if delay > 0:
                return delay, None
            return 0.0, self._budget.spend(tokens)

    def settle(self, ticket: int, tokens: int, moved_to_now: bool = False) -> None:
        """Replace the room a request took with what it used (`TokenBudget.settle`)."""
        with self._lock:
            self._budget.settle(ticket, tokens, moved_to_now=moved_to_now)

    def sending(self) -> None:
        """Count a request that goes out now."""
        with self._lock:
            self._sent += 1
            self._in_flight += 1
            self._most_in_flight = max(self._most_in_flight, self._in_flight)

    def answered(self) -> None:
        """A request in flight came back, or failed."""
        with self._lock:
            self._in_flight -= 1
