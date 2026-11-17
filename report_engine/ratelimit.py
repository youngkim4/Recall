"""Tokens-per-minute budget.

A sliding window of one minute. A call is entered at its send time with an
estimate, and the estimate is replaced by the real number when the reply
arrives. One run owns one budget; it is not shared between threads.
"""

from __future__ import annotations

from collections.abc import Callable

from report_engine.errors import ConfigError

WINDOW_SECONDS = 60.0
WAIT_SLACK_SECONDS = 0.01


class TokenBudget:
    def __init__(
        self,
        limit: int | None,
        clock: Callable[[], float],
        window_s: float = WINDOW_SECONDS,
    ) -> None:
        if limit is not None and limit <= 0:
            raise ValueError("tokens-per-minute limit must be positive")
        self._limit = limit
        self._clock = clock
        self._window = window_s
        self._entries: dict[int, tuple[float, int]] = {}
        self._tickets = 0

    def _live(self) -> dict[int, tuple[float, int]]:
        horizon = self._clock() - self._window
        self._entries = {
            ticket: entry for ticket, entry in self._entries.items() if entry[0] > horizon
        }
        return self._entries

    def used(self) -> int:
        """Tokens counted in the window that ends now."""
        return sum(tokens for _, tokens in self._live().values())

    def delay(self, tokens: int) -> float:
        """Seconds to wait before `tokens` more fit. 0.0 means send now."""
        if tokens < 0:
            raise ValueError("token amount can not be negative")
        if self._limit is None:
            return 0.0
        if tokens > self._limit:
            raise ConfigError(
                f"one request of {tokens} tokens is larger than the limit of {self._limit} tokens per minute"
            )
        live = self._live()
        excess = sum(amount for _, amount in live.values()) + tokens - self._limit
        if excess <= 0:
            return 0.0
        freed = 0
        for sent_at, amount in sorted(live.values()):
            freed += amount
            if freed >= excess:
                return max(0.0, sent_at + self._window - self._clock()) + WAIT_SLACK_SECONDS
        return self._window

    def spend(self, tokens: int) -> int:
        """Enter a call at the current time. Returns a ticket for `settle`."""
        if tokens < 0:
            raise ValueError("token amount can not be negative")
        self._tickets += 1
        if self._limit is not None:
            self._entries = {**self._live(), self._tickets: (self._clock(), tokens)}
        return self._tickets

    def settle(self, ticket: int, tokens: int) -> None:
        """Replace an estimate with the real number. The send time stays."""
        entry = self._live().get(ticket)
        if entry is not None:
            self._entries = {**self._entries, ticket: (entry[0], max(0, tokens))}
