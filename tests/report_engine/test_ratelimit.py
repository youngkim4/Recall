"""Tokens-per-minute budget over a sliding window. Time is a fake clock."""

import pytest

from report_engine.errors import ConfigError
from report_engine.ratelimit import TokenBudget


class Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_no_limit_never_waits() -> None:
    budget = TokenBudget(None, Clock())
    for _ in range(5):
        assert budget.delay(10**9) == 0.0
        budget.spend(10**9)
    assert budget.used() == 0


def test_under_the_limit_goes_now() -> None:
    budget = TokenBudget(100, Clock())
    assert budget.delay(60) == 0.0
    budget.spend(60)
    assert budget.delay(40) == 0.0
    assert budget.used() == 60


def test_over_the_limit_waits_until_enough_has_left_the_window() -> None:
    clock = Clock()
    budget = TokenBudget(100, clock)
    budget.spend(60)
    clock.advance(10)
    budget.spend(30)
    clock.advance(5)
    wait = budget.delay(50)
    # 60 was spent 15 s ago and leaves the window 45 s from now
    assert wait == pytest.approx(45, abs=0.1)
    clock.advance(wait)
    assert budget.delay(50) == 0.0
    assert budget.used() == 30


def test_the_wait_covers_as_many_old_entries_as_needed() -> None:
    clock = Clock()
    budget = TokenBudget(100, clock)
    budget.spend(40)
    clock.advance(20)
    budget.spend(40)
    clock.advance(20)
    # 80 used; 90 more needs both entries gone; the second leaves 40 s from now
    assert budget.delay(90) == pytest.approx(40, abs=0.1)


def test_a_request_larger_than_the_limit_can_never_go() -> None:
    budget = TokenBudget(100, Clock())
    with pytest.raises(ConfigError):
        budget.delay(101)
    assert budget.delay(100) == 0.0


def test_settle_replaces_the_estimate_with_the_real_number() -> None:
    clock = Clock()
    budget = TokenBudget(100, clock)
    ticket = budget.spend(90)
    assert budget.delay(20) > 0
    budget.settle(ticket, 50)
    assert budget.used() == 50
    assert budget.delay(20) == 0.0
    budget.settle(ticket, 95)
    assert budget.used() == 95


def test_settle_after_the_entry_left_the_window_changes_nothing() -> None:
    clock = Clock()
    budget = TokenBudget(100, clock)
    ticket = budget.spend(90)
    clock.advance(61)
    budget.settle(ticket, 10)
    assert budget.used() == 0


def test_the_window_slides() -> None:
    clock = Clock()
    budget = TokenBudget(100, clock)
    for _ in range(10):
        assert budget.delay(100) == 0.0
        budget.spend(100)
        clock.advance(60)
    assert budget.used() == 0


@pytest.mark.parametrize("tokens", [-1])
def test_negative_amounts_are_refused(tokens: int) -> None:
    budget = TokenBudget(100, Clock())
    with pytest.raises(ValueError):
        budget.spend(tokens)


@pytest.mark.parametrize("limit", [0, -10])
def test_the_limit_must_be_positive(limit: int) -> None:
    with pytest.raises(ValueError):
        TokenBudget(limit, Clock())
