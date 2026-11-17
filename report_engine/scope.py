"""Scope helpers: turn a caller's dates into the bounds a source compares against."""

from __future__ import annotations

from typing import Mapping

from report_engine.dates import DAY_LENGTH
from report_engine.models import Scope

DAY_START = " 00:00:00"
DAY_END = " 23:59:59"


def coerce_scope(value: Scope | Mapping[str, str | None] | None) -> Scope:
    """Accept a Scope, a mapping with since/until, or None for all dates."""
    if value is None:
        return Scope()
    if isinstance(value, Scope):
        return value
    return Scope.model_validate(dict(value))


def lower_bound(since: str | None) -> str | None:
    if since is None:
        return None
    return since + DAY_START if len(since) == DAY_LENGTH else since


def upper_bound(until: str | None) -> str | None:
    if until is None:
        return None
    return until + DAY_END if len(until) == DAY_LENGTH else until


def inclusive_bounds(scope: Scope) -> tuple[str | None, str | None]:
    """Full timestamps, both included. Timestamps are fixed width, so strings compare."""
    return lower_bound(scope.since), upper_bound(scope.until)


def in_bounds(timestamp: str, low: str | None, high: str | None) -> bool:
    if low is not None and timestamp < low:
        return False
    if high is not None and timestamp > high:
        return False
    return True
