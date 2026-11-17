"""Scope helpers: turn a caller's dates into the bounds a source compares against."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from report_engine.dates import lower_bound, upper_bound
from report_engine.errors import ConfigError
from report_engine.models import Scope

__all__ = ["coerce_scope", "in_bounds", "inclusive_bounds", "lower_bound", "upper_bound"]


def coerce_scope(value: Scope | Mapping[str, str | None] | None) -> Scope:
    """Accept a Scope, a mapping with since/until, or None for all dates.
    Anything else is a ConfigError: a scope that is not understood must not
    turn into "all dates"."""
    if value is None:
        return Scope()
    if isinstance(value, Scope):
        return value
    if not isinstance(value, Mapping):
        raise ConfigError(
            f"scope must be a mapping with since and until, not {type(value).__name__}"
        )
    try:
        return Scope.model_validate(dict(value))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, item['loc'])) or 'scope'}: {item['msg']}" for item in exc.errors()
        )
        raise ConfigError(f"scope is not valid: {problems}") from exc


def inclusive_bounds(scope: Scope) -> tuple[str | None, str | None]:
    """Full timestamps, both included. Timestamps are fixed width, so strings compare."""
    return lower_bound(scope.since), upper_bound(scope.until)


def in_bounds(timestamp: str, low: str | None, high: str | None) -> bool:
    if low is not None and timestamp < low:
        return False
    return high is None or timestamp <= high
