"""Make a caller's figures safe to store as JSON.

A report keeps the caller's stats as given. In practice they hold pandas
timestamps and numpy numbers, which JSON can not carry. Values keep their
meaning; only their type changes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any


def jsonable(value: Any) -> Any:
    """A deep copy made of dict, list, str, int, float, bool and None only."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return None if value != value else value.isoformat()  # NaT is not equal to itself
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [jsonable(item) for item in sorted(value, key=str)]
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, type):
        try:
            return jsonable(item())
        except (TypeError, ValueError):
            return str(value)
    return str(value)
