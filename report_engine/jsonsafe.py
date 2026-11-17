"""Make a caller's figures safe to store as JSON.

A report keeps the caller's stats as given. In practice they hold pandas
timestamps and numpy numbers, which JSON can not carry. Values keep their
meaning; only their type changes. This function does not raise.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)

MAX_DEPTH = 32
MISSING_TYPE_NAMES = frozenset({"NAType", "NaTType"})
UNWRAP_METHODS = ("to_dict", "tolist", "item")


def _mapping(value: Mapping[Any, Any], depth: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in value.items():
        name = str(key)
        if name in out:
            logger.warning(
                "stats key %r appears twice once keys are text; the later value is kept", name
            )
        out[name] = _convert(item, depth + 1)
    return out


def _unwrap(value: Any, depth: int) -> Any:
    """numpy and pandas values know how to turn into plain Python."""
    for name in UNWRAP_METHODS:
        method = getattr(value, name, None)
        if not callable(method) or isinstance(value, type):
            continue
        try:
            return _convert(method(), depth + 1)
        except (TypeError, ValueError):
            continue
    return str(value)


def _convert(value: Any, depth: int) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if type(value).__name__ in MISSING_TYPE_NAMES:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if depth >= MAX_DEPTH:
        return str(value)
    if isinstance(value, Mapping):
        return _mapping(value, depth)
    if isinstance(value, (list, tuple)):
        return [_convert(item, depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_convert(item, depth + 1) for item in sorted(value, key=str)]
    return _unwrap(value, depth)


def jsonable(value: Any) -> Any:
    """A deep copy made of dict, list, str, int, float, bool and None only."""
    return _convert(value, 0)
