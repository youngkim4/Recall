"""Checks on tool arguments. The model wrote them, so nothing is trusted."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Optional, Sequence

from .scope import parse_day

_MAX_TEXT = 300
_MAX_ITEMS = 30
_MAX_ITEM_CHARS = 120


class ToolInputError(ValueError):
    """The arguments cannot be used. The message goes back to the model."""


def _not_a_number(constant: str) -> None:
    raise ValueError(f"{constant} is not a JSON number")


def parse_arguments(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}", parse_constant=_not_a_number)
    except (TypeError, ValueError, RecursionError) as error:
        raise ToolInputError("the arguments are not valid JSON.") from error
    if not isinstance(value, dict):
        raise ToolInputError("the arguments must be a JSON object.")
    return value


def text_arg(arguments: Mapping[str, Any], key: str) -> str:
    value = arguments.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ToolInputError(f"{key} must be text.")
    return " ".join(value.split())[:_MAX_TEXT]


def day_arg(arguments: Mapping[str, Any], key: str) -> Optional[str]:
    value = arguments.get(key)
    if value is None or value == "":
        return None
    try:
        return parse_day(value)
    except (TypeError, ValueError) as error:
        raise ToolInputError(f"{key} must be a day like 2024-03-31.") from error


def bool_arg(arguments: Mapping[str, Any], key: str) -> Optional[bool]:
    value = arguments.get(key)
    if value is None or isinstance(value, bool):
        return value
    raise ToolInputError(f"{key} must be true, false or null.")


def int_arg(arguments: Mapping[str, Any], key: str, *, default: int, low: int, high: int) -> int:
    value = arguments.get(key)
    if value is None:
        return default
    whole = isinstance(value, int) or (isinstance(value, float) and math.isfinite(value) and value == int(value))
    if isinstance(value, bool) or not whole:
        raise ToolInputError(f"{key} must be a whole number.")
    return max(low, min(high, int(value)))


def choice_arg(
    arguments: Mapping[str, Any], key: str, choices: Sequence[str], default: Optional[str]
) -> Optional[str]:
    value = arguments.get(key)
    if value is None or value == "":
        return default
    if value not in choices:
        raise ToolInputError(f"{key} must be one of: {', '.join(choices)}.")
    return str(value)


def list_arg(arguments: Mapping[str, Any], key: str) -> Optional[list[str]]:
    value = arguments.get(key)
    if value is None or value in ([], ""):
        return None
    items = [value] if isinstance(value, str) else value
    fine = isinstance(items, list) and len(items) <= _MAX_ITEMS
    if not fine or not all(isinstance(item, str) and len(item) <= _MAX_ITEM_CHARS for item in items):
        raise ToolInputError(f"{key} must be a short list of conversation aliases like c3.")
    return list(items)
