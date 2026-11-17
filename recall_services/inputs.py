"""Values as a route receives them: text from a query string, or a value from code."""

from __future__ import annotations


def whole(name: str, value: object, default: int | None) -> int | None:
    """A whole number. Nothing given (None or "") is the default. Anything else raises ValueError."""
    if value is None or value == "":
        return default
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise ValueError(f"{name} must be a whole number, got {value!r}")


def text(value: object) -> str:
    """Text as it is, and "" for nothing given."""
    return "" if value is None else str(value)
