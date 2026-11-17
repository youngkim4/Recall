"""Values as a route receives them: text from a query string, or a value from code."""

from __future__ import annotations

import re

# beyond any place in an archive, and inside what SQLite holds as a whole number
LARGEST = 2**62
_WHOLE = re.compile(r"[+-]?[0-9]+")
_ECHO = 40


def _held(number: int) -> int:
    return max(-LARGEST, min(LARGEST, number))


def shown(value: object) -> str:
    """A value for an error text: its start only, so that a long value does not come back to the client."""
    whole_text = repr(value)
    return whole_text if len(whole_text) <= _ECHO else f"{whole_text[:_ECHO]}... ({len(whole_text)} characters)"


def whole(name: str, value: object, default: int | None) -> int | None:
    """A whole number. Nothing given (None or "") is the default. Anything else raises ValueError.

    Text is a whole number when it is plain digits with or without a sign. A
    number beyond `LARGEST` is held to it: a place past the end is a place.
    """
    if value is None or value == "":
        return default
    if isinstance(value, int) and not isinstance(value, bool):
        return _held(value)
    if isinstance(value, str):
        text = value.strip()
        if _WHOLE.fullmatch(text):
            # more digits than any place has: the sign is all that counts
            many = len(text) > 30
            return _held((-LARGEST if text.startswith("-") else LARGEST) if many else int(text))
    raise ValueError(f"{name} must be a whole number, got {shown(value)}")


def text(value: object) -> str:
    """Text as it is, and "" for nothing given."""
    return "" if value is None else str(value)
