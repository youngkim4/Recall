"""The /api/day payload from the archive: one day of one conversation.

The keys are those of `day_payload` in ui_server.py at 89ab504: `contact`,
`day`, `total`, `offset`, `previousDay`, `nextDay`, `messages`. Each message
has the shape of `messages.payload`.

The paging rule of "A day of a conversation" in CONTRACTS.md:
- The conversation id matches whole, never in part.
- A place given as `offset` is used as it is, from 0 to the total. The answer
  can hold fewer rows than `limit`, and a place past the end gives none.
- With no `offset`, the part is a whole window around the time `around`, or
  from the start of the day.
- A `limit` below 1 is refused. A `limit` above 400 is 400.
- `previousDay` and `nextDay` are the nearest days that hold messages, or None.
"""

from __future__ import annotations

import re
from typing import Any

from archive import Archive
from day_view import DAY_LIMIT
from recall_services import inputs, messages
from recall_services.names import Names

_CLOCK = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})(:\d{2})?")


def _clock(around: object) -> str | None:
    """`around` as "YYYY-MM-DD HH:MM:SS", or None when it is no such time.

    A fraction of a second and a zone are dropped: every time of the archive
    is the time on the clock. A time with no seconds is at second 0.
    """
    if not isinstance(around, str):
        return None
    found = _CLOCK.match(around.strip())
    return f"{found.group(1)} {found.group(2)}{found.group(3) or ':00'}" if found else None


def payload(
    archive: Archive,
    names: Names,
    contact: str,
    day: str,
    around: str | None = None,
    offset: int | None = None,
    limit: int = DAY_LIMIT,
) -> dict[str, Any]:
    """One day of one conversation, oldest message first. A long day is given in part.

    Raises ValueError when no conversation is named, when `day` is not
    written as YYYY-MM-DD, and when `limit` is below 1.
    """
    conversation = inputs.text(contact)
    if not conversation.strip():
        raise ValueError("A conversation is needed")
    size = inputs.whole("limit", limit, DAY_LIMIT)
    if size is None or size < 1:
        raise ValueError(f"Not a number of messages: {limit!r}")
    place = inputs.whole("offset", offset, None)
    found = archive.day(
        conversation,
        day,
        limit=size,
        offset=None if place is None else max(0, place),
        around=_clock(around),
    )
    return {
        "contact": conversation,
        "day": found["day"],
        "total": found["total"],
        "offset": found["offset"],
        "previousDay": found["previous_day"],
        "nextDay": found["next_day"],
        "messages": messages.payloads(archive, names, found["messages"]),
    }
