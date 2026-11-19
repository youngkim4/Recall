"""The figures of one profile of the people index, from the archive only. No model reads them.

The messages of all the conversations of the profile are read as one
stream in time order (`Archive.iter_messages` with a list of ids), group
events left out, as every figure of the archive leaves them out. A message
with no text is a message.

- `first_day`, `last_day`: local days, `YYYY-MM-DD`, or None.
- `messages`, `sent`, `received`: counts; `sent` is what the owner wrote.
- `active_days`: the local days with a message.
- `longest_gap_days`: the longest time between two messages that follow
  each other, in days, elapsed (`sent_at` is UTC), to 2 decimals.
- `busiest_month`: `YYYY-MM` with the most messages; a tie goes to the earliest.
- `wrote_first_share`: the share of the active days whose first message the
  owner wrote, to 3 decimals.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from archive import Archive

SECONDS_PER_DAY = 86_400
DAY_CHARS = 10
MONTH_CHARS = 7
GAP_DECIMALS = 2
SHARE_DECIMALS = 3


@dataclass(frozen=True)
class Figures:
    first_day: str | None
    last_day: str | None
    messages: int
    sent: int
    received: int
    active_days: int
    longest_gap_days: float
    busiest_month: str | None
    wrote_first_share: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _busiest(months: Mapping[str, int]) -> str | None:
    if not months:
        return None
    most = max(months.values())
    return min(month for month, count in months.items() if count == most)


def figures_of(messages: Iterable[Mapping[str, Any]]) -> Figures:
    """The figures of messages in time order: each has `timestamp`, `sent_at` and `is_from_me`."""
    count = sent = owner_first = 0
    days: list[str] = []
    months: dict[str, int] = {}
    gap = 0
    last_moment: int | None = None
    for row in messages:
        day = str(row["timestamp"])[:DAY_CHARS]
        own = bool(row["is_from_me"])
        moment = int(row["sent_at"])
        count += 1
        sent += own
        if not days or days[-1] != day:
            days.append(day)
            owner_first += own
        months[day[:MONTH_CHARS]] = months.get(day[:MONTH_CHARS], 0) + 1
        if last_moment is not None:
            gap = max(gap, moment - last_moment)
        last_moment = moment
    active = len(set(days))
    return Figures(
        first_day=days[0] if days else None,
        last_day=days[-1] if days else None,
        messages=count,
        sent=sent,
        received=count - sent,
        active_days=active,
        longest_gap_days=round(gap / SECONDS_PER_DAY, GAP_DECIMALS),
        busiest_month=_busiest(months),
        wrote_first_share=round(owner_first / active, SHARE_DECIMALS) if active else 0.0,
    )


def read_figures(archive: Archive, conversations: Sequence[str]) -> Figures:
    """The figures of these conversations together. Raises ArchiveError."""
    return figures_of(archive.iter_messages(list(conversations)))
