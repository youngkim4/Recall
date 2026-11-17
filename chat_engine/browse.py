"""Read one conversation from its start or its end, with no search word.

The archive pages a conversation by message id: `before`, `after`, `around`.
It has no call for "the first messages" or "the messages of that week". So
this module finds the place itself:

1. It asks for one page. Which edge that page shows does not matter.
2. It goes to the period: by a search for a frequent word when that works,
   else page by page.
3. It walks to the edge that was asked for, then back until it has enough.

Every walk has a limit. When the limit stops it, the result says so.

An archive that can read a conversation in order (`iter_messages`) does not
need the walk for the first messages, and is the way out when a walk stops.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from itertools import islice
from typing import Optional

from .ports import ChatArchive, Message, MessagePage
from .render import time_key
from .textnorm import surface_words

PAGE = 200
MAX_HOPS = 60
_PROBES = 3
OLDER, NEWER = "older", "newer"


@dataclass(frozen=True)
class Browsed:
    messages: tuple[Message, ...]
    reached_edge: bool  # False: a walk stopped at its limit before the edge
    anchored: bool = True  # False: the period was not found within the limit


@dataclass(frozen=True)
class _Period:
    since: Optional[str]
    until: Optional[str]

    def holds(self, message: Message) -> bool:
        day = _day(message)
        return not ((self.since and day < self.since) or (self.until and day > self.until))

    def side_of(self, page: MessagePage) -> str:
        """Where the period lies from a page: OLDER, NEWER, or "" when the page reaches into it."""
        if self.until and _day(page["messages"][0]) > self.until:
            return OLDER
        if self.since and _day(page["messages"][-1]) < self.since:
            return NEWER
        return ""

    def goes_past(self, page: MessagePage, direction: str) -> bool:
        """True when the period can hold messages beyond this page in `direction`."""
        if direction == OLDER:
            return bool(page["has_older"]) and not (self.since and _day(page["messages"][0]) < self.since)
        return bool(page["has_newer"]) and not (self.until and _day(page["messages"][-1]) > self.until)


class _Walk:
    """Pages of one conversation, with a limit on how many are fetched."""

    def __init__(self, archive: ChatArchive, conversation_id: str, max_hops: int) -> None:
        self._archive = archive
        self._conversation = conversation_id
        self._left = max_hops

    @property
    def spent(self) -> bool:
        return self._left <= 0

    def first(self) -> MessagePage:
        return self._archive.messages(self._conversation, limit=PAGE)

    def around(self, message_id: str) -> MessagePage:
        return self._archive.messages(self._conversation, around=message_id, limit=PAGE)

    def step(self, page: MessagePage, direction: str) -> MessagePage:
        self._left -= 1
        if direction == OLDER:
            return self._archive.messages(self._conversation, before=str(page["messages"][0]["id"]), limit=PAGE)
        return self._archive.messages(self._conversation, after=str(page["messages"][-1]["id"]), limit=PAGE)

    def probe(self, sample: list[Message], edge: str, period: _Period) -> Optional[Message]:
        """A message near the wanted edge: the oldest (or newest) hit of a frequent word."""
        counts = Counter(token for message in sample for token in set(surface_words(message.get("text") or "")))
        best: Optional[Message] = None
        for token, _ in counts.most_common(_PROBES):
            result = self._archive.search(
                token, conversations=[self._conversation], since=period.since, until=period.until,
                from_me=None, limit=1, offset=0, order=edge,
            )
            for hit in result["hits"][:1]:
                best = _nearer(best, hit["message"], edge == "oldest")
        return best


def browse(
    archive: ChatArchive,
    conversation_id: str,
    *,
    edge: str,
    limit: int,
    since: Optional[str] = None,
    until: Optional[str] = None,
    max_hops: int = MAX_HOPS,
) -> Browsed:
    """The first (`edge="oldest"`) or last messages of a conversation or of a period in it."""
    reader = getattr(archive, "iter_messages", None)
    if callable(reader) and edge == "oldest":
        return Browsed(tuple(islice(reader(conversation_id, since=since, until=until), max(0, limit))), True)
    found = _walk_to(archive, conversation_id, edge, limit, since, until, max_hops)
    if callable(reader) and not (found.anchored and found.reached_edge):
        rows = reader(conversation_id, since=since, until=until)
        return Browsed(tuple(deque(rows, maxlen=max(1, limit))), True)
    return found


def _walk_to(
    archive: ChatArchive,
    conversation_id: str,
    edge: str,
    limit: int,
    since: Optional[str],
    until: Optional[str],
    max_hops: int,
) -> Browsed:
    walk = _Walk(archive, conversation_id, max_hops)
    period = _Period(since, until)
    toward, away = (OLDER, NEWER) if edge == "oldest" else (NEWER, OLDER)
    page = walk.first()
    if not page["messages"]:
        return Browsed((), True)
    if period.side_of(page) or period.goes_past(page, toward):
        anchor = walk.probe(page["messages"], edge, period)
        jumped = walk.around(str(anchor["id"])) if anchor is not None else None
        page = jumped if jumped and jumped["messages"] else page
    previous = ""
    while period.side_of(page):
        side = period.side_of(page)
        if previous and side != previous:
            return Browsed((), True)  # the period lies between two pages: no message is in it
        previous = side
        if not page["has_older" if side == OLDER else "has_newer"]:
            return Browsed((), True)  # the conversation ends before the period starts
        if walk.spent:
            return Browsed((), False, anchored=False)
        page = walk.step(page, side)
        if not page["messages"]:
            return Browsed((), True)
    while period.goes_past(page, toward) and not walk.spent:
        further = walk.step(page, toward)
        if not further["messages"]:
            break
        page = further
    reached = not period.goes_past(page, toward)
    kept = [message for message in page["messages"] if period.holds(message)]
    while len(kept) < limit and period.goes_past(page, away) and not walk.spent:
        page = walk.step(page, away)
        if not page["messages"]:
            break
        more = [message for message in page["messages"] if period.holds(message)]
        kept = kept + more if away == NEWER else more + kept
    return Browsed(tuple(kept[:limit] if edge == "oldest" else kept[-limit:]), reached)


def _day(message: Message) -> str:
    return str(message.get("timestamp") or "")[:10]


def _nearer(current: Optional[Message], candidate: Message, oldest: bool) -> Message:
    if current is None:
        return candidate
    one, other = time_key(candidate), time_key(current)
    if oldest:
        return candidate if one < other else current
    return candidate if one > other else current
