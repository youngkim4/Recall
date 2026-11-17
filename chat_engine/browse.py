"""Read one conversation from its start or its end, with no search word.

The archive pages a conversation by message id: `before`, `after`, `around`.
It has no call for "the first messages". So this module finds the edge itself:
it asks for one page, jumps close to the edge with a search for a frequent
word, then walks page by page until no older (or newer) page is left.
The result is right whichever edge the first page shows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

from .ports import ChatArchive, Message, MessagePage
from .textnorm import words

PAGE = 200
MAX_HOPS = 60
_PROBES = 3


@dataclass(frozen=True)
class Browsed:
    messages: tuple[Message, ...]
    reached_edge: bool  # False: the walk stopped at the hop limit
    anchored: bool = True  # False: no way was found into the period


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
    """The first (`edge="oldest"`) or last messages of a conversation or a period."""
    backwards = edge == "oldest"
    page = archive.messages(conversation_id, limit=PAGE)
    if not page["messages"]:
        return Browsed((), True)
    inside = _inside(since, until)
    whole = not page["has_older"] and not page["has_newer"]
    in_period = any(inside(message) for message in page["messages"])
    if not whole and (_more(page, backwards, since, until) or not in_period):
        anchor = _probe(archive, conversation_id, page["messages"], edge, since, until)
        if anchor is not None:
            jumped = archive.messages(conversation_id, around=str(anchor["id"]), limit=PAGE)
            page = jumped if jumped["messages"] else page
        elif not in_period:
            return Browsed((), False, anchored=False)
    kept = _edge([m for m in page["messages"] if inside(m)], backwards, limit)
    hops = 0
    while _more(page, backwards, since, until) and hops < max_hops:
        page = _next_page(archive, conversation_id, page, backwards)
        if not page["messages"]:
            break
        fresh = [m for m in page["messages"] if inside(m)]
        kept = _edge(fresh + kept if backwards else kept + fresh, backwards, limit)
        hops += 1
    return Browsed(tuple(kept), not _more(page, backwards, since, until))


def _edge(messages: list[Message], backwards: bool, limit: int) -> list[Message]:
    return messages[:limit] if backwards else messages[-limit:]


def _inside(since: Optional[str], until: Optional[str]) -> Callable[[Message], bool]:
    def check(message: Message) -> bool:
        day = str(message.get("timestamp") or "")[:10]
        return not ((since and day < since) or (until and day > until))

    return check


def _more(page: MessagePage, backwards: bool, since: Optional[str], until: Optional[str]) -> bool:
    """True while the edge we want is not on this page."""
    if not page["messages"]:
        return False
    if backwards:
        first_day = str(page["messages"][0].get("timestamp") or "")[:10]
        return bool(page["has_older"]) and (since is None or first_day >= since)
    last_day = str(page["messages"][-1].get("timestamp") or "")[:10]
    return bool(page["has_newer"]) and (until is None or last_day <= until)


def _next_page(archive: ChatArchive, conversation_id: str, page: MessagePage, backwards: bool) -> MessagePage:
    if backwards:
        return archive.messages(conversation_id, before=str(page["messages"][0]["id"]), limit=PAGE)
    return archive.messages(conversation_id, after=str(page["messages"][-1]["id"]), limit=PAGE)


def _probe(
    archive: ChatArchive,
    conversation_id: str,
    sample: list[Message],
    edge: str,
    since: Optional[str],
    until: Optional[str],
) -> Optional[Message]:
    """A message near the edge: the oldest (or newest) hit of a frequent word."""
    counts = Counter(token for message in sample for token in set(words(message.get("text") or "")))
    best: Optional[Message] = None
    for token, _ in counts.most_common(_PROBES):
        result = archive.search(
            token, conversations=[conversation_id], since=since, until=until, from_me=None,
            limit=1, offset=0, order=edge,
        )
        for hit in result["hits"][:1]:
            best = _nearer(best, hit["message"], edge == "oldest")
    return best


def _nearer(current: Optional[Message], candidate: Message, backwards: bool) -> Message:
    if current is None:
        return candidate
    key = lambda m: (str(m.get("timestamp") or ""), m.get("sent_at") or 0)  # noqa: E731
    if backwards:
        return candidate if key(candidate) < key(current) else current
    return candidate if key(candidate) > key(current) else current
