"""The /api/search payload from the archive.

The keys are those of today (`query`, `contact`, `count`, `total`, `results`)
plus `offset`. `total` is the number of matches and `count` is the number of
matches that the response holds (CONTRACTS.md "Figures"). Each result has the
shape of `messages.payload`. The newest match is first.

What a caller may send as `query`:
- Words. A message must hold all of them, each as a whole word, in any order.
- "Quoted text": the words as a phrase, in this order.
- A star at the end of a word or of a phrase: a prefix, `fam*`, `"good mor"*`.
- Nothing else is syntax. AND, OR, brackets and colons are plain words, and no
  text raises. Case and accents do not matter.
- A term in a script that writes no spaces, or with no letter or digit, is
  matched as plain text. A query is cut to 1,000 characters and 32 terms.

Differences from the server at 89ab504, which matched the whole query as one
piece of plain text:
- `piz` no longer finds `pizza`. Send `piz*`, or set `as_you_type` while the
  reader types: the last word is then a prefix.
- A tapback is not a message, so its summary line is never a result.
- No words and a conversation still list that conversation from its newest
  message. No words and no conversation give nothing: no screen asks for the
  newest messages of the whole archive.
"""

from __future__ import annotations

from typing import Any

from archive import Archive
from recall_services import inputs, messages
from recall_services.names import Names

DEFAULT_LIMIT = 60
MAX_LIMIT = 200
# how far into one conversation a search with no words reads
BROWSE_DEPTH = 5_000
_CLOSING = ('*', '"', "“", "”", "„", "‟", "＂")


def as_typed(query: str) -> str:
    """The query with its last word as a prefix: the reader may still be typing it.

    A word followed by a space is whole. A query that ends with a star or
    with a quote mark is left as it is.
    """
    raw = inputs.text(query)
    needle = raw.strip()
    if not needle or raw != raw.rstrip() or needle.endswith(_CLOSING):
        return needle
    return f"{needle}*"


def _browse(archive: Archive, contact: str, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
    """The messages of one conversation from its newest one, and how many it holds."""
    total = int(archive.count(conversations=[contact]))  # type: ignore[arg-type]
    reach = min(offset + limit, BROWSE_DEPTH)
    if reach <= offset:
        return [], total
    newest_first = archive.messages(contact, limit=reach)["messages"][::-1]
    return newest_first[offset:reach], total


def _matches(archive: Archive, asked: str, contact: str, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
    found = archive.search(
        asked, conversations=[contact] if contact else None, limit=limit, offset=offset, order="newest"
    )
    return [hit["message"] for hit in found["hits"]], int(found["total"])


def payload(
    archive: Archive,
    names: Names,
    query: str = "",
    contact: str = "",
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    *,
    as_you_type: bool = False,
) -> dict[str, Any]:
    """Search the archive, or one conversation of it. See the module docstring for the query.

    `limit` is held between 1 and 200. `offset` is the place of the first
    result among all matches. A `limit` or an `offset` that is no whole number
    raises ValueError. A conversation id matches whole, never in part.
    """
    size = inputs.whole("limit", limit, DEFAULT_LIMIT) or DEFAULT_LIMIT
    size = max(1, min(size, MAX_LIMIT))
    place = max(0, inputs.whole("offset", offset, 0) or 0)
    scope = inputs.text(contact)
    raw = inputs.text(query)
    needle = raw.strip()
    if needle:
        found, total = _matches(archive, as_typed(raw) if as_you_type else needle, scope, size, place)
    elif scope:
        found, total = _browse(archive, scope, size, place)
    else:
        found, total = [], 0
    results = messages.payloads(archive, names, found)
    return {
        "query": needle,
        "contact": scope,
        "count": len(results),
        "total": total,
        "offset": place,
        "results": results,
    }
