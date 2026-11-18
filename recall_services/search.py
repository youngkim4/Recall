"""The /api/search payload from the archive.

The keys: `query`, `contact`, `count`, `total`, `offset`, `results`, and
`filters`, `order` and `prefix`, which say what the search applied. `total`
is the number of matches and `count` is the number of matches that the
response holds (CONTRACTS.md "Figures"). Each result has the shape of
`messages.payload` and one more key, `ranges`.

What a caller may send as `query`:
- Words. A message must hold all of them, each as a whole word, in any order.
- "Quoted text": the words as a phrase, in this order.
- A star at the end of a word or of a phrase: a prefix, `fam*`, `"good mor"*`.
- Nothing else is syntax. AND, OR, brackets and colons are plain words, and no
  text raises. Case and accents do not matter.
- A term in a script that writes no spaces, or with no letter or digit, is
  matched as plain text. A query is cut to 1,000 characters and 32 terms.

`prefix=true` makes the last word a prefix: the reader is still typing it.
With `prefix=false` (the default) every word is whole. A star typed by the
reader is always a prefix.

`ranges` are the [start, end) places of the matched words in the `text` of
the result, in UTF-16 units, as JavaScript counts. A word that the shown text
cut off has no range. With no words, `ranges` is [].

No words and a filter (a conversation, days, a kind, attachments) list the
messages of the filter. No words and no filter give nothing: no screen asks
for the newest messages of the whole archive. The filters are those of
`search_filters`. A tapback is not a message, so it is never a result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from archive import Archive
from archive.search import utf16_ranges
from message_text import ATTACHMENT_PLACEHOLDER, display_text
from recall_services import inputs, messages
from recall_services.names import Names
from recall_services.search_filters import NO_FILTERS, Filters, read_filters, read_order, read_prefix

DEFAULT_LIMIT = 60
MAX_LIMIT = 200
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


def _shown_places(raw: str, visible: int) -> dict[int, int]:
    """For each character of the stored text that the result shows: its place in the shown text."""
    places: dict[int, int] = {}
    for index, char in enumerate(raw):
        if len(places) >= visible:
            break
        if char == ATTACHMENT_PLACEHOLDER or (not places and char.isspace()):
            continue  # display_text drops the placeholder and the outer space
        places[index] = len(places)
    return places


def shown_ranges(raw: str, shown: str, ranges: Sequence[Sequence[int]]) -> list[list[int]]:
    """Code point ranges of the stored text, moved onto the shown text and given in UTF-16 units.

    `shown` is `messages.payload(...)["text"]`: the text without the placeholder
    and the outer space, cut with "…" when it is long. A range is cut at the cut.
    """
    cut = len(display_text(raw)) > messages.MESSAGE_TEXT_LIMIT
    places = _shown_places(raw, len(shown) - 1 if cut else len(shown))
    moved: list[list[int]] = []
    for start, end in ranges:
        inside = [places[index] for index in range(start, end) if index in places]
        if inside:
            moved.append([inside[0], inside[-1] + 1])
    return utf16_ranges(shown, moved)


def _results(archive: Archive, names: Names, hits: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    shaped = messages.payloads(archive, names, [hit["message"] for hit in hits])
    return [
        {**result, "ranges": shown_ranges(str(hit["message"]["text"]), result["text"], hit["ranges"])}
        for result, hit in zip(shaped, hits, strict=True)
    ]


def payload(
    archive: Archive,
    names: Names,
    query: str = "",
    contact: str = "",
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    *,
    as_you_type: bool = False,
    filters: Filters = NO_FILTERS,
    order: str = "newest",
) -> dict[str, Any]:
    """Search the archive. See the module docstring for the query.

    `limit` is held between 1 and 200. `offset` is the place of the first
    result among all matches. A `limit` or an `offset` that is no whole number
    raises ValueError. `contact` is one more conversation of `filters`.
    """
    size = max(1, min(inputs.whole("limit", limit, DEFAULT_LIMIT) or DEFAULT_LIMIT, MAX_LIMIT))
    place = max(0, inputs.whole("offset", offset, 0) or 0)
    scope = inputs.text(contact)
    applied = filters.with_contact(scope)
    raw = inputs.text(query)
    needle = raw.strip()
    hits: list[dict[str, Any]] = []
    total = 0
    if needle or not applied.is_empty:
        found = archive.search(
            as_typed(raw) if as_you_type else needle,
            limit=size,
            offset=place,
            order=order,
            browse=not applied.is_empty,
            **applied.archive_args(),
        )
        hits, total = found["hits"], int(found["total"])
    results = _results(archive, names, hits)
    return {
        "query": needle,
        "contact": scope,
        "count": len(results),
        "total": total,
        "offset": place,
        "results": results,
        "filters": applied.payload(),
        "order": order,
        "prefix": as_you_type,
    }


def answer(archive: Archive, names: Names, params: Mapping[str, object]) -> dict[str, Any]:
    """The payload for the query string of /api/search. A value that is not allowed raises ValueError."""
    filters = read_filters({**params, "contact": ""})
    return payload(
        archive,
        names,
        inputs.text(params.get("query")),
        inputs.text(params.get("contact")),
        params.get("limit", ""),  # type: ignore[arg-type]
        params.get("offset", ""),  # type: ignore[arg-type]
        as_you_type=read_prefix(params),
        filters=filters,
        order=read_order(params),
    )
