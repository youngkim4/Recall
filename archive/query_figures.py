"""search(), count() and stats() of Archive: the part of the query API that filters and counts.

Archive (archive/queries.py) takes these methods from FigureQueries. Each one
reads inside one read transaction of the calling thread.

`kind` is `direct` or `group`. `attachments` is `any`, `none`, `image`,
`video`, `audio` or `other` (archive/narrow.py). None is no filter.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from typing import Any

from archive import figures, filters
from archive import search as search_module
from archive.filters import as_id_list, check_bool, check_choice, conversation_keys, day_bounds


class FigureQueries:
    """Search and figures. The class that takes these methods gives `_reading`."""

    _reading: Callable[[], AbstractContextManager[sqlite3.Connection]]

    def search(
        self,
        query: str,
        *,
        conversations: str | Iterable[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        from_me: bool | None = None,
        limit: int = 50,
        offset: int = 0,
        order: str = "newest",
        include_system: bool = False,
        no_conversation: bool = False,
        kind: str | None = None,
        attachments: str | None = None,
        browse: bool = False,
    ) -> dict[str, Any]:
        """`{"total": every match, "hits": [{"message", "ranges"}]}`.

        Words are AND-ed, "quoted text" is a phrase, a trailing * is a prefix.
        Text typed by a user never raises. `ranges` are [start, end) code point offsets in the text.
        `no_conversation=True` searches only the messages that belong to no conversation.
        `browse=True` lists every message of the filters when the query has no words.
        """
        with self._reading() as conn:
            return search_module.search(
                conn,
                query,
                conversations=conversations,
                since=since,
                until=until,
                from_me=from_me,
                limit=limit,
                offset=offset,
                order=order,
                include_system=include_system,
                no_conversation=no_conversation,
                kind=kind,
                attachments=attachments,
                browse=browse,
            )

    def count(
        self,
        *,
        conversations: str | Iterable[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        from_me: bool | None = None,
        text: str | None = None,
        group_by: str | None = None,
        include_system: bool = False,
        no_conversation: bool = False,
        kind: str | None = None,
        attachments: str | None = None,
    ) -> int | list[dict[str, Any]]:
        """Number of messages, or `[{"key", "total", "sent", "received"}]` per bucket.

        `group_by` is one of figures.GROUPS. `text` follows the rules of search().
        `no_conversation=True` counts only the messages that belong to no conversation.
        """
        if group_by is not None:
            check_choice("group_by", group_by, figures.GROUPS)
        check_bool("include_system", include_system)
        parsed = None if text is None else search_module.parse_query(text)
        lower, upper = day_bounds(since, until)
        direction = filters.check_flag("from_me", from_me)
        plain = kind is None and attachments is None and not include_system and not no_conversation
        from_totals = plain and parsed is None and lower is None and upper is None
        with self._reading() as conn:
            if from_totals and group_by in (None, "conversation"):
                keys = None if conversations is None else conversation_keys(conn, conversations)
                return figures.count_from_totals(conn, keys, direction, group_by == "conversation")
            where = filters.scope(
                conn,
                conversations=conversations,
                since=since,
                until=until,
                from_me=from_me,
                text=parsed,
                include_system=include_system,
                no_conversation=no_conversation,
                kind=kind,
                attachments=attachments,
            )
            return figures.count(conn, where, group_by, lower, upper)

    def stats(
        self,
        conversation_id: str | Iterable[str],
        *,
        since: str | None = None,
        until: str | None = None,
        include_system: bool = False,
    ) -> dict[str, Any]:
        """The figures of compute_stats in conversation.py. The README lists the keys."""
        check_bool("include_system", include_system)
        lower, upper = day_bounds(since, until)
        label: object = conversation_id if isinstance(conversation_id, str) else as_id_list("ids", conversation_id)
        with self._reading() as conn:
            keys = conversation_keys(conn, conversation_id)
            return figures.stats(conn, label, keys, lower, upper, include_system)
