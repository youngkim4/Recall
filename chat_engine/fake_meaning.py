"""A scripted search by meaning for tests. It reads no index and uses no network.

It returns the hits of the script as they are. It does not apply the filters
of the call, so a test can script a hit that the engine must drop.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .ports import MeaningHit


def _copy(row: Any) -> Any:
    """A hit of the script. A row in a wrong form is kept as it is: a test scripts it on purpose."""
    return dict(row) if isinstance(row, Mapping) else row


def hit(conversation: str, first_message_id: str, last_message_id: str, score: float = 1.0) -> MeaningHit:
    return {
        "conversation": conversation,
        "first_message_id": first_message_id,
        "last_message_id": last_message_id,
        "score": float(score),
    }


class FakeMeaning:
    """`hits` answers every query. `by_query` answers the queries it names, and no other.

    `error` is raised by every search. Every call is recorded in `calls`.
    """

    def __init__(
        self,
        hits: Sequence[MeaningHit] = (),
        *,
        by_query: Optional[Mapping[str, Sequence[MeaningHit]]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self._hits = [_copy(row) for row in hits]
        self._by_query = (
            None if by_query is None else {key: [_copy(row) for row in rows] for key, rows in by_query.items()}
        )
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        *,
        conversations: Optional[Sequence[str]],
        since: Optional[str],
        until: Optional[str],
        limit: int,
    ) -> list[MeaningHit]:
        self.calls.append(
            {
                "query": query,
                "conversations": None if conversations is None else list(conversations),
                "since": since,
                "until": until,
                "limit": limit,
            }
        )
        if self._error is not None:
            raise self._error
        rows = self._hits if self._by_query is None else self._by_query.get(query, [])
        return [_copy(row) for row in rows[:max(0, limit)]]
