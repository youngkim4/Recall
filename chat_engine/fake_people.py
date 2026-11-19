"""A scripted search of people for tests. It reads no index and uses no network.

It returns the people of the script as they are. It does not apply the kinds,
the days or the conversations of the call, so a test can script a person that
the engine must drop.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional, Sequence

from .ports import PersonHit, TurningPoint


def _copy(row: Any) -> Any:
    return dict(row) if isinstance(row, Mapping) else row


def turning_point(date: str, what: str) -> TurningPoint:
    return {"date": date, "what": what}


def person(
    key: str,
    conversations: Sequence[str],
    *,
    label: str = "",
    kind: str = "unclear",
    status: str = "unclear",
    summary: str = "",
    turning_points: Sequence[TurningPoint] = (),
    figures: Optional[Mapping[str, Any]] = None,
    score: float = 1.0,
) -> PersonHit:
    return {
        "key": key,
        "conversations": list(conversations),
        "label": label,
        "kind": kind,
        "status": status,
        "summary": summary,
        # a row in a wrong form is kept as it is: a test scripts it on purpose
        "turning_points": [_copy(row) for row in turning_points],
        "figures": dict(figures or {}),
        "score": float(score),
    }


class FakePeople:
    """`people` answers every description. `by_query` answers the descriptions it names, and no other.

    `error` is raised by every search. Every call is recorded in `calls`.
    """

    def __init__(
        self,
        people: Sequence[Any] = (),
        *,
        by_query: Optional[Mapping[str, Sequence[Any]]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self._people = copy.deepcopy(list(people))
        self._by_query = (
            None if by_query is None else {key: copy.deepcopy(list(rows)) for key, rows in by_query.items()}
        )
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        description: str,
        *,
        kinds: Optional[Sequence[str]],
        since: Optional[str],
        until: Optional[str],
        limit: int,
        conversations: Optional[Sequence[str]] = None,
    ) -> list[PersonHit]:
        self.calls.append(
            {
                "description": description,
                "kinds": None if kinds is None else list(kinds),
                "since": since,
                "until": until,
                "limit": limit,
                "conversations": None if conversations is None else list(conversations),
            }
        )
        if self._error is not None:
            raise self._error
        rows = self._people if self._by_query is None else self._by_query.get(description, [])
        return copy.deepcopy(rows[:max(0, limit)])
