"""Find the segments nearest to a query vector.

    index = MeaningIndex.load(store)      # None when there is no index
    hits = index.search(vector, conversations=[...], since="2024-06-01", limit=20)

The vectors stay on disk as float16. A search reads them in chunks, so the
memory of a search does not grow with the index. A loaded index answers from
the build that it loaded: load again after a build.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import date

import numpy as np

from meaning_index.errors import StoreError
from meaning_index.shapes import IndexMeta, SearchHit, Segment
from meaning_index.store import IndexStore

DEFAULT_LIMIT = 20
SCORE_CHUNK = 4096

_DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _day_number(day: object, name: str) -> int:
    """A local day "YYYY-MM-DD" as a number that sorts like the day. Other text is a ValueError."""
    if not isinstance(day, str) or not _DAY.fullmatch(day):
        raise ValueError(f"{name} must be a day, YYYY-MM-DD")
    try:
        date.fromisoformat(day)
    except ValueError:
        raise ValueError(f"{name} must be a day, YYYY-MM-DD") from None
    return int(day.replace("-", ""))


def _day_of(timestamp: str) -> int:
    try:
        return int(timestamp[:10].replace("-", ""))
    except ValueError:
        raise StoreError("a segment of the index has a time that can not be read") from None


def _id_list(value: object) -> list[str]:
    """One id or several ids, as the archive takes them."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (bytes, dict)) or not isinstance(value, Iterable):
        raise TypeError("conversations must be an id or a list of ids")
    items = list(value)
    if not all(isinstance(item, str) for item in items):
        raise TypeError("conversations must be an id or a list of ids")
    return list(dict.fromkeys(items))


class MeaningIndex:
    """One loaded index. `segments[n]` belongs to row `n` of `vectors`."""

    def __init__(self, meta: IndexMeta, segments: Sequence[Segment], vectors: np.ndarray) -> None:
        if vectors.shape != (len(segments), meta.dimensions):
            raise ValueError("the vectors do not fit the segments and the dimensions")
        self._meta = meta
        self._segments = tuple(segments)
        self._vectors = vectors
        self._first_days = np.array([_day_of(s.first_timestamp) for s in self._segments], dtype=np.int64)
        self._last_days = np.array([_day_of(s.last_timestamp) for s in self._segments], dtype=np.int64)
        rows: dict[str, list[int]] = {}
        for row, segment in enumerate(self._segments):
            rows.setdefault(segment.conversation_id, []).append(row)
        self._rows = {key: np.array(value, dtype=np.int64) for key, value in rows.items()}

    @classmethod
    def load(cls, store: IndexStore) -> MeaningIndex | None:
        """The index of a store, or None when the store holds no whole build."""
        stored = store.read()
        if stored is None:
            return None
        return cls(stored.meta, stored.segments, stored.vectors)

    @property
    def meta(self) -> IndexMeta:
        return self._meta

    @property
    def model(self) -> str:
        return self._meta.model

    @property
    def dimensions(self) -> int:
        return self._meta.dimensions

    def __len__(self) -> int:
        return len(self._segments)

    def _query(self, vector: object) -> np.ndarray:
        try:
            query = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError):
            raise ValueError("the query must be a vector of numbers") from None
        if query.shape != (self.dimensions,):
            raise ValueError(f"the query must have {self.dimensions} dimensions")
        length = float(np.linalg.norm(query))
        if not np.isfinite(query).all() or length == 0.0:
            raise ValueError("the query must be a vector with a length")
        return query / length

    def _chosen(self, conversations: object, since: str | None, until: str | None) -> np.ndarray | None:
        """The rows that pass the filters, in order. None means every row."""
        rows: np.ndarray | None = None
        if conversations is not None:
            found = [self._rows[key] for key in _id_list(conversations) if key in self._rows]
            rows = np.sort(np.concatenate(found)) if found else np.empty(0, dtype=np.int64)
        if since is None and until is None:
            return rows
        keep = np.ones(len(self._segments), dtype=bool)
        if since is not None:
            keep &= self._last_days >= _day_number(since, "since")
        if until is not None:
            keep &= self._first_days <= _day_number(until, "until")
        return np.flatnonzero(keep) if rows is None else rows[keep[rows]]

    def _scores(self, query: np.ndarray, rows: np.ndarray | None) -> np.ndarray:
        count = len(self._segments) if rows is None else len(rows)
        scores = np.empty(count, dtype=np.float32)
        for start in range(0, count, SCORE_CHUNK):
            stop = min(count, start + SCORE_CHUNK)
            chunk = self._vectors[start:stop] if rows is None else self._vectors[rows[start:stop]]
            scores[start:stop] = np.asarray(chunk, dtype=np.float32) @ query
        return scores

    def search(
        self,
        vector: Sequence[float],
        *,
        conversations: str | Iterable[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[SearchHit]:
        """The `limit` nearest segments, nearest first.

        `conversations` is one id or a list of ids; `[]` matches nothing.
        `since` and `until` are local days, YYYY-MM-DD: a segment matches when
        it overlaps the days, both included. The score is the dot product of
        the stored vector and the query made unit length. Equal scores come in
        the order of the rows.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a whole number, 1 or more")
        query = self._query(vector)
        rows = self._chosen(conversations, since, until)
        scores = self._scores(query, rows)
        if len(scores) > limit:
            floor = np.partition(scores, len(scores) - limit)[len(scores) - limit]
            places = np.flatnonzero(scores >= floor)
        else:
            places = np.arange(len(scores))
        order = places[np.lexsort((places, -scores[places]))][:limit]
        return [
            SearchHit(segment=self._segments[int(place if rows is None else rows[place])], score=float(scores[place]))
            for place in order
        ]
