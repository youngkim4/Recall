"""The search of the people index: profiles ranked by meaning, for chat.

`search` embeds the description with the model and the dimensions of the
store, and ranks the profiles by cosine (each vector has length 1). It keeps
only the kinds asked for, and the profiles whose span (first to last day)
overlaps `since` to `until`. The caller masks the description first.

It never fails a question. With no profile, a store that can not be read, or
a description that can not be embedded, it gives an empty list whose
`reason` says why: `no_index`, `no_key` or `embed_failed`. `reason` is None
when the search ran. It raises ValueError only for arguments that can not be
used: a kind that is not one of `KINDS`, a day that is not `YYYY-MM-DD`,
`since` after `until`, a `limit` that is not a whole number from 1 to 50.

A process keeps the store that it read last, for each file, and reads it
again only when the file changed.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Sequence
from datetime import date
from typing import Any

import numpy as np

from archive import Archive
from recall_services import meaning
from recall_services.meaning_failures import NoKey
from recall_services.names import Names
from recall_services.people_prompt import KINDS
from recall_services.people_store import Held, PeopleStore, PeopleStoreError, Profile, people_folder, profile_text, text_hash

logger = logging.getLogger(__name__)

NO_INDEX, NO_KEY, EMBED_FAILED = "no_index", "no_key", "embed_failed"
MOST_ROWS = 50
SEARCH_TIMEOUT_SECONDS = 20.0
SCORE_DECIMALS = 4
DAY_CHARS = 10

_loaded: dict[str, tuple[tuple[int, int] | None, Held | None]] = {}
_loaded_lock = threading.Lock()


class Found(list):  # type: ignore[type-arg]
    """The rows of a search, best first, and why there are none when that is so."""

    def __init__(self, rows: Iterable[dict[str, Any]] = (), reason: str | None = None) -> None:
        super().__init__(rows)
        self._reason = reason

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def index_missing(self) -> bool:
        return self._reason == NO_INDEX


def _kinds(kinds: Sequence[str] | None) -> frozenset[str] | None:
    if kinds is None:
        return None
    if isinstance(kinds, str) or not all(isinstance(kind, str) and kind in KINDS for kind in kinds):
        raise ValueError(f"kinds must be a list of these words: {', '.join(KINDS)}")
    return frozenset(kinds)


def _day(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    try:
        if not isinstance(value, str) or len(value) != DAY_CHARS:
            raise ValueError
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise ValueError(f"{name} must be a day as YYYY-MM-DD") from None


def _limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MOST_ROWS:
        raise ValueError(f"limit must be a whole number from 1 to {MOST_ROWS}")
    return limit


def _held(store: PeopleStore) -> Held | None:
    """The store as it is on disk, read again only when the file changed. None when it can not be read."""
    signature = store.signature()
    key = str(store.path)
    with _loaded_lock:
        found = _loaded.get(key)
        if found is not None and found[0] == signature:
            return found[1]
    try:
        held = store.read()
    except PeopleStoreError as error:
        logger.warning("the people index can not be read (%s)", type(error).__name__)
        held = None
    with _loaded_lock:
        _loaded[key] = (signature, held)
    return held


def _searchable(held: Held) -> list[Profile]:
    """The rows with a vector of the embedding model and the dimensions of the store."""
    meta = held.meta
    return [
        profile
        for profile in held.profiles
        if profile.vector is not None
        and profile.text_hash == text_hash(profile_text(profile), meta.embed_model, meta.dimensions)
    ]


def _overlaps(profile: Profile, since: str | None, until: str | None) -> bool:
    first, last = profile.figures.first_day, profile.figures.last_day
    if first is None or last is None:
        return since is None and until is None
    return (since is None or last >= since) and (until is None or first <= until)


def _row(profile: Profile, score: float) -> dict[str, Any]:
    return {
        "key": profile.key,
        "conversations": list(profile.conversations),
        "label": profile.label,
        "kind": profile.kind,
        "status": profile.status,
        "summary": profile.summary,
        "turning_points": [{"date": point.date, "what": point.what} for point in profile.turning_points],
        "figures": profile.figures.as_dict(),
        "score": round(float(score), SCORE_DECIMALS),
    }


def _app_embedder(model: str, dimensions: int) -> Any:
    return meaning.embedder_for(model, dimensions, timeout_s=SEARCH_TIMEOUT_SECONDS)


def _query(description: str, held: Held, factory: Callable[[str, int], Any]) -> np.ndarray | str:
    """The vector of the description, or the reason why there is none."""
    try:
        embedder = factory(held.meta.embed_model, held.meta.dimensions)
        values = np.asarray(embedder.embed([description])[0], dtype="<f4")
    except NoKey:
        return NO_KEY
    except Exception as error:  # noqa: BLE001 - a search never fails a question
        logger.warning("the description of a people search could not be embedded (%s)", type(error).__name__)
        return EMBED_FAILED
    length = float(np.linalg.norm(values))
    return values / length if length > 0 else values


def search(
    archive: Archive,
    names: Names,
    description: str,
    *,
    kinds: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 10,
    store: PeopleStore | None = None,
    embedder_factory: Callable[[str, int], Any] | None = None,
) -> Found:
    """The profiles nearest to the description, best first. `names` is for the adapters; the labels are stored."""
    if not isinstance(description, str) or not description.strip():
        raise ValueError("a description is needed")
    wanted, first, last, size = _kinds(kinds), _day(since, "since"), _day(until, "until"), _limit(limit)
    if first is not None and last is not None and first > last:
        raise ValueError("since is after until")
    held = _held(store or PeopleStore(people_folder(archive.path)))
    if held is None or not held.profiles:
        return Found((), NO_INDEX)
    chosen = [
        profile
        for profile in _searchable(held)
        if (wanted is None or profile.kind in wanted) and _overlaps(profile, first, last)
    ]
    if not chosen:
        return Found(())
    query = _query(description, held, embedder_factory or _app_embedder)
    if isinstance(query, str):
        return Found((), query)
    vectors = np.stack([np.frombuffer(profile.vector or b"", dtype="<f4") for profile in chosen])
    scores = vectors @ query
    order = sorted(range(len(chosen)), key=lambda place: (-float(scores[place]), place))[:size]
    return Found(_row(chosen[place], scores[place]) for place in order)
