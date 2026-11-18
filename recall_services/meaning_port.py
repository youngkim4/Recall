"""The search by meaning of chat: `chat_engine.ports.MeaningSearch` over the index of one archive.

`MeaningPort` embeds the description with the model and the dimensions of the
index. The process keeps one loaded index for each index folder, and the
vectors of the last descriptions.

Rules:
- The description that reaches `MeaningPort.search` is masked by the engine.
  Only the description goes to the embedding provider. The conversation ids
  are handles: they filter the search on this machine and go nowhere.
- Every failure is raised. The engine then ranks by words alone.
- No log line holds a description, a name, a handle or an id.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence

from chat_engine.ports import MeaningHit
from meaning_index import Embedder, IndexStore, MeaningIndex, MeaningIndexError, SearchHit
from recall_services import meaning

logger = logging.getLogger(__name__)

QUERY_VECTORS = 64
QUERY_TIMEOUT_SECONDS = 10.0
# segments read for each hit that is asked for: a repeat inside one conversation is one hit
FETCH_FACTOR = 2
LOAD_TRIES = 3

EmbedderFactory = Callable[[str, int], Embedder]
_VectorKey = tuple[str, int, str]


class MeaningNotReady(Exception):
    """The folder holds no whole index, and the process loaded none before."""


def _query_embedder(model: str, dimensions: int) -> Embedder:
    """The embedder of a description. A question does not wait as long as a build does."""
    return meaning.embedder_for(model, dimensions, timeout_s=QUERY_TIMEOUT_SECONDS)


def _load(store: IndexStore) -> MeaningIndex | None:
    """One whole build, or None while a build swaps the files. Files of two builds are never read as one index."""
    for _ in range(LOAD_TRIES):
        index = MeaningIndex.load(store)
        if index is None:
            return None
        now = store.meta()
        if now is not None and now.build_id == index.meta.build_id:
            return index
    return None


def _one_vector(reply: Sequence[Sequence[float]], dimensions: int) -> tuple[float, ...]:
    """The vector of one description. A reply that a search can not use is not kept."""
    rows = list(reply)
    if len(rows) != 1:
        raise ValueError("the embedder must give one vector for one description")
    vector = tuple(float(value) for value in rows[0])
    if len(vector) != dimensions or not all(math.isfinite(value) for value in vector):
        raise ValueError("the embedder must give one vector of finite numbers with the dimensions of the index")
    return vector


class _Kept:
    """What the process keeps for one index folder: the loaded index and the vectors of the last descriptions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._index: MeaningIndex | None = None
        self._vectors: OrderedDict[_VectorKey, tuple[float, ...]] = OrderedDict()

    def index(self, store: IndexStore) -> MeaningIndex | None:
        """The index of the build on disk, loaded again when that build changed.

        While a build swaps the files the folder holds no whole build: the
        index that was loaded before answers then. None when there is none.
        """
        with self._lock:
            on_disk = store.meta()
            held = self._index
            if on_disk is None or (held is not None and held.meta.build_id == on_disk.build_id):
                return held
            try:
                fresh = _load(store)
            except MeaningIndexError as error:
                if held is None:
                    raise
                logger.warning("the new index can not be read (%s); the one before it answers", type(error).__name__)
                return held
            if fresh is not None:  # an index with no segment is an index
                self._index = fresh
            return self._index

    def vector(self, index: MeaningIndex, query: str, factory: EmbedderFactory) -> tuple[float, ...]:
        """The vector of a description. One of the last QUERY_VECTORS descriptions is not embedded again."""
        text = hashlib.sha256(query.encode("utf-8", "surrogatepass")).hexdigest()
        key = (index.model, index.dimensions, text)
        with self._lock:
            found = self._vectors.get(key)
            if found is not None:
                self._vectors.move_to_end(key)
                return found
        made = _one_vector(factory(index.model, index.dimensions).embed([query]), index.dimensions)
        with self._lock:
            self._vectors[key] = made
            self._vectors.move_to_end(key)
            while len(self._vectors) > QUERY_VECTORS:
                self._vectors.popitem(last=False)
        return made


_KEPT: dict[str, _Kept] = {}
_KEPT_LOCK = threading.Lock()


def _kept_for(store: IndexStore) -> _Kept:
    with _KEPT_LOCK:
        return _KEPT.setdefault(str(store.folder), _Kept())


def ready(store: IndexStore) -> bool:
    """True when a search of this folder can answer now. A folder that can not be read raises."""
    return _kept_for(store).index(store) is not None


def forget() -> None:
    """Let go of every loaded index and every vector of a description."""
    with _KEPT_LOCK:
        _KEPT.clear()


def _hits(found: Sequence[SearchHit], limit: int) -> list[MeaningHit]:
    """The best `limit` hits, best first.

    A hit with the content hash of a better hit of the same conversation is
    left out: the same words said again. The hash holds no conversation, so
    the same words in two conversations are two hits.
    """
    seen: set[tuple[str, str]] = set()
    rows: list[MeaningHit] = []
    for hit in found:
        segment = hit.segment
        said = (segment.conversation_id, segment.content_hash)
        if said in seen:
            continue
        seen.add(said)
        rows.append(
            {
                "conversation": segment.conversation_id,
                "first_message_id": segment.first_message_id,
                "last_message_id": segment.last_message_id,
                "score": float(hit.score),
            }
        )
        if len(rows) >= limit:
            break
    return rows


class MeaningPort:
    """`chat_engine.ports.MeaningSearch` over the index of one folder."""

    def __init__(self, store: IndexStore, embedder_factory: EmbedderFactory | None = None) -> None:
        self._store = store
        self._factory = embedder_factory or _query_embedder
        self._kept = _kept_for(store)

    def search(
        self,
        query: str,
        *,
        conversations: Sequence[str] | None,
        since: str | None,
        until: str | None,
        limit: int,
    ) -> list[MeaningHit]:
        """The exchanges nearest to the description, best first. `since` and `until` are days, YYYY-MM-DD."""
        if limit < 1:
            return []
        index = self._kept.index(self._store)
        if index is None:
            raise MeaningNotReady("there is no index to search")
        vector = self._kept.vector(index, query, self._factory)
        found = index.search(
            vector, conversations=conversations, since=since, until=until, limit=limit * FETCH_FACTOR
        )
        return _hits(found, limit)
