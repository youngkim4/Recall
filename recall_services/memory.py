"""The two optional ports of chat: the search by meaning and the digest of one conversation.

`ports(archive, names)` gives `{"meaning": ..., "digests": ...}` for
`ChatEngine`. A port that can not work is None: the engine then does not offer
its tool, and nothing else changes.

- `MeaningPort` is `chat_engine.ports.MeaningSearch` over `meaning_index`. It
  embeds the description with the model and the dimensions of the index. The
  process keeps the loaded index and the vectors of the last descriptions.
- `DigestPort` is `chat_engine.ports.DigestSource` over
  `recall_services.digests`. A digest is wired as a report is wired, so the
  two share the notes of the parts that were read.

Rules:
- The description that reaches `MeaningPort.search` is masked by the engine.
  Only the description goes to the embedding provider. The conversation ids
  are handles: they filter the search on this machine and go nowhere.
- Every failure of a port is raised. The engine then ranks by words alone, or
  prints its one line for a digest. No failure here fails a question.
- No log line holds a description, a name, a handle or an id.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from archive import Archive
from chat_engine.cancel import CancelToken
from chat_engine.ports import DigestCost, DigestMoment, DigestPeriod, DigestResult, MeaningHit
from meaning_index import Embedder, IndexStore, MeaningIndex, MeaningIndexError, SearchHit
from meaning_index.store_files import usable_meta
from recall_services import digests, keys, meaning, models, reports
from recall_services.names import Names
from report_engine import CancelToken as ReportCancelToken
from report_engine import PeriodCache, ReportOptions, Runtime, StoreError
from report_engine.chat_digest import MomentNote, PeriodNote
from report_engine.tokens import TokenCounter

logger = logging.getLogger(__name__)

# The model of a digest: an id of models.py, or "" for the model of the reports.
# WARNING: the notes of a part belong to one model. With another model than the
# reports use, a digest pays again for each part that a report read.
DIGEST_MODEL = ""
QUERY_VECTORS = 64
QUERY_TIMEOUT_SECONDS = 10.0
# segments read for each hit that is asked for: the same words said again are one hit
FETCH_FACTOR = 2
LOAD_TRIES = 3

EmbedderFactory = Callable[[str, int], Embedder]
_VectorKey = tuple[str, int, str]


class MeaningNotReady(Exception):
    """The folder holds no whole index, and the process loaded none before."""


@dataclass(frozen=True)
class DigestSettings:
    """What a digest of chat is built with. Each None is the value of a report run of the app.

    WARNING: a digest shares the notes of a report only when both use the same
    model, options, counter and places.
    """

    model: str | None = None  # None: DIGEST_MODEL
    places: reports.ReportPlaces | None = None
    options: ReportOptions | None = None
    counter: TokenCounter | None = None
    llm_factory: reports.LLMFactory | None = None
    runtime: Runtime | None = None


def _query_embedder(model: str, dimensions: int) -> Embedder:
    """The embedder of a description. A question does not wait as long as a build does."""
    return meaning.embedder_for(model, dimensions, timeout_s=QUERY_TIMEOUT_SECONDS)


def _load(store: IndexStore) -> MeaningIndex | None:
    """One whole build, or None while a build swaps the files. Files of two builds are never read as one index."""
    for _ in range(LOAD_TRIES):
        index = MeaningIndex.load(store)
        if index is None:
            return None
        now = usable_meta(store.folder)
        if now is not None and now.build_id == index.meta.build_id:
            return index
    return None


def _one_vector(reply: Sequence[Sequence[float]]) -> tuple[float, ...]:
    rows = list(reply)
    if len(rows) != 1:
        raise ValueError("the embedder must give one vector for one description")
    return tuple(float(value) for value in rows[0])


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
            on_disk = usable_meta(store.folder)
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
        made = _one_vector(factory(index.model, index.dimensions).embed([query]))
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


def forget() -> None:
    """Let go of every loaded index and every vector of a description."""
    with _KEPT_LOCK:
        _KEPT.clear()


def _hits(found: Sequence[SearchHit], limit: int) -> list[MeaningHit]:
    """The best `limit` hits, best first. A hit with the content hash of a better hit is left out."""
    seen: set[str] = set()
    rows: list[MeaningHit] = []
    for hit in found:
        segment = hit.segment
        if segment.content_hash in seen:
            continue
        seen.add(segment.content_hash)
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


@contextmanager
def _report_cancel(cancel: CancelToken | None) -> Iterator[ReportCancelToken]:
    """A cancel token of the report engine that is set when the token of chat is cancelled."""
    token = ReportCancelToken()
    forget_it = cancel.on_cancel(token.cancel) if cancel is not None else None
    try:
        yield token
    finally:
        if forget_it is not None:
            forget_it()


def _moment(note: MomentNote) -> DigestMoment:
    return {
        "date": note.date,
        "title": note.title,
        "detail": note.detail,
        "quote": note.quote,
        "message_ids": list(note.message_ids),
    }


def _period(note: PeriodNote) -> DigestPeriod:
    return {
        "label": note.label,
        "first_day": note.first_day,
        "last_day": note.last_day,
        "messages": note.messages,
        "summary": note.summary,
        "moments": [_moment(moment) for moment in note.moments],
    }


class DigestPort:
    """`chat_engine.ports.DigestSource` over `recall_services.digests`."""

    def __init__(self, archive: Archive, names: Names, settings: DigestSettings | None = None) -> None:
        self._archive = archive
        self._names = names
        self._settings = settings or DigestSettings()

    @property
    def model(self) -> str:
        """The model id that goes to `digests`: "" is the model of the reports."""
        return DIGEST_MODEL if self._settings.model is None else self._settings.model

    def _same(self) -> dict[str, Any]:
        """What the estimate and the digest both get, so that they plan the same parts."""
        chosen = self._settings
        return {
            "archive": self._archive,
            "names": self._names,
            "model": self.model,
            "places": chosen.places,
            "options": chosen.options,
            "counter": chosen.counter,
        }

    def estimate(self, conversation_id: str, *, since: str | None, until: str | None) -> DigestCost:
        """The parts of the digest and the cost of those that are not read yet. No model call."""
        found = digests.estimate(conversation_id, since, until, **self._same())
        return {"periods": found.periods, "periods_cached": found.periods_cached, "usd": found.usd}

    def digest(
        self, conversation_id: str, *, since: str | None, until: str | None, cancel: CancelToken
    ) -> DigestResult:
        """Build the digest. It stops before its next model call when `cancel` is cancelled."""
        chosen = self._settings
        with _report_cancel(cancel) as token:
            found = digests.build(
                conversation_id,
                since,
                until,
                token,
                llm_factory=chosen.llm_factory,
                runtime=chosen.runtime,
                **self._same(),
            )
        logger.info(
            "a digest for chat: %d parts read now, %d from the notes, %d failed",
            found.periods_built, found.periods_cached, found.periods_failed,
        )  # fmt: skip
        return {"periods": [_period(note) for note in found.periods], "periods_failed": found.periods_failed}


def _meaning_port(store: IndexStore, factory: EmbedderFactory | None) -> MeaningPort | None:
    """The port when an index can be searched now. A folder that can not be used gives None and one log line."""
    port = MeaningPort(store, factory)
    try:
        ready = _kept_for(store).index(store) is not None
    except (MeaningIndexError, OSError, ValueError) as error:
        logger.warning("the index can not be used (%s); chat searches by words only", type(error).__name__)
        return None
    return port if ready else None


def _digest_port(archive: Archive, names: Names, settings: DigestSettings) -> DigestPort | None:
    """The port when a digest can be built now: its model can be run, and the notes of earlier runs can be opened."""
    port = DigestPort(archive, names, settings)
    notes = (settings.places or reports.default_places()).cache_path
    try:
        models.model_info(port.model.strip() or models.default_id())
        if notes.exists():  # a check makes no file: the first digest makes the notes
            PeriodCache(notes)
    except (models.ModelChoiceError, models.ModelFactsError, StoreError) as error:
        logger.warning("a digest can not be built (%s); chat has no digest", type(error).__name__)
        return None
    return port


def ports(
    archive: Archive,
    names: Names,
    *,
    store: IndexStore | None = None,
    embedder_factory: EmbedderFactory | None = None,
    digest: DigestSettings | None = None,
) -> dict[str, Any]:
    """`{"meaning": ..., "digests": ...}` for `ChatEngine`, for one question.

    `meaning` is None when no key is set, or when no index can be searched.
    `digests` is None when no key is set, when the model of the digest can not
    be run, or when the notes of earlier report runs can not be opened. A test
    gives `store`, `embedder_factory` and `digest`.
    """
    if not keys.api_key():
        return {"meaning": None, "digests": None}
    return {
        "meaning": _meaning_port(store or meaning.index_store(), embedder_factory),
        "digests": _digest_port(archive, names, digest or DigestSettings()),
    }
