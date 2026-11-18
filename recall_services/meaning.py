"""The index of meanings for the routes: its status, the cost of a build, and the build.

`status` says what the index holds: `none`, `ready`, or `behind` when the
archive holds a later message with text than the index covers. `estimate`
says what a build that starts now would send and cost. `build` builds the
index, or brings it up to date: it embeds only what is new. `payload` is the
answer of `GET /api/meaning`.

Each archive has an index of its own: `index_folder` gives its folder, and no
other code builds that path. `meaning_index` does the work: this module gives
it the messages, the key and the model facts, and turns its results into the
shapes of the routes.

Rules:
- The text of a message is read as a report reads it: each handle that is not
  saved is masked (`reports._Masked`). So no such handle goes to the provider.
- A status and an estimate make no request and no file. They need no key.
- A build asks for the key when the first text must be embedded. A build with
  nothing new needs no key.
- Each failure is a `MeaningFailure`: a code and one plain sentence. What the
  provider said never goes to the person.
- No line of the log, no line for a job and no sentence holds text of a
  message, a name, a handle or an id. The lines for a job hold counts only.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from archive import Archive
from archive import ArchiveError as ArchiveFault
from meaning_index import (
    ArchiveSource,
    BatchLimits,
    BuildProgress,
    Embedder,
    EmbedderFacts,
    IndexStatus,
    IndexStore,
    MeaningIndexError,
    SourceMessage,
    StoreError,
    build_index,
    cut_segments,
    default_count_tokens,
    embedder_facts,
    estimate_build,
)
from recall_services import keys, reports
from recall_services.meaning_failures import MeaningFailure as MeaningFailure
from recall_services.meaning_failures import NoKey as NoKey
from recall_services.meaning_failures import failed
from recall_services.meaning_failures import failure_of as failure_of
from recall_services.names import Names
from report_engine import ReportEngineError

logger = logging.getLogger(__name__)

STORE_LEAF = "meaning"
FOLDER_NAME_CHARS = 16
BUILD_TIMEOUT_SECONDS = 120.0
# The server brings an index up to date by itself only when a build would embed
# at most this many segments, or at most this share of the segments, whichever is more.
REFRESH_SEGMENTS = 2_000
REFRESH_SHARE = 0.10
# the three steps of a build, as a job shows them
STEPS = {
    "read": (1, "Reading your conversations"),
    "embed": (2, "Reading the meaning of your messages"),
    "write": (3, "Saving the index"),
}
_FAULTS = (MeaningIndexError, ArchiveFault, ReportEngineError)

EmbedderFactory = Callable[[str, int], Embedder]
Listener = Callable[[dict[str, Any]], None]
CountTokens = Callable[[str], int]


def index_folder(archive_file: str | os.PathLike[str]) -> Path:
    """The folder of the index of one archive: `meaning/<name>` in the saves folder, read at each call.

    The name is the first 16 hex digits of the SHA-256 of the resolved path of
    the archive file. No other code builds this path.
    """
    import recall_paths

    resolved = str(Path(archive_file).expanduser().resolve())
    name = hashlib.sha256(resolved.encode("utf-8", "surrogatepass")).hexdigest()[:FOLDER_NAME_CHARS]
    return Path(recall_paths.SAVES_DIR) / STORE_LEAF / name


def index_store(archive_file: str | os.PathLike[str]) -> IndexStore:
    """The index of one archive."""
    return IndexStore(index_folder(archive_file))


def default_facts() -> EmbedderFacts:
    """The model and the dimensions that the app builds with, and the price of the model."""
    return embedder_facts()


def embedder_for(model: str, dimensions: int, *, timeout_s: float = BUILD_TIMEOUT_SECONDS) -> Embedder:
    """The embedder of the app. The SDK is loaded here. The client goes to the address of the provider only."""
    from meaning_index.openai_embedder import OpenAIEmbedder
    from report_engine.openai_llm import build_client

    key = keys.api_key()
    if not key:
        raise NoKey("no key is set")
    return OpenAIEmbedder(build_client(key, timeout_s=timeout_s), model, dimensions)


class _OnDemand:
    """The embedder of a build, made when the first text must be embedded."""

    def __init__(self, factory: EmbedderFactory, facts: EmbedderFacts) -> None:
        self.model = facts.model
        self.dimensions = facts.dimensions
        self._factory = factory
        self._made: Embedder | None = None

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self._made is None:
            self._made = self._factory(self.model, self.dimensions)
        return self._made.embed(texts)


class _Masked:
    """The messages of the archive for the index, with the text that a report sends.

    The rows come through `reports._Masked`, so a handle that is not saved is
    masked by the same function in a report, in a digest and here. A message
    with no text is in no segment. With `note_ids`, `segment_ids` holds the id
    of each segment that the messages read so far make.
    """

    def __init__(self, archive: Archive, names: Names, since: str | None = None, *, note_ids: bool = False) -> None:
        redact = names.chat_labeler(archive.conversations()).redact
        self._archive = archive
        self._rows = reports._Masked(archive, redact)
        self._since = since
        self._note_ids = note_ids
        self.segment_ids: set[str] = set()

    def conversation_ids(self) -> Sequence[str]:
        return ArchiveSource(self._archive).conversation_ids()

    def messages(self, conversation_id: str) -> Iterator[SourceMessage]:
        read: list[SourceMessage] = []
        for row in self._rows.messages(conversation_id, self._since, None):
            if not row["text"].strip():
                continue
            message: SourceMessage = {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "text": row["text"],
                "is_from_me": row["is_from_me"],
            }
            if self._note_ids:
                read.append(message)
            yield message
        self.segment_ids.update(cut.segment_id for cut in cut_segments(conversation_id, read))


def _held(store: IndexStore) -> IndexStatus:
    """What the folder holds. An index that this build of the app can not read counts as none.

    With no folder there is no index, and no file is asked for: an import
    that looks for an index opens nothing then.
    """
    if not store.folder.is_dir():
        return IndexStatus(state="none")
    try:
        return store.status()
    except StoreError as error:  # a newer format is one of these
        logger.warning("the index can not be read (%s); it counts as none", type(error).__name__)
        return IndexStatus(state="none")


def _shown(held: IndexStatus, state: str) -> dict[str, Any]:
    return {
        "state": state,
        "segments": held.segments,
        "model": held.model,
        "dimensions": held.dimensions,
        "builtAt": held.built_at,
        "coveredUntil": held.covered_until,
    }


def _later_text(archive: Archive, names: Names, covered_until: str | None) -> bool:
    """True when the archive holds a message that a build would index, later than the index covers."""
    covered = covered_until or ""
    later = [str(row["id"]) for row in archive.conversations() if str(row.get("last_timestamp") or "") > covered]
    if not later:
        return False
    source = _Masked(archive, names, covered[:10] or None)
    return any(message["timestamp"] > covered for chat in later for message in source.messages(chat))


def _stored_ids(store: IndexStore) -> set[str]:
    """The id of each segment in the store. No folder and no whole build hold none."""
    held = store.read() if store.folder.is_dir() else None
    return set() if held is None else {segment.segment_id for segment in held.segments}


def built(archive_file: str | os.PathLike[str], *, store: IndexStore | None = None) -> bool:
    """True when this archive has a whole index that this build of the app reads. It reads no archive."""
    return _held(store or index_store(archive_file)).state == "ready"


def status(archive: Archive, names: Names, *, store: IndexStore | None = None) -> dict[str, Any]:
    """`state`, `segments`, `model`, `dimensions`, `builtAt` and `coveredUntil` of the index.

    `state` is `none`, `ready` or `behind`. This is the quick look: `behind`
    means a later message with text. `payload` also knows what the estimate
    found. Raises MeaningFailure (`archive_error`).
    """
    held = _held(store or index_store(archive.path))
    if held.state != "ready":
        return _shown(held, held.state)
    try:
        behind = _later_text(archive, names, held.covered_until)
    except _FAULTS as error:
        raise failed(error, "the status of the index") from None
    return _shown(held, "behind" if behind else "ready")


def estimate(
    archive: Archive,
    names: Names,
    *,
    store: IndexStore | None = None,
    facts: EmbedderFacts | None = None,
    count_tokens: CountTokens | None = None,
) -> dict[str, Any]:
    """What a build that starts now would do: `segments`, `newSegments`, `staleSegments`, `tokens`,
    `estimatedCost` and `upperBound`.

    `newSegments` are embedded, and they cost. `staleSegments` are the
    segments whose id the store does not hold: a build writes them, and it
    pays only for those that are also new. `upperBound` is true when the
    vectors that a stopped build paid for could not be read: the build then
    sends at most this much. It reads every conversation. It makes no request
    and no file. Raises MeaningFailure.
    """
    try:
        chosen = store or index_store(archive.path)
        source = _Masked(archive, names, note_ids=True)
        found = estimate_build(source, chosen, facts or default_facts(), count_tokens or default_count_tokens)
        stale = len(source.segment_ids - _stored_ids(chosen))
    except _FAULTS as error:
        raise failed(error, "the estimate of an index build") from None
    return {
        "segments": found.segments,
        "newSegments": found.new_segments,
        "staleSegments": stale,
        "tokens": found.tokens,
        "estimatedCost": found.usd,
        "upperBound": bool(found.upper_bound),
    }


def small_enough(found: Mapping[str, Any]) -> bool:
    """True when the server can bring the index up to date by itself: a build embeds little.

    `found` is an estimate. Above the limit the reader starts the build, and
    sees the cost first.
    """
    return int(found["newSegments"]) <= max(REFRESH_SEGMENTS, REFRESH_SHARE * int(found["segments"]))


def payload(archive: Archive, names: Names, **more: Any) -> dict[str, Any]:
    """The answer of `GET /api/meaning`: `status`, `estimate`, `error` and `code`.

    The state is `behind` also when a build would embed or write a segment.
    When the status or the estimate fails, `estimate` is None, and `error` and
    `code` say why. `more` goes to `estimate`; `store` goes to both.
    """
    store = more.get("store") or index_store(archive.path)
    try:
        shown = status(archive, names, store=store)
        cost = estimate(archive, names, **{**more, "store": store})
    except MeaningFailure as failure:
        held = _held(store)
        return {"status": _shown(held, held.state), "estimate": None, "error": str(failure), "code": failure.code}
    changes = bool(cost["newSegments"] or cost["staleSegments"])
    state = "behind" if shown["state"] == "ready" and changes else shown["state"]
    return {"status": {**shown, "state": state}, "estimate": cost, "error": None, "code": None}


class _Teller:
    """Turns the steps of a build into updates for a listener. A listener that fails does not stop the build."""

    def __init__(self, listener: Listener | None) -> None:
        self._listener = listener
        self._warned = False

    def tell(self, update: dict[str, Any]) -> None:
        if self._listener is None:
            return
        try:
            self._listener(update)
        except Exception as error:  # noqa: BLE001 - the build is worth more than its progress line
            if not self._warned:
                logger.warning("the progress listener of an index build failed (%s)", type(error).__name__)
                self._warned = True

    def step(self, event: BuildProgress) -> None:
        number, label = STEPS[event.phase]
        progress = {"step": number, "of": len(STEPS), "label": label, "done": event.done, "total": event.total}
        update: dict[str, Any] = {"progress": progress}
        if event.done == 0:
            update["log"] = _started(event)
        self.tell(update)


def _count(number: int, word: str) -> str:
    return f"{number:,} {word}{'' if number == 1 else 's'}"


def _started(event: BuildProgress) -> str:
    """The line of a step that starts. An exchange is a segment of the index."""
    if event.phase == "read":
        return f"Reading {_count(event.total, 'conversation')}"
    if event.phase == "embed":
        return f"Reading the meaning of {_count(event.total, 'exchange')}" if event.total else "No new exchange to read"
    return f"Saving {_count(event.total, 'exchange')}"


def build(
    archive: Archive,
    names: Names,
    cancel: Any = None,
    on_progress: Listener | None = None,
    *,
    store: IndexStore | None = None,
    facts: EmbedderFacts | None = None,
    embedder_factory: EmbedderFactory | None = None,
    count_tokens: CountTokens | None = None,
    sleep: Callable[[float], None] = time.sleep,
    limits: BatchLimits | None = None,
) -> dict[str, Any]:
    """Build the index, or bring it up to date. Gives `segments`, `embedded`, `reused`, `removed` and `tokens`.

    `cancel` is a callable that returns a bool, or a token with `cancelled`.
    `on_progress` gets `{"progress": {"step", "of", "label", "done", "total"}}`,
    with a line for the job under `log` when a step starts and when the build ends.

    Raises MeaningFailure when no index was written. The index is then as it
    was, and the vectors that were paid for are kept for the next build.
    """
    teller = _Teller(on_progress)
    try:
        chosen = facts or default_facts()
        found = build_index(
            _Masked(archive, names),
            _OnDemand(embedder_factory or embedder_for, chosen),
            store or index_store(archive.path),
            progress=teller.step,
            cancel=cancel,
            count_tokens=count_tokens or default_count_tokens,
            sleep=sleep,
            limits=limits or BatchLimits(),
        )
    except _FAULTS as error:
        raise failed(error, "an index build") from None
    teller.tell(
        {
            "log": f"The index holds {_count(found.segments, 'exchange')}: {found.embedded:,} new, "
            f"{found.reused:,} kept, {found.removed:,} removed."
        }
    )
    return {
        "segments": found.segments,
        "embedded": found.embedded,
        "reused": found.reused,
        "removed": found.removed,
        "tokens": found.tokens,
    }
