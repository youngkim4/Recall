"""The index of meanings for the routes: its status, the cost of a build, and the build.

`status` says what the index holds: `none`, `ready`, or `behind` when the
archive holds a later message with text than the index covers. `estimate`
says what a build that starts now would send and cost. `build` builds the
index, or brings it up to date: it embeds only what is new. `payload` is the
answer of `GET /api/meaning`.

The index is in the folder `meaning` of the saves folder. `meaning_index`
does the work: this module gives it the archive, the key and the model facts,
and turns its results into the shapes of the routes.

Rules:
- A status and an estimate make no request and no file. They need no key.
- A build asks for the key when the first text must be embedded. A build with
  nothing new needs no key.
- Each failure is a `MeaningFailure`: a code and one plain sentence. What the
  provider said never goes to the person.
- No line of the log, no line for a job and no sentence holds text of a
  message, a name, a handle or an id. The lines for a job hold counts only.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from archive import Archive
from archive import ArchiveError as ArchiveFault
from meaning_index import (
    ArchiveSource,
    BatchLimits,
    BuildCancelled,
    BuildProgress,
    Embedder,
    EmbedderFacts,
    EmbeddingError,
    EmbeddingKeyError,
    EmbeddingNetworkError,
    EmbeddingRateLimited,
    IndexBusyError,
    IndexFormatError,
    IndexStatus,
    IndexStore,
    MeaningIndexError,
    SourceError,
    StoreError,
    build_index,
    default_count_tokens,
    embedder_facts,
    estimate_build,
)
from recall_services import keys

logger = logging.getLogger(__name__)

STORE_LEAF = "meaning"
# the embedder of the app is the adapter of this provider
PROVIDER = "OpenAI"
BUILD_TIMEOUT_SECONDS = 120.0
AGAIN = "then build the index again."
NO_KEY = f"No API key is set. Add {keys.KEY_NAME} to {keys.KEY_PLACE}, {AGAIN}"
KEY_REJECTED = f"{PROVIDER} rejected the API key. Check {keys.KEY_NAME} in {keys.KEY_PLACE}, {AGAIN}"
RATE_LIMIT = f"{PROVIDER} is limiting requests, or the account has no credit left. Wait a moment, {AGAIN}"
OFFLINE = f"{PROVIDER} could not be reached. Check the internet connection, {AGAIN}"
MODEL_ERROR = "The embedding model did not answer. What was read is kept: build the index again."
BUSY = "The index is being built now. Wait until that ends."
CANCELLED = "The build was stopped. What was read is kept for the next build."
ARCHIVE_UNREADABLE = f"The message archive could not be read. Update the archive, {AGAIN}"
INDEX_NEWER = "A newer version of Recall made this index. Update Recall, then try again."
INDEX_ERROR = f"The index could not be written. Check the free space on the disk, {AGAIN}"
BUILD_FAILED = "The index could not be built. Build it again."
_TOO_MANY = 429
# the three steps of a build, as a job shows them
STEPS = {
    "read": (1, "Reading your conversations"),
    "embed": (2, "Reading the meaning of your messages"),
    "write": (3, "Saving the index"),
}

EmbedderFactory = Callable[[str, int], Embedder]
Listener = Callable[[dict[str, Any]], None]
CountTokens = Callable[[str], int]


class MeaningFailure(Exception):
    """A status, an estimate or a build that gave no result. The message is the sentence for the person."""

    def __init__(self, code: str, sentence: str) -> None:
        super().__init__(sentence)
        self.code = code


class NoKey(EmbeddingKeyError):
    """No key is set. No request was made."""


def failure_of(error: BaseException) -> MeaningFailure:
    """The failure for the person. It holds nothing of what the error said."""
    if isinstance(error, NoKey):
        return MeaningFailure("no_key", NO_KEY)
    if isinstance(error, EmbeddingKeyError):
        return MeaningFailure("key_rejected", KEY_REJECTED)
    if isinstance(error, EmbeddingRateLimited) or getattr(error, "status_code", None) == _TOO_MANY:
        return MeaningFailure("rate_limit", RATE_LIMIT)
    if isinstance(error, EmbeddingNetworkError):
        return MeaningFailure("offline", OFFLINE)
    if isinstance(error, EmbeddingError):
        return MeaningFailure("model_error", MODEL_ERROR)
    if isinstance(error, IndexBusyError):
        return MeaningFailure("busy", BUSY)
    if isinstance(error, BuildCancelled):
        return MeaningFailure("cancelled", CANCELLED)
    if isinstance(error, (SourceError, ArchiveFault)):
        return MeaningFailure("archive_error", ARCHIVE_UNREADABLE)
    if isinstance(error, IndexFormatError):
        return MeaningFailure("index_newer", INDEX_NEWER)
    if isinstance(error, StoreError):
        return MeaningFailure("index_error", INDEX_ERROR)
    return MeaningFailure("build_failed", BUILD_FAILED)


def _failed(error: BaseException, doing: str) -> MeaningFailure:
    """The failure for the person, and one line for the log: the type of the error and the facts of the service.

    The caller does not chain the error: what an archive raises can hold an id.
    """
    facts = ""
    if isinstance(error, EmbeddingError):
        facts = f", HTTP {error.status_code}, code {error.code}"
    logger.warning("%s failed: %s%s", doing, type(error).__name__, facts)
    return failure_of(error)


def index_store() -> IndexStore:
    """The index of the app: the folder `meaning` in the saves folder, read at each call."""
    import recall_paths

    return IndexStore(Path(recall_paths.SAVES_DIR) / STORE_LEAF)


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


def _held(store: IndexStore) -> IndexStatus:
    """What the folder holds. An index of a newer build of the app is not read: it counts as none."""
    try:
        return store.status()
    except IndexFormatError:
        logger.warning("the index was made by a newer version of the app; it is not read")
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


class _Since:
    """The archive as `ArchiveSource` reads it, from one day on."""

    def __init__(self, archive: Archive, day: str | None) -> None:
        self._archive = archive
        self._day = day

    def iter_messages(self, conversation_id: str) -> Any:
        return self._archive.iter_messages(conversation_id, since=self._day)


def _later_text(archive: Archive, covered_until: str | None) -> bool:
    """True when the archive holds a message that a build would index, later than the index covers."""
    covered = covered_until or ""
    source = ArchiveSource(_Since(archive, covered[:10] or None))
    for conversation in archive.conversations():
        if str(conversation.get("last_timestamp") or "") <= covered:
            continue
        if any(message["timestamp"] > covered for message in source.messages(str(conversation["id"]))):
            return True
    return False


def built(store: IndexStore | None = None) -> bool:
    """True when the folder holds a whole index that this build of the app reads. It reads no archive."""
    return _held(store or index_store()).state == "ready"


def status(archive: Archive, *, store: IndexStore | None = None) -> dict[str, Any]:
    """`state`, `segments`, `model`, `dimensions`, `builtAt` and `coveredUntil` of the index.

    `state` is `none`, `ready` or `behind`. Raises MeaningFailure (`archive_error`).
    """
    held = _held(store or index_store())
    if held.state != "ready":
        return _shown(held, held.state)
    try:
        behind = _later_text(archive, held.covered_until)
    except (MeaningIndexError, ArchiveFault) as error:
        raise _failed(error, "the status of the index") from None
    return _shown(held, "behind" if behind else "ready")


def estimate(
    archive: Archive,
    *,
    store: IndexStore | None = None,
    facts: EmbedderFacts | None = None,
    count_tokens: CountTokens | None = None,
) -> dict[str, Any]:
    """`segments`, `newSegments`, `tokens` and `estimatedCost` of a build that starts now.

    It reads every conversation. It makes no request and no file. Raises MeaningFailure.
    """
    try:
        found = estimate_build(
            ArchiveSource(archive),
            store or index_store(),
            facts or default_facts(),
            count_tokens or default_count_tokens,
        )
    except (MeaningIndexError, ArchiveFault) as error:
        raise _failed(error, "the estimate of an index build") from None
    return {
        "segments": found.segments,
        "newSegments": found.new_segments,
        "tokens": found.tokens,
        "estimatedCost": found.usd,
    }


def payload(archive: Archive, **more: Any) -> dict[str, Any]:
    """The answer of `GET /api/meaning`: `status`, `estimate`, `error` and `code`.

    When the status or the estimate fails, `estimate` is None, and `error` and
    `code` say why. `more` goes to `estimate`; `store` goes to both.
    """
    store = more.get("store") or index_store()
    try:
        return {
            "status": status(archive, store=store),
            "estimate": estimate(archive, **{**more, "store": store}),
            "error": None,
            "code": None,
        }
    except MeaningFailure as failure:
        held = _held(store)
        return {"status": _shown(held, held.state), "estimate": None, "error": str(failure), "code": failure.code}


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
            ArchiveSource(archive),
            _OnDemand(embedder_factory or embedder_for, chosen),
            store or index_store(),
            progress=teller.step,
            cancel=cancel,
            count_tokens=count_tokens or default_count_tokens,
            sleep=sleep,
            limits=limits or BatchLimits(),
        )
    except (MeaningIndexError, ArchiveFault) as error:
        raise _failed(error, "an index build") from None
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
