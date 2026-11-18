"""The two optional ports of chat: the search by meaning and the digest of one conversation.

`ports(archive, names)` gives `{"meaning": ..., "digests": ...}` for
`ChatEngine`. A port that can not work is None: the engine then does not offer
its tool, and nothing else changes.

- `MeaningPort` is `chat_engine.ports.MeaningSearch` over `meaning_index`. It
  embeds the description with the model and the dimensions of the index. The
  process keeps the loaded index and the vectors of the last descriptions.
- `DigestPort` is `chat_engine.ports.DigestSource` over
  `recall_services.digests`. A digest is wired as a report is wired, so the
  two share the notes of the parts that were read. It tells the engine each
  change of a part (`progress_relay`). Nothing here starts a reading outside
  the question.

Rules:
- The description that reaches `MeaningPort.search` is masked by the engine.
  Only the description goes to the embedding provider. The conversation ids
  are handles: they filter the search on this machine and go nowhere.
- Every failure of a port is raised. The engine then ranks by words alone, or
  prints its one line for a digest. No failure here fails a question.
- No log line holds a description, a name, a handle or an id.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from archive import Archive
from chat_engine.cancel import CancelToken
from chat_engine.ports import DigestCost, DigestMoment, DigestPeriod, DigestProgressCallback, DigestResult
from meaning_index import IndexStore
from recall_services import digests, keys, meaning, models, reports
from recall_services.meaning_port import QUERY_TIMEOUT_SECONDS as QUERY_TIMEOUT_SECONDS
from recall_services.meaning_port import QUERY_VECTORS as QUERY_VECTORS
from recall_services.meaning_port import EmbedderFactory
from recall_services.meaning_port import MeaningNotReady as MeaningNotReady
from recall_services.meaning_port import MeaningPort as MeaningPort
from recall_services.meaning_port import forget as forget
from recall_services.meaning_port import ready
from recall_services.names import Names
from recall_services.report_failures import NO_KEY, ReportFailure
from report_engine import LLM, ModelInfo, PeriodCache, ReportOptions, Runtime
from report_engine import CancelToken as ReportCancelToken
from report_engine.chat_digest import MomentNote, PeriodNote
from report_engine.control import ProgressCallback, ProgressEvent
from report_engine.tokens import TokenCounter

logger = logging.getLogger(__name__)

# The model of a digest: an id of models.py, or "" for the default model of the reports.
# The digest only sums up the periods; the chat model answers and checks against the messages.
# WARNING: the notes of a part belong to one model. A digest shares the notes of
# a report only when the report was run with this model.
DIGEST_MODEL = "gpt-6-luna"
# The most parts not yet read that one digest inside a question reads. `chat.py`
# gives the engine the same number, so the two can not differ. A longer
# conversation is not read by a question: the payload names it, and the reader decides.
# The parts of one digest are read one at a time (each takes the notes of the one before):
# about 20 s each on DIGEST_MODEL, so about 80 s for one digest.
DIGEST_PARTS_LIMIT = 4
# the states of a part that the engine is told; "waiting" (for the rate budget) is not a change of the part
TOLD_STATES = frozenset({"start", "done", "cached", "failed"})
# The model port of a digest inside a question: a provider that hangs holds a
# question for two tries of this long for each part. A report job keeps its own.
DIGEST_TIMEOUT_SECONDS = 120.0
DIGEST_RETRIES = 1
LOCK_POLL_SECONDS = 0.1
# the log of what a question spent on a digest: counts and a cost, at the level INFO
SPEND_LOG = "recall.spend"

spend = logging.getLogger(SPEND_LOG)


class DigestStopped(Exception):
    """The question was cancelled while its digest waited for the digest of another question."""


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
    llm_factory: reports.LLMFactory | None = None  # None: question_llm
    runtime: Runtime | None = None
    parts_limit: int | None = None  # None: DIGEST_PARTS_LIMIT


def question_llm(info: ModelInfo) -> LLM:
    """The model port of a digest inside a question: a short timeout and at most one more try."""
    from report_engine.openai_llm import OpenAIResponsesLLM, build_client

    key = keys.api_key()
    if not key:
        raise ReportFailure("no_key", NO_KEY)
    client = build_client(key, timeout_s=DIGEST_TIMEOUT_SECONDS)
    return OpenAIResponsesLLM(client, info.id, timeout_s=DIGEST_TIMEOUT_SECONDS, max_retries=DIGEST_RETRIES)


_LOCKS: dict[tuple[str, str, str], threading.Lock] = {}
_LOCKS_LOCK = threading.Lock()


def digest_lock(archive: Archive, conversation_id: str, model: str) -> threading.Lock:
    """The one lock of this process for the digests of one conversation of one archive by one model."""
    key = (str(getattr(archive, "path", "")), conversation_id, model)
    with _LOCKS_LOCK:
        return _LOCKS.setdefault(key, threading.Lock())


@contextmanager
def _turn(lock: threading.Lock, cancel: CancelToken | None) -> Iterator[None]:
    """Hold the lock. A question that is cancelled while it waits stops waiting."""
    while not lock.acquire(timeout=LOCK_POLL_SECONDS):
        if cancel is not None and cancel.cancelled:
            raise DigestStopped("the question was cancelled while its digest waited")
    try:
        yield
    finally:
        lock.release()


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


def progress_relay(progress: DigestProgressCallback) -> ProgressCallback:
    """The listener of the report engine that tells `progress` each change of a part of the digest.

    Only a map event with a part is told: "start", "done", "cached" or
    "failed". The report engine can call it from more than one thread.
    """

    def relay(event: ProgressEvent) -> None:
        if event.step == "map" and event.state in TOLD_STATES and event.period is not None:
            progress({"part": event.index, "of": event.total, "label": event.period, "state": event.state})

    return relay


def _spent_for_nothing(failure: ReportFailure) -> float:
    """What a digest that failed or was stopped cost, from the usage that its failure carries.

    A failure that was paid for leaves one line in the spend log. A failure
    before any call carries no usage: it cost nothing and leaves no line.
    """
    usage = failure.usage
    if not usage:
        return 0.0
    usd = float(usage.get("cost_usd") or 0.0)
    spend.info(
        "a digest inside a question gave nothing (%s): %d model calls, %.4f USD",
        failure.code, int(usage.get("calls") or 0), usd,
    )  # fmt: skip
    return usd


class DigestPort:
    """`chat_engine.ports.DigestSource` over `recall_services.digests`."""

    def __init__(
        self,
        archive: Archive,
        names: Names,
        settings: DigestSettings | None = None,
    ) -> None:
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
        self,
        conversation_id: str,
        *,
        since: str | None,
        until: str | None,
        cancel: CancelToken,
        progress: DigestProgressCallback | None = None,
    ) -> DigestResult:
        """Build the digest, and say what it cost. It stops before its next model call when `cancel` is cancelled.

        `progress` is told each change of a part, as the report engine reads it.

        A digest that fails or is stopped raises the failure of `digests.build`
        with `usd` set on it: what the run cost until then.

        It reads at most `DIGEST_PARTS_LIMIT` parts that are not read yet: a
        digest that needs more raises, and no call is made. One lock for the
        archive, the conversation and the model is held, so a second question
        waits and then finds the parts read.
        """
        chosen = self._settings
        limit = DIGEST_PARTS_LIMIT if chosen.parts_limit is None else chosen.parts_limit
        lock = digest_lock(self._archive, conversation_id, self.model)
        try:
            with _turn(lock, cancel), _report_cancel(cancel) as token:
                found = digests.build(
                    conversation_id,
                    since,
                    until,
                    token,
                    on_progress=None if progress is None else progress_relay(progress),
                    llm_factory=chosen.llm_factory or question_llm,
                    runtime=chosen.runtime,
                    max_new_periods=limit,
                    **self._same(),
                )
        except ReportFailure as failure:
            failure.usd = _spent_for_nothing(failure)  # type: ignore[attr-defined]
            raise
        usd = float(found.usage.cost_usd)
        spend.info(
            "a digest inside a question: %d parts read now, %d from the notes, %d failed, %d model calls, %.4f USD",
            found.periods_built, found.periods_cached, found.periods_failed, found.usage.calls, usd,
        )  # fmt: skip
        periods = [_period(note) for note in found.periods]
        return {"periods": periods, "periods_failed": found.periods_failed, "usd": usd}  # type: ignore[typeddict-unknown-key]

def _meaning_port(store: IndexStore, factory: EmbedderFactory | None) -> MeaningPort | None:
    """The port when an index can be searched now. A folder that can not be used gives None and one log line."""
    try:
        usable = ready(store)
    except Exception as error:  # noqa: BLE001 - whatever a folder raises, the question goes on by words
        logger.warning("the index can not be used (%s); chat searches by words only", type(error).__name__)
        return None
    return MeaningPort(store, factory) if usable else None


def _digest_port(archive: Archive, names: Names, settings: DigestSettings) -> DigestPort | None:
    """The port when a digest can be built now: its model can be run, and the notes of earlier runs can be opened."""
    port = DigestPort(archive, names, settings)
    notes = (settings.places or reports.default_places()).cache_path
    try:
        models.model_info(port.model.strip() or models.default_id())
        if notes.exists():  # a check makes no file: the first digest makes the notes
            PeriodCache(notes)
    except Exception as error:  # noqa: BLE001 - a model that can not be run, notes that can not be opened
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
        "meaning": _meaning_port(store or meaning.index_store(archive.path), embedder_factory),
        "digests": _digest_port(archive, names, digest or DigestSettings()),
    }
