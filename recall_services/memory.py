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

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from archive import Archive
from chat_engine.cancel import CancelToken
from chat_engine.ports import DigestCost, DigestMoment, DigestPeriod, DigestResult
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
from report_engine import CancelToken as ReportCancelToken
from report_engine import PeriodCache, ReportOptions, Runtime
from report_engine.chat_digest import MomentNote, PeriodNote
from report_engine.tokens import TokenCounter

logger = logging.getLogger(__name__)

# The model of a digest: an id of models.py, or "" for the default model of the reports.
# The digest only sums up the periods; the chat model answers and checks against the messages.
# WARNING: the notes of a part belong to one model. A digest shares the notes of
# a report only when the report was run with this model.
DIGEST_MODEL = "gpt-6-luna"
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
