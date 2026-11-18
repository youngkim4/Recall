"""The digest of one conversation for chat, on the archive.

`estimate` says what a digest that starts now would send and cost. `build`
builds the digest: the summary of each part of the conversation, and the
moments of each part with the ids of the messages that back them. A digest
writes no report and no file but the notes of the parts that it read.

A digest is wired as a report is wired in `reports.py`, with the functions of
that module: the same messages with the same masks, the same names for the
model, the same key, the same model facts and prices, the same token counter
and the same file of notes.

WARNING: Name the model of the reports. The notes of a part belong to one
model. With no model named, the default model is used, as in a report.

The dates do not cut the conversation, as the dates of a report do. They
select the parts that overlap them, and each selected part is read whole. The
parts are the parts of a report on all dates.

A digest shares the notes of the reports when the model reads what it reads
in a report: the dates start at the first part, or every part before them was
read. Then a part that a report read costs a digest nothing, and a part that
a digest read costs a report nothing. With other dates the digest keeps notes
of its own, which only a digest of dates with the same first part reads again.
See "Digest for chat" in `report_engine/README.md`.

A failure is a `ReportFailure` with the codes and the sentences of a report
run, and `too_many_parts` when `max_new_periods` refuses the digest.

Several conversations can be read at once: run one `build` in each thread
and give them one pool (`call_pool`). The parts of one conversation are read
one at a time, because the prompt of a part holds the notes of the parts
before it. See "Several conversations at once" in `report_engine/README.md`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from archive import Archive
from recall_services import inputs, models, reports
from recall_services.names import Names
from recall_services.report_failures import ReportFailure, failure_of
from report_engine import CallPool, PeriodCache, ReportEngineError, ReportOptions, Runtime, StoreError
from report_engine.chat_digest import (
    ConversationDigest,
    DigestEstimate,
    DigestTooLarge,
    conversation_digest,
    estimate_digest,
)
from report_engine.control import ProgressCallback
from report_engine.tokens import TokenCounter

__all__ = ["ReportFailure", "build", "call_pool", "estimate"]

logger = logging.getLogger(__name__)


def _notes_to_read(cache_path: Path) -> PeriodCache | None:
    """The notes of earlier runs, for an estimate. An estimate makes no file,
    and notes that can not be read hold nothing, as in the estimate of a report."""
    if not cache_path.exists():
        return None
    try:
        return PeriodCache(cache_path)
    except StoreError as error:
        logger.warning("The notes of earlier report runs could not be read: %s", type(error).__name__)
        return None


def _too_many(error: DigestTooLarge) -> ReportFailure:
    """The failure of a digest that needs more model calls than its limit.

    The engine refuses before the first call, or it stops at the limit after
    calls that were paid for. After a stop the error carries the usage of
    those calls, and it does not know how many parts are left: the sentence
    then names no such number.
    """
    usage = getattr(error, "usage", None)
    if usage is not None:
        return ReportFailure(
            "too_many_parts",
            f"This conversation has more parts that are not read yet than the {error.limit} that can be read now. "
            "What was read is kept. Choose fewer dates.",
            usage.model_dump(mode="json"),
        )
    new = error.new_periods
    waiting = "1 part that is not read yet" if new == 1 else f"{new} parts that are not read yet"
    return ReportFailure(
        "too_many_parts",
        f"This conversation has {waiting}, and {error.limit} can be read now. Choose fewer dates.",
    )


def _reading(
    archive: Archive, asked: reports._Asked, options: ReportOptions | None, counter: TokenCounter | None
) -> dict[str, Any]:
    """What the estimate and the digest both give the engine, as a report run
    gives it. The two then plan the same parts, with the cache keys of a report."""
    return {
        "source": reports._Masked(archive, asked.redact),
        "conversation": asked.conversation,
        "model_info": asked.info,
        "labeler": asked.labeler,
        "since": asked.since,
        "until": asked.until,
        "options": options,
        "counter": counter or reports.shared_counter(asked.info.tokenizer_family),
    }


def estimate(
    conversation_id: str,
    since: str | None = None,
    until: str | None = None,
    *,
    archive: Archive,
    names: Names,
    model: str = "",
    places: reports.ReportPlaces | None = None,
    options: ReportOptions | None = None,
    counter: TokenCounter | None = None,
) -> DigestEstimate:
    """What a digest that starts now would send and cost. Makes no model call and writes nothing.

    Raises ValueError for a missing conversation or a bad day, a
    ModelChoiceError for a model that can not be run, and ReportFailure
    (`too_large`, `archive_error`) when the conversation can not be planned.
    """
    cache_path = (places or reports.default_places()).cache_path
    with reports._archive_faults():
        asked = reports._asked(conversation_id, model, since, until, archive, names)
        try:
            return estimate_digest(
                cache=_notes_to_read(cache_path), **_reading(archive, asked, options, counter)
            )
        except ReportEngineError as error:
            raise failure_of(error, None, reports._provider(asked.info), asked.info.id) from error


def call_pool(model: str = "", max_parallel: int = 1, *, clock: Callable[[], float] = time.monotonic) -> CallPool:
    """One pool for the digests that are read at the same time with one model.

    At most `max_parallel` requests are in flight, and the digests share the
    tokens per minute of the model. Give the same model to each `build`.
    `clock` is the clock of the `runtime` of the builds. Raises a
    ModelChoiceError for a model that can not be run.
    """
    info = models.model_info(inputs.text(model).strip() or models.default_id())
    return CallPool(max_parallel=max_parallel, tokens_per_minute=info.tokens_per_minute, clock=clock)


def build(
    conversation_id: str,
    since: str | None = None,
    until: str | None = None,
    cancel: Any = None,
    on_progress: ProgressCallback | None = None,
    *,
    archive: Archive,
    names: Names,
    model: str = "",
    places: reports.ReportPlaces | None = None,
    llm_factory: reports.LLMFactory | None = None,
    options: ReportOptions | None = None,
    counter: TokenCounter | None = None,
    runtime: Runtime | None = None,
    max_new_periods: int | None = None,
    pool: CallPool | None = None,
) -> ConversationDigest:
    """Build the digest of one conversation. A part in the notes of earlier runs costs no call.

    `on_progress` gets each `ProgressEvent` of the engine, in the thread of
    the build. `cancel` is a token with `cancelled`: the digest stops before
    its next model call, and the parts that were read stay in the notes.
    `max_new_periods` refuses a digest that needs more model calls, before
    any call. `pool` is shared by the builds that run at the same time
    (`call_pool`); its `stop()` stops each of them before its next call.

    Raises ReportFailure, with a code and one plain sentence, when no digest
    was built. Raises ValueError for a missing conversation or a bad day, and
    a ModelChoiceError for a model that can not be run.
    """
    where = places or reports.default_places()
    with reports._archive_faults():
        asked = reports._asked(conversation_id, model, since, until, archive, names)
    notes = reports._notes(where.cache_path)
    watched = reports._Watched(reports._port(llm_factory or reports.default_llm, asked.info))
    try:
        return conversation_digest(
            llm=watched,
            cache=notes,
            on_progress=on_progress,
            cancel=cancel,
            runtime=runtime,
            max_new_periods=max_new_periods,
            pool=pool,
            **_reading(archive, asked, options, counter),
        )
    except DigestTooLarge as error:
        raise _too_many(error) from error
    except ReportEngineError as error:
        raise failure_of(error, watched.last, reports._provider(asked.info), asked.info.id) from error
