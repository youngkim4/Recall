"""The memory of chat, read ahead of time: the `memory` job, and what `GET /api/memory` says.

The memory is the digest of each conversation with at least
`PREPARE_MIN_MESSAGES` messages, read with the model of a digest
(`memory.DIGEST_MODEL`). A question then finds every part in the notes.

- `payload` says what is read: `none` (no part), `partial` or `ready` (every
  part), the counts, and the cost of the parts that are not read. It makes no
  model call and needs no key.
- `run` is the `memory` job: it reads the conversations that are not read
  whole, busiest first, `PREPARE_PARALLEL` at a time. The parts of one
  conversation are read one at a time: the prompt of a part holds the notes
  of the part before it. All the conversations share one pool of calls, so
  at most `PREPARE_PARALLEL` requests are in flight and they share the tokens
  per minute of the model.
- `worth_starting` says whether the server starts the job by itself after an
  import: only when something was read before, and the rest costs at most
  `PREPARE_AUTO_USD`.

A conversation that fails is counted, and the others are read. A failure
that every conversation would have (`FATAL`) stops the job: no new request
is sent, and the requests in flight end and are kept.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from archive import Archive
from archive import ArchiveError as ArchiveFault
from recall_services import digests, memory, models, reports
from recall_services import memory_prep_words as words
from recall_services.memory_prep_read import Planned, Read, Reading, tally
from recall_services.memory_prep_words import MemoryFailure as MemoryFailure
from recall_services.models import ModelChoiceError
from recall_services.names import Names
from recall_services.report_failures import ReportFailure
from report_engine import PeriodCache, StoreError, run_in_parallel

logger = logging.getLogger(__name__)

PREPARE_MIN_MESSAGES = 200
PREPARE_PARALLEL = 4
PREPARE_AUTO_USD = 0.25
# what an estimate takes of the settings of a build
ESTIMATE_KEYS = ("places", "options", "counter")


def model() -> str:
    """The id of the model of the memory, read at each call."""
    return memory.DIGEST_MODEL or models.default_id()


@contextmanager
def _archive_faults() -> Iterator[None]:
    try:
        yield
    except ArchiveFault as error:
        raise MemoryFailure("archive_error", words.ARCHIVE_UNREADABLE) from error


def least(min_messages: int | None) -> int:
    """The least number of messages of a conversation of the memory.
    None is `PREPARE_MIN_MESSAGES`, read at each call."""
    return PREPARE_MIN_MESSAGES if min_messages is None else min_messages


def candidates(archive: Archive, min_messages: int | None = None) -> tuple[str, ...]:
    """The ids of the conversations of the memory, busiest first. Raises MemoryFailure (`archive_error`)."""
    with _archive_faults():
        rows = archive.conversations(order="count")
    return tuple(str(row["id"]) for row in rows if int(row["message_count"] or 0) >= least(min_messages))


def read_before(
    archive: Archive, *, places: reports.ReportPlaces | None = None, min_messages: int | None = None
) -> bool:
    """True when the notes hold a part that the model of the memory read, of a conversation of the memory.

    This holds after an import that gave each part that was read a new key.
    Notes that are not there or can not be read say False, and no file is made.
    """
    where = (places or reports.default_places()).cache_path
    if not where.exists():
        return False
    try:
        read = PeriodCache(where).conversations_read(model())
    except StoreError as error:
        logger.warning("the notes of earlier runs can not be read (%s)", type(error).__name__)
        return False
    return bool(read & set(candidates(archive, min_messages)))


def _planned(
    archive: Archive, names: Names, chosen: Sequence[str], more: Mapping[str, Any]
) -> tuple[list[Planned], int]:
    """What reading each conversation needs. One that can not be planned is skipped and counted."""
    kept = {name: more[name] for name in ESTIMATE_KEYS if name in more}
    planned: list[Planned] = []
    skipped = 0
    for conversation in chosen:
        try:
            found = digests.estimate(conversation, model=model(), archive=archive, names=names, **kept)
        except (ReportFailure, ValueError) as error:
            why = getattr(error, "code", type(error).__name__)
            logger.info("a conversation of the memory can not be planned (%s)", why)
            skipped += 1
            continue
        planned.append(Planned(conversation, found.periods, found.periods_cached, float(found.usd)))
    return planned, skipped


def _shown(planned: Sequence[Planned], skipped: int, min_messages: int) -> dict[str, Any]:
    parts = sum(one.periods for one in planned)
    read = sum(one.periods_cached for one in planned)
    state = "none" if read == 0 else "ready" if read == parts else "partial"
    return {
        "state": state,
        "model": model(),
        "minMessages": min_messages,
        "conversations": len(planned) + skipped,
        "conversationsRead": sum(one.not_read == 0 for one in planned),
        "conversationsSkipped": skipped,
        "parts": parts,
        "partsRead": read,
        "partsNotRead": parts - read,
        "estimatedCost": float(sum(one.usd for one in planned)),
        "error": None,
        "code": None,
    }


def payload(archive: Archive, names: Names, *, min_messages: int | None = None, **more: Any) -> dict[str, Any]:
    """The answer of `GET /api/memory`. No model call, no key, no file.

    When the archive can not be read or the model can not be run, the counts
    are zeros, the state is `none`, and `error` and `code` say why. `more`
    gives the places, the options and the counter of the reports (tests).
    """
    fewest = least(min_messages)
    try:
        planned, skipped = _planned(archive, names, candidates(archive, fewest), more)
    except ModelChoiceError as error:
        return {**_shown((), 0, fewest), "error": words.MODEL_NOT_RUN, "code": error.code}
    except MemoryFailure as failure:
        return {**_shown((), 0, fewest), "error": str(failure), "code": failure.code}
    return _shown(planned, skipped, fewest)


def worth_starting(found: Mapping[str, Any], *, read_before: bool = False) -> bool:
    """True when the server starts a `memory` job by itself after an import.

    `found` is a `payload` after the import. Something was read before: the
    state is `partial` or `ready`, or `read_before` says that the notes hold
    parts of the memory under keys that the import changed. Something is not
    read now, and it costs at most `PREPARE_AUTO_USD`. The first reading is
    the choice of the reader.
    """
    cost = found.get("estimatedCost")
    if found.get("error") or not (read_before or found.get("state") in ("partial", "ready")):
        return False
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        return False
    return int(found.get("partsNotRead") or 0) > 0 and cost <= PREPARE_AUTO_USD


def _provider(model_id: str) -> str:
    return models.provider_name(models.facts(model_id).provider)


def run(
    on_progress: words.Listener | None = None,
    cancel: Any = None,
    *,
    archive: Archive,
    names: Names,
    parallel: int | None = None,
    min_messages: int | None = None,
    **more: Any,
) -> dict[str, Any]:
    """Read the memory. Gives `conversations`, `conversationsRead`, `conversationsFailed`,
    `periodsBuilt`, `periodsCached`, `periodsFailed`, `usd` and `usage`.

    `cancel` is a token with `cancelled`: no new request is sent, and the
    requests in flight end. `more` goes to `digests.build`: tests give the
    places, the options, the counter, a model port and a runtime. Raises
    MemoryFailure with a code and one sentence, and a ModelChoiceError for a
    model that can not be run.
    """
    teller = words.Teller(on_progress)
    slots = PREPARE_PARALLEL if parallel is None else parallel
    chosen = candidates(archive, min_messages)
    teller.tell(step="look", total=len(chosen), log=words.looking(len(chosen), least(min_messages)))
    planned, skipped = _planned(archive, names, chosen, more)
    waiting = [one for one in planned if one.not_read > 0]
    parts = sum(one.not_read for one in waiting)
    read = sum(one.periods_cached for one in planned)
    teller.tell(step="read", total=parts, log=words.to_read(parts, len(waiting), read))
    settings = {"model": model(), **more}
    model_id = str(settings["model"] or "") or models.default_id()
    runtime = settings.get("runtime")
    pool = digests.call_pool(model_id, slots, **({"clock": runtime.clock} if runtime else {}))

    def build(conversation: str, heard: Callable[[Any], None]) -> Any:
        return digests.build(
            conversation, None, None, cancel, heard, archive=archive, names=names, pool=pool, **settings
        )

    reading = Reading(teller, parts, len(waiting), cancel, pool, build)
    outcomes = run_in_parallel([reading.task(one) for one in waiting], max_parallel=slots, stop=reading.stopped)
    return _ended(outcomes, (planned, skipped), reading, teller, model_id)


def _ended(
    outcomes: Sequence[Any],
    looked: tuple[Sequence[Planned], int],
    reading: Reading,
    teller: words.Teller,
    model_id: str,
) -> dict[str, Any]:
    """The result of the job, or the failure that stopped it. `looked` is what was planned, and the count of skipped."""
    unplanned = [outcome.error for outcome in outcomes if outcome.error is not None]
    if unplanned:
        raise unplanned[0]
    reads: list[Read] = [outcome.value for outcome in outcomes if outcome.value is not None]
    result = tally(looked[0], reads, looked[1])
    stop = next((read.failure for read in reads if read.failure is not None and read.failure.code in words.FATAL), None)
    if stop is not None:
        raise MemoryFailure(stop.code, words.sentence_of(stop.code, _provider(model_id), model_id), result["usage"])
    if reading.cancelled():
        raise MemoryFailure("cancelled", words.STOPPED, result["usage"])
    teller.tell(
        step="read", done=reading.total, total=reading.total,
        log=words.all_read(result["periodsBuilt"], result["periodsCached"], result["periodsFailed"]),
    )  # fmt: skip
    return result
