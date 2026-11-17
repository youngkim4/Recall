"""The estimate, a report run and the moments of a report, for the routes of the server.

`estimate` gives the keys that the app reads today (`msg_count`,
`input_tokens`, `output_tokens`, `estimated_cost`, `needs_chunking`,
`token_budget`, `long_context_pricing`, `years`) with the numbers of
`report_engine.estimate`, plus `periods` and `periods_cached`.

`run` runs `run_report` on the archive and then:
- saves the report as JSON with `ReportStore`;
- writes `analysis_<name>.md` with `render_markdown`, which the server reads;
- writes `events_timeline_<name>.csv` from the moments, or removes an older
  one when the report has no moment (CONTRACTS.md "Reports");
- removes an older `analysis_<name>.html`: the engine writes no page, and an
  older page would be shown beside the new report as its own.
`<name>` is `sanitize_filename` of the conversation id, as today. Two ids that
differ only in other characters than letters, digits, `-`, `_` and `+` share
one name, as today.

The report file and the events file are one report: when one of them can not
be written, the files of the older report stay as they were.

`saved_report` gives the saved report of a report file, and `shown_plain` says
whether the server shows the story of a report file as plain paragraphs.

`moments_for` gives each moment what the Reports screen needs to open its
day at the quoted message.

Names for the model come from `names.report_labeler`. The model, its limits
and its prices come from `models.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from archive import Archive
from archive import ArchiveError as ArchiveFault
from recall_services import inputs, keys, models, report_files
from recall_services.names import Names, ReportLabeler
from recall_services.report_failures import NO_KEY, ReportFailure, failure_of
# part of this module for its callers
from recall_services.report_files import EVENT_COLUMNS as EVENT_COLUMNS
from recall_services.report_files import PAGE_KEPT as PAGE_KEPT
from recall_services.report_files import PAGE_REMOVED as PAGE_REMOVED
from recall_services.report_files import shown_plain as shown_plain
from recall_services.report_files import story_of as story_of
from recall_services.report_progress import Listener, Teller
from report_engine import (
    LLM,
    ChatMessage,
    LLMError,
    LLMResult,
    LLMUsage,
    ModelInfo,
    PeriodCache,
    Report,
    ReportEngineError,
    ReportOptions,
    ReportStore,
    Runtime,
    StoreError,
    render_markdown,
    run_report,
)
from report_engine.archive_source import ArchiveSource
from report_engine.estimate import planned_calls
from report_engine.pricing import call_cost_usd
from report_engine.tokens import TokenCounter, load_counter

logger = logging.getLogger(__name__)

LLMFactory = Callable[[ModelInfo], LLM]
SHOWN_PLAIN = "The story of this report is long and dense. The app shows it as plain paragraphs, with no chapters."
NOT_SAVED = "The report was written but could not be saved. Run the report again."
NOT_STARTED = "The model could not be started. No report was written."
SAVED = "The report is saved."
ARCHIVE_UNREADABLE = "The message archive could not be read. Update the archive, then try again."


@dataclass(frozen=True)
class ReportPlaces:
    """Where a run writes: the files the server reads, the saved reports, and the notes of earlier runs."""

    out_dir: Path
    store_dir: Path
    cache_path: Path


def default_places() -> ReportPlaces:
    """The places in the data folder of Recall, read at each call."""
    import recall_paths

    saves = Path(recall_paths.SAVES_DIR)
    return ReportPlaces(
        out_dir=Path(recall_paths.OUT_DIR), store_dir=saves / "reports", cache_path=saves / "report-cache.sqlite3"
    )


@dataclass(frozen=True)
class ReportRun:
    """A run that wrote a report. `shown_plain` says that the server will show its story as plain paragraphs."""

    report: Report
    report_path: Path
    events_path: Path | None
    json_path: Path
    shown_plain: bool

    @property
    def report_id(self) -> str:
        return self.report.id

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.report.warnings

    def as_dict(self) -> dict[str, Any]:
        """The keys of a job result. `htmlPath` is always None: the engine writes no page."""
        return {
            "reportPath": str(self.report_path),
            "eventsPath": None if self.events_path is None else str(self.events_path),
            "htmlPath": None,
            "jsonPath": str(self.json_path),
            "reportId": self.report.id,
            "status": self.report.status,
            "warnings": list(self.report.warnings),
            "shownPlain": self.shown_plain,
            "usage": self.report.usage.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class _Asked:
    conversation: str
    since: str | None
    until: str | None
    info: ModelInfo
    labeler: ReportLabeler

    @property
    def scope(self) -> dict[str, str | None] | None:
        return {"since": self.since, "until": self.until} if (self.since or self.until) else None


@lru_cache(maxsize=8)
def shared_counter(family: str) -> TokenCounter:
    """One token counter for each tokenizer family, for the estimate and for the run.

    The engine loads a counter for each call that gets none. A load can fall
    back to a rough counter, and an estimate and a run with two counters plan
    two sets of parts.
    """
    return load_counter(family)


class _Watched:
    """The model port of a run, with a note of the failure that its last call had. The engine says only that a part failed."""

    def __init__(self, inner: LLM) -> None:
        self._inner = inner
        self.last: LLMError | None = None

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: Mapping[str, Any] | None = None,
        max_output_tokens: int,
        reasoning: str | None = None,
    ) -> LLMResult:
        try:
            result = self._inner.generate(
                messages, schema=schema, max_output_tokens=max_output_tokens, reasoning=reasoning
            )
        except LLMError as error:
            self.last = error
            raise
        self.last = None  # a failure before a call that worked did not end the run
        return result


def default_llm(info: ModelInfo) -> LLM:
    """The model port of the app. It goes to the address of the provider only."""
    from report_engine.openai_llm import OpenAIResponsesLLM, build_client

    key = keys.api_key()
    if not key:
        raise ReportFailure("no_key", NO_KEY)
    return OpenAIResponsesLLM(build_client(key), info.id)


def _port(factory: LLMFactory, info: ModelInfo) -> LLM:
    """The model port of a run. A factory that fails in a way of its own is a failure with a sentence."""
    try:
        return factory(info)
    except ReportFailure:
        raise
    except Exception as error:
        logger.exception("the model port of a report run could not be made")
        raise ReportFailure("run_failed", NOT_STARTED) from error


@contextmanager
def _archive_faults() -> Iterator[None]:
    """A failure of the archive itself, outside the engine, as the failure for the person."""
    try:
        yield
    except ArchiveFault as error:
        raise ReportFailure("archive_error", ARCHIVE_UNREADABLE) from error


def _day(value: object) -> str | None:
    text = inputs.text(value).strip()
    if not text:
        return None
    try:
        written = date.fromisoformat(text).isoformat()
    except ValueError:
        written = ""
    if written != text:
        raise ValueError(f"Not a day: {inputs.shown(text)}")
    return text


def _asked(contact: str, model: str, since: object, until: object, archive: Archive, names: Names) -> _Asked:
    conversation = inputs.text(contact)
    if not conversation.strip():
        raise ValueError("A conversation is needed")
    first, last = _day(since), _day(until)
    if first and last and first > last:
        raise ValueError("The first day is after the last day.")
    info = models.model_info(inputs.text(model).strip() or models.default_id())
    record = archive.conversation(conversation) or {"id": conversation, "is_group": False}
    return _Asked(conversation, first, last, info, names.report_labeler(record))


def _provider(info: ModelInfo) -> str:
    return models.provider_name(models.facts(info.id).provider)


def _planned(
    archive: Archive, asked: _Asked, cache_path: Path, options: ReportOptions, counter: TokenCounter | None
) -> list[Any]:
    """The calls of a run, without the parts that the notes of earlier runs hold. Notes that can not be read hold none."""

    def plan(cache: PeriodCache | None) -> list[Any]:
        return planned_calls(
            ArchiveSource(archive),
            asked.conversation,
            asked.scope,
            asked.info,
            cache,
            labeler=asked.labeler,
            options=options,
            counter=counter or shared_counter(asked.info.tokenizer_family),
        )

    if not cache_path.exists():
        return plan(None)
    try:
        return plan(PeriodCache(cache_path))
    except StoreError as error:
        logger.warning("The notes of earlier report runs could not be read: %s", type(error).__name__)
        return plan(None)


def _notes(cache_path: Path) -> PeriodCache:
    """The notes of earlier runs, for a run. A file that can not be opened stops the run before any call."""
    try:
        return PeriodCache(cache_path)
    except StoreError as error:
        raise ReportFailure(
            "cache_unreadable",
            f"The file {cache_path.name}, which holds what earlier report runs read, can not be opened. "
            "No report was written.",
        ) from error


def estimate(
    contact: str,
    model: str = "",
    since: str | None = None,
    until: str | None = None,
    *,
    archive: Archive,
    names: Names,
    places: ReportPlaces | None = None,
    options: ReportOptions | None = None,
    counter: TokenCounter | None = None,
) -> dict[str, Any]:
    """What a run that starts now would send and cost. Makes no model call and writes no report.

    A part that an earlier run read is not counted again: the notes of
    earlier runs are opened when that file is there. When they can not be
    read, every part is counted.

    Raises ValueError for a missing conversation or a bad day, a
    ModelChoiceError for a model that can not be run, and ReportFailure
    (`too_large`, `archive_error`) when the conversation can not be planned.
    """
    chosen = options or ReportOptions()
    cache_path = (places or default_places()).cache_path
    with _archive_faults():
        asked = _asked(contact, model, since, until, archive, names)
        try:
            calls = _planned(archive, asked, cache_path, chosen, counter)
        except ReportEngineError as error:
            raise failure_of(error, None, _provider(asked.info), asked.info.id) from error
        within = {"conversations": [asked.conversation], "since": asked.since, "until": asked.until}
        parts = [call for call in calls if call.kind == "map"]
        years = archive.count(group_by="year", **within) if len(parts) > 1 else []
        messages = int(archive.count(**within))  # type: ignore[arg-type]
    to_send = [call for call in calls if not call.cached]
    costs = (
        call_cost_usd(asked.info, LLMUsage(input_tokens=call.input_tokens, output_tokens=call.output_tokens))
        for call in to_send
    )
    rule = asked.info.long_context
    return {
        "msg_count": messages,
        "input_tokens": sum(call.input_tokens for call in to_send),
        "output_tokens": sum(call.output_tokens for call in to_send),
        "estimated_cost": float(sum(costs)),
        "needs_chunking": len(parts) > 1,
        "token_budget": chosen.period_token_budget,
        "long_context_pricing": rule is not None and any(call.input_tokens > rule.threshold_tokens for call in to_send),
        "years": [int(row["key"]) for row in years],  # type: ignore[union-attr]
        "periods": len(parts),
        "periods_cached": sum(1 for call in parts if call.cached),
    }


def saved_report(contact: str, report_text: str, places: ReportPlaces | None = None) -> Report | None:
    """The saved report that a report file shows, or None.

    The file holds no id. A saved report of the conversation is the one of
    the file when the file is its rendering, letter for letter. A report of
    the older kind has no saved report.
    """
    conversation = inputs.text(contact)
    if not conversation.strip():
        return None
    store = ReportStore((places or default_places()).store_dir)
    try:
        for entry in store.list_reports(conversation):
            held = store.load_report(conversation, entry.id)
            if render_markdown(held) == report_text:
                return held
    except StoreError as error:
        logger.warning("The saved reports could not be read: %s", type(error).__name__)
    return None


def _write(report: Report, places: ReportPlaces, teller: Teller) -> ReportRun:
    """Save the report, then make the files that the server reads hold it."""
    json_path = ReportStore(places.store_dir).save(report)
    report_path, events_path, text, said = report_files.write(report, places.out_dir)
    if said:
        teller.tell(log=said)
    return ReportRun(report, report_path, events_path, json_path, shown_plain(text))


def run(
    contact: str,
    model: str = "",
    since: str | None = None,
    until: str | None = None,
    on_progress: Listener | None = None,
    cancel: Any = None,
    *,
    archive: Archive,
    names: Names,
    places: ReportPlaces | None = None,
    llm_factory: LLMFactory | None = None,
    options: ReportOptions | None = None,
    counter: TokenCounter | None = None,
    runtime: Runtime | None = None,
) -> ReportRun:
    """Write a report on one conversation and save it. See the module docstring for the files.

    `on_progress` gets the updates of `report_progress.Teller`. `cancel` is a
    token with `cancelled`: the run stops before its next model call.

    Raises ReportFailure, with a code and one plain sentence, when no report
    was written. Raises ValueError for a missing conversation or a bad day,
    and a ModelChoiceError for a model that can not be run.
    """
    where = places or default_places()
    teller = Teller(on_progress)
    with _archive_faults():
        asked = _asked(contact, model, since, until, archive, names)
        figures = archive.stats(asked.conversation, since=asked.since, until=asked.until)
    notes = _notes(where.cache_path)
    watched = _Watched(_port(llm_factory or default_llm, asked.info))
    try:
        report = run_report(
            source=ArchiveSource(archive),
            conversation=asked.conversation,
            scope=asked.scope,
            llm=watched,
            model_info=asked.info,
            labeler=asked.labeler,
            cache=notes,
            stats=figures,
            options=options,
            counter=counter or shared_counter(asked.info.tokenizer_family),
            on_progress=teller.engine_event,
            cancel=cancel,
            runtime=runtime,
        )
    except ReportEngineError as error:
        raise failure_of(error, watched.last, _provider(asked.info), asked.info.id) from error
    try:
        done = _write(report, where, teller)
    except (StoreError, OSError) as error:
        raise ReportFailure("not_saved", NOT_SAVED, report.usage.model_dump(mode="json")) from error
    for warning in done.warnings:
        teller.tell(log=f"Note: {warning}")
    if done.shown_plain:
        teller.tell(log=SHOWN_PLAIN)
    teller.tell(log=SAVED)
    return done


def moments_for(report: Report, archive: Archive) -> list[dict[str, Any]]:
    """Each moment of a report with what the Reports screen needs to open its day.

    `chatId` and `day` open the day. For a verified quote, `day` is the day of
    the quoted message, `at` is its exact time and `pickedId` is its id, so
    the day opens at that message. With no verified quote, or when the
    archive no longer holds the message, `at` is None and `pickedId` is "".
    """
    shaped = []
    for moment in report.moments:
        quoted = archive.message(moment.quote_message_id) if moment.quote_verified else None
        if quoted is not None and quoted["conversation"] != report.conversation:
            quoted = None
        shaped.append(
            {
                "date": moment.date,
                "title": moment.title,
                "detail": moment.detail,
                "category": moment.category,
                "score": moment.score,
                "quote": moment.quote,
                "chatId": report.conversation,
                "day": quoted["timestamp"][:10] if quoted else moment.date,
                "at": quoted["timestamp"] if quoted else None,
                "pickedId": quoted["id"] if quoted else "",
            }
        )
    return shaped
