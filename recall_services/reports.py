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
`<name>` is `sanitize_filename` of the conversation id, as today.

`moments_for` gives each moment what the Reports screen needs to open its
day at the quoted message.

Names for the model come from `names.report_labeler`. The model, its limits
and its prices come from `models.py`.
"""

from __future__ import annotations

import csv
import io
import logging
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import safe_html
from archive import Archive
from archive import ArchiveError as ArchiveFault
from conversation import sanitize_filename
from recall_services import inputs, models
from recall_services.names import Names, ReportLabeler
from recall_services.report_failures import NO_KEY, ReportFailure, failure_of
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
from report_engine.store import write_text_atomic
from report_engine.tokens import TokenCounter

logger = logging.getLogger(__name__)

LLMFactory = Callable[[ModelInfo], LLM]
EVENT_COLUMNS = ("date", "title", "detail", "category", "score", "quote")
STORY_MARKER = "## Summary"
SHOWN_PLAIN = "The story of this report is long and dense. The app shows it as plain paragraphs, with no chapters."
PAGE_REMOVED = "The page of an older report was removed: this report has no page."
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


class _Watched:
    """The model port of a run, with a note of its last failure. The engine says only that a part failed."""

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
            return self._inner.generate(
                messages, schema=schema, max_output_tokens=max_output_tokens, reasoning=reasoning
            )
        except LLMError as error:
            self.last = error
            raise


def default_llm(info: ModelInfo) -> LLM:
    """The model port of the app. It goes to the address of the provider only."""
    from report_engine.openai_llm import OpenAIResponsesLLM, build_client

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ReportFailure("no_key", NO_KEY)
    return OpenAIResponsesLLM(build_client(key), info.id)


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
        raise ValueError(f"Not a day: {text!r}")
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
    """What a run that starts now would send and cost. Makes no model call and writes no file.

    A part that an earlier run read is not counted again. Raises ValueError
    for a missing conversation or a bad day, and a ModelChoiceError for a
    model that can not be run.
    """
    chosen = options or ReportOptions()
    cache_path = (places or default_places()).cache_path
    with _archive_faults():
        asked = _asked(contact, model, since, until, archive, names)
        try:
            calls = planned_calls(
                ArchiveSource(archive),
                asked.conversation,
                asked.scope,
                asked.info,
                PeriodCache(cache_path) if cache_path.exists() else None,
                labeler=asked.labeler,
                options=chosen,
                counter=counter,
            )
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


def story_of(report_text: str) -> str:
    """The story of a report file: what stands after `## Summary`, as `markdown_section` in the server reads it."""
    start = report_text.find(STORY_MARKER)
    start = report_text.find("\n", start) if start >= 0 else -1
    return report_text[start:].strip() if start >= 0 else ""


def _shown_plain(report_text: str) -> bool:
    return safe_html.story_work(safe_html.prepared(story_of(report_text))) > safe_html.STORY_WORK_LIMIT


def _write_events(path: Path, report: Report) -> Path | None:
    """The events file that the server reads, or no file when the report has no moment."""
    if not report.moments:
        _remove(path)
        return None
    lines = io.StringIO(newline="")
    writer = csv.writer(lines, lineterminator="\n")
    writer.writerow(EVENT_COLUMNS)
    for moment in report.moments:
        writer.writerow([moment.date, moment.title, moment.detail, moment.category, moment.score, moment.quote])
    return write_text_atomic(path, lines.getvalue())


def _remove(path: Path) -> bool:
    """Remove a file of an older report. True when there was one. The new report is written either way."""
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("A file of an older report could not be removed (%s): %s", path.name, type(exc).__name__)
        return False
    return True


def _write(report: Report, places: ReportPlaces, teller: Teller) -> ReportRun:
    name = sanitize_filename(report.conversation)
    text = render_markdown(report)
    json_path = ReportStore(places.store_dir).save(report)
    report_path = write_text_atomic(places.out_dir / f"analysis_{name}.md", text)
    events_path = _write_events(places.out_dir / f"events_timeline_{name}.csv", report)
    if _remove(places.out_dir / f"analysis_{name}.html"):
        teller.tell(log=PAGE_REMOVED)
    return ReportRun(report, report_path, events_path, json_path, _shown_plain(text))


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
    watched = _Watched((llm_factory or default_llm)(asked.info))
    try:
        report = run_report(
            source=ArchiveSource(archive),
            conversation=asked.conversation,
            scope=asked.scope,
            llm=watched,
            model_info=asked.info,
            labeler=asked.labeler,
            cache=PeriodCache(where.cache_path),
            stats=figures,
            options=options,
            counter=counter,
            on_progress=teller.engine_event,
            cancel=cancel,
            runtime=runtime,
        )
    except ReportEngineError as error:
        raise failure_of(error, watched.last, _provider(asked.info), asked.info.id) from error
    try:
        done = _write(report, where, teller)
    except StoreError as error:
        raise ReportFailure(
            "not_saved",
            "The report was written but could not be saved. Run the report again.",
            report.usage.model_dump(mode="json"),
        ) from error
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
