"""The summaries that the memory read before, for the context line of the segments. No model call.

The memory (`memory_prep`) reads each part of a conversation one time and
keeps its summary in the notes of the reports. This module reads them back:
it lays out the parts as a digest of the whole conversation does, with the
model, the options and the token counter of the memory, and takes the summary
of each part that the notes hold. A part that is not read yet has no summary.

Rules:
- It makes no request and no file. With no notes file there is no summary.
- A summary is masked by the function that masks the messages of the index,
  and cut to about SUMMARY_CHARS characters at a space.
- A conversation that can not be laid out, or notes that can not be read,
  give no summary and one log line with the kind of the failure. The build
  goes on: the context line then holds no memory.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from archive import Archive
from archive import ArchiveError as ArchiveFault
from recall_services import models, reports
from recall_services.names import Names
from report_engine import PeriodCache, ReportEngineError, ReportOptions, StoreError
from report_engine.chat_digest_refs import without_refs
from report_engine.chat_digest_span import span_plan
from report_engine.digest import DigestError, parse_digest
from report_engine.tokens import TokenCounter

logger = logging.getLogger(__name__)

SUMMARY_CHARS = 200
CUT_MARK = "..."
_FAULTS = (ReportEngineError, ArchiveFault, ValueError, models.ModelChoiceError)


@dataclass(frozen=True)
class MemoryNotes:
    """Where and how the memory read its parts. None is what the memory job uses, read at each call."""

    places: reports.ReportPlaces | None = None
    model: str | None = None
    options: ReportOptions | None = None
    counter: TokenCounter | None = None
    min_messages: int | None = None


@dataclass(frozen=True)
class PartSummary:
    """The summary of one part of a conversation, from its first day to its last day."""

    first_day: str
    last_day: str
    summary: str


def shortened(text: str, limit: int = SUMMARY_CHARS) -> str:
    """The text on one line, cut at the last space before `limit` characters, with a mark when cut."""
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit].rsplit(" ", 1)[0] if " " in flat[:limit] else flat[:limit]
    return cut.rstrip(" ,;:.") + CUT_MARK


def _memory_model(chosen: MemoryNotes) -> str:
    from recall_services import memory_prep  # memory_prep reaches this module through meaning

    return chosen.model or memory_prep.model()


def _least(chosen: MemoryNotes) -> int:
    from recall_services import memory_prep

    return memory_prep.least(chosen.min_messages)


def open_notes(chosen: MemoryNotes) -> PeriodCache | None:
    """The notes of the reports, or None when there is no file or it can not be opened. It makes no file."""
    path = (chosen.places or reports.default_places()).cache_path
    if not path.exists():
        return None
    try:
        return PeriodCache(path)
    except StoreError as error:
        logger.warning("the notes of the memory can not be read (%s); no context line holds a memory", type(error).__name__)
        return None


def part_summaries(
    archive: Archive,
    names: Names,
    row: Mapping[str, Any],
    redact: Callable[[str], str],
    notes: PeriodCache,
    chosen: MemoryNotes,
) -> tuple[PartSummary, ...]:
    """The summary of each part of one conversation that the notes hold, oldest first.

    `redact` is the mask of chat, as a digest reads the messages: the parts
    then have the keys of the notes.
    """
    try:
        info = models.model_info(_memory_model(chosen))
        span = span_plan(
            reports._Masked(archive, redact),
            str(row["id"]),
            None,
            None,
            names.report_labeler(row),
            info,
            chosen.options or ReportOptions(),
            chosen.counter or reports.shared_counter(info.tokenizer_family),
        )
        planned = span.whole.periods
        entries = notes.get_many([part.key for part in planned])
    except _FAULTS as error:
        logger.info("a conversation gives no memory to its context lines (%s)", type(error).__name__)
        return ()
    found: list[PartSummary] = []
    for part in planned:
        entry = entries.get(part.key)
        try:
            digest = None if entry is None else parse_digest(entry.payload)
        except DigestError:
            digest = None
        if digest is not None:
            found.append(PartSummary(part.period.first_day, part.period.last_day, without_refs(digest.summary)))
    return tuple(found)


class Memory:
    """The summary of the part of a day, for the context line. Each conversation is laid out one time."""

    def __init__(
        self, archive: Archive, names: Names, redact: Callable[[str], str], chosen: MemoryNotes | None = None
    ) -> None:
        self._archive = archive
        self._names = names
        self._redact = redact
        self._chosen = chosen or MemoryNotes()
        self._notes = open_notes(self._chosen)
        self._parts: dict[str, tuple[PartSummary, ...]] = {}

    def _read(self, row: Mapping[str, Any]) -> tuple[PartSummary, ...]:
        if self._notes is None or int(row.get("message_count") or 0) < _least(self._chosen):
            return ()
        return part_summaries(self._archive, self._names, row, self._redact, self._notes, self._chosen)

    def summary(self, row: Mapping[str, Any], day: str) -> str:
        """The masked, short summary of the part that holds `day`, or "" when the memory has none."""
        if self._notes is None:
            return ""
        key = str(row.get("id") or "")
        if key not in self._parts:
            self._parts[key] = self._read(row)
        part = next((one for one in self._parts[key] if one.first_day <= day <= one.last_day), None)
        return "" if part is None else shortened(self._redact(part.summary))
