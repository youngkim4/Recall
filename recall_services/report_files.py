"""The files of a report that the server reads: the report file, the events file and the page of an older report.

`write` makes the report file and the events file hold one report. They
change together: when one of them can not be written, the files of the older
report stay as they were. The report file is written last, because the server
finds a report by that file.
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

import safe_html
from conversation import sanitize_filename
from report_engine import Report, StoreError, render_markdown
from report_engine.store import write_text_atomic

logger = logging.getLogger(__name__)

EVENT_COLUMNS = ("date", "title", "detail", "category", "score", "quote")
STORY_MARKER = "## Summary"
PAGE_REMOVED = "The page of an older report was removed: this report has no page."
PAGE_KEPT = "The page of an older report could not be removed. It is not the page of this report."


def story_of(report_text: str) -> str:
    """The story of a report file: what stands after `## Summary`, as `markdown_section` in the server reads it."""
    start = report_text.find(STORY_MARKER)
    start = report_text.find("\n", start) if start >= 0 else -1
    return report_text[start:].strip() if start >= 0 else ""


def shown_plain(report_text: str) -> bool:
    """True when the server shows the story of this report file as plain paragraphs: `story_html` does not render it."""
    return safe_html.story_work(safe_html.prepared(story_of(report_text))) > safe_html.STORY_WORK_LIMIT


def _events_text(report: Report) -> str | None:
    """The events file that the server reads, or None when the report has no moment."""
    if not report.moments:
        return None
    lines = io.StringIO(newline="")
    writer = csv.writer(lines, lineterminator="\n")
    writer.writerow(EVENT_COLUMNS)
    for moment in report.moments:
        writer.writerow([moment.date, moment.title, moment.detail, moment.category, moment.score, moment.quote])
    return lines.getvalue()


def _read(path: Path) -> str | None:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return handle.read()
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        raise StoreError(f"an older file can not be read: {path.name}") from error


def _put(path: Path, text: str | None) -> Path | None:
    """Make a file hold a text, or be gone for None. Raises StoreError when it can not."""
    if text is not None:
        return write_text_atomic(path, text)
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        raise StoreError(f"an older file can not be removed: {path.name}") from error
    return None


def _put_back(path: Path, text: str | None) -> None:
    try:
        _put(path, text)
    except (StoreError, OSError) as error:
        logger.warning("A file of the older report could not be put back (%s): %s", path.name, type(error).__name__)


def _remove_page(path: Path) -> str | None:
    """Remove the page of an older report. The sentence for the log, or None when there was no page."""
    try:
        path.unlink()
    except FileNotFoundError:
        return None
    except OSError as error:
        logger.warning("The page of an older report could not be removed (%s): %s", path.name, type(error).__name__)
        return PAGE_KEPT
    return PAGE_REMOVED


def write(report: Report, out_dir: Path) -> tuple[Path, Path | None, str, str | None]:
    """Make the files of a conversation hold this report.

    Returns the report file, the events file or None when the report has no
    moment, the text of the report file, and a sentence for the log about the
    page of an older report or None. Raises StoreError or OSError when the
    report could not be written; the files of the older report then stay.
    """
    name = sanitize_filename(report.conversation)
    report_path = out_dir / f"analysis_{name}.md"
    events_path = out_dir / f"events_timeline_{name}.csv"
    text, events_text = render_markdown(report), _events_text(report)
    older_events = _read(events_path)
    try:
        written_events = _put(events_path, events_text)
        write_text_atomic(report_path, text)  # last: the server finds a report by this file
    except (StoreError, OSError):
        _put_back(events_path, older_events)
        raise
    return report_path, written_events, text, _remove_page(out_dir / f"analysis_{name}.html")
