"""Load a legacy export into a temporary archive and print numbers only.

    ./venv/bin/python scripts/archive_bridge_check.py /path/to/messages.csv

The export is read and never changed. The archive is made in a temporary
folder outside the project and is deleted at the end. Nothing of the export
is printed: no text, no name, no phone number, no address. The search probes
are fixed common words, not words taken from the export.
"""

from __future__ import annotations

import argparse
import resource
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from archive import Archive, connection, import_legacy_csv, mirror

MIB = 1024 * 1024
REPEATS = 5
# generic probes: (label, query). The label is printed, the hits are only counted.
SEARCH_PROBES = (
    ("search common word", "the"),
    ("search plain word", "tomorrow"),
    ("search two words", "happy birthday"),
    ("search phrase", '"see you"'),
    ("search prefix", "happ*"),
    ("search emoji (substring)", chr(0x1F602)),
    ("search CJK (substring)", chr(0x7684)),
)


def peak_rss_mib() -> float:
    """Peak resident memory of this process. macOS reports bytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MIB


def median_ms(action: Callable[[], Any]) -> tuple[float, Any]:
    """Median time of REPEATS calls after one warm-up call, in milliseconds, and the last result."""
    result = action()
    times = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        result = action()
        times.append((time.perf_counter() - started) * 1000)
    return statistics.median(times), result


def show(label: str, value: object) -> None:
    print(f"  {label}: {value}")


def import_numbers(csv_path: Path, archive_path: Path) -> None:
    started = time.perf_counter()
    result = import_legacy_csv(csv_path, archive_path)
    seconds = time.perf_counter() - started
    print("load")
    show("export bytes", csv_path.stat().st_size)
    show("seconds", round(seconds, 2))
    show("rows per second", round(result.rows_seen / seconds) if seconds else 0)
    show("peak memory MiB", round(peak_rss_mib(), 1))
    show("archive bytes on disk", sum(item.stat().st_size for item in archive_path.parent.iterdir()))
    for name in (
        "rows_seen",
        "messages_added",
        "system_added",
        "messages_updated",
        "attachments_added",
        "conversations_added",
        "handles_added",
        "oversize_texts",
    ):
        show(name, getattr(result, name))
    show("reaction_phrases", result.counts.get("reaction_phrases", 0))
    for reason, count in sorted(result.skipped.items()):
        show(f"skipped {reason}", count)


def query_numbers(archive: Archive) -> None:
    print("archive")
    summary = archive.summary()
    show("messages", summary["messages"])
    show("conversations with messages", summary["conversations"])
    show("first day", (summary["first_timestamp"] or "")[:10])
    show("last day", (summary["last_timestamp"] or "")[:10])
    show("people", len(archive.people()))
    biggest = archive.conversations(limit=1)
    if not biggest:
        return
    big = biggest[0]["id"]
    show("messages in the largest conversation", biggest[0]["message_count"])
    figures = archive.stats(big)
    busiest = archive.day(big, figures["busiest_day"])
    if not busiest["messages"]:
        return
    anchor = busiest["messages"][len(busiest["messages"]) // 2]["id"]
    show("messages on its busiest day", busiest["total"])

    print("query latency, median of 5, milliseconds")
    probes: list[tuple[str, Callable[[], Any], Callable[[Any], object] | None]] = [
        ("summary", archive.summary, None),
        ("conversations(limit=50)", lambda: archive.conversations(limit=50), len),
        ("conversations()", archive.conversations, len),
        ("messages newest page", lambda: archive.messages(big, limit=50), None),
        ("messages oldest page", lambda: archive.messages(big, oldest=True, limit=50), None),
        ("messages(around=...)", lambda: archive.messages(big, around=anchor, limit=50), None),
        ("context(radius=6)", lambda: archive.context(anchor, radius=6), len),
        ("day (busiest)", lambda: archive.day(big, figures["busiest_day"]), lambda found: found["total"]),
        ("reactions of a page", lambda: archive.reactions([m["id"] for m in busiest["messages"][:50]]), None),
        ("count()", archive.count, None),
        ("count(conversation)", lambda: archive.count(conversations=big), None),
        ("count(since, until: one year)", lambda: archive.count(since="2024-01-01", until="2024-12-31"), None),
        ("count(group_by=day)", lambda: archive.count(group_by="day"), len),
        ("count(group_by=month)", lambda: archive.count(group_by="month"), len),
        ("count(group_by=weekday_hour)", lambda: archive.count(group_by="weekday_hour"), len),
        ("count(group_by=conversation)", lambda: archive.count(group_by="conversation"), len),
        ("count(group_by=sender)", lambda: archive.count(group_by="sender"), len),
        ("count(text=common word)", lambda: archive.count(text="the"), None),
        ("stats(largest)", lambda: archive.stats(big), None),
        ("iter_messages(largest)", lambda: sum(1 for _ in archive.iter_messages(big)), None),
        ("message_by_source_rowid", lambda: archive.message_by_source_rowid(1), None),
    ]
    for label, query in SEARCH_PROBES:
        probes.append((label, lambda query=query: archive.search(query, limit=50), lambda found: found["total"]))
    for label, action, size in probes:
        elapsed, result = median_ms(action)
        extra = "" if size is None else f"  (rows or hits: {size(result)})"
        show(label, f"{elapsed:.2f}{extra}")


def replace_numbers(archive_path: Path) -> None:
    """Time of the step that a first pull from the database runs: remove every bridge row. It is rolled back."""
    conn = connection.open_writer(archive_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        started = time.perf_counter()
        removed = mirror.clear_bridge_rows(conn)
        seconds = time.perf_counter() - started
        conn.execute("ROLLBACK")
    finally:
        conn.close()
    print("replace step of the first pull (rolled back)")
    show("bridge rows", removed)
    show("seconds", round(seconds, 2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Load a legacy export into a temporary archive. Print numbers only.")
    parser.add_argument("csv_path", help="path to messages.csv")
    csv_path = Path(parser.parse_args().csv_path).expanduser()
    folder = Path(tempfile.mkdtemp(prefix="recall-bridge-check-"))
    try:
        archive_path = folder / "archive.sqlite3"
        import_numbers(csv_path, archive_path)
        with Archive.open(archive_path) as archive:
            query_numbers(archive)
        replace_numbers(archive_path)
    except Exception as exc:  # noqa: BLE001 - the message of an error can hold private text: print its type only
        print(f"failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
