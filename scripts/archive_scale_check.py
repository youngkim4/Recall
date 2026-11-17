"""Measure the archive importer and the query API on a large synthetic database.

    ./venv/bin/python scripts/archive_scale_check.py
    ./venv/bin/python scripts/archive_scale_check.py --messages 400000 --compare 50000 100000 200000

The script makes synthetic chat.db files of several sizes (invented data
only), imports each in its own process, and prints time and peak memory.
Memory is "flat" when the peak stops growing as the database gets larger.
The peak includes SQLite's page cache, which fills up to a fixed limit.

It never reads a real Messages database.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from archive import Archive, import_chat_db

MIB = 1024 * 1024


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # macOS reports bytes, Linux kibibytes


def timed(action: Callable[[], Any]) -> tuple[float, Any]:
    started = time.perf_counter()
    value = action()
    return time.perf_counter() - started, value


def child_import(source: Path, archive_path: Path, full: bool) -> dict[str, Any]:
    baseline = peak_rss_bytes()
    seconds, result = timed(lambda: import_chat_db(source, archive_path, full=full))
    return {
        "seconds": seconds,
        "baseline_rss": baseline,
        "peak_rss": peak_rss_bytes(),
        "mode": result.mode,
        "rows_seen": result.rows_seen,
        "messages_added": result.messages_added,
        "messages_seen": result.messages_seen,
        "reactions_seen": result.reactions_seen,
        "system_seen": result.system_seen,
        "attachments_added": result.attachments_added,
        "blob_texts": result.blob_texts,
        "decode_failures": result.decode_failures,
        "skipped": dict(result.skipped),
    }


def child_queries(archive_path: Path) -> dict[str, Any]:
    timings: dict[str, float] = {}
    with Archive.open(archive_path) as archive:

        def measure(name: str, action: Callable[[], Any]) -> Any:
            best, value = min((timed(action) for _ in range(3)), key=lambda pair: pair[0])
            timings[name] = best * 1000
            return value

        measure("summary()", archive.summary)
        top = measure("conversations(limit=50)", lambda: archive.conversations(limit=50))
        biggest = top[0]["id"]
        measure("conversations() all", archive.conversations)
        page = measure("messages(biggest) newest page", lambda: archive.messages(biggest))
        middle = page["messages"][0]["id"]
        measure("messages(biggest, before=...)", lambda: archive.messages(biggest, before=middle))
        measure("messages(biggest, around=...)", lambda: archive.messages(biggest, around=middle))
        measure("context(id)", lambda: archive.context(middle))
        measure('search("ferry")', lambda: archive.search("ferry"))
        measure('search("ferry ticket")', lambda: archive.search("ferry ticket"))
        measure('search("fer*")', lambda: archive.search("fer*"))
        measure('search("coffee", relevance)', lambda: archive.search("coffee", order="relevance"))
        measure('search("만나요") substring', lambda: archive.search("만나요"))
        measure('search("ferry", in biggest)', lambda: archive.search("ferry", conversations=[biggest]))
        measure("count()", archive.count)
        measure('count(group_by="day")', lambda: archive.count(group_by="day"))
        measure('count(group_by="month")', lambda: archive.count(group_by="month"))
        measure('count(group_by="weekday_hour")', lambda: archive.count(group_by="weekday_hour"))
        measure('count(group_by="conversation")', lambda: archive.count(group_by="conversation"))
        measure('count(group_by="sender")', lambda: archive.count(group_by="sender"))
        measure('count(text="ferry", group_by="year")', lambda: archive.count(text="ferry", group_by="year"))
        measure("count(biggest, group_by=month)", lambda: archive.count(conversations=[biggest], group_by="month"))
        measure("stats(biggest)", lambda: archive.stats(biggest))
        measure("people()", archive.people)
        before_stream = peak_rss_bytes()
        streamed = measure("iter_messages(biggest) all", lambda: sum(1 for _ in archive.iter_messages(biggest)))
        total = archive.summary()["messages"]
    return {
        "timings_ms": timings,
        "biggest_conversation_messages": streamed,
        "messages": total,
        "peak_rss": peak_rss_bytes(),
        "rss_growth_while_streaming": peak_rss_bytes() - before_stream,
    }


def run_child(*args: str) -> dict[str, Any]:
    result = subprocess.run([sys.executable, __file__, *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"child failed: {result.stderr[-2000:]}")
    return json.loads(result.stdout.splitlines()[-1])


def make_source(folder: Path, messages: int) -> tuple[Path, Any, float]:
    from tests.archive.synthetic import build_synthetic_chat_db

    path = folder / f"source-{messages}" / "chat.db"
    path.parent.mkdir(parents=True)
    seconds, counts = timed(lambda: build_synthetic_chat_db(path, messages))
    return path, counts, seconds


def report_import(label: str, source: Path, counts: Any, data: dict[str, Any]) -> None:
    rate = data["rows_seen"] / data["seconds"] if data["seconds"] else 0.0
    print(f"{label}")
    print(f"  source file: {source.stat().st_size / MIB:.1f} MiB, {counts.source_rows:,} rows")
    print(f"  mode: {data['mode']}, rows seen: {data['rows_seen']:,}, messages added: {data['messages_added']:,}")
    print(f"  time: {data['seconds']:.2f} s ({rate:,.0f} rows/s)")
    print(
        f"  peak memory: {data['peak_rss'] / MIB:.1f} MiB (before the import started: {data['baseline_rss'] / MIB:.1f} MiB)"
    )


def check_counts(counts: Any, data: dict[str, Any]) -> None:
    expected = (
        counts.source_rows,
        counts.messages,
        counts.reactions,
        counts.system_rows,
        counts.attachments,
        counts.blob_texts,
    )
    found = (
        data["rows_seen"],
        data["messages_seen"],
        data["reactions_seen"],
        data["system_seen"],
        data["attachments_added"],
        data["blob_texts"],
    )
    if expected != found or data["decode_failures"] or data["skipped"]:
        raise RuntimeError(
            f"import counts differ from the generator: expected {expected}, found {found}, {data['skipped']}"
        )


def parent(messages: int, compare: list[int]) -> None:
    with tempfile.TemporaryDirectory(prefix="recall-scale-") as name:
        folder = Path(name)
        results: dict[int, dict[str, Any]] = {}
        archives: dict[int, Path] = {}
        for size in (*sorted(compare), messages):
            source, counts, build_seconds = make_source(folder, size)
            archives[size] = folder / f"archive-{size}" / "archive.sqlite3"
            data = run_child("import", str(source), str(archives[size]))
            check_counts(counts, data)
            results[size] = data
            print(f"made a synthetic database with {size:,} messages in {build_seconds:.1f} s")
            report_import(f"first import, {size:,} messages", source, counts, data)
            if size == messages:
                again = run_child("import", str(source), str(archives[size]))
                report_import("second pull of the same database (full, the default)", source, counts, again)
                partial = run_child("import", str(source), str(archives[size]), "--incremental")
                report_import("third run with full=False (new rows only)", source, counts, partial)
                print(f"  archive file: {archives[size].stat().st_size / MIB:.1f} MiB")
        print("first import by size")
        print("  messages | seconds | rows per second | peak memory MiB")
        for size, data in results.items():
            rate = data["rows_seen"] / data["seconds"]
            print(f"  {size:>8,} | {data['seconds']:>7.2f} | {rate:>15,.0f} | {data['peak_rss'] / MIB:>15.1f}")
        sizes = list(results)
        growth = (results[sizes[-1]]["peak_rss"] - results[sizes[-2]]["peak_rss"]) / MIB
        print(f"  peak memory from {sizes[-2]:,} to {sizes[-1]:,} messages: {growth:+.1f} MiB")
        queries = run_child("queries", str(archives[messages]))
        print(f"queries on {queries['messages']:,} messages (best of 3, milliseconds)")
        for label, value in queries["timings_ms"].items():
            print(f"  {label}: {value:.2f}")
        print(f"  biggest conversation: {queries['biggest_conversation_messages']:,} messages")
        print(f"  memory growth while streaming it: {queries['rss_growth_while_streaming'] / MIB:+.1f} MiB")
        print(f"  peak memory of the query process: {queries['peak_rss'] / MIB:.1f} MiB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scale check on synthetic data.")
    parser.add_argument("mode", nargs="?", default="parent", choices=("parent", "import", "queries"))
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--messages", type=int, default=200_000)
    parser.add_argument("--compare", type=int, nargs="+", default=[50_000, 100_000])
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()
    if args.mode == "import":
        print(json.dumps(child_import(Path(args.paths[0]), Path(args.paths[1]), not args.incremental)))
    elif args.mode == "queries":
        print(json.dumps(child_queries(Path(args.paths[0]))))
    else:
        parent(args.messages, args.compare)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
