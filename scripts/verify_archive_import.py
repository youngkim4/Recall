"""Check the archive importer against your own Messages database.

    ./venv/bin/python scripts/verify_archive_import.py
    ./venv/bin/python scripts/verify_archive_import.py --db /path/to/chat.db

The script imports the database into a temporary archive, then compares the
old attributedBody decoder (parse_imessage.py) with the new one.

It prints NUMBERS ONLY. It never prints message text, names, phone numbers or
email addresses. It does not change the database, and it deletes the
temporary archive when it ends. It needs Full Disk Access for the live
database, like the app does.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from archive import ImportResult, import_chat_db
from archive.classify import classify
from archive.errors import TypedStreamError
from archive.records import MessageRecord
from archive.source import open_snapshot
from archive.typedstream import read_text, scan_text

DEFAULT_DB = Path("~/Library/Messages/chat.db")
BATCH_ROWS = 1000
BUCKETS = ("under 128 bytes", "128 to 255 bytes", "256 bytes and more")
CHECK_TITLE = "decoder check (rows whose text is only in the blob)"


def bucket_of(byte_length: int) -> str:
    if byte_length < 128:
        return BUCKETS[0]
    return BUCKETS[1] if byte_length < 256 else BUCKETS[2]


def legacy_cleanup(text: str) -> str:
    """The cleanup that the old decoder applies to its result."""
    return "".join(char for char in text if char.isprintable() or char in "\n\t ").strip()


def load_legacy_decoder():  # type: ignore[no-untyped-def]
    try:
        from parse_imessage import extract_text_from_attributed_body
    except Exception:  # noqa: BLE001 - the check still runs without the old decoder
        return None
    return extract_text_from_attributed_body


def decode(blob: bytes) -> tuple[str | None, str]:
    """(text, how it was read): structure, marker scan, or not readable."""
    try:
        return read_text(blob), "read by structure"
    except TypedStreamError:
        pass
    try:
        return scan_text(blob), "read by marker scan"
    except TypedStreamError:
        return None, "not readable"


def check_decoders(db_path: Path, work_dir: Path) -> Counter[str]:
    """Count where the old decoder and the new one give text of another length."""
    legacy = load_legacy_decoder()
    counts: Counter[str] = Counter({"legacy decoder loaded": int(legacy is not None)})
    with open_snapshot(db_path, work_dir) as reader:
        for batch in reader.messages_after(0, BATCH_ROWS):
            for row in batch:
                record = classify(row, has_attachments=False)
                if not isinstance(record, MessageRecord) or record.text_source != "blob" or not row.body:
                    continue
                counts["rows checked"] += 1
                text, how = decode(row.body)
                counts[how] += 1
                if text is None:
                    continue
                bucket = bucket_of(len(text.encode("utf-8")))
                counts[f"{bucket}, rows"] += 1
                if legacy is None:
                    continue
                old_length = len(legacy(row.body) or "")
                counts[f"{bucket}, legacy length differs"] += int(old_length != len(text))
                counts[f"{bucket}, differs after the same cleanup"] += int(old_length != len(legacy_cleanup(text)))
    return counts


def import_lines(result: ImportResult, seconds: float) -> list[tuple[str, float]]:
    lines: list[tuple[str, float]] = [
        ("rows seen", result.rows_seen),
        ("messages", result.messages_seen),
        ("tapbacks", result.reactions_seen),
        ("system rows", result.system_seen),
        ("attachments", result.attachments_added),
        ("conversations", result.conversations_added),
        ("handles", result.handles_added),
        ("text from the blob", result.blob_texts),
        ("blobs not readable", result.decode_failures),
    ]
    lines.append(("messages with no conversation", result.counts.get("without_conversation", 0)))
    lines.append(("texts longer than the search limit", result.oversize_texts))
    lines.extend(
        (f"rows not stored, {reason.replace('_', ' ')}", count) for reason, count in sorted(result.skipped.items())
    )
    lines.append(("seconds", round(seconds, 2)))
    return lines


def check_lines(counts: Counter[str]) -> list[tuple[str, float]]:
    names = ["legacy decoder loaded", "rows checked", "read by structure", "read by marker scan", "not readable"]
    for bucket in BUCKETS:
        names.extend(
            f"{bucket}, {what}" for what in ("rows", "legacy length differs", "differs after the same cleanup")
        )
    return [(name, counts[name]) for name in names]


def print_section(title: str, lines: list[tuple[str, float]]) -> None:
    print(title)
    for label, value in lines:
        print(f"  {label}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import a Messages database into a temporary archive. Print numbers only."
    )
    parser.add_argument("--db", default=str(DEFAULT_DB), help="path to chat.db (default: ~/Library/Messages/chat.db)")
    db_path = Path(parser.parse_args().db).expanduser()
    try:
        with tempfile.TemporaryDirectory(prefix="recall-verify-") as folder:
            started = time.monotonic()
            result = import_chat_db(db_path, Path(folder) / "archive.sqlite3", full=True)
            seconds = time.monotonic() - started
            counts = check_decoders(db_path, Path(folder) / "decoder-check")
    except Exception as exc:  # noqa: BLE001 - the message of an error can wrap text of the database: print its type only
        print(f"failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print_section("import", import_lines(result, seconds))
    print_section(CHECK_TITLE, check_lines(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
