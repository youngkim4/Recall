#!/usr/bin/env python3
"""Compare the old attributedBody decoder with the current one on a Messages database.

For each message whose text column is NULL and whose attributedBody is not,
this decodes the blob with the old rule (kept below) and with
parse_imessage.extract_text_from_attributed_body, and counts where they differ.

It prints numbers only: no text, no names, no handles. It opens the database
read-only and writes nothing.

Usage (Terminal needs Full Disk Access to read the live database):
  python scripts/verify_decoder_counts.py
  python scripts/verify_decoder_counts.py --db /path/to/chat.db
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Iterable, Iterator, Optional
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parse_imessage import extract_text_from_attributed_body  # noqa: E402

DEFAULT_DB = Path.home() / "Library" / "Messages" / "chat.db"
QUERY = "SELECT attributedBody FROM message WHERE text IS NULL AND attributedBody IS NOT NULL"
MARKER = b"NSString"

# length of the text that the new rule gives, in UTF-8 bytes
BUCKETS = ("under 128", "128 to 255", "256 and more")
# how the old text differs from the new text
ENDS = "only the ends"
CHARACTERS = "characters dropped"
DAMAGED = "cut or fragment"
KINDS = (ENDS, CHARACTERS, DAMAGED)


def old_extract_text(blob: bytes) -> Optional[str]:
    """The decoder as it was before the fix. It is here for the comparison only.

    It reads 0x81 as "one length byte follows", drops each character that is
    not printable, trims the ends, and falls back to the longest run of ASCII.
    """
    if not blob:
        return None
    try:
        ns_string_marker = b'NSString'
        idx = blob.find(ns_string_marker)
        if idx != -1:
            search_start = idx + len(ns_string_marker)
            plus_idx = blob.find(b'+', search_start, search_start + 20)
            if plus_idx != -1 and plus_idx + 2 < len(blob):
                first_byte = blob[plus_idx + 1]
                if first_byte < 0x80:
                    text_len = first_byte
                    text_start = plus_idx + 2
                else:
                    num_len_bytes = first_byte & 0x7F
                    if 1 <= num_len_bytes <= 4 and plus_idx + 2 + num_len_bytes <= len(blob):
                        text_len = int.from_bytes(
                            blob[plus_idx + 2:plus_idx + 2 + num_len_bytes], 'big'
                        )
                        text_start = plus_idx + 2 + num_len_bytes
                    else:
                        text_len = first_byte
                        text_start = plus_idx + 2

                text_end = text_start + text_len
                if 0 < text_len and text_end <= len(blob):
                    try:
                        text = blob[text_start:text_end].decode('utf-8')
                        text = ''.join(c for c in text if c.isprintable() or c in '\n\t ')
                        if text.strip():
                            return text.strip()
                    except UnicodeDecodeError:
                        pass

        decoded = blob.decode('utf-8', errors='ignore')
        cleaned = re.sub(r'(streamtyped|NSAttributedString|NSMutableString|NSString|NSObject|NSDictionary|NSNumber|NSArray)', '', decoded)
        matches = re.findall(r'[a-zA-Z0-9\s\.\,\!\?\'\"\-\:\;\@\#\$\%\&\*\(\)\[\]\{\}\/\\]+', cleaned)
        matches = [m.strip() for m in matches if len(m.strip()) >= 2 and m.strip() not in ['iI', 'II', 'ii', 'NS']]
        if matches:
            return max(matches, key=len)
        return None
    except Exception:
        return None


def _old_filter(text: str) -> str:
    """What the old rule did to a text that it read whole."""
    return "".join(c for c in text if c.isprintable() or c in "\n\t ").strip()


def _bucket(text: str) -> str:
    size = len(text.encode("utf-8"))
    if size < 128:
        return BUCKETS[0]
    return BUCKETS[1] if size < 256 else BUCKETS[2]


def _kind(old: Optional[str], new: str) -> str:
    if old == new.strip():
        return ENDS
    if old == _old_filter(new):
        return CHARACTERS
    return DAMAGED


def tally(blobs: Iterable[object]) -> Counter:
    """Count how the two rules agree. The keys are labels, never message text."""
    counts: Counter = Counter()
    for blob in blobs:
        old = old_extract_text(blob)
        new = extract_text_from_attributed_body(blob)
        counts["checked"] += 1
        counts["old none"] += old is None
        counts["new none"] += new is None
        if isinstance(blob, (bytes, bytearray)) and MARKER not in blob:
            counts["no marker"] += 1
            counts["no marker, new text"] += new is not None
        if old == new:
            continue
        counts["differ"] += 1
        if new is None:
            counts["differ, new none"] += 1
            continue
        bucket = _bucket(new)
        counts[bucket] += 1
        counts[bucket, _kind(old, new)] += 1
    return counts


def report(counts: Counter) -> str:
    width = max(len(bucket) for bucket in BUCKETS)
    lines = [
        f"Rows with text NULL and attributedBody not NULL: {counts['checked']:,}",
        f"Rows where the old rule and the new rule differ: {counts['differ']:,}",
        "",
        "Rows that differ, by the UTF-8 length of the text that the new rule gives:",
        f"  {'bytes':<{width}}  {'differ':>10}" + "".join(f"  {kind:>18}" for kind in KINDS),
    ]
    for bucket in BUCKETS:
        cells = "".join(f"  {counts[bucket, kind]:>18,}" for kind in KINDS)
        lines.append(f"  {bucket:<{width}}  {counts[bucket]:>10,}{cells}")
    lines += [
        f"  the new rule gives None and the old rule gave text: {counts['differ, new none']:,}",
        "",
        f"  {ENDS}: the old rule trimmed spaces or line ends; the text is the same.",
        f"  {CHARACTERS}: the old rule dropped characters that are not printable.",
        f"  {DAMAGED}: the old text is cut, comes from another place in the blob, or is missing.",
        "",
        f"Rows where the new rule gives None: {counts['new none']:,}",
        f"Rows where the old rule gave None: {counts['old none']:,}",
        f"Rows with no NSString marker: {counts['no marker']:,}"
        f" (the new rule gives text for {counts['no marker, new text']:,} of them)",
    ]
    return "\n".join(lines)


def open_read_only(db_path: Path) -> sqlite3.Connection:
    """Open the database so that no statement can change it."""
    return sqlite3.connect(f"file:{quote(str(db_path))}?mode=ro", uri=True)


def read_blobs(conn: sqlite3.Connection) -> Iterator[object]:
    for (blob,) in conn.execute(QUERY):
        yield blob


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Count where the old and the new attributedBody decoder differ")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"Messages database (default: {DEFAULT_DB})")
    args = parser.parse_args(argv)

    try:
        with closing(open_read_only(args.db)) as conn:
            counts = tally(read_blobs(conn))
    except sqlite3.Error as exc:
        print(
            f"Cannot read {args.db}: {exc}\n"
            "For the live database, give Terminal Full Disk Access in System Settings > Privacy & Security.",
            file=sys.stderr,
        )
        return 1

    print(report(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
