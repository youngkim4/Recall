"""Invented exports and invented Messages databases for the tests of the joined branch.

An export has the ten columns that parse_imessage.py writes. `load_both` reads
one export the two ways the app can read it: with the loader of
conversation.py (pandas), and into a new archive through the legacy bridge.

Every name, number and text here is invented.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from archive import Archive, ImportResult, import_legacy_csv
from conversation import in_conversation, load_messages
from tests.archive.builders import ChatDbBuilder, unix_seconds, utc

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_COLUMNS: tuple[str, ...] = (
    "message_id",
    "timestamp",
    "sender",
    "text",
    "chat_id",
    "chat_display_name",
    "is_from_me",
    "service",
    "attachment_types",
    "attachment_files",
)
ROWAN = "+15550100131"
# another person whose id starts with the whole id of Rowan: an id never matches in part
NEIGHBOUR = "+155501001319"
MIRA = "mira@example.com"
GROUP = "chat700100"
GROUP_TITLE = "Harbor crew"
PLACEHOLDER = chr(0xFFFC)
CURLY_OPEN, CURLY_CLOSE = chr(0x201C), chr(0x201D)
LEGACY_PREFIX = "legacy:"
STAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
APPLE_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "typedstream"


def export_row(
    number: int | str,
    stamp: str,
    text: str,
    chat: str = ROWAN,
    *,
    from_me: bool = False,
    sender: str | None = None,
    **more: Any,
) -> dict[str, Any]:
    """One row of an export. A sent row of a one-to-one chat names the other person, as chat.db does."""
    named = sender if sender is not None else ("" if chat.startswith("chat") else chat)
    row = {
        "message_id": number,
        "timestamp": stamp,
        "sender": named or None,
        "text": text,
        "chat_id": chat or None,
        "is_from_me": int(from_me),
        "service": "iMessage",
    }
    return {**row, **more}


def write_export(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write the rows with pandas, as the import of an earlier build of the app wrote an export."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows), columns=list(EXPORT_COLUMNS)).to_csv(path, index=False)
    return path


def stamps(first: str, count: int, *, step_seconds: int = 60) -> list[str]:
    """`count` local times, `step_seconds` apart, from `first`."""
    start = datetime.strptime(first, STAMP_FORMAT)
    return [(start + timedelta(seconds=step_seconds * place)).strftime(STAMP_FORMAT) for place in range(count)]


def legacy_id(number: object) -> str:
    """The id the bridge gives the row with this ROWID."""
    return f"{LEGACY_PREFIX}{number}"


def frame_ids(rows: pd.DataFrame) -> list[str]:
    """The ids of the rows of an export frame, written as the bridge writes them."""
    return [legacy_id(value) for value in rows["message_id"]]


def archive_ids(messages: Iterable[Mapping[str, Any]]) -> list[str]:
    return [str(message["id"]) for message in messages]


def frame_times(rows: pd.DataFrame) -> list[str]:
    """The times of the rows of a frame, written as the server writes them into a payload."""
    return [str(stamp) for stamp in rows["timestamp"]]


@dataclass(frozen=True)
class Loaded:
    """One export, read by the loader of the app and loaded into an archive."""

    csv_path: Path
    frame: pd.DataFrame
    archive: Archive
    result: ImportResult

    def conversation(self, chat: str) -> pd.DataFrame:
        """The rows of one conversation, as /api/day selects them."""
        return self.frame[in_conversation(self.frame, chat)]


def load_both(
    folder: Path, rows: Sequence[Mapping[str, Any]], *, contacts: Sequence[Mapping[str, object]] | None = None
) -> Loaded:
    """Write an export into `folder` and read it both ways. The caller closes `archive`.

    The bridge reads the local times of the export in the time zone of the
    process: call this inside the time zone that the test reads the archive in.
    """
    csv_path = write_export(folder / "messages.csv", rows)
    archive_path = folder / "data" / "archive.sqlite3"
    result = import_legacy_csv(csv_path, archive_path, contacts=contacts)
    return Loaded(csv_path, load_messages(str(csv_path)), Archive.open(archive_path), result)


@dataclass(frozen=True)
class AppleFixture:
    """One archive made by Apple's NSArchiver (tests/fixtures/typedstream)."""

    name: str
    blob: bytes
    utf8_bytes: int | None
    sha256: str | None
    text: str | None  # None when the manifest gives only the size and the hash of the text


def apple_fixtures() -> dict[str, AppleFixture]:
    """Every fixture of the decoder fix, by name."""
    listed = json.loads((APPLE_FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))["fixtures"]
    return {entry["name"]: _apple_fixture(entry) for entry in listed}


def _apple_fixture(entry: Mapping[str, Any]) -> AppleFixture:
    name = str(entry["name"])
    hexadecimal = (APPLE_FIXTURE_DIR / f"{name}.hex").read_text(encoding="ascii")
    stored = entry.get("utf8_hex")
    return AppleFixture(
        name=name,
        blob=bytes.fromhex("".join(hexadecimal.split())),
        utf8_bytes=entry.get("utf8_bytes"),
        sha256=entry.get("sha256"),
        text=None if stored is None else bytes.fromhex(stored).decode("utf-8"),
    )


@dataclass(frozen=True)
class BlobDatabase:
    """An invented Messages database whose texts are only in attributedBody."""

    path: Path
    rowids: Mapping[str, int]  # label of a row -> its ROWID
    sent_at: Mapping[str, int]  # label of a row -> UTC unix seconds


def blob_database(path: Path, blobs: Mapping[str, bytes], *, tapback_text: str) -> BlobDatabase:
    """A one-to-one chat with Rowan: one message for each blob, then a photo and a tapback.

    The text column of a blob row is NULL, as a current macOS writes it. The
    photo and the tapback are there because a real database has both, and the
    exporter writes a file for each.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    builder = ChatDbBuilder(path)
    try:
        rowan = builder.handle(ROWAN)
        chat = builder.chat(ROWAN, members=[rowan])
        rowids: dict[str, int] = {}
        sent_at: dict[str, int] = {}

        def add(label: str, minute: int, **message: Any) -> int:
            when = utc(2024, 5, 6, 17, minute)
            rowids[label] = builder.message(chat, when=when, handle=rowan, **message)
            sent_at[label] = unix_seconds(when)
            return rowids[label]

        first = add("text_column", 0, text="See you at eight", from_me=True)
        for place, (name, blob) in enumerate(blobs.items(), start=1):
            add(name, place, body=blob)
        photo = add("photo", len(blobs) + 1)
        builder.attachment(photo, mime_type="image/jpeg", transfer_name="IMG_0042.jpeg")
        when = utc(2024, 5, 6, 17, len(blobs) + 2)
        rowids["tapback"] = builder.tapback(
            chat, builder.guid_of(first), 2000, when=when, handle=rowan, text=tapback_text
        )
        sent_at["tapback"] = unix_seconds(when)
    finally:
        builder.close()
    return BlobDatabase(path, rowids, sent_at)
