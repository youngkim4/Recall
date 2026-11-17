"""Invented exports and invented Messages databases for the tests of archive_state."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from tests.archive.builders import ChatDbBuilder

ZONE = ZoneInfo("America/Los_Angeles")
PERSON = "+15550190001"
OTHER = "+15550190002"
COLUMNS = (
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
START = datetime(2024, 7, 1, 9, 0, tzinfo=ZONE)


def write_export(path: Path, texts: Sequence[str], *, chat: str = PERSON, first_id: int = 1) -> Path:
    """An export of the ten columns that parse_imessage.py writes, one received message for each text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for place, text in enumerate(texts):
            stamp = (START + timedelta(minutes=place)).strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([first_id + place, stamp, chat, text, chat, "", 0, "iMessage", "", ""])
    return path


def write_database(path: Path, texts: Sequence[str], *, chat: str = PERSON) -> Path:
    """A Messages database with one received message for each text. A second call makes a new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    builder = ChatDbBuilder(path)
    handle = builder.handle(chat)
    thread = builder.chat(chat, members=[handle])
    for place, text in enumerate(texts):
        # the guid follows the text, so the same text is the same message in a later database
        builder.message(thread, when=START + timedelta(minutes=place), text=text, handle=handle, guid=f"guid-{text}")
    builder.close()
    return path


def texts_of(archive: object, chat: str = PERSON) -> list[str]:
    return [message["text"] for message in archive.messages(chat, limit=1_000)["messages"]]  # type: ignore[attr-defined]


def ids_of(archive: object, chat: str = PERSON) -> list[str]:
    return [message["id"] for message in archive.messages(chat, limit=1_000)["messages"]]  # type: ignore[attr-defined]


def write_database_of_events(path: Path, *, chat: str = "chat190000") -> Path:
    """A Messages database that holds rows and no message: one event of a group."""
    path.unlink(missing_ok=True)
    builder = ChatDbBuilder(path)
    handle = builder.handle(PERSON)
    group = builder.chat(chat, members=[handle])
    builder.message(group, when=START, handle=handle, item_type=2, group_title="Renamed")
    builder.close()
    return path


def files_in(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def kept_files(folder: Path) -> list[str]:
    """The files of a folder, without the two that SQLite makes beside an archive that is open for reading."""
    made_by_sqlite = {"archive.sqlite3-wal", "archive.sqlite3-shm"}
    return [name for name in files_in(folder) if name not in made_by_sqlite]
