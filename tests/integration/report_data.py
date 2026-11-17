"""Invented conversations in the real archive, for the report tests of the joined branch.

Two ways fill an archive here. Each test says which one it uses.

- `Pulled`: an invented chat.db, made with the builder of the archive tests and
  read with `import_chat_db`. This is the path of the app. A message id is
  Apple's guid, and rows can be added for a second pull.
- `bridged`: an invented export with the ten columns of `parse_imessage.py`,
  loaded with `import_legacy_csv`. A message id is `legacy:<ROWID>`.

The lines are the invented chatter of the report engine's own tests
(`tests/report_engine/synth.py`). Their time stamps are local times of the
test zone. A report runs with the engine's own test helper, so it uses the
fake model, a counter of one token for each word and a fixed clock.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from archive import Archive, ImportResult, import_chat_db, import_legacy_csv
from report_engine import PeriodCache, Report, estimate
from report_engine.archive_source import ArchiveSource
from report_engine.fake_llm import FakeLLM
from report_engine.ports import SourceMessage
from tests import guards
from tests.archive.builders import ChatDbBuilder
from tests.report_engine.support import SMALL_OPTIONS, run
from tests.report_engine.synth import MINA, RAVI, THEO, WordCounter, model_info

LOCAL_ZONE = ZoneInfo(guards.TEST_TIME_ZONE)
STAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
GROUP_TITLE = "Harbor Walkers"
MINA_NAME = "Mina"
MINA_CONTACT = {"name": MINA_NAME, "phones": ["(202) 555-0143"], "emails": []}
EXPORT_COLUMNS = (
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
# every handle of the invented people, in the spellings a prompt must never hold
HANDLE_PARTS = (MINA, RAVI, THEO, "2025550143", "2025550171", "555-0143", "555-0171")

Labeler = Callable[[str], str]


def moment(stamp: str) -> datetime:
    """The UTC moment of a local time stamp of the test zone."""
    return datetime.strptime(stamp, STAMP_FORMAT).replace(tzinfo=LOCAL_ZONE).astimezone(UTC)


def is_group(identifier: str) -> bool:
    """The ids of the invented groups all start with `chat`. No invented handle does."""
    return identifier.startswith("chat")


def report_labeler(saved: Mapping[str, str], contact_name: str = "") -> Labeler:
    """The labeler that the wiring must give the report engine.

    - A sender with a saved name gets that name.
    - A sender with no saved name gets the empty string, never words such as
      "an unsaved contact": the engine reads one name as one person, and it
      gives each empty name a label of its own.
    - The empty sender is a received message with no handle. In a one-to-one
      conversation it is the contact, so it gets the contact's name. In a
      group nobody knows who it is: `contact_name` stays empty.
    """

    def label(sender: str) -> str:
        return saved.get(sender, "") if sender else contact_name

    return label


class Pulled:
    """An invented chat.db and the archive that `import_chat_db` makes of it."""

    def __init__(self, folder: Path) -> None:
        self.source = folder / "source" / "chat.db"
        self.archive_path = folder / "data" / "archive.sqlite3"
        self.source.parent.mkdir(parents=True)
        self.builder = ChatDbBuilder(self.source)
        self._handles: dict[str, int] = {}
        self._chats: dict[str, int] = {}

    def handle(self, text: str) -> int:
        if text not in self._handles:
            self._handles[text] = self.builder.handle(text)
        return self._handles[text]

    def chat(self, identifier: str, members: Sequence[str] = (), title: str | None = None) -> int:
        """The chat row of a conversation. The member of a one-to-one conversation is its handle."""
        if identifier not in self._chats:
            people = members or (() if is_group(identifier) else (identifier,))
            rows = [self.handle(person) for person in people]
            self._chats[identifier] = self.builder.chat(identifier, display_name=title, members=rows)
        return self._chats[identifier]

    def write(self, identifier: str, messages: Sequence[SourceMessage]) -> list[int]:
        """Add messages to a conversation. Returns their ROWIDs in the source."""
        chat = self.chat(identifier)
        direct = not is_group(identifier)
        rowids = []
        for message in messages:
            own = bool(message["is_from_me"])
            # chat.db names the other person on an own one-to-one message
            sender = identifier if own and direct else message["sender"]
            rowids.append(
                self.builder.message(
                    chat,
                    when=moment(message["timestamp"]),
                    text=message["text"],
                    from_me=own,
                    handle=self.handle(sender) if sender else 0,
                )
            )
        return rowids

    def pull(self, contacts: Sequence[Mapping[str, object]] | None = None) -> ImportResult:
        """A full pull of the source as it is now."""
        self.builder.commit()
        return import_chat_db(self.source, self.archive_path, contacts=contacts)

    def source_guids(self) -> set[str]:
        """The guid of every row of the source, as it is now."""
        return {str(row[0]) for row in self.builder.raw.execute("SELECT guid FROM message")}

    def open(self) -> Archive:
        return Archive.open(self.archive_path)

    def close(self) -> None:
        self.builder.close()


def export_row(message: SourceMessage, chat_id: str, title: str = "") -> dict[str, Any]:
    """One row of the export. A sent one-to-one row names the other person, as the exporter does."""
    own = bool(message["is_from_me"])
    direct = not is_group(chat_id)
    return {
        "message_id": message["id"],
        "timestamp": message["timestamp"],
        "sender": chat_id if own and direct else message["sender"],
        "text": message["text"],
        "chat_id": chat_id,
        "chat_display_name": title,
        "is_from_me": int(own),
        "service": "iMessage",
        "attachment_types": "",
        "attachment_files": "",
    }


def write_export(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def bridged(folder: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """An archive that the legacy bridge loads from an invented export. Returns its path."""
    archive_path = folder / "bridge" / "archive.sqlite3"
    import_legacy_csv(write_export(folder / "export" / "messages.csv", rows), archive_path)
    return archive_path


def run_on(
    archive: Archive,
    conversation: str,
    labeler: Labeler,
    *,
    llm: FakeLLM | None = None,
    cache: PeriodCache | None = None,
    scope: Mapping[str, str | None] | None = None,
) -> Report:
    """One report through ArchiveSource, with the engine's small test options."""
    return run(
        (),
        source=ArchiveSource(archive),
        conversation_id=conversation,
        labeler=labeler,
        llm=llm,
        cache=cache,
        scope=scope,
    )


def estimate_on(
    archive: Archive,
    conversation: str,
    labeler: Labeler,
    cache: PeriodCache | None,
    scope: Mapping[str, str | None] | None = None,
) -> dict[str, int | float]:
    """The estimate with the labeler, options and counter of `run_on`."""
    return estimate(
        ArchiveSource(archive),
        conversation,
        scope,
        model_info(),
        cache,
        labeler=labeler,
        options=SMALL_OPTIONS,
        counter=WordCounter(),
    )


def prompt_text(llm: FakeLLM) -> str:
    """Everything the fake model was sent, in one text."""
    return "\n".join(call.text for call in llm.calls)


def labels_seen(llm: FakeLLM) -> set[str]:
    """The speaker names in the transcripts that the fake model read."""
    return {line.label for call in llm.map_calls for line in call.lines}
