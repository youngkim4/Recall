"""Write the eval world into a synthetic chat.db and pull it into a real archive.

`world()` gives the messages with no file at all: the questions read it.
`build_world(folder)` writes `folder/source/chat.db` with the builder of the
tests (`tests.archive.builders.ChatDbBuilder`) and imports it with
`archive.import_chat_db`, the way the app pulls a real history. The same
call always gives the same rows, guids and archive.

Times are local times of America/Los_Angeles. Run the import and the
questions in that zone (`evals.world.zone`), so each message keeps its day.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from archive import import_chat_db
from evals.world.filler import filler
from evals.world.people import BY_KEY, CHATS, CONTACTS
from evals.world.scenes_friends import friend_scenes
from evals.world.scenes_home import home_scenes
from evals.world.scenes_wren import WREN_QUIET_DAYS, wren_scenes
from evals.world.script import ZONE, Item, Line, Tapback
from recall_services.names import Names
from tests.archive.builders import ChatDbBuilder

# the two days of the trip with no scene: Sam and Wren were together
TRIP_QUIET_DAYS = frozenset({"2024-05-19", "2024-05-20"})
# the text Messages keeps on a tapback row; the archive stores tapbacks apart from messages
TAPBACK_TEXT = "Reacted"


class WorldError(Exception):
    """The world can not be built, or a label is unknown."""


@dataclass(frozen=True)
class World:
    """Every message and tapback of the eval world, oldest first."""

    lines: tuple[Line, ...]
    tapbacks: tuple[Tapback, ...]

    def labelled(self, label: str) -> Line:
        for line in self.lines:
            if line.label == label:
                return line
        raise WorldError(f"no message has the label {label}")

    def guids(self, labels: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(self.labelled(label).guid for label in labels)

    def of(self, chat: str) -> tuple[Line, ...]:
        return tuple(line for line in self.lines if line.conversation == chat)


@dataclass(frozen=True)
class Built:
    """A world in files: the source database, the archive and the saved names."""

    world: World
    source_path: Path
    archive_path: Path
    names: Names


def scenes() -> tuple[Item, ...]:
    """Every written scene, before the filler."""
    return (*wren_scenes(), *friend_scenes(), *home_scenes())


@lru_cache(maxsize=1)
def world() -> World:
    """The world, built once for each process. It is immutable."""
    written = scenes()
    quiet = {"wren": WREN_QUIET_DAYS | TRIP_QUIET_DAYS}
    items: list[Item] = [*written, *filler(written, quiet)]
    ordered = sorted(items, key=lambda item: item.when)
    return World(
        lines=tuple(item for item in ordered if isinstance(item, Line)),
        tapbacks=tuple(item for item in ordered if isinstance(item, Tapback)),
    )


@contextmanager
def zone() -> Iterator[None]:
    """Run a block in the time zone of the world."""
    previous = os.environ.get("TZ")
    os.environ["TZ"] = ZONE.key
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


def _write(made: World, source: Path) -> None:
    builder = ChatDbBuilder(source)
    try:
        handles = {person.key: builder.handle(person.handle) for person in BY_KEY.values()}
        chats = {
            chat.key: builder.chat(
                chat.identifier,
                display_name=chat.title or None,
                members=[handles[member] for member in chat.members],
            )
            for chat in CHATS
        }
        ordered: list[Item] = sorted([*made.lines, *made.tapbacks], key=lambda item: item.when)
        for item in ordered:
            sender = 0 if item.sender == "me" else handles[item.sender]
            builder.message(
                chats[item.conversation], when=item.when, from_me=item.sender == "me", handle=sender,
                guid=item.guid, text=item.text if isinstance(item, Line) else TAPBACK_TEXT,
                associated_guid=f"p:0/{item.target}" if isinstance(item, Tapback) else None,
                associated_type=item.code if isinstance(item, Tapback) else 0,
            )  # fmt: skip
    finally:
        builder.close()


def build_world(folder: Path) -> Built:
    """Write the world under `folder` and pull it into `folder/data/archive.sqlite3`."""
    made = world()
    source = Path(folder) / "source" / "chat.db"
    archive_path = Path(folder) / "data" / "archive.sqlite3"
    if source.exists() or archive_path.exists():
        raise WorldError(f"the folder already holds a world: {folder}")
    source.parent.mkdir(parents=True)
    archive_path.parent.mkdir(parents=True)
    with zone():
        _write(made, source)
        import_chat_db(source, archive_path, contacts=[dict(record) for record in CONTACTS])
    return Built(made, source, archive_path, Names.from_records(CONTACTS))
