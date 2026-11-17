"""The "Orchard" history: a small invented archive for the service tests.

Every name, number and text is invented. The history is written into a
synthetic chat.db and pulled into a real archive with `import_chat_db`.

Conversations:
- a one-to-one thread with Ines's phone number, with a long day of 450 messages
- a one-to-one thread with Ines's email address (the same person)
- a one-to-one thread with a number nobody saved
- the group "Garden Club" with Ines, Tomas and the unsaved number
- a group with no title: Tomas, Ines and an unsaved address
- one message that belongs to no conversation

Times are local times of America/Los_Angeles, the zone of the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from archive import import_chat_db
from tests.archive.builders import ChatDbBuilder

ZONE = ZoneInfo("America/Los_Angeles")
INES_PHONE = "+15550170001"
INES_EMAIL = "ines@example.org"
TOMAS = "+15550170002"
UNSAVED = "+15550179999"
UNSAVED_EMAIL = "stranger@example.net"
GARDEN = "chat900100"
GARDEN_TITLE = "Garden Club"
UNTITLED = "chat900200"
UNTITLED_NAME = "Tomas Reyes, Ines Calder and 1 other"

RECORDS: list[dict[str, Any]] = [
    {"name": "Ines Calder", "phones": ["(555) 017-0001"], "emails": ["Ines@Example.org"]},
    {"name": "Tomas Reyes", "phones": ["+1 555 017 0002"], "emails": []},
]

LONG_DAY = "2024-03-02"
LONG_DAY_MESSAGES = 450
QUIET_DAY = "2024-03-03"
LONG_TEXT_WORD = "quince"
# longer than the 600 characters a result shows; the rare word stands after the cut
LONG_TEXT = ("The pear trees need a slow and careful winter prune. " * 14) + f"Then we plant the {LONG_TEXT_WORD}."


def la(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=ZONE)


@dataclass
class Orchard:
    """The archive file and the ids of the messages that the tests name."""

    archive_path: Path
    source_path: Path
    guids: dict[str, str] = field(default_factory=dict)
    rowids: dict[str, int] = field(default_factory=dict)

    def id(self, label: str) -> str:
        return self.guids[label]


class _Writer:
    def __init__(self, builder: ChatDbBuilder, world: Orchard) -> None:
        self.builder = builder
        self.world = world

    def say(self, label: str, chat: int | None, when: datetime, text: str | None, sender: int | None, **more: Any) -> int:
        rowid = self.builder.message(chat, when=when, text=text, from_me=sender is None, handle=sender or 0, **more)
        self.world.rowids[label] = rowid
        self.world.guids[label] = self.builder.guid_of(rowid)
        return rowid


def _ines_thread(w: _Writer, chat: int, ines: int) -> None:
    w.say("i1", chat, la(2024, 3, 1, 9, 0), "Are you bringing the ladder on Saturday?", ines)
    w.say("i2", chat, la(2024, 3, 1, 9, 5), "Yes, and the pruning saw", None)
    photo = w.say("i3", chat, la(2024, 3, 1, 9, 6), None, ines)
    w.builder.attachment(photo, mime_type="image/heic", transfer_name="IMG_0301.HEIC")
    w.say("i4", chat, la(2024, 3, 1, 18, 30), "  Picked the first lemons today  ", None)
    start = la(2024, 3, 2, 8, 0)
    for number in range(LONG_DAY_MESSAGES):
        sender = ines if number % 3 else None
        w.say(f"long{number}", chat, start + timedelta(minutes=number), f"pruning note {number}", sender)
    w.say("i5", chat, la(2024, 3, 5, 7, 45), LONG_TEXT, ines)
    liked = w.say("i6", chat, la(2024, 3, 5, 7, 50), "The apricot is in bloom", None)
    w.builder.tapback(chat, w.builder.guid_of(liked), 2000, when=la(2024, 3, 5, 7, 51), handle=ines)
    w.say("i7", chat, la(2024, 3, 5, 7, 52), "Same second, first", ines)
    w.say("i8", chat, la(2024, 3, 5, 7, 52), "Same second, second", ines)
    two = w.say("i9", chat, la(2024, 3, 5, 8, 0), None, None)
    w.builder.attachment(two, mime_type="image/jpeg", transfer_name="IMG_0302.jpeg")
    w.builder.attachment(two, mime_type="image/png", transfer_name="IMG_0303.png")


def _groups(w: _Writer, garden: int, untitled: int, people: dict[str, int]) -> None:
    ines, tomas, unsaved, stranger = (people[name] for name in ("ines", "tomas", "unsaved", "stranger"))
    w.say("g_event", garden, la(2024, 4, 1, 9, 0), None, ines, item_type=2, group_title=GARDEN_TITLE)
    w.say("g1", garden, la(2024, 4, 1, 10, 0), "Who has the wheelbarrow?", ines)
    w.say("g2", garden, la(2024, 4, 1, 10, 1), "I do, behind the shed", tomas)
    w.say("g3", garden, la(2024, 4, 1, 10, 2), "The wheelbarrow has a flat tire", unsaved)
    w.say("g4", garden, la(2024, 4, 1, 10, 3), "I will bring a pump", None)
    w.say("g5", garden, la(2024, 4, 2, 10, 0), "Tire fixed", tomas)
    for number in range(5):
        w.say(f"u_t{number}", untitled, la(2024, 5, 1, 9, number), f"compost turn {number}", tomas)
    for number in range(3):
        w.say(f"u_i{number}", untitled, la(2024, 5, 1, 10, number), f"mulch order {number}", ines)
    w.say("u_s0", untitled, la(2024, 5, 1, 11, 0), "Is the compost ready?", stranger)
    w.say("u_me", untitled, la(2024, 5, 1, 11, 5), "Almost, two more weeks", None)


def build_orchard(folder: Path) -> Orchard:
    """Write the history and pull it into an archive under `folder`."""
    source = folder / "source" / "chat.db"
    source.parent.mkdir(parents=True)
    world = Orchard(archive_path=folder / "data" / "archive.sqlite3", source_path=source)
    builder = ChatDbBuilder(source)
    w = _Writer(builder, world)
    people = {
        "ines": builder.handle(INES_PHONE),
        "ines_mail": builder.handle(INES_EMAIL),
        "tomas": builder.handle(TOMAS),
        "unsaved": builder.handle(UNSAVED),
        "stranger": builder.handle(UNSAVED_EMAIL),
    }
    ines_chat = builder.chat(INES_PHONE, members=[people["ines"]])
    mail_chat = builder.chat(INES_EMAIL, members=[people["ines_mail"]])
    unsaved_chat = builder.chat(UNSAVED, members=[people["unsaved"]])
    garden = builder.chat(GARDEN, display_name=GARDEN_TITLE, members=[people[n] for n in ("ines", "tomas", "unsaved")])
    untitled = builder.chat(UNTITLED, members=[people[n] for n in ("tomas", "ines", "stranger")])

    _ines_thread(w, ines_chat, people["ines"])
    w.say("m1", mail_chat, la(2024, 3, 6, 12, 0), "Sending the seed list from my laptop", people["ines_mail"])
    w.say("m2", mail_chat, la(2024, 3, 6, 12, 5), "Got it, thank you", None)
    w.say("s1", unsaved_chat, la(2024, 3, 7, 15, 0), "Your delivery of mulch is on its way", people["unsaved"])
    w.say("s2", unsaved_chat, la(2024, 3, 7, 15, 5), "Leave it by the gate please", None)
    w.say("s3", unsaved_chat, la(2024, 3, 7, 15, 9), "Call 555 017 9999 when you arrive", None)
    _groups(w, garden, untitled, people)
    w.say("lost", None, la(2024, 6, 1, 9, 0), "A note that no conversation holds", people["ines"])
    builder.close()
    import_chat_db(source, world.archive_path, contacts=RECORDS)
    return world
