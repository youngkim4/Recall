"""Make a large synthetic chat.db for scale and memory checks.

Everything is invented: numbers in the 555-01xx range, example.com
addresses, and sentences drawn at random from a short word list. The blobs
are the Apple-made typedstream fixtures.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from tests.archive.builders import APPLE_EPOCH_UNIX, ChatDbBuilder
from tests.archive.typedstream_fixtures import blob

WORDS = [
    "harbor",
    "ferry",
    "ticket",
    "morning",
    "coffee",
    "later",
    "tonight",
    "weekend",
    "lake",
    "canoe",
    "dinner",
    "seven",
    "movie",
    "train",
    "station",
    "late",
    "early",
    "sorry",
    "thanks",
    "great",
    "sure",
    "maybe",
    "tomorrow",
    "today",
    "rain",
    "sunny",
    "picnic",
    "garden",
    "kitchen",
    "recipe",
    "soup",
    "bread",
    "market",
    "street",
    "corner",
    "library",
    "book",
    "chapter",
    "music",
    "concert",
    "guitar",
    "piano",
    "dance",
    "practice",
    "school",
    "office",
    "meeting",
    "report",
    "lunch",
    "break",
    "walk",
    "river",
    "bridge",
    "mountain",
    "trail",
    "camp",
    "fire",
    "story",
    "laugh",
    "photo",
    "birthday",
    "cake",
    "candle",
    "gift",
    "surprise",
    "party",
    "friend",
    "sister",
    "brother",
    "cousin",
    "uncle",
    "window",
    "door",
    "key",
    "phone",
]
BLOB_NAMES = ("part_attribute", "korean", "accented", "newline", "emoji_end", "ascii_127", "ascii_128", "ascii_300")
START_UNIX = 1_483_228_800  # 2017-01-01T00:00:00Z
ONE_TO_ONE = 280
GROUPS = 20
BATCH = 5000
MESSAGE_SQL = (
    "INSERT INTO message (ROWID, guid, text, attributedBody, handle_id, is_from_me, date, service, item_type,"
    " associated_message_guid, associated_message_type, thread_originator_guid, date_edited, date_retracted,"
    " group_title) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


@dataclass(frozen=True)
class SyntheticCounts:
    """What the importer must find in the generated database."""

    source_rows: int
    messages: int
    reactions: int
    system_rows: int
    attachments: int
    blob_texts: int
    conversations: int
    handles: int


def _guid(number: int) -> str:
    return f"{number:08X}-5EED-4000-8000-{number:012X}"


def _apple_ns(unix_seconds: int) -> int:
    return (unix_seconds - APPLE_EPOCH_UNIX) * 1_000_000_000


def _directory(builder: ChatDbBuilder, rng: random.Random) -> tuple[list[int], list[list[int]]]:
    """Handles and chats. Returns (chat ROWIDs, member handle ROWIDs of each chat)."""
    phones = [builder.handle(f"+1555010{index:04d}") for index in range(300)]
    emails = [builder.handle(f"person{index:03d}@example.com") for index in range(100)]
    handles = phones + emails
    chats: list[int] = []
    members: list[list[int]] = []
    for index in range(ONE_TO_ONE):
        chats.append(builder.chat(f"+1555010{index:04d}", members=[phones[index]]))
        members.append([phones[index]])
    for index in range(GROUPS):
        crew = rng.sample(handles, rng.randint(3, 8))
        chats.append(builder.chat(f"chat9{index:09d}", display_name=f"Invented Group {index}", members=crew))
        members.append(crew)
    return chats, members


def _rows(
    messages: int, chats: list[int], members: list[list[int]], rng: random.Random, tally: dict[str, int]
) -> Iterator[tuple]:
    """Yield ("message" | "link" | "attachment" | "file_link", values) in ROWID order."""
    weights = [1 / (rank + 1) ** 1.1 for rank in range(len(chats))]
    blobs = [blob(name) for name in BLOB_NAMES]
    photo, caption = blob("attachment_only"), blob("attachment_with_caption")
    recent: dict[int, str] = {}
    rowid = 0
    attachment_rowid = 0
    now = START_UNIX
    made = 0
    while made < messages:
        now += rng.randint(30, 2370)
        slot = rng.choices(range(len(chats)), weights)[0]
        chat, crew = chats[slot], members[slot]
        from_me = rng.random() < 0.45
        handle = 0 if from_me and len(crew) > 1 else rng.choice(crew)
        rowid += 1
        guid = _guid(rowid)
        text: str | None = None
        body: bytes | None = None
        files = 0
        edited = retracted = 0
        roll = rng.random()
        if roll < 0.55:
            text = " ".join(rng.choices(WORDS, k=rng.randint(2, 14)))
        elif roll < 0.90:
            body = blobs[rowid % len(blobs)]
            tally["blob_texts"] += 1
        elif roll < 0.98:
            body, files = photo, rng.randint(1, 3)
        elif roll < 0.999:
            body, files = caption, 1
            tally["blob_texts"] += 1
        else:
            retracted = _apple_ns(now + 60)
        if text is not None and rng.random() < 0.01:
            edited = _apple_ns(now + 120)
        reply_to = recent.get(chat) if rng.random() < 0.02 else None
        yield (
            "message",
            (
                rowid,
                guid,
                text,
                body,
                handle,
                int(from_me),
                _apple_ns(now),
                "iMessage",
                0,
                None,
                0,
                reply_to,
                edited,
                retracted,
                None,
            ),
        )
        yield "link", (chat, rowid)
        made += 1
        tally["messages"] += 1
        recent[chat] = guid
        for _ in range(files):
            attachment_rowid += 1
            tally["attachments"] += 1
            name = f"IMG_{attachment_rowid:07d}.HEIC"
            yield (
                "attachment",
                (
                    attachment_rowid,
                    f"at_{attachment_rowid:012d}",
                    f"~/Library/Messages/Attachments/{name}",
                    "image/heic",
                    name,
                    1_000_000 + attachment_rowid,
                ),
            )
            yield "file_link", (rowid, attachment_rowid)
        if rng.random() < 0.06:
            rowid += 1
            tally["reactions"] += 1
            yield (
                "message",
                (
                    rowid,
                    _guid(rowid),
                    "Loved an invented message",
                    None,
                    rng.choice(crew),
                    0,
                    _apple_ns(now + 5),
                    "iMessage",
                    0,
                    f"p:0/{guid}",
                    2000 + rng.randint(0, 5),
                    None,
                    0,
                    0,
                    None,
                ),
            )
            yield "link", (chat, rowid)
        if len(crew) > 1 and rng.random() < 0.01:
            rowid += 1
            tally["system_rows"] += 1
            yield (
                "message",
                (
                    rowid,
                    _guid(rowid),
                    None,
                    None,
                    rng.choice(crew),
                    0,
                    _apple_ns(now + 9),
                    "iMessage",
                    2,
                    None,
                    0,
                    None,
                    0,
                    0,
                    "Invented Group Name",
                ),
            )
            yield "link", (chat, rowid)


def build_synthetic_chat_db(path: Path, messages: int, *, seed: int = 7) -> SyntheticCounts:
    """Write a chat.db with `messages` messages plus tapbacks, group events and attachments."""
    rng = random.Random(seed)
    builder = ChatDbBuilder(path)
    tally = {"messages": 0, "reactions": 0, "system_rows": 0, "attachments": 0, "blob_texts": 0}
    try:
        chats, members = _directory(builder, rng)
        statements = {
            "message": MESSAGE_SQL,
            "link": "INSERT INTO chat_message_join (chat_id, message_id) VALUES (?, ?)",
            "attachment": "INSERT INTO attachment (ROWID, guid, filename, mime_type, transfer_name, total_bytes) VALUES (?, ?, ?, ?, ?, ?)",
            "file_link": "INSERT INTO message_attachment_join (message_id, attachment_id) VALUES (?, ?)",
        }
        stream = _rows(messages, chats, members, rng, tally)
        while True:
            chunk = list(islice(stream, BATCH))
            if not chunk:
                break
            for kind, sql in statements.items():
                builder.raw.executemany(sql, [values for name, values in chunk if name == kind])
            builder.raw.commit()
    finally:
        builder.close()
    return SyntheticCounts(
        source_rows=tally["messages"] + tally["reactions"] + tally["system_rows"],
        messages=tally["messages"],
        reactions=tally["reactions"],
        system_rows=tally["system_rows"],
        attachments=tally["attachments"],
        blob_texts=tally["blob_texts"],
        conversations=ONE_TO_ONE + GROUPS,
        handles=400,
    )
