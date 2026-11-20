"""The threads the insight engine reads, made from the archive.

One thread for each person: every one-to-one conversation of a person (a
phone and an email) joins into one stream. A conversation with no person
stays a thread of its own. Groups are left out.

The name of a thread is the saved name of its handle (`Names.saved`). Left
out, before any message is read:
- a person nobody saved: a relationship worth a nudge is one you saved, and
  a thread with no name has nothing to call the person on the screen
- a person saved under a name that is itself a phone number or an email: the
  name goes to the model, and a handle never does
Left out after the read: a thread where I never wrote (codes, deliveries).

The archive streams messages in UTC order, and a note keeps local time. On
the night the clocks go back, local time runs back an hour, so the notes are
sorted by local time before they make a thread.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from archive import Archive
from chat_engine.handles import looks_like_handle
from insights.thread import Note, Thread
from recall_services.names import Names

ATTACHMENT_TEXT = "[attachment]"


def _groups(conversations: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for conversation in conversations:
        if conversation.get("is_group"):
            continue
        person = conversation.get("person_id")
        key = f"person:{person}" if person is not None else f"chat:{conversation['id']}"
        grouped[key].append(conversation)
    return grouped


def _note(message: Mapping[str, Any]) -> Note:
    text = str(message.get("text") or "") or (ATTACHMENT_TEXT if message.get("attachments") else "")
    return Note(str(message["id"]), datetime.fromisoformat(str(message["timestamp"])), bool(message["is_from_me"]), text)


def name_of(members: Sequence[Mapping[str, Any]], names: Names) -> str:
    """The saved name of a person's conversations, or "" when there is none or it is a handle."""
    for conversation in members:
        saved = " ".join(names.saved(str(conversation["id"])).split())
        if saved and not looks_like_handle(saved):
            return saved
    return ""


def threads(archive: Archive, names: Names) -> list[Thread]:
    """Every saved person thread with at least one message of mine, by name."""
    made: list[Thread] = []
    for key, members in _groups(archive.conversations()).items():
        name = name_of(members, names)
        if not name:
            continue
        ids = [str(conversation["id"]) for conversation in members]
        streamed = [_note(message) for message in archive.iter_messages(ids) if not message.get("unsent")]
        notes = tuple(sorted(streamed, key=lambda note: note.when))
        if any(note.from_me for note in notes):
            made.append(Thread(key, name, notes))
    return sorted(made, key=lambda thread: (thread.name, thread.key))
