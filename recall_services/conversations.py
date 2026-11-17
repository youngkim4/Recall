"""The rows of /api/contacts and the totals of the archive.

`rows` has the shape of `contacts_from_messages` in ui_server.py at 89ab504,
as the route sends it: `chat_id`, `display_name`, `is_group`, `first_msg`,
`last_msg`, `message_count`, `sent_count`, `received_count`. The conversation
with the most messages is first.

`summary` has the shape of `archive_summary`: `messages`, `conversations`,
`firstMessage`, `lastMessage`. The list of rows is cut at a limit, so totals
come from here and never from adding up that list.

The times of both are written as the route writes them today: with a `T`
between the day and the time.

Differences from the server at 89ab504:
- A tapback is not a message. The old export counted its summary line.
- A message that no conversation holds is in no row and in no total.
- The archive has two directions only. `sent_count` and `received_count`
  add up to `message_count`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from archive import Archive
from recall_services import inputs
from recall_services.names import Names

DEFAULT_LIMIT = 40


def _iso(timestamp: object) -> str | None:
    """A time of the archive ("YYYY-MM-DD HH:MM:SS") in the ISO form of the route."""
    return str(timestamp).replace(" ", "T", 1) if timestamp else None


def _row(conversation: Mapping[str, Any], names: Names) -> dict[str, Any]:
    return {
        "chat_id": conversation["id"],
        "display_name": names.conversation(conversation),
        "is_group": int(bool(conversation["is_group"])),
        "first_msg": _iso(conversation["first_timestamp"]),
        "last_msg": _iso(conversation["last_timestamp"]),
        "message_count": conversation["message_count"],
        "sent_count": conversation["sent_count"],
        "received_count": conversation["received_count"],
    }


def rows(archive: Archive, names: Names, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    """The conversations with the most messages, each as a row of the conversation list."""
    size = inputs.whole("limit", limit, DEFAULT_LIMIT)
    listed = archive.conversations(limit=max(0, size if size is not None else DEFAULT_LIMIT), order="count")
    return [_row(conversation, names) for conversation in listed]


def summary(archive: Archive) -> dict[str, Any]:
    """Size and span of the whole archive."""
    found = archive.summary()
    return {
        "messages": found["messages"],
        "conversations": found["conversations"],
        "firstMessage": _iso(found["first_timestamp"]),
        "lastMessage": _iso(found["last_timestamp"]),
    }


def name_count(listed: Iterable[Mapping[str, Any]]) -> int:
    """How many rows have a name."""
    return sum(1 for row in listed if str(row.get("display_name") or ""))


def payload(archive: Archive, names: Names, limit: int = DEFAULT_LIMIT, messages_path: str = "") -> dict[str, Any]:
    """The whole answer of /api/contacts when there is an archive."""
    listed = rows(archive, names, limit)
    return {
        "contacts": listed,
        "source": "messages",
        "messagesPath": str(messages_path),
        "contactNameCount": name_count(listed),
        "archive": summary(archive),
    }
