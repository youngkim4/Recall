"""Tapbacks: reactions that chat.db stores as rows of the message table."""

from __future__ import annotations

import re

# associated_message_type 2000 + n adds kind n, 3000 + n removes it
TAPBACK_KINDS: tuple[str, ...] = (
    "love",
    "like",
    "dislike",
    "laugh",
    "emphasis",
    "question",
    "emoji",
    "sticker",
)
ADD_BASE = 2000
REMOVE_BASE = 3000

# "p:0/<guid>" points at one part of a message, "bp:<guid>" at a whole bubble
_TARGET_PREFIX = re.compile(r"p:(\d+)/|bp:")


def tapback(code: int | None) -> tuple[str, bool] | None:
    """(kind, removed) for a tapback code. None when the row is not a tapback."""
    if code is None:
        return None
    if ADD_BASE <= code < ADD_BASE + len(TAPBACK_KINDS):
        return TAPBACK_KINDS[code - ADD_BASE], False
    if REMOVE_BASE <= code < REMOVE_BASE + len(TAPBACK_KINDS):
        return TAPBACK_KINDS[code - REMOVE_BASE], True
    return None


def split_target(associated_guid: str | None) -> tuple[str, int | None] | None:
    """(guid of the target message, part index) from associated_message_guid."""
    text = (associated_guid or "").strip()
    prefix = _TARGET_PREFIX.match(text)
    part = prefix.group(1) if prefix is not None else None
    guid = text[prefix.end() :].strip() if prefix is not None else text
    if not guid:
        return None
    return guid, (int(part) if part is not None else None)
