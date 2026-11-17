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
RANGE_END = 4000  # 2000-3999 is the range of tapback codes, also for kinds that are not in TAPBACK_KINDS

# "p:0/<guid>" points at one part of a message, "bp:<guid>" at a whole bubble.
# A part index has 1 to 4 ASCII digits. A longer number is not a part index and would not fit SQLite.
_TARGET_PREFIX = re.compile(r"p:([0-9]{1,4})/|bp:")


def in_tapback_range(code: object) -> bool:
    """True for an int code from 2000 to 3999: a tapback, also of a kind that `tapback` does not know.

    A bool is not a code.
    """
    return isinstance(code, int) and not isinstance(code, bool) and ADD_BASE <= code < RANGE_END


def tapback(code: int | None) -> tuple[str, bool] | None:
    """(kind, removed) for a tapback code. None when the row is not a tapback.

    A code is an int. Any other value is not a tapback, also a bool and a float.
    """
    if not isinstance(code, int) or isinstance(code, bool):
        return None
    if ADD_BASE <= code < ADD_BASE + len(TAPBACK_KINDS):
        return TAPBACK_KINDS[code - ADD_BASE], False
    if REMOVE_BASE <= code < REMOVE_BASE + len(TAPBACK_KINDS):
        return TAPBACK_KINDS[code - REMOVE_BASE], True
    return None


def split_target(associated_guid: str | None) -> tuple[str, int | None] | None:
    """(guid of the target message, part index) from associated_message_guid.

    Text with no known prefix is the guid as it is. "p:" with more than 4 digits before the slash is
    not a known prefix: that text is then the guid as a whole, and it matches no message.
    """
    text = (associated_guid or "").strip()
    prefix = _TARGET_PREFIX.match(text)
    part = prefix.group(1) if prefix is not None else None
    guid = text[prefix.end() :].strip() if prefix is not None else text
    if not guid:
        return None
    return guid, (int(part) if part is not None else None)
