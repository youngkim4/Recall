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


# The English summary that Messages writes into the text of a tapback row. The legacy export
# keeps that text and nothing else of the tapback, so this rule is the one way to know such a row.
_QUOTED = '["' + chr(0x201C) + '].*["' + chr(0x201D) + "]"  # text in straight or curly double quotes
_TARGET = rf"(?:{_QUOTED}|an image|a movie|an audio message|an attachment|a contact)"
_REMOVALS = "a heart|a like|a dislike|a laugh|an exclamation|a question mark"
_EMOJI = r"[^\sA-Za-z0-9]{1,16}"  # an emoji, also one made of several code points. Never a word or a number.
_REACTION_PHRASE = re.compile(
    rf"(?:Loved|Liked|Disliked|Laughed at|Emphasized|Questioned) {_TARGET}"
    rf"|Removed (?:{_REMOVALS}) from {_TARGET}"
    rf"|Reacted {_EMOJI} to {_QUOTED}"
    rf"|Removed {_EMOJI} from {_QUOTED}",
    re.DOTALL,
)


def is_reaction_phrase(text: object) -> bool:
    """True when the whole text is the English summary of a tapback.

    The forms, where the target is text in double quotes (straight or curly),
    `an image`, `a movie`, `an audio message`, `an attachment` or `a contact`:
    - `Loved`, `Liked`, `Disliked`, `Laughed at`, `Emphasized` or `Questioned`, then the target
    - `Removed a heart from`, `Removed a like from`, `Removed a dislike from`, `Removed a laugh from`,
      `Removed an exclamation from` or `Removed a question mark from`, then the target
    - `Reacted <emoji> to` or `Removed <emoji> from`, then text in double quotes. The emoji is
      1 to 16 characters with no space, no ASCII letter and no ASCII digit.

    The whole text must match. Text that starts like a phrase and does not end
    as one (`Loved it`, `Loved "Dune" a lot`) is a message, not a tapback. The
    rule is case-sensitive and knows English only.
    """
    return isinstance(text, str) and _REACTION_PHRASE.fullmatch(text) is not None
