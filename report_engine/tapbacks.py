"""Tapback rows of an export that stores them as text.

Messages keeps a tapback as a row of its own. The text of the row is a fixed
phrase, and then the whole text of the message that got the tapback, in quote
marks. The person who tapped did not write those words. Such a row is not a
message: in a transcript it would put the words of one person under the name
of another, and a quote check would find them there.

Only the English phrases are known here. A typed message that is exactly one
of these phrases with quoted text reads as a tapback too.
"""

from __future__ import annotations

import re

ADDED = ("Loved", "Liked", "Disliked", "Laughed at", "Emphasized", "Questioned")
REMOVED = (
    "Removed a heart from",
    "Removed a like from",
    "Removed a dislike from",
    "Removed a laugh from",
    "Removed an exclamation from",
    "Removed a question mark from",
)
THINGS = ("an image", "a movie", "an audio message", "an attachment")
STICKER = "Reacted with a sticker to"

_OPEN_MARKS = chr(0x201C) + '"'
_CLOSE_MARKS = chr(0x201D) + '"'
# a tapback with a symbol of the person's choice: the symbol has no letter and no digit
_SYMBOL = r"[^\sA-Za-z0-9]{1,16}"
_PHRASES = "|".join(
    [
        *(re.escape(phrase) for phrase in (*ADDED, *REMOVED, STICKER)),
        rf"Reacted {_SYMBOL} to",
        rf"Removed {_SYMBOL} from",
    ]
)
_THINGS = "|".join(re.escape(thing) for thing in THINGS)
_TAPBACK = re.compile(
    rf"(?:{_PHRASES}) (?:[{_OPEN_MARKS}].*[{_CLOSE_MARKS}]|{_THINGS})",
    re.DOTALL,
)
_FIRST_LETTERS = frozenset(phrase[0] for phrase in (*ADDED, *REMOVED, STICKER))


def is_tapback(text: str) -> bool:
    """True when the whole text is a tapback phrase and what it points at."""
    stripped = text.strip()
    return stripped[:1] in _FIRST_LETTERS and _TAPBACK.fullmatch(stripped) is not None
