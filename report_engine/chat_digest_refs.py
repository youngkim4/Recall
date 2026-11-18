"""Take the line references of the report engine out of text that goes to chat.

A map reply cites lines of its transcript: "you welcomed the message
(m1–m2)". In a report those references mean lines of the period. Chat has
references of the same form for other messages, so a digest gives its text
without them.

The form is the one of `transcript.py`: `render_period` writes `m` and the
number of the line, and `resolve_ref` reads `m` and 1 to 9 digits. A
reference is removed whether or not the period has a line with that number:
a number that the model made up means a message to chat all the same.

A reference goes with what only holds it: its brackets, a range (`m1–m2`,
`m1-2`), a list (`m3, m7 and m9`) and the words before it ("see m12",
"lines m3–m18"). The text is changed only where a reference stood.

Known limits: the words before a reference are English, and a bare lower-case
word such as `m25` in other text reads as a reference. `M25`, `m4a` and
`40 m2` do not.
"""

from __future__ import annotations

import re

# as `render_period` writes a reference; `resolve_ref` reads 1 to 9 digits
_ONE = r"m[0-9]{1,9}(?!\w)"
_DASH = "-‐‑‒–—−"
# m1–m2 and m1 – m2; with no space also m1–2
_RANGE = rf"{_ONE}(?:[{_DASH}][0-9]{{1,9}}(?!\w)|\s*[{_DASH}]\s*{_ONE})?"
_MORE = r"(?i:and|or|also|plus)"
_JOIN = rf"(?:\s*[,;/&]\s*(?:{_MORE}\s+)*|\s+(?i:and|or|to|through|plus)\s+(?:(?i:also)\s+)?)"
_LIST = rf"{_RANGE}(?:{_JOIN}{_RANGE})*"
# words that only hold a reference: "see m12", "in lines m3–m18", "refs: m3"
_POINTERS = r"see(?:\s+also)?|cf\.?|e\.g\.,?|as\s+in|as\s+shown\s+in|shown\s+in|in|at|from|per"
_POINTER = rf"(?i:(?:{_POINTERS})\s+)?"
_NOUN = r"(?i:(?:the\s+)?(?:lines?|refs?|references?)[.:]?\s+)?"
# in brackets also "messages m3–m18". Outside them "the message m12" can be a
# sentence about a message, so only the reference goes there.
_NOUN_IN_BRACKETS = r"(?i:(?:the\s+)?(?:lines?|refs?|references?|messages?|msgs?)[.:]?\s+)?"
# not part of a word, a path, a tag or a measure such as "40 m2"
_BARE = rf"(?<![\w/#@.\-])(?<![0-9]\s){_POINTER}{_NOUN}{_LIST}(?![\w/#@])(?!-\w)"
_BRACKETED = rf"[(\[]\s*{_POINTER}{_NOUN_IN_BRACKETS}{_LIST}\s*(?:[.,;]\s*)?[)\]]"
_GROUP = re.compile(rf"{_BRACKETED}|{_BARE}")

_BLANKS = " \t"
_OPENING = "(["
_CLOSING = ")]"
_SEPARATORS = ",;:"
_NO_SPACE_BEFORE = ".,;:!?)]\n"


def _joined(left: str, right: str) -> str:
    """The text on both sides of a reference that was taken out, put together:
    with no empty brackets, no separator left alone and no space before a mark."""
    spaced = left[-1:] in _BLANKS or right[:1] in _BLANKS
    left, right = left.rstrip(_BLANKS), right.lstrip(_BLANKS)
    while left and right and left[-1] in _OPENING and right[0] in _CLOSING:
        left, right = left[:-1].rstrip(_BLANKS), right[1:].lstrip(_BLANKS)
        spaced = True
    if not left or left[-1] == "\n" or left[-1] in _OPENING:
        return left + right.lstrip(_SEPARATORS + _BLANKS)
    if not right or right[0] in _NO_SPACE_BEFORE:
        return left.rstrip(_SEPARATORS + _BLANKS) + right
    return f"{left}{' ' if spaced else ''}{right}"


def without_refs(text: str) -> str:
    """`text` with no line reference of the engine. Text that holds none comes
    back as it is."""
    cleaned = ""
    cursor = 0
    found = False
    for match in _GROUP.finditer(text):
        piece = text[cursor : match.start()]
        cleaned = _joined(cleaned, piece) if found else piece
        cursor = match.end()
        found = True
    if not found:
        return text
    return _joined(cleaned, text[cursor:])
