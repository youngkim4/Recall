"""Mask every phone number and every email address of a text, also one that the archive does not hold.

`Labeler.redact` masks the handles it knows: those of the people of the
archive who are not saved. That is right for message text. A summary of a
digest is other text: a model wrote it from a whole conversation, so it can
hold the handle of anyone. Here every span that has the form of a phone
number or of an email address goes:

- a handle the labeler knows gets what the labeler gives for it: the saved
  name, or its mask;
- any other handle gets a fixed phrase.

A range of years, a date and a clock time are not phone numbers. They stay.
"""

from __future__ import annotations

import re
from typing import Callable

from .handles import EMAIL_RE, MIN_PHONE_DIGITS, PHONE_SPAN_RE, is_dates, is_mask
from .ports import Labeler

SOME_PHONE = "a phone number"
SOME_EMAIL = "an email address"
# 9:30, 18:22:05. The digits of a clock time are never part of a phone number.
_CLOCK_RE = re.compile(r"((?<![\d:])\d{1,2}:\d{2}(?::\d{2})?(?![\d:]))")
_BETWEEN_NUMBERS_RE = re.compile(r"(\s*[,;/|]\s*)")  # a list of numbers: "415-555-0123, 415-555-0124"


def mask_every_handle(text: str, labeler: Labeler) -> str:
    source = labeler.redact(str(text or ""))  # first the handles the labeler knows, in every spelling it knows
    source = EMAIL_RE.sub(lambda match: _saved_name(labeler, match.group()) or SOME_EMAIL, source)
    pieces = _CLOCK_RE.split(source)  # every second piece is a clock time
    return "".join(
        piece if index % 2 else PHONE_SPAN_RE.sub(_phone(labeler), piece) for index, piece in enumerate(pieces)
    )


def _phone(labeler: Labeler) -> Callable[[re.Match[str]], str]:
    def one(span: str) -> str:
        if is_dates(span):
            return span
        return _saved_name(labeler, span) or SOME_PHONE

    def swap(match: re.Match[str]) -> str:
        """One run of digits. When it is a list of whole numbers, each number gets its own words."""
        pieces = _BETWEEN_NUMBERS_RE.split(match.group())
        whole = all(sum(char.isdecimal() for char in piece) >= MIN_PHONE_DIGITS for piece in pieces[::2])
        if len(pieces) == 1 or not whole:
            return one(match.group())
        return "".join(piece if index % 2 else one(piece) for index, piece in enumerate(pieces))

    return swap


def _saved_name(labeler: Labeler, handle: str) -> str:
    """The saved name of a handle, or "" when nobody saved it.

    `MaskingLabeler.saved_name` answers and learns nothing. A labeler of
    another kind is asked for the sender: it gives a name or a mask.
    """
    lookup = getattr(labeler, "saved_name", None)
    if callable(lookup):
        return str(lookup(handle) or "")
    given = str(labeler.sender(handle) or "")
    return "" if is_mask(given) else given
