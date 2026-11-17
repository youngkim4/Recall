"""Names the model reads.

A saved name is used as it is. Anyone else is masked: "a number ending in
1234" or "an unsaved email contact". The caller supplies the saved names; this
module supplies the masking, for labels and for free text.

The words are the same as `contact_label` in `ui_server.py` uses.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

from .handles import (
    EMAIL_MASK,
    EMAIL_RE,
    GROUP_MASK,
    PHONE_MASK,
    PHONE_SPAN_RE,
    UNKNOWN_MASK,
    Phone,
    PhoneIndex,
    email_key,
    handle_kind,
    handle_spans,
    is_mask,
    looks_like_handle,
    mask_handle,
    match_in_run,
    parse_phone,
)
from .ports import Conversation

__all__ = [
    "EMAIL_MASK",
    "GROUP_MASK",
    "PHONE_MASK",
    "UNKNOWN_MASK",
    "MaskingLabeler",
    "handle_kind",
    "handle_spans",
    "is_mask",
    "looks_like_handle",
    "mask_handle",
]

_GROUP_NAMES_SHOWN = 3
_COUNTRY_DIGITS = 3


def _clean_name(name: object) -> str:
    """A name with single spaces. A name that is itself a handle is no name."""
    text = " ".join(str(name or "").split())
    return "" if looks_like_handle(text) else text


class MaskingLabeler:
    """The default `Labeler`: saved names as given, a mask for everyone else."""

    def __init__(
        self,
        saved_names: Mapping[str, str] | None = None,
        known_handles: Iterable[str] = (),
    ) -> None:
        self._saved_phones: PhoneIndex[str] = PhoneIndex()
        self._saved_emails: dict[str, str] = {}
        self._saved_ids: dict[str, str] = {}
        self._unsaved_phones: PhoneIndex[Phone] = PhoneIndex()
        self._unsaved_emails: set[str] = set()
        for handle, name in (saved_names or {}).items():
            self._save(str(handle), name)
        for handle in known_handles:
            self._remember(handle)

    def saved_name(self, handle: str) -> str:
        """The saved name of a handle in any spelling, or "" when nobody saved it."""
        raw = str(handle or "")
        if raw in self._saved_ids:
            return self._saved_ids[raw]
        if handle_kind(raw) == "email":
            return self._saved_emails.get(email_key(raw), "")
        phone = parse_phone(raw)
        return (self._saved_phones.get(phone) or "") if phone else ""

    def sender(self, handle: str) -> str:
        name = self.saved_name(handle)
        if name:
            return name
        self._remember(handle)
        return mask_handle(handle)

    def conversation(self, conversation: Conversation) -> str:
        if conversation.get("is_group"):
            return self._group_label(conversation)
        return self._direct_label(conversation)

    def redact(self, text: str) -> str:
        source = str(text or "")
        if self._unsaved_emails:
            source = EMAIL_RE.sub(self._mask_email, source)
        if self._unsaved_phones:
            source = PHONE_SPAN_RE.sub(self._mask_phones, source)
        return source

    # --- labels ---

    def _direct_label(self, conversation: Conversation) -> str:
        handles = [str(conversation.get("id") or ""), *map(str, conversation.get("members") or [])]
        for handle in handles:
            name = self.saved_name(handle)
            if name:
                return name
        archive_name = _clean_name(conversation.get("name"))
        if archive_name and archive_name not in handles:
            return self._scrub(archive_name)
        for handle in handles:
            if handle_kind(handle) != "other":
                self._remember(handle)
                return mask_handle(handle)
        return UNKNOWN_MASK

    def _group_label(self, conversation: Conversation) -> str:
        saved = self.saved_name(str(conversation.get("id") or ""))
        name = saved or self._scrub(_clean_name(conversation.get("name")))
        if name:
            return name
        members = [self.sender(str(handle)) for handle in conversation.get("members") or []]
        if not members:
            return GROUP_MASK
        shown = members[:_GROUP_NAMES_SHOWN]
        rest = len(members) - len(shown)
        if rest > 0:
            return f"{GROUP_MASK} with {', '.join(shown)} and {rest} other{'s' if rest > 1 else ''}"
        if len(shown) == 1:
            return f"{GROUP_MASK} with {shown[0]}"
        return f"{GROUP_MASK} with {', '.join(shown[:-1])} and {shown[-1]}"

    def _scrub(self, name: str) -> str:
        """A name from the archive, with any handle inside it masked unless it is saved."""

        def swap(match: re.Match[str]) -> str:
            found = match.group()
            return found if self.saved_name(found) or not looks_like_handle(found) else mask_handle(found)

        return PHONE_SPAN_RE.sub(swap, EMAIL_RE.sub(swap, name))

    # --- what is known ---

    def _save(self, handle: str, name: object) -> None:
        clean = _clean_name(name)
        if not clean:
            return
        self._saved_ids[handle] = clean
        phone = parse_phone(handle)
        if phone is not None:
            self._saved_phones.add(phone, clean)
        elif handle_kind(handle) == "email":
            self._saved_emails[email_key(handle)] = clean

    def _remember(self, handle: str) -> None:
        """Keep a handle nobody saved, so that `redact` can find it in text."""
        raw = str(handle or "")
        if self.saved_name(raw):
            return
        phone = parse_phone(raw)
        if phone is not None:
            self._unsaved_phones.add(phone, phone)
        elif handle_kind(raw) == "email":
            self._unsaved_emails.add(email_key(raw))

    # --- redaction ---

    def _mask_email(self, match: re.Match[str]) -> str:
        found = match.group()
        return EMAIL_MASK if found.lower() in self._unsaved_emails else found

    def _mask_phones(self, match: re.Match[str]) -> str:
        """Mask every unsaved number inside one run of digits and separators."""
        span = match.group()
        positions = [index for index, char in enumerate(span) if char.isdigit()]
        digits = "".join(span[index] for index in positions)
        found = match_in_run(self._unsaved_phones, digits, span.lstrip().startswith("+"))
        parts: list[str] = []
        cursor = 0
        previous_end = 0
        for start, end, phone in found:
            if self._saved_phones.get(Phone(digits[start:end], False)):
                continue  # the same digits also spell a saved number
            first = _replace_from(span, positions, start, previous_end, cursor)
            parts.append(span[cursor:first])
            parts.append(PHONE_MASK.format(tail=phone.tail))
            cursor = positions[end - 1] + 1
            previous_end = end
        parts.append(span[cursor:])
        return "".join(parts)


def _replace_from(span: str, positions: list[int], start: int, previous_end: int, cursor: int) -> int:
    """The character where a mask begins: a country code and "+(" before the number go with it."""
    lead = start - previous_end
    if lead > _COUNTRY_DIGITS:
        first = positions[start]
    elif previous_end == 0:
        return 0
    else:
        first = positions[previous_end]
    while first > cursor and span[first - 1] in "+(":
        first -= 1
    return first
