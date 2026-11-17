"""Names the model reads.

A saved name is used as it is. Anyone else is masked: "a number ending in
1234" or "an unsaved email contact". The caller supplies the saved names; this
module supplies the masking, for labels and for free text.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

from .ports import Conversation

EMAIL_MASK = "an unsaved email contact"
UNKNOWN_MASK = "an unsaved contact"
PHONE_MASK = "a number ending in {tail}"
GROUP_MASK = "a group chat"

_EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")
_SCHEME_RE = re.compile(r"^(?:tel|sms|mailto|imessage|p|e):", re.IGNORECASE)
_PHONE_ONLY_RE = re.compile(r"^[\d\s+().\-]+$")
_PHONE_SPAN_RE = re.compile(r"\+?\d(?:[\s().\-]{0,3}\d){6,}")
_MIN_PHONE_DIGITS = 7
_NATIONAL_DIGITS = 10
_GROUP_NAMES_SHOWN = 3


def _strip_scheme(handle: str) -> str:
    return _SCHEME_RE.sub("", str(handle or "").strip())


def handle_kind(handle: str) -> str:
    """ "email", "phone" or "other" (a group id, a name, an empty value)."""
    text = _strip_scheme(handle)
    if _EMAIL_RE.fullmatch(text):
        return "email"
    digits = re.sub(r"\D", "", text)
    if len(digits) >= _MIN_PHONE_DIGITS and _PHONE_ONLY_RE.fullmatch(text):
        return "phone"
    return "other"


def _phone_key(digits: str) -> str:
    return digits[-_NATIONAL_DIGITS:] if len(digits) > _NATIONAL_DIGITS else digits


def handle_key(handle: str) -> str:
    """One lookup key for all spellings of a handle. "" when it is not a handle."""
    text = _strip_scheme(handle)
    kind = handle_kind(text)
    if kind == "email":
        return text.lower()
    if kind == "phone":
        return _phone_key(re.sub(r"\D", "", text))
    return ""


def looks_like_handle(text: str) -> bool:
    return handle_kind(text) != "other"


def mask_handle(handle: str) -> str:
    kind = handle_kind(handle)
    if kind == "email":
        return EMAIL_MASK
    if kind == "phone":
        return PHONE_MASK.format(tail=handle_key(handle)[-4:])
    return UNKNOWN_MASK


def is_mask(label: str) -> bool:
    """True when a label is one of the masks above, not a real name."""
    text = str(label or "")
    prefix = PHONE_MASK.format(tail="")
    return text in (EMAIL_MASK, UNKNOWN_MASK) or text.startswith((prefix, GROUP_MASK))


def _clean_name(name: object) -> str:
    text = " ".join(str(name or "").split())
    return "" if looks_like_handle(text) else text


def _index_names(saved_names: Mapping[str, str]) -> dict[str, str]:
    index: dict[str, str] = {}
    for handle, name in saved_names.items():
        clean = _clean_name(name)
        if not clean:
            continue
        index[str(handle)] = clean
        key = handle_key(str(handle))
        if key:
            index[key] = clean
    return index


class MaskingLabeler:
    """The default `Labeler`: saved names as given, a mask for everyone else."""

    def __init__(
        self,
        saved_names: Mapping[str, str] | None = None,
        known_handles: Iterable[str] = (),
    ) -> None:
        self._saved = _index_names(saved_names or {})
        self._unsaved_phones: set[str] = set()
        self._unsaved_emails: set[str] = set()
        for handle in known_handles:
            self._remember(handle)

    def saved_name(self, handle: str) -> str:
        """The saved name of a handle, or "" when nobody saved it."""
        text = str(handle or "")
        return self._saved.get(text) or self._saved.get(handle_key(text), "")

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
            source = _EMAIL_RE.sub(self._mask_email, source)
        if self._unsaved_phones:
            source = _PHONE_SPAN_RE.sub(self._mask_phone, source)
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
            return archive_name
        for handle in handles:
            if looks_like_handle(handle):
                self._remember(handle)
                return mask_handle(handle)
        return UNKNOWN_MASK

    def _group_label(self, conversation: Conversation) -> str:
        name = self.saved_name(str(conversation.get("id") or "")) or _clean_name(conversation.get("name"))
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

    # --- redaction ---

    def _remember(self, handle: str) -> None:
        text = str(handle or "")
        if self.saved_name(text):
            return
        kind = handle_kind(text)
        if kind == "email":
            self._unsaved_emails.add(handle_key(text))
        elif kind == "phone":
            self._unsaved_phones.add(handle_key(text))

    def _mask_email(self, match: re.Match[str]) -> str:
        found = match.group()
        return EMAIL_MASK if found.lower() in self._unsaved_emails else found

    def _mask_phone(self, match: re.Match[str]) -> str:
        span = match.group()
        positions = [i for i, ch in enumerate(span) if ch.isdigit()]
        digits = "".join(span[i] for i in positions)
        for size in sorted({len(key) for key in self._unsaved_phones}, reverse=True):
            for start in range(len(digits) - size + 1):
                key = digits[start:start + size]
                if key not in self._unsaved_phones:
                    continue
                mask = PHONE_MASK.format(tail=key[-4:])
                if len(digits) - size <= 3:  # only a country code is left over
                    return mask
                first, last = positions[start], positions[start + size - 1]
                return span[:first] + mask + span[last + 1:]
        return span
