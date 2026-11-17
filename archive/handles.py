"""Rules for handles (phone numbers, email addresses) and conversation ids."""

from __future__ import annotations

import re

_GROUP_ID = re.compile(r"chat\d+")
_PHONE = re.compile(r"\+?[\d\s().\-‐‑]*\d[\d\s().\-‐‑]*")
PHONE_MATCH_DIGITS = 10


def is_group_identifier(chat_identifier: str) -> bool:
    """A group conversation id is `chat` followed by digits only. Anything else is one person."""
    return _GROUP_ID.fullmatch(chat_identifier) is not None


def handle_kind(handle: str) -> str:
    """`email`, `phone` or `other`."""
    text = handle.strip()
    if "@" in text:
        return "email"
    if _PHONE.fullmatch(text) is not None:
        return "phone"
    return "other"


def phone_key(value: str) -> str | None:
    """Match key of a phone number: its digits, the last ten when there are more."""
    digits = "".join(str(int(char)) for char in value if char.isdecimal())
    if not digits:
        return None
    return "p:" + digits[-PHONE_MATCH_DIGITS:]


def email_key(value: str) -> str | None:
    """Match key of an email address: lower case."""
    text = value.strip().lower()
    return "e:" + text if "@" in text else None


def handle_match_key(handle: str) -> str:
    """The key that links a handle from chat.db to a phone or email of a contact."""
    kind = handle_kind(handle)
    if kind == "email":
        return email_key(handle) or "o:" + handle.strip().lower()
    if kind == "phone":
        return phone_key(handle) or "o:" + handle.strip().lower()
    return "o:" + handle.strip().lower()
