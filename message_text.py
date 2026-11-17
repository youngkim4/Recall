"""What the interface shows of a stored message.

An export keeps text exactly as Messages stored it. An attachment leaves
U+FFFC, an invisible placeholder, in that text, and a message that is only a
photo is that one character. Nothing shown to a person should carry it.
"""
from __future__ import annotations

import pandas as pd

ATTACHMENT_PLACEHOLDER = "￼"
# what stands in the place of a message that its writer took back: it has no words and no attachment
UNSENT_LABEL = "Unsent message"
_SEPARATOR = "|"

# kind -> (one of them, several of them)
_LABELS = {
    "gif": ("GIF", "GIFs"),
    "photo": ("Photo", "photos"),
    "video": ("Video", "videos"),
    "audio": ("Audio", "audio clips"),
    "contact": ("Contact", "contacts"),
    "location": ("Location", "locations"),
    "file": ("File", "files"),
}
_CONTACT_TYPES = frozenset({"text/vcard", "text/x-vcard", "text/directory"})


def _is_missing(value) -> bool:
    return value is None or (not isinstance(value, str) and bool(pd.isna(value)))


def display_text(value) -> str:
    """Message text with the attachment placeholder removed."""
    if _is_missing(value):
        return ""
    return str(value).replace(ATTACHMENT_PLACEHOLDER, "").strip()


def _kind(mime_type: str) -> str:
    if mime_type == "image/gif":
        return "gif"
    if mime_type in _CONTACT_TYPES:
        return "contact"
    if mime_type == "text/x-vlocation":
        return "location"
    family = mime_type.split("/", 1)[0]
    return {"image": "photo", "video": "video", "audio": "audio"}.get(family, "file")


def attachment_label(mime_types) -> str:
    """ "Photo", "3 photos", "2 attachments": what a message carried, from the
    export's attachment_types column. Empty when it carried nothing."""
    if _is_missing(mime_types):
        return ""
    kinds = [_kind(part.strip().lower()) for part in str(mime_types).split(_SEPARATOR) if part.strip()]
    if not kinds:
        return ""
    if len(kinds) == 1:
        return _LABELS[kinds[0]][0]
    if len(set(kinds)) == 1:
        return f"{len(kinds)} {_LABELS[kinds[0]][1]}"
    return f"{len(kinds)} attachments"


def clip_text(text: str, limit: int) -> str:
    """Shorten text at a word boundary and mark the cut, so clipped words
    never read as the whole message."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    if not text[limit].isspace() and any(char.isspace() for char in head):
        head = head.rsplit(None, 1)[0]
    return head.rstrip(" \t\n.,;:!?-") + "…"
