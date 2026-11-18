"""SQL for two filters over `messages m`: the kind of conversation and the attachments.

- `kind`: `direct` (one person) or `group`.
- `attachments`: `any` (one attachment or more), `none`, or a kind of attachment.
  The kind comes from the mime type: `image/*`, `video/*`, `audio/*`, and
  `other` for every other type and for an attachment with no type.

The clauses read only `m.id` and `m.conversation_id`. Both indexes of messages
hold these columns, so a count with these filters still reads no table row.
The caller checks each value against KINDS and ATTACHMENT_CHOICES first.
"""

from __future__ import annotations

KINDS = ("direct", "group")
ATTACHMENT_KINDS = ("image", "video", "audio", "other")
ATTACHMENT_CHOICES = ("any", "none", *ATTACHMENT_KINDS)

_FAMILIES = {kind: f"a.mime_type LIKE '{kind}/%'" for kind in ATTACHMENT_KINDS[:-1]}
_OTHER = f"a.mime_type IS NULL OR NOT ({' OR '.join(_FAMILIES.values())})"
_HOLDERS = "(SELECT a.message_id FROM attachments a{where})"


def kind_clause(kind: str) -> str:
    """SQL for one kind of conversation."""
    return f"m.conversation_id IN (SELECT c.id FROM conversations c WHERE c.is_group = {int(kind == 'group')})"


def attachment_clause(choice: str) -> str:
    """SQL for the attachments of a message."""
    if choice == "any":
        return "m.id IN " + _HOLDERS.format(where="")
    if choice == "none":
        return "m.id NOT IN " + _HOLDERS.format(where="")
    test = _OTHER if choice == "other" else _FAMILIES[choice]
    return "m.id IN " + _HOLDERS.format(where=f" WHERE {test}")
