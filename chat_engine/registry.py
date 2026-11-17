"""References that live for the whole chat.

Every message the model reads gets a reference `m1`, `m2`, ... and every
conversation an alias `c1`, `c2`, .... The model never sees a raw id. A
follow-up question can cite or re-read anything found earlier.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional

from .ports import Message

_REF_RE = re.compile(r"^\[?\s*m0*(\d{1,9})\s*\]?$", re.IGNORECASE)
_ALIAS_RE = re.compile(r"^\[?\s*c0*(\d{1,9})\s*\]?$", re.IGNORECASE)


def normalize_ref(value: object) -> str:
    """ "M3", "[m3]" and "m03" all mean "m3". "" when it is not a reference."""
    match = _REF_RE.match(str(value or "").strip())
    return f"m{int(match.group(1))}" if match else ""


def normalize_alias(value: object) -> str:
    match = _ALIAS_RE.match(str(value or "").strip())
    return f"c{int(match.group(1))}" if match else ""


class EvidenceRegistry:
    """Messages shown to the model, by reference. Grows; never renumbers."""

    def __init__(self) -> None:
        self._by_ref: dict[str, Message] = {}
        self._ref_by_id: dict[str, str] = {}

    def add(self, message: Message) -> str:
        message_id = str(message["id"])
        known = self._ref_by_id.get(message_id)
        if known:
            return known
        ref = f"m{len(self._by_ref) + 1}"
        self._by_ref[ref] = _copy(message)
        self._ref_by_id[message_id] = ref
        return ref

    def checkpoint(self) -> int:
        """A mark to roll back to when rendered lines are not shown after all."""
        return len(self._by_ref)

    def rollback(self, mark: int) -> None:
        """Forget every message added after `mark`. A mark out of range changes nothing."""
        if not 0 <= mark <= len(self._by_ref):
            return
        for ref in list(self._by_ref)[mark:]:
            message = self._by_ref.pop(ref)
            self._ref_by_id.pop(str(message["id"]), None)

    def get(self, ref: str) -> Optional[Message]:
        found = self._by_ref.get(normalize_ref(ref))
        return _copy(found) if found else None

    def ref_of(self, message_id: str) -> Optional[str]:
        return self._ref_by_id.get(str(message_id))

    def messages(self) -> list[Message]:
        return [_copy(message) for message in self._by_ref.values()]

    def __len__(self) -> int:
        return len(self._by_ref)

    def __contains__(self, ref: object) -> bool:
        return normalize_ref(ref) in self._by_ref

    def snapshot(self) -> list[dict[str, Any]]:
        return [{"ref": ref, "message": _copy(message)} for ref, message in self._by_ref.items()]

    @classmethod
    def restore(cls, rows: Iterable[Mapping[str, Any]]) -> EvidenceRegistry:
        """A registry from `snapshot()`. Raises ValueError when the rows are not a snapshot."""
        registry = cls()
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("message"), Mapping):
                raise ValueError("an evidence row must hold a message")
            ref = registry.add(_checked(row["message"]))
            if row.get("ref") not in (None, ref):
                raise ValueError("the evidence rows are not in the order of their references")
        return registry


class ConversationAliases:
    """Conversation ids behind short aliases, so the model never reads a handle."""

    def __init__(self) -> None:
        self._id_by_alias: dict[str, str] = {}
        self._alias_by_id: dict[str, str] = {}

    def alias(self, conversation_id: str) -> str:
        key = str(conversation_id)
        known = self._alias_by_id.get(key)
        if known:
            return known
        alias = f"c{len(self._id_by_alias) + 1}"
        self._id_by_alias[alias] = key
        self._alias_by_id[key] = alias
        return alias

    def resolve(self, alias: object) -> Optional[str]:
        return self._id_by_alias.get(normalize_alias(alias))

    def checkpoint(self) -> int:
        """A mark to roll back to when a question does not finish."""
        return len(self._id_by_alias)

    def rollback(self, mark: int) -> None:
        """Forget every alias given after `mark`. A mark out of range changes nothing."""
        if not 0 <= mark <= len(self._id_by_alias):
            return
        for alias in list(self._id_by_alias)[mark:]:
            self._alias_by_id.pop(self._id_by_alias.pop(alias), None)

    def snapshot(self) -> list[str]:
        return list(self._id_by_alias.values())

    @classmethod
    def restore(cls, conversation_ids: Iterable[str]) -> ConversationAliases:
        """Aliases from `snapshot()`. Raises ValueError when an id is not text."""
        aliases = cls()
        for conversation_id in conversation_ids:
            if not isinstance(conversation_id, str) or not conversation_id:
                raise ValueError("a conversation id must be text")
            aliases.alias(conversation_id)
        return aliases


def _checked(message: Mapping[str, Any]) -> Message:
    """A stored message with the fields the engine reads, each of the right type."""
    for key in ("id", "conversation"):
        if not isinstance(message.get(key), str) or not message[key]:
            raise ValueError(f"a stored message needs its {key}")
    for key in ("timestamp", "text", "sender", "kind"):
        if not isinstance(message.get(key, ""), str):
            raise ValueError(f"the {key} of a stored message must be text")
    attachments = message.get("attachments") or []
    if not isinstance(attachments, (list, tuple)) or not all(isinstance(item, str) for item in attachments):
        raise ValueError("the attachments of a stored message must be a list of text")
    return _copy(message)


def _copy(message: Mapping[str, Any]) -> Message:
    copied = dict(message)
    copied["attachments"] = list(message.get("attachments") or [])
    return copied  # type: ignore[return-value]
