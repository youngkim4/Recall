"""What one chat remembers between questions: the turns and the references."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from .registry import ConversationAliases, EvidenceRegistry
from .scope import EVERYTHING, Scope

_SHOWN_CITATION_RE = re.compile(r"[ \t]*\[\d+\]")


@dataclass(frozen=True)
class Turn:
    """One question and its answer. The answer keeps the m-references."""

    question: str
    answer: str
    cited: tuple[str, ...] = ()
    scope: Scope = EVERYTHING  # the scope the question was asked in
    limited: bool = False  # the answer stopped at the lookup limit of its question


class ChatSession:
    """Keep one per chat and pass it to every question of that chat.

    Not safe for two questions at the same time. `to_dict()` holds the whole
    text of every message the model has read: keep it on this machine.
    """

    def __init__(self) -> None:
        self.registry = EvidenceRegistry()
        self.aliases = ConversationAliases()
        self.turns: tuple[Turn, ...] = ()
        self._started = 0  # questions started on this session
        self._open: Optional[tuple[int, int]] = None  # where the open question began in the registry and the aliases

    def add_turn(self, turn: Turn) -> None:
        self.turns = (*self.turns, turn)

    def begin(self) -> int:
        """Start a question and get its number. A question that was left open before is rolled back first."""
        self._roll_back()
        self._open = (self.registry.checkpoint(), self.aliases.checkpoint())
        self._started += 1
        return self._started

    def finish(self, question: int, keep: bool) -> None:
        """End the question with this number. `keep=False` forgets the evidence and the aliases it added.

        A call for a question that is not the newest one does nothing: its
        marks are not valid any more. So a stream that is closed late cannot
        take away what a later question found.
        """
        if question != self._started or self._open is None:
            return
        if keep:
            self._open = None
        else:
            self._roll_back()

    def _roll_back(self) -> None:
        if self._open is not None:
            self.registry.rollback(self._open[0])
            self.aliases.rollback(self._open[1])
            self._open = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": [_turn_row(turn) for turn in self.turns],
            "evidence": self.registry.snapshot(),
            "conversations": self.aliases.snapshot(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChatSession:
        """A session from `to_dict()`. Raises ValueError when the data is not one."""
        if not isinstance(data, Mapping):
            raise ValueError("a stored session must be a mapping")
        session = cls()
        session.registry = EvidenceRegistry.restore(_rows(data, "evidence"))
        session.aliases = ConversationAliases.restore(_rows(data, "conversations"))
        session.turns = tuple(_turn(row) for row in _rows(data, "turns"))
        return session

    @classmethod
    def from_history(cls, history: Iterable[Mapping[str, Any]], scope: Scope = EVERYTHING) -> ChatSession:
        """A session from plain role/content rows. The evidence is not restored.

        A "user" row is a question and the "assistant" row after it is its
        answer. Every other row is ignored. `scope` is the scope the chat was
        asked in: give it, or a question with a scope leaves the earlier
        answers out. The numbers [1], [2] of the shown answers are removed:
        the messages they stood for are not in the new session.
        """
        session = cls()
        question = ""
        for row in history:
            if not isinstance(row, Mapping):
                raise ValueError("a history row must be a mapping")
            role = str(row.get("role") or "")
            content = str(row.get("content") or "").strip()
            if role == "user":
                question = content
            elif role == "assistant":
                answer = _SHOWN_CITATION_RE.sub("", content).strip()
                if question and answer:
                    session.add_turn(Turn(question, answer, (), scope))
                question = ""
        return session


def _turn_row(turn: Turn) -> dict[str, Any]:
    return {
        "question": turn.question,
        "answer": turn.answer,
        "cited": list(turn.cited),
        "scope": {
            "conversations": list(turn.scope.conversations),
            "since": turn.scope.since,
            "until": turn.scope.until,
        },
        "limited": turn.limited,
    }


def _rows(data: Mapping[str, Any], key: str) -> list[Any]:
    rows = data.get(key)
    if rows is None:
        return []
    if not isinstance(rows, (list, tuple)):
        raise ValueError(f"{key} must be a list")
    return list(rows)


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text")
    return value


def _texts(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of text")
    return tuple(value)


def _turn(row: Any) -> Turn:
    if not isinstance(row, Mapping):
        raise ValueError("a turn must be a mapping")
    limited = row.get("limited", False)
    if not isinstance(limited, bool):
        raise ValueError("limited must be true or false")
    return Turn(
        _text(row, "question"), _text(row, "answer"), _texts(row.get("cited"), "cited"), _scope(row), limited
    )


def _scope(row: Mapping[str, Any]) -> Scope:
    stored = row.get("scope")
    if stored is None:
        return EVERYTHING
    if not isinstance(stored, Mapping):
        raise ValueError("scope must be a mapping")
    return Scope(
        _texts(stored.get("conversations"), "scope conversations"),
        _text(stored, "since") or None,
        _text(stored, "until") or None,
    )
