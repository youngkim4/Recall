"""What one chat remembers between questions: the turns and the references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .registry import ConversationAliases, EvidenceRegistry


@dataclass(frozen=True)
class Turn:
    """One question and its answer. The answer keeps the m-references."""

    question: str
    answer: str
    cited: tuple[str, ...] = ()


class ChatSession:
    """Keep one per chat and pass it to every question of that chat.

    Not safe for two questions at the same time.
    """

    def __init__(self) -> None:
        self.registry = EvidenceRegistry()
        self.aliases = ConversationAliases()
        self.turns: tuple[Turn, ...] = ()

    def add_turn(self, turn: Turn) -> None:
        self.turns = (*self.turns, turn)

    def questions(self) -> list[str]:
        return [turn.question for turn in self.turns]

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": [
                {"question": turn.question, "answer": turn.answer, "cited": list(turn.cited)}
                for turn in self.turns
            ],
            "evidence": self.registry.snapshot(),
            "conversations": self.aliases.snapshot(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChatSession":
        session = cls()
        session.registry = EvidenceRegistry.restore(data.get("evidence") or [])
        session.aliases = ConversationAliases.restore(data.get("conversations") or [])
        session.turns = tuple(
            Turn(str(row.get("question") or ""), str(row.get("answer") or ""), tuple(row.get("cited") or ()))
            for row in data.get("turns") or []
        )
        return session

    @classmethod
    def from_history(cls, history: Iterable[Mapping[str, Any]]) -> "ChatSession":
        """A session from plain role/content turns. The evidence is not restored."""
        session = cls()
        question = ""
        for row in history:
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            if str(row.get("role")) == "assistant":
                if question:
                    session.add_turn(Turn(question, content))
                question = ""
            else:
                question = content
        return session
