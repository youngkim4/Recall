"""A tool box with the memory ports, for the tests of the two memory tools."""

from __future__ import annotations

import json
from typing import Any, Optional

from chat_engine.cancel import CancelToken
from chat_engine.fake_archive import FakeArchive
from chat_engine.labeler import MaskingLabeler
from chat_engine.registry import ConversationAliases, EvidenceRegistry
from chat_engine.render import Limits
from chat_engine.scope import Scope
from chat_engine.tool_context import Memory, ToolOutcome
from chat_engine.tools import ToolBox
from tests.chat_engine.tiny import SAVED_NAMES, tiny_archive, tiny_labeler

RAW_HANDLES = ["7205550199", "720555", "555-0199", "+1720", "+1 720", "rk.unsaved", "example.org"]


class MemoryBench:
    def __init__(
        self,
        *,
        meaning: Any = None,
        digests: Any = None,
        scope: Optional[Scope] = None,
        archive: Optional[FakeArchive] = None,
        labeler: Any = None,
        limits: Optional[Limits] = None,
        digest_usd_limit: float = 2.0,
        digest_tokens: int = 24_000,
    ) -> None:
        self.archive = archive or tiny_archive()
        self.registry = EvidenceRegistry()
        self.aliases = ConversationAliases()
        self.cancel = CancelToken()
        memory = Memory(meaning, digests, digest_tokens, digest_usd_limit, self.cancel)
        self.box = ToolBox(
            self.archive, labeler or tiny_labeler(), self.registry, self.aliases, scope or Scope(),
            limits or Limits(), memory,
        )

    def prepare(self, tool: str, /, **arguments: object) -> Any:
        return self.box.prepare(tool, json.dumps(arguments))

    def run(self, tool: str, /, budget: int = 3000, **arguments: object) -> ToolOutcome:
        prepared = self.prepare(tool, **arguments)
        return prepared.run(budget if prepared.own_budget is None else prepared.own_budget)

    def alias(self, conversation_id: str) -> str:
        return self.aliases.alias(conversation_id)

    def read(self, kind: str) -> list[dict]:
        """The arguments of every archive call of one kind."""
        return [arguments for name, arguments in self.archive.calls if name == kind]


class GrowingLabeler(MaskingLabeler):
    """A labeler whose masks make every text much longer: no result fits after the masking."""

    def __init__(self) -> None:
        super().__init__(SAVED_NAMES)

    def redact(self, text: str) -> str:
        return super().redact(text) + " pad" * len(text)


def shown_refs(text: str) -> set[str]:
    """The references at the start of the message lines of a result."""
    refs = set()
    for line in text.splitlines():
        head = line.strip().split(" ", 1)[0]
        if head[:1] == "m" and head[1:].isdigit():
            refs.add(head)
    return refs
