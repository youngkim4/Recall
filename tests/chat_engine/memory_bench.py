"""A tool box with the memory ports, for the tests of the two memory tools."""

from __future__ import annotations

import json
from typing import Any, Optional

from chat_engine.cancel import CancelToken
from chat_engine.fake_archive import FakeArchive
from chat_engine.fake_digests import FakeDigests, moment, period
from chat_engine.labeler import MaskingLabeler
from chat_engine.registry import ConversationAliases, EvidenceRegistry
from chat_engine.render import Limits
from chat_engine.scope import Scope
from chat_engine.tool_context import Memory, ToolOutcome
from chat_engine.tools import ToolBox
from tests.chat_engine.tiny import MAYA_PHONE, SAVED_NAMES, tiny_archive, tiny_labeler

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


def maya_story() -> list:
    return [
        period(
            "2021", "2021-03-04", "2021-03-04", 4, "Maya signs a lease and plans the move.",
            [
                moment(
                    "2021-03-04", "The lease is signed", "Maya tells you the news.", "the lease got signed btw",
                    ["a2", "a3"],
                ),
                moment("2021-03-04", "Moving day is set", "They move in April.", "", ["a4"]),
            ],
        ),
        period(
            "2023", "2023-07-11", "2023-07-11", 3, "The landlord raises the rent.",
            [moment("2023-07-11", "Rent goes up", "Fifty more a month.", "fifty more a month", ["a7"])],
        ),
        period("2025", "2025-12-24", "2025-12-24", 2, "A Christmas greeting and a plan for the move."),
    ]


FAKE_OPTIONS = {"usd", "cached", "failed", "error", "estimate_error", "during", "honour_cancel"}


def bench_for(story: list | None = None, **options: object) -> MemoryBench:
    fake_options = {key: options.pop(key) for key in list(options) if key in FAKE_OPTIONS}
    digests = FakeDigests({MAYA_PHONE: maya_story() if story is None else story}, **fake_options)
    return MemoryBench(digests=digests, **options)
