"""The tools the model may call, and the box that runs them.

Each result is compact, names people by label only and gives every message a
reference from the evidence registry. The scope of the user limits every tool.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable, Mapping, Optional

from .errors import ArchiveError, ChatEngineError
from .ports import ChatArchive, Labeler
from .prompt import DIGEST, MEANING
from .registry import ConversationAliases, EvidenceRegistry
from .render import Limits, squeeze
from .scope import Scope
from .tool_args import ToolInputError, parse_arguments
from .tool_context import Memory, Prepared, ToolContext, ToolOutcome
from .tool_digest import prepare_digest
from .tool_figures import prepare_count, prepare_list, prepare_stats
from .tool_meaning import prepare_search_or_meaning
from .tool_search import prepare_read, prepare_search
from .tool_specs import TOOL_SPECS, tool_specs

__all__ = ["TOOL_SPECS", "ToolBox"]

logger = logging.getLogger(__name__)
_WRITES = 2  # a result is written at most this many times to fit its budget
_TOO_LARGE = "The result does not fit. Ask for less: a smaller limit, fewer conversations, or a shorter period."

_ECHO_CHARS = 40
Builder = Callable[[ToolContext, Mapping[str, Any]], Prepared]

_BUILDERS: dict[str, Builder] = {
    "search_messages": prepare_search,
    "read_around": prepare_read,
    "count_messages": prepare_count,
    "list_conversations": prepare_list,
    "conversation_stats": prepare_stats,
}


class ToolBox:
    """Checks and runs tool calls for one question.

    `memory` holds the optional ports. A tool of a port that is missing is
    not in `specs`, and a call of it is refused as an unknown tool.
    """

    specs = TOOL_SPECS
    offered: frozenset[str] = frozenset()

    def __init__(
        self,
        archive: ChatArchive,
        labeler: Labeler,
        registry: EvidenceRegistry,
        aliases: ConversationAliases,
        scope: Scope,
        limits: Optional[Limits] = None,
        memory: Optional[Memory] = None,
    ) -> None:
        self.context = ToolContext(archive, labeler, registry, aliases, scope, limits or Limits(), memory)
        self._builders = dict(_BUILDERS)
        ports = self.context.memory
        if ports.meaning is not None:
            self._builders["search_messages"] = prepare_search_or_meaning
        if ports.digests is not None:
            self._builders[DIGEST] = prepare_digest
        names = {MEANING: ports.meaning, DIGEST: ports.digests}
        self.offered = frozenset(name for name, port in names.items() if port is not None)
        if self.offered:
            self.specs = tool_specs(meaning=MEANING in self.offered, digest=DIGEST in self.offered)

    def prepare(self, name: str, raw_arguments: str) -> Prepared:
        """Check a call. A refused call still runs: it tells the model why."""
        builder = self._builders.get(name)
        if builder is None:
            return self._refuse(f'There is no tool named "{_brief(name)}".', name)
        try:
            prepared = self.guard(lambda: builder(self.context, parse_arguments(raw_arguments)))
        except ToolInputError as error:
            return self._refuse(f"Cannot run {name}: {error}", name)
        return self.wrap(name, prepared)

    def wrap(self, name: str, prepared: Prepared) -> Prepared:
        """Give a lookup the guard and the redaction every tool result gets."""
        redact = self.context.labeler.redact
        return replace(
            prepared, status=redact(prepared.status), run=lambda budget: self._run(name, prepared, budget)
        )

    def guard(self, action: Callable[[], Any]) -> Any:
        """Run archive work. An unexpected failure becomes a typed ArchiveError."""
        try:
            return action()
        except (ToolInputError, ChatEngineError):
            raise
        except Exception as error:
            logger.exception("archive lookup failed")
            raise ArchiveError(detail=f"{type(error).__name__}: {error}") from error

    def _run(self, name: str, prepared: Prepared, budget: int) -> ToolOutcome:
        """Run a lookup and mask its result. The result never counts more tokens than `budget`.

        A mask is longer than the handle it stands for. When the labeler
        learns a handle while a result is written, the lines before it grow
        in the last masking and the result can pass its budget. Then the
        result is written again: now every line is masked as it is written.
        """
        registry = self.context.registry
        redact = self.context.labeler.redact
        count = self.context.limits.count_tokens
        room = max(0, budget)
        for _ in range(_WRITES):
            mark = registry.checkpoint()
            try:
                outcome = self.guard(lambda: prepared.run(room))
            except ToolInputError as error:
                return self._refuse(f"Cannot run {name}: {error}", name).run(budget)
            text = redact(outcome.text)
            if count(text) <= room:
                return ToolOutcome(text, redact(outcome.summary), outcome.ok)
            registry.rollback(mark)  # no message stays registered without its line
        logger.warning("the result of %s does not fit its budget after masking", name)
        return self._refuse(_TOO_LARGE, name).run(budget)

    def _refuse(self, reason: str, name: str) -> Prepared:
        """A call that is not run. The model gets the reason; no raw handle is echoed."""
        text = self.context.labeler.redact(reason)
        count = self.context.limits.count_tokens
        summary = f"Could not run {_brief(name)}"
        return Prepared("", lambda budget: ToolOutcome(squeeze(text, max(0, budget), count), summary, ok=False))


def _brief(value: object) -> str:
    """Model input cut short before it is echoed."""
    text = " ".join(str(value).split())
    return text if len(text) <= _ECHO_CHARS else text[:_ECHO_CHARS] + "..."
