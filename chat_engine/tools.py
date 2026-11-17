"""The tools the model may call, and the box that runs them.

Each result is compact, names people by label only and gives every message a
reference from the evidence registry. The scope of the user limits every tool.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional

from .errors import ArchiveError
from .ports import ChatArchive, Labeler, ToolSpec
from .registry import ConversationAliases, EvidenceRegistry
from .render import Limits
from .scope import Scope
from .tool_args import ToolInputError, parse_arguments
from .tool_context import Prepared, ToolContext, ToolOutcome
from .tool_figures import GROUPS, LIST_LIMIT_MAX, prepare_count, prepare_list, prepare_stats
from .tool_search import READ_RADIUS_MAX, SEARCH_LIMIT_MAX, prepare_read, prepare_search

logger = logging.getLogger(__name__)

_DAY = "A day as YYYY-MM-DD in local time. The day itself is included."
_CONVERSATIONS = (
    "Conversation aliases such as c3, from list_conversations or an earlier result. "
    "Null means every conversation in the scope."
)


def _nullable(kind: str, description: str, **extra: Any) -> dict[str, Any]:
    return {"type": [kind, "null"], "description": description, **extra}


def _schema(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": dict(properties),
        "required": list(properties),
    }


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "search_messages",
        "Find messages by their words. Each hit comes with the messages around it; every line has a "
        "reference such as m12. All words must be in the same message. Put a phrase in double quotes. "
        "End a word with * to match its beginning. Case and accents do not matter. Use few, distinctive "
        'words. Set match to "any" to rank messages that hold any of the words: good for synonyms and '
        "for words in other languages. For the first or the last time, set order. "
        "To read one conversation with no search word, leave query empty and give that one conversation: "
        'order "oldest" reads it from its start, "newest" from its end; since and until read a period.',
        _schema(
            {
                "query": _nullable("string", "Words to look for. Empty or null reads one conversation."),
                "match": _nullable("string", '"all" (default) or "any".', enum=["all", "any", None]),
                "conversations": _nullable("array", _CONVERSATIONS, items={"type": "string"}),
                "since": _nullable("string", _DAY),
                "until": _nullable("string", _DAY),
                "from_me": _nullable(
                    "boolean", "true: only messages the user sent. false: only messages from others."
                ),
                "order": _nullable(
                    "string", '"newest" (default), "oldest" or "relevance".',
                    enum=["newest", "oldest", "relevance", None],
                ),
                "limit": _nullable("integer", f"Hits to show, 1 to {SEARCH_LIMIT_MAX}. Default 8."),
            }
        ),
    ),
    ToolSpec(
        "read_around",
        "Read the exchange around one message: the messages before and after it in the same "
        "conversation. Use it before you rely on a hit, and to read on from the edge of a result.",
        _schema(
            {
                "reference": {"type": "string", "description": "A message reference such as m12."},
                "radius": _nullable(
                    "integer", f"Messages on each side, 1 to {READ_RADIUS_MAX}. Default 6."
                ),
            }
        ),
    ),
    ToolSpec(
        "count_messages",
        "Count messages exactly. Use it for every number: how many, how often, who most, which month. "
        "Without group_by it gives one number. With group_by it gives one row per group, each with the "
        "total, the count the user sent and the count from others.",
        _schema(
            {
                "conversations": _nullable("array", _CONVERSATIONS, items={"type": "string"}),
                "since": _nullable("string", _DAY),
                "until": _nullable("string", _DAY),
                "from_me": _nullable(
                    "boolean", "true: only messages the user sent. false: only messages from others."
                ),
                "text": _nullable(
                    "string", "Count only messages that match these words. Same rules as search_messages."
                ),
                "group_by": _nullable("string", "One row per group.", enum=[*GROUPS, None]),
            }
        ),
    ),
    ToolSpec(
        "list_conversations",
        "Find conversations and people by name. Gives the alias (c3), the name, direct or group, message "
        "counts and the first and last day. One person can have several threads and be in groups: all "
        "are listed. Digits match part of a phone number. Without a name it lists the conversations "
        "with the most messages.",
        _schema(
            {
                "name": _nullable("string", "A name, a part of a name, or digits of a phone number."),
                "limit": _nullable("integer", f"Rows to show, 1 to {LIST_LIMIT_MAX}. Default 10."),
            }
        ),
    ),
    ToolSpec(
        "conversation_stats",
        "Figures for one conversation: totals, sent and received, first and last time, active days, "
        "busiest day, longest gap. Give days to limit the period.",
        _schema(
            {
                "conversation": {"type": "string", "description": "A conversation alias such as c3."},
                "since": _nullable("string", _DAY),
                "until": _nullable("string", _DAY),
            }
        ),
    ),
)

_BUILDERS: dict[str, Callable[[ToolContext, Mapping[str, Any]], Prepared]] = {
    "search_messages": prepare_search,
    "read_around": prepare_read,
    "count_messages": prepare_count,
    "list_conversations": prepare_list,
    "conversation_stats": prepare_stats,
}


class ToolBox:
    """Checks and runs tool calls for one question."""

    specs = TOOL_SPECS

    def __init__(
        self,
        archive: ChatArchive,
        labeler: Labeler,
        registry: EvidenceRegistry,
        aliases: ConversationAliases,
        scope: Scope,
        limits: Optional[Limits] = None,
    ) -> None:
        self.context = ToolContext(archive, labeler, registry, aliases, scope, limits or Limits())

    def prepare(self, name: str, raw_arguments: str) -> Prepared:
        """Check a call. A refused call still runs: it tells the model why."""
        builder = _BUILDERS.get(name)
        if builder is None:
            return _refused(f'There is no tool named "{name}".', name)
        try:
            prepared = self.guard(lambda: builder(self.context, parse_arguments(raw_arguments)))
        except ToolInputError as error:
            return _refused(f"Cannot run {name}: {error}", name)
        return self.wrap(name, prepared)

    def wrap(self, name: str, prepared: Prepared) -> Prepared:
        """Give a lookup the guard and the redaction every tool result gets."""
        redact = self.context.labeler.redact
        return Prepared(redact(prepared.status), lambda budget: self._run(name, prepared, budget))

    def guard(self, action: Callable[[], Any]) -> Any:
        """Run archive work. An unexpected failure becomes a typed ArchiveError."""
        try:
            return action()
        except ToolInputError:
            raise
        except Exception as error:
            logger.exception("archive lookup failed")
            raise ArchiveError(detail=f"{type(error).__name__}: {error}") from error

    def _run(self, name: str, prepared: Prepared, budget: int) -> ToolOutcome:
        try:
            outcome = self.guard(lambda: prepared.run(max(0, budget)))
        except ToolInputError as error:
            return _refused(f"Cannot run {name}: {error}", name).run(budget)
        redact = self.context.labeler.redact
        return ToolOutcome(redact(outcome.text), redact(outcome.summary), outcome.ok)


def _refused(reason: str, name: str) -> Prepared:
    outcome = ToolOutcome(reason, f"Could not run {name}", ok=False)
    return Prepared("", lambda _budget: outcome)
