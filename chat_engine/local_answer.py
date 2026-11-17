"""An answer from keyword search only. No model is called.

The engine never uses this by itself. The caller may offer it when the model
call failed, and must say that it did.
"""

from __future__ import annotations

from typing import Any, Optional

from .labeler import MaskingLabeler
from .ports import ChatArchive, Labeler
from .registry import ConversationAliases, EvidenceRegistry
from .render import Limits
from .result import citations
from .scope import EVERYTHING, Scope
from .search import SearchRequest, run_search
from .seed import question_units
from .tool_context import ToolContext

_EXCERPT_CHARS = 200
_NO_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cached_input_tokens": 0,
    "reasoning_tokens": 0,
    "model_calls": 0,
    "tool_calls": 0,
}


def local_answer(
    archive: ChatArchive,
    question: str,
    scope: Scope = EVERYTHING,
    *,
    labeler: Optional[Labeler] = None,
    limit: int = 6,
) -> dict[str, Any]:
    """The messages that match the words of the question best, as a plain list."""
    names = labeler or MaskingLabeler()
    registry = EvidenceRegistry()
    context = ToolContext(archive, names, registry, ConversationAliases(), scope, Limits())
    units = question_units(question, names.redact)
    if not units:
        return _result("No AI model was used. Give at least one word to search for.", [], "No search was run")
    since, until = context.days(None, None)
    request = SearchRequest(
        query=" ".join(units),
        match="any",
        conversations=context.conversations(None),
        since=since,
        until=until,
        limit=max(1, limit),
    )
    outcome = run_search(archive, request)
    words = ", ".join(outcome.used or units)
    if not outcome.hits:
        where = " in this scope" if scope.is_limited else ""
        return _result(
            f"No AI model was used. I found no message{where} with these words: {words}.",
            [],
            f"Searched for “{request.query}”: 0 matches",
        )
    lines = [f"No AI model was used. These messages match your words best ({words}):"]
    refs = []
    for number, found in enumerate(outcome.hits, start=1):
        refs.append(registry.add(found.message))
        lines.append(_line(context, found.message, number))
    cited = citations(refs, registry, context.book)
    step = f"Searched for “{request.query}”: {len(cited)} matches shown"
    return _result(names.redact("\n".join(lines)), cited, step)


def _line(context: ToolContext, message: Any, number: int) -> str:
    text = " ".join(str(message.get("text") or "").split())
    excerpt = text if len(text) <= _EXCERPT_CHARS else text[:_EXCERPT_CHARS].rstrip() + "…"
    day = str(message.get("timestamp") or "")[:10]
    speaker = context.book.speaker(message)
    place = context.book.label(str(message["conversation"]))
    who = speaker if speaker == place else f"{speaker} in {place}"
    return f'- {who}, {day}: "{excerpt}" [{number}]'


def _result(answer: str, cited: list[dict[str, Any]], step: str) -> dict[str, Any]:
    return {
        "answer": answer,
        "citations": cited,
        "unverified_quotes": [],
        "invalid_references": [],
        "steps": [{"tool": "search_messages", "text": step, "seeded": True}],
        "usage": dict(_NO_USAGE),
        "cancelled": False,
        "truncated": False,
        "mode": "local",
    }
