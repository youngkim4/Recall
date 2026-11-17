"""The result of one question: answer, citations, flagged quotes, steps, usage."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from .registry import EvidenceRegistry
from .render import ConversationBook
from .verify import unverified_quotes


def citations(order: Sequence[str], registry: EvidenceRegistry, book: ConversationBook) -> list[dict[str, Any]]:
    """The messages the answer cites, numbered in order of first use."""
    cited = []
    for number, ref in enumerate(order, start=1):
        message = registry.get(ref)
        if message is None:
            continue
        cited.append(
            {
                **message,
                "number": number,
                "reference": ref,
                "conversation_label": book.label(str(message["conversation"])),
                "sender_label": book.speaker(message),
            }
        )
    return cited


def quote_sources(
    registry: EvidenceRegistry,
    book: ConversationBook,
    questions: Sequence[str],
    redact: Callable[[str], str],
) -> list[str]:
    """Text a quote may come from: shown messages, the user's own words, names."""
    sources: list[str] = []
    for message in registry.messages():
        text = str(message.get("text") or "")
        sources.append(text)
        masked = redact(text)
        if masked != text:
            sources.append(masked)
    sources.extend(questions)
    sources.extend(book.labels())
    return sources


def build_payload(
    *,
    answer: str,
    raw_answer: str,
    order: Sequence[str],
    unknown: Sequence[str],
    registry: EvidenceRegistry,
    book: ConversationBook,
    questions: Sequence[str],
    redact: Callable[[str], str],
    steps: Sequence[dict[str, Any]],
    usage: dict[str, int],
    cancelled: bool,
    truncated: bool,
) -> dict[str, Any]:
    sources = quote_sources(registry, book, questions, redact)
    return {
        "answer": answer,
        "citations": citations(order, registry, book),
        "unverified_quotes": unverified_quotes(raw_answer, sources),
        "invalid_references": list(unknown),
        "steps": [dict(step) for step in steps],
        "usage": dict(usage),
        "cancelled": cancelled,
        "truncated": truncated,
    }
