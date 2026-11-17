"""The result of one question: answer, citations, flagged quotes, steps, usage."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from .ports import Message
from .registry import EvidenceRegistry
from .render import ConversationBook
from .verify import unverified_quotes


def citations(
    order: Sequence[str], registry: EvidenceRegistry, book: ConversationBook, text_cap: int = 0
) -> list[dict[str, Any]]:
    """The messages the answer cites, numbered in order of first use.

    `text_cap` is the most characters of text one citation holds; 0 is no cap.
    A cut text keeps its start and has "text_truncated".
    """
    cited = []
    for number, ref in enumerate(order, start=1):
        message = registry.get(ref)
        if message is None:
            continue
        text = str(message.get("text") or "")
        cut = 0 < text_cap < len(text)
        cited.append(
            {
                **message,
                "text": text[:text_cap] if cut else text,
                "text_truncated": cut,
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
    allowed: Callable[[Message], bool],
) -> list[str]:
    """Text a quote may come from: shown messages inside the scope, the user's own words, names.

    The whole stored text of a message counts, also the part a cap cut away.
    """
    sources: list[str] = []
    for message in registry.messages():
        if not allowed(message):
            continue
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
    allowed: Callable[[Message], bool],
    text_cap: int,
    steps: Sequence[dict[str, Any]],
    usage: dict[str, int],
    cancelled: bool,
    truncated: bool,
) -> dict[str, Any]:
    sources = quote_sources(registry, book, questions, redact, allowed)
    return {
        "answer": answer,
        "citations": citations(order, registry, book, text_cap),
        "unverified_quotes": unverified_quotes(raw_answer, sources),
        "invalid_references": list(unknown),
        "steps": [dict(step) for step in steps],
        "usage": dict(usage),
        "cancelled": cancelled,
        "truncated": truncated,
        "mode": "ai",
    }
