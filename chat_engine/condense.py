"""Condense older lookup results, so a question can keep reading without filling the prompt.

When the results held in the prompt pass `held_tool_tokens`, every result
before the newest round goes to one model call with no tools. The call keeps
each fact that can matter for the question, with the m-reference of each
message it comes from. The first older result takes the condensed text; the
others take `CONDENSED_ABOVE`. Each tool call keeps its answer, so the
transcript stays well formed.

The evidence registry is not touched: a reference that the condensed text
keeps can still be cited and read around.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

from .ports import LLM, ChatMessage, Completed, LLMRequest, TextDelta, Usage
from .cancel import CancelToken

CONDENSED_MARK = "[Condensed results of earlier lookups]"
CONDENSED_ABOVE = "[Condensed into the first earlier result.]"
CONDENSE_SYSTEM = (
    "You condense the results of lookups in a person's message archive, so that an assistant can keep "
    "working on a question. Keep every fact, date, count, name label and quote that can matter for the "
    "question, and the reference (such as m12) of each message it comes from. Keep a message whole when "
    "its exact words can matter. Drop what cannot matter. Write short lines, one fact a line, in the "
    "order of time. Do not answer the question. Do not add anything that the results do not say."
)


@dataclass(frozen=True)
class Condensed:
    """The new transcript and what the call used."""

    transcript: tuple[ChatMessage, ...]
    usage: Usage


def older_results(transcript: Sequence[ChatMessage]) -> tuple[int, ...]:
    """The places of the tool results before the newest round, that hold text to condense."""
    last_round = max((place for place, m in enumerate(transcript) if m.role == "assistant" and m.tool_calls), default=-1)
    return tuple(
        place
        for place, message in enumerate(transcript)
        if message.role == "tool" and place < last_round and message.content != CONDENSED_ABOVE
    )


def held_tokens(transcript: Iterable[ChatMessage], count: Callable[[str], int]) -> int:
    return sum(count(message.content) for message in transcript if message.role == "tool")


def needs_condensing(transcript: Sequence[ChatMessage], limit: int, count: Callable[[str], int]) -> bool:
    return held_tokens(transcript, count) > limit and bool(older_results(transcript))


def request(
    model: str, question: str, transcript: Sequence[ChatMessage], max_output_tokens: int, cancel: CancelToken
) -> LLMRequest:
    blocks = "\n\n".join(transcript[place].content for place in older_results(transcript))
    user = f"The question: {question}\n\nThe results to condense:\n\n{blocks}"
    return LLMRequest(
        model=model,
        messages=(ChatMessage("system", CONDENSE_SYSTEM), ChatMessage("user", user)),
        tools=(),
        max_output_tokens=max_output_tokens,
        allow_tools=False,
        cancel=cancel,
    )


def condense(llm: LLM, asked: LLMRequest, transcript: Sequence[ChatMessage]) -> Condensed:
    """Run the call and put its text in the place of the older results. Raises what the model port raises."""
    pieces: list[str] = []
    usage = Usage()
    for event in llm.stream(asked):
        if isinstance(event, TextDelta) and not event.commentary:
            pieces.append(event.text)
        elif isinstance(event, Completed):
            usage = event.usage
    text = "".join(pieces).strip()
    places = older_results(transcript)
    if not text or not places:
        return Condensed(tuple(transcript), usage)
    first, rest = places[0], frozenset(places[1:])
    changed = tuple(
        replace(message, content=f"{CONDENSED_MARK}\n{text}")
        if place == first
        else replace(message, content=CONDENSED_ABOVE)
        if place in rest
        else message
        for place, message in enumerate(transcript)
    )
    return Condensed(changed, usage)
