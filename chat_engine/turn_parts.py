"""The small parts of a turn: its settings, what one model call gave, a status event."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .errors import ModelError
from .ports import ToolCall

Event = dict[str, Any]
TOO_MANY = (
    "[{count} more lookups of this step were not run: one step runs at most {limit}. "
    "Ask for them in the next step if you still need them.]"
)


@dataclass(frozen=True)
class TurnSettings:
    model: str
    max_output_tokens: int
    max_steps: int
    max_calls_per_step: int
    result_tokens: int
    total_tool_tokens: int
    history_tokens: int
    max_message_chars: int
    timezone: str
    count_tokens: Callable[[str], int]


@dataclass(frozen=True)
class Reply:
    """What one model call gave."""

    text: str = ""  # as the model wrote it, with m-references
    shown: str = ""  # as streamed to the reader, with numbers
    calls: tuple[ToolCall, ...] = ()
    order: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    stop_reason: str = "stop"
    provider_items: tuple[Any, ...] = ()
    streamed: bool = False
    cancelled: bool = False
    forced: bool = False  # the call that had to answer without tools


def status(text: str, reset: bool = False) -> Event:
    event: Event = {"type": "status", "text": text}
    if reset:
        event["reset"] = True
    return event


def no_answer(reply: Reply) -> ModelError:
    """Why a model call that had to answer gave no text."""
    if reply.stop_reason == "length":
        message = "The model used all of its output before it wrote an answer."
    elif reply.stop_reason == "filtered":
        message = "The model provider blocked the answer."
    elif reply.forced:
        message = "The model gave no answer after its last lookup."
    else:
        message = "The model gave no answer."
    return ModelError(message, detail=f"stop_reason={reply.stop_reason} forced={reply.forced}")
