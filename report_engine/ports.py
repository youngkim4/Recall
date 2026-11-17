"""Ports: what the engine needs from the outside, and nothing more.

The engine never opens a database, a CSV file or a network connection itself.
A caller gives it a message source, a labeler and a model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict


class SourceMessage(TypedDict):
    """One message. `timestamp` is local time with no zone: YYYY-MM-DD HH:MM:SS."""

    id: str
    timestamp: str
    is_from_me: bool
    sender: str
    text: str


class ReportSource(Protocol):
    """Yields the messages of one conversation, oldest first.

    `since` and `until` are days (YYYY-MM-DD) or full timestamps. Both ends are
    included, and a day covers the whole day.
    """

    def messages(
        self, conversation: str, since: str | None, until: str | None
    ) -> Iterator[SourceMessage]: ...


Labeler = Callable[[str], str]
"""Maps a raw sender to the name the model reads. Own messages are never passed in."""


class ChatMessage(TypedDict):
    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True)
class LLMUsage:
    """Tokens of one call. `input_tokens` includes the cached ones."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0


LLMStatus = Literal["completed", "incomplete", "refused", "failed"]


@dataclass(frozen=True)
class LLMResult:
    """One reply. `parsed` is set only when a schema was given and the text is valid JSON."""

    text: str
    parsed: Mapping[str, Any] | None
    usage: LLMUsage
    status: LLMStatus
    attempts: int = 1


class LLM(Protocol):
    """One model, already bound to its id. Raises LLMError when it can not answer."""

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: Mapping[str, Any] | None = None,
        max_output_tokens: int,
        reasoning: str | None = None,
    ) -> LLMResult: ...
