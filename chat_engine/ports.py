"""Ports of the chat engine.

The engine reads the archive, calls a model and names people only through the
first three interfaces below. Tests replace all three.

Two more ports are optional: a search by meaning and a digest of one
conversation. An engine that gets none of them offers no tool for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, Protocol, Sequence, TypedDict, Union

from .cancel import CancelToken

# --- archive ---------------------------------------------------------------


class Message(TypedDict):
    id: str
    conversation: str
    timestamp: str  # local time, no zone, "YYYY-MM-DD HH:MM:SS"
    sent_at: int
    is_from_me: bool
    sender: str  # "" for own messages
    text: str
    kind: str  # "text" | "attachment" | "system"
    attachments: list[str]
    reply_to: Optional[str]
    edited: bool
    unsent: bool


class Conversation(TypedDict):
    id: str
    is_group: bool
    name: str
    message_count: int
    sent_count: int
    received_count: int
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]
    members: list[str]
    person_id: Optional[int]


class ArchiveSummary(TypedDict):
    messages: int
    conversations: int
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]


class SearchHit(TypedDict):
    message: Message
    ranges: list[list[int]]


class SearchResult(TypedDict):
    total: int
    hits: list[SearchHit]


class MessagePage(TypedDict):
    messages: list[Message]
    has_older: bool
    has_newer: bool


class CountRow(TypedDict):
    key: str
    total: int
    sent: int
    received: int


class ChatArchive(Protocol):
    """The part of `archive.queries.Archive` that chat uses.

    `since` and `until` are days "YYYY-MM-DD" in local time, both included.
    """

    def summary(self) -> ArchiveSummary: ...

    def conversations(
        self, *, limit: Optional[int] = None, offset: int = 0, order: str = "count"
    ) -> list[Conversation]: ...

    def conversation(self, conversation_id: str) -> Optional[Conversation]: ...

    def messages(
        self,
        conversation_id: str,
        *,
        around: Optional[str] = None,
        before: Optional[str] = None,
        after: Optional[str] = None,
        limit: int = 50,
    ) -> MessagePage: ...

    def context(self, message_id: str, *, radius: int = 6) -> list[Message]: ...

    def search(
        self,
        query: str,
        *,
        conversations: Optional[Sequence[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        from_me: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
        order: str = "newest",
    ) -> SearchResult: ...

    def count(
        self,
        *,
        conversations: Optional[Sequence[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        from_me: Optional[bool] = None,
        text: Optional[str] = None,
        group_by: Optional[str] = None,
    ) -> Union[int, list[CountRow]]: ...

    def stats(
        self, conversation_id: str, *, since: Optional[str] = None, until: Optional[str] = None
    ) -> dict[str, Any]: ...

    def message(self, message_id: str) -> Optional[Message]:
        """One message by its id. None when the archive does not hold it, or holds it as a system row."""
        ...


# --- model -----------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """One tool the model may call. `parameters` is a JSON schema object."""

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A tool call as the model wrote it. `arguments` is JSON text, unchecked."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatMessage:
    """One message of the model input.

    role is "system", "user", "assistant" or "tool". `provider_items` belongs
    to the model adapter that produced an assistant message; the engine hands
    it back unread on the next call.
    """

    role: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str = ""
    provider_items: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class TextDelta:
    """A piece of answer text. `commentary` marks a note written before a tool call."""

    text: str
    commentary: bool = False


@dataclass(frozen=True)
class Completed:
    """End of one model call. stop_reason is "stop", "length" or "filtered"."""

    usage: Usage = Usage()
    stop_reason: str = "stop"
    provider_items: tuple[Mapping[str, Any], ...] = ()


LLMEvent = Union[TextDelta, ToolCall, Completed]


@dataclass(frozen=True)
class LLMRequest:
    model: str
    messages: tuple[ChatMessage, ...]
    tools: tuple[ToolSpec, ...]
    max_output_tokens: int
    allow_tools: bool = True
    cancel: Optional[CancelToken] = None


class LLM(Protocol):
    """One streaming model call.

    Yields text deltas and tool calls as they arrive, then one `Completed`.
    Raises a `ChatEngineError` subclass on failure. Closing the iterator stops
    the call.
    """

    def stream(self, request: LLMRequest) -> Iterator[LLMEvent]: ...


# --- labels ----------------------------------------------------------------


class Labeler(Protocol):
    """The name a conversation or a sender goes by in text the model reads.

    A saved name is used as it is. Anyone else is masked: the raw phone number
    or email address must never be returned.

    Before any text goes to the model, the engine calls `conversation` for
    every conversation of the archive and `sender` for every member. A
    labeler can learn the handles there. `redact` must then mask each handle
    of a person who is not saved, in every spelling, anywhere in a text.
    """

    def conversation(self, conversation: Conversation) -> str: ...

    def sender(self, handle: str) -> str: ...

    def redact(self, text: str) -> str:
        """Mask raw handles of people who are not saved inside free text."""
        ...


# --- memory: search by meaning ---------------------------------------------


class MeaningHit(TypedDict):
    """One short exchange that a search by meaning found."""

    conversation: str  # conversation id
    first_message_id: str  # the exchange runs from this message
    last_message_id: str  # to this message, in the same conversation
    score: float  # higher is nearer


class MeaningSearch(Protocol):
    """Finds short exchanges by what they are about. The hits come best first.

    `since` and `until` are days, both included. The engine checks every hit
    against the scope again, so a hit outside it is dropped. A failure here
    does not fail the question: the engine then ranks by words alone.
    """

    def search(
        self,
        query: str,
        *,
        conversations: Optional[Sequence[str]],
        since: Optional[str],
        until: Optional[str],
        limit: int,
    ) -> list[MeaningHit]: ...


# --- memory: digest of one conversation ------------------------------------


class DigestMoment(TypedDict):
    date: str  # "YYYY-MM-DD"
    title: str
    detail: str
    quote: str  # verified by the source, or ""
    message_ids: list[str]  # archive ids, verified by the source


class DigestPeriod(TypedDict):
    label: str
    first_day: str
    last_day: str
    messages: int
    summary: str
    moments: list[DigestMoment]  # the strongest first


class DigestCost(TypedDict):
    periods: int
    periods_cached: int
    usd: float  # cost of the periods that are not cached


class DigestResult(TypedDict):
    periods: list[DigestPeriod]
    periods_failed: int


class DigestSource(Protocol):
    """The story of one conversation, period by period.

    `estimate` builds nothing and costs nothing. `digest` builds the periods
    that are not cached. It stops when `cancel` is cancelled. A failure here
    does not fail the question.
    """

    def estimate(
        self, conversation_id: str, *, since: Optional[str], until: Optional[str]
    ) -> DigestCost: ...

    def digest(
        self, conversation_id: str, *, since: Optional[str], until: Optional[str], cancel: CancelToken
    ) -> DigestResult: ...
