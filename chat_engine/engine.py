"""The chat engine: ask a question about the archive, get a streamed answer.

    engine = ChatEngine(archive, llm, labeler, EngineConfig(model=..., max_output_tokens=...))
    result = engine.ask("When did we first talk about the lease?", on_event=send)

The model, its output cap and every budget come from the caller. Nothing about
a model is written here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Generator, Iterator, Optional

from .cancel import CancelToken
from .digest_fit import MIN_DIGEST_TOKENS
from .errors import ChatEngineError
from .ports import LLM, ChatArchive, DigestSource, Labeler, MeaningSearch
from .render import estimate_tokens
from .scope import EVERYTHING, Scope
from .session import ChatSession
from .tool_context import Memory
from .turn import Event, QuestionTurn, TurnSettings

MAX_QUESTION_CHARS = 4000


@dataclass(frozen=True)
class EngineConfig:
    """Settings of the engine. `model` and `max_output_tokens` have no default.

    max_steps: model calls that may use tools; one more call must answer.
    max_calls_per_step: the most lookups one model call may start.
    result_tokens: the most one tool result may hold.
    total_tool_tokens: the most all tool results of one question may hold.
    history_tokens: the most the earlier turns of the chat may hold.
    digest_tokens: the most one conversation digest may hold. It has this budget of its own, at least 600.
    digest_parts_limit: the most periods not yet read that one digest reads inside a question. 0 reads none.
    digest_usd_limit: a digest that costs more to build is not built.
    timezone: the name shown in the prompt; "" takes the zone of the clock.
    count_tokens: a tokenizer; the default is a cautious estimate.
    """

    model: str
    max_output_tokens: int
    max_steps: int = 8
    max_calls_per_step: int = 8
    result_tokens: int = 3000
    total_tool_tokens: int = 30000
    history_tokens: int = 6000
    max_message_chars: int = 4000
    timezone: str = ""
    count_tokens: Callable[[str], int] = estimate_tokens
    digest_tokens: int = 24_000
    digest_usd_limit: float = 0.5
    digest_parts_limit: int = 2

    def __post_init__(self) -> None:
        if not str(self.model or "").strip():
            raise ValueError("model is required")
        for name in (
            "max_output_tokens", "max_steps", "max_calls_per_step", "result_tokens", "total_tool_tokens",
            "max_message_chars",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.history_tokens < 0:
            raise ValueError("history_tokens must not be negative")
        if int(self.digest_tokens) < MIN_DIGEST_TOKENS:  # less does not hold the header, the notes and a few periods
            raise ValueError(f"digest_tokens must be at least {MIN_DIGEST_TOKENS}")
        if int(self.digest_parts_limit) < 0:
            raise ValueError("digest_parts_limit must not be negative")
        if not float(self.digest_usd_limit) >= 0:  # also refuses a value that is not a number
            raise ValueError("digest_usd_limit must not be negative")

    def settings(self) -> TurnSettings:
        return TurnSettings(
            model=self.model,
            max_output_tokens=self.max_output_tokens,
            max_steps=self.max_steps,
            max_calls_per_step=self.max_calls_per_step,
            result_tokens=self.result_tokens,
            total_tool_tokens=self.total_tool_tokens,
            history_tokens=self.history_tokens,
            max_message_chars=self.max_message_chars,
            timezone=self.timezone,
            count_tokens=self.count_tokens,
        )


class ChatEngine:
    def __init__(
        self,
        archive: ChatArchive,
        llm: LLM,
        labeler: Labeler,
        config: EngineConfig,
        *,
        meaning: Optional[MeaningSearch] = None,
        digests: Optional[DigestSource] = None,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        """`meaning` and `digests` are optional. Without one, its tool is not offered."""
        self._archive = archive
        self._llm = llm
        self._labeler = labeler
        self._config = config
        self._meaning = meaning
        self._digests = digests
        self._clock = clock

    def ask(
        self,
        question: str,
        *,
        scope: Scope = EVERYTHING,
        session: Optional[ChatSession] = None,
        on_event: Optional[Callable[[Event], None]] = None,
        cancel: Optional[CancelToken] = None,
    ) -> dict[str, Any]:
        """Answer one question. Returns the result. Raises a ChatEngineError on failure.

        `on_event` gets status, delta, done and error events while it runs.
        A cancelled question returns what was written so far, with "cancelled".
        """
        payload: dict[str, Any] = {}
        events = self._turn(question, scope, session, cancel).events()
        try:
            for event in events:
                if on_event is not None:
                    on_event(event)
                if event["type"] == "done":
                    payload = event["payload"]
        except ChatEngineError as error:
            if on_event is not None:
                on_event(error.to_event())
            raise
        finally:
            events.close()
        return payload

    def stream(
        self,
        question: str,
        *,
        scope: Scope = EVERYTHING,
        session: Optional[ChatSession] = None,
        cancel: Optional[CancelToken] = None,
    ) -> Iterator[Event]:
        """The same as `ask`, as an iterator of events.

        The last event is "done" or "error". A failure of the model or of the
        archive is not raised here: it is the "error" event. An empty question
        is refused at once, with ValueError. Closing the iterator stops the
        model call.
        """
        return _relay(self._turn(question, scope, session, cancel).events())

    def _turn(
        self,
        question: str,
        scope: Scope,
        session: Optional[ChatSession],
        cancel: Optional[CancelToken],
    ) -> QuestionTurn:
        text = str(question or "").strip()
        if not text:
            raise ValueError("the question is empty")
        note = ""
        if len(text) > MAX_QUESTION_CHARS:
            text = text[:MAX_QUESTION_CHARS]
            note = f"[The question was cut at {MAX_QUESTION_CHARS:,} characters.]"
        token = cancel if cancel is not None else CancelToken()
        config = self._config
        return QuestionTurn(
            archive=self._archive,
            llm=self._llm,
            labeler=self._labeler,
            settings=self._config.settings(),
            now=self._clock(),
            question=text,
            note=note,
            scope=scope,
            session=session if session is not None else ChatSession(),
            cancel=token,
            memory=Memory(
                self._meaning, self._digests, config.digest_tokens, config.digest_usd_limit, token,
                config.digest_parts_limit,
            ),
        )


def _relay(events: Generator[Event, None, None]) -> Generator[Event, None, None]:
    """Pass the events on; a typed failure becomes the last event."""
    try:
        yield from events
    except ChatEngineError as error:
        yield error.to_event()
    finally:
        events.close()
