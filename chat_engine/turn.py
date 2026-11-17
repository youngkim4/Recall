"""One question, from the seeded lookups to the final answer.

The turn is a generator of events. It raises a typed error when the model or
the archive fails. It never switches to another way of answering by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Callable, Generator, Iterator, Optional

from .cancel import CancelToken
from .citations import CitationStream
from .errors import ChatEngineError, ModelError
from .ports import (
    LLM,
    ChatArchive,
    ChatMessage,
    Completed,
    Labeler,
    LLMRequest,
    Message,
    TextDelta,
    ToolCall,
    Usage,
)
from .prompt import LIMIT_NOTE, question_block, scope_line, system_prompt, timezone_label
from .render import Limits, ResultWriter, render_block
from .result import build_payload
from .scope import Scope
from .seed import plan_seed
from .session import ChatSession, Turn
from .tool_context import ToolOutcome
from .tools import ToolBox

Event = dict[str, Any]
MIN_RESULT_TOKENS = 100
_CARRIED_TOKENS = 1200
_REPEATED = "Same lookup as before: its result is above. Try a different lookup, or answer."
_SKIPPED = "Lookup skipped: the limit on lookups for this question is reached."


@dataclass(frozen=True)
class TurnSettings:
    model: str
    max_output_tokens: int
    max_steps: int
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


def status(text: str, reset: bool = False) -> Event:
    event: Event = {"type": "status", "text": text}
    if reset:
        event["reset"] = True
    return event


class QuestionTurn:
    def __init__(
        self,
        *,
        archive: ChatArchive,
        llm: LLM,
        labeler: Labeler,
        settings: TurnSettings,
        now: datetime,
        question: str,
        note: str,
        scope: Scope,
        session: ChatSession,
        cancel: CancelToken,
    ) -> None:
        self._archive = archive
        self._llm = llm
        self._labeler = labeler
        self._settings = settings
        self._now = now
        self._question = question
        self._note = note
        self._scope = scope
        self._session = session
        self._cancel = cancel
        limits = Limits(settings.max_message_chars, settings.result_tokens, settings.count_tokens)
        self._box = ToolBox(archive, labeler, session.registry, session.aliases, scope, limits)
        self._limits = limits
        self._transcript: list[ChatMessage] = []
        self._steps: list[Event] = []
        self._ran: set[tuple[str, str]] = set()
        self._left = settings.total_tool_tokens
        self._totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
            "model_calls": 0,
            "tool_calls": 0,
        }

    # --- the turn ---

    def events(self) -> Iterator[Event]:
        if self._cancel.cancelled:
            yield self._done(Reply(cancelled=True))
            return
        yield status("Reading your archive")
        summary = self._box.guard(self._archive.summary)
        lookups = yield from self._seed()
        if self._cancel.cancelled:
            yield self._done(Reply(cancelled=True))
            return
        self._transcript = self._box.guard(lambda: self._opening(summary, lookups))
        reply = yield from self._loop()
        yield self._done(reply)

    def _seed(self) -> Generator[Event, None, list[tuple[str, str]]]:
        lookups: list[tuple[str, str]] = []
        for seed in plan_seed(self._box, self._question):
            if self._cancel.cancelled:
                break
            yield status(seed.prepared.status)
            outcome = seed.prepared.run(self._allowance())
            self._record(seed.tool, outcome, seeded=True)
            lookups.append((seed.label, outcome.text))
        return lookups

    def _loop(self) -> Generator[Event, None, Reply]:
        for round_number in range(self._settings.max_steps + 1):
            forced = round_number >= self._settings.max_steps or self._left < MIN_RESULT_TOKENS
            if forced:
                self._transcript.append(ChatMessage("user", LIMIT_NOTE))
            reply = yield from self._call_model(allow_tools=not forced)
            if reply.cancelled or forced or not reply.calls:
                return reply
            yield from self._run_tools(reply)
            if self._cancel.cancelled:
                return Reply(cancelled=True)
        raise ModelError(detail="the tool loop ended without a forced answer")

    # --- the model ---

    def _call_model(self, allow_tools: bool) -> Generator[Event, None, Reply]:
        request = LLMRequest(
            model=self._settings.model,
            messages=tuple(self._outbound()),
            tools=self._box.specs,
            max_output_tokens=self._settings.max_output_tokens,
            allow_tools=allow_tools,
            cancel=self._cancel,
        )
        renumber = CitationStream(lambda ref: ref in self._session.registry)
        raw: list[str] = []
        shown: list[str] = []
        calls: list[ToolCall] = []
        end = Completed()
        stream: Optional[Iterator[Any]] = None
        try:
            stream = iter(self._llm.stream(request))
            for event in stream:
                if self._cancel.cancelled:
                    break
                if isinstance(event, TextDelta):
                    if event.commentary:
                        continue
                    raw.append(event.text)
                    piece = renumber.feed(event.text)
                    if piece:
                        shown.append(piece)
                        yield {"type": "delta", "text": piece}
                elif isinstance(event, ToolCall):
                    calls.append(event)
                elif isinstance(event, Completed):
                    end = event
        except ChatEngineError:
            raise
        except Exception as error:
            raise ModelError(detail=f"{type(error).__name__}: {error}") from error
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        self._count_usage(end.usage)
        tail = renumber.finish()
        if tail:
            shown.append(tail)
            yield {"type": "delta", "text": tail}
        return Reply(
            text="".join(raw),
            shown="".join(shown),
            calls=tuple(calls),
            order=renumber.order,
            unknown=renumber.unknown,
            stop_reason=end.stop_reason,
            provider_items=tuple(end.provider_items),
            streamed=bool(shown),
            cancelled=self._cancel.cancelled,
        )

    def _outbound(self) -> list[ChatMessage]:
        """Every message the model gets passes here: no raw handle goes out."""
        redact = self._labeler.redact
        return [replace(message, content=redact(message.content)) for message in self._transcript]

    def _count_usage(self, usage: Usage) -> None:
        self._totals["input_tokens"] += usage.input_tokens
        self._totals["output_tokens"] += usage.output_tokens
        self._totals["cached_input_tokens"] += usage.cached_input_tokens
        self._totals["reasoning_tokens"] += usage.reasoning_tokens
        self._totals["model_calls"] += 1

    # --- the tools ---

    def _run_tools(self, reply: Reply) -> Generator[Event, None, None]:
        self._transcript.append(
            ChatMessage("assistant", reply.text, tool_calls=reply.calls, provider_items=reply.provider_items)
        )
        reset = reply.streamed
        for call in reply.calls:
            if self._cancel.cancelled:
                return
            key = (call.name, call.arguments)
            if key in self._ran:
                self._answer(call, _REPEATED)
                continue
            if self._left < MIN_RESULT_TOKENS:
                self._answer(call, _SKIPPED)
                continue
            prepared = self._box.prepare(call.name, call.arguments)
            if prepared.status:
                yield status(prepared.status, reset=reset)
                reset = False
            if self._cancel.cancelled:
                return
            outcome = prepared.run(self._allowance())
            self._ran.add(key)
            self._totals["tool_calls"] += 1
            self._record(call.name, outcome, seeded=False)
            self._answer(call, outcome.text)

    def _answer(self, call: ToolCall, text: str) -> None:
        self._transcript.append(ChatMessage("tool", text, tool_call_id=call.id))

    def _allowance(self) -> int:
        return max(0, min(self._settings.result_tokens, self._left))

    def _record(self, tool: str, outcome: ToolOutcome, *, seeded: bool) -> None:
        self._left -= self._settings.count_tokens(outcome.text)
        self._steps.append({"tool": tool, "text": outcome.summary, "seeded": seeded})

    # --- the prompt ---

    def _opening(self, summary: Any, lookups: list[tuple[str, str]]) -> list[ChatMessage]:
        context = self._box.context
        where = context.where(self._scope.conversations) if self._scope.conversations else None
        system = system_prompt(
            now=self._now,
            timezone=timezone_label(self._now, self._settings.timezone),
            summary=summary,
            scope=scope_line(where, self._scope.since, self._scope.until),
        )
        turns = self._recent_turns()
        messages = [ChatMessage("system", system)]
        for turn in turns:
            messages.append(ChatMessage("user", turn.question))
            messages.append(ChatMessage("assistant", turn.answer))
        block = question_block(self._question, lookups, self._carried(turns), self._note)
        return [*messages, ChatMessage("user", block)]

    def _recent_turns(self) -> list[Turn]:
        """The newest turns that fit the history budget. The last one always fits."""
        count = self._settings.count_tokens
        kept: list[Turn] = []
        used = 0
        for turn in reversed(self._session.turns):
            cost = count(turn.question) + count(turn.answer)
            if kept and used + cost > self._settings.history_tokens:
                break
            kept.append(turn)
            used += cost
        return list(reversed(kept))

    def _carried(self, turns: list[Turn]) -> str:
        """Messages the earlier answers cited, still inside the scope."""
        context = self._box.context
        refs = dict.fromkeys(ref for turn in turns for ref in turn.cited)
        found = [self._session.registry.get(ref) for ref in refs]
        by_conversation: dict[str, list[Message]] = {}
        for message in found:
            if message is not None and context.allows(message):
                by_conversation.setdefault(str(message["conversation"]), []).append(message)
        writer = ResultWriter(_CARRIED_TOKENS, self._settings.count_tokens)
        for messages in by_conversation.values():
            ordered = sorted(messages, key=lambda m: (str(m.get("timestamp") or ""), m.get("sent_at") or 0))
            block = render_block(
                ordered, book=context.book, registry=self._session.registry, limits=self._limits, marks={}
            )
            if not writer.add("\n".join(block)):
                break
        return self._labeler.redact(writer.text())

    # --- the result ---

    def _done(self, reply: Reply) -> Event:
        if not reply.cancelled and not reply.shown:
            raise ModelError("The model gave no answer.", detail=f"stop_reason={reply.stop_reason}")
        context = self._box.context
        payload = build_payload(
            answer=reply.shown,
            raw_answer=reply.text,
            order=reply.order,
            unknown=reply.unknown,
            registry=self._session.registry,
            book=context.book,
            questions=[self._question, *self._session.questions()],
            redact=self._labeler.redact,
            steps=self._steps,
            usage=self._totals,
            cancelled=reply.cancelled,
            truncated=reply.stop_reason == "length",
        )
        if not reply.cancelled:
            self._session.add_turn(Turn(self._question, reply.text.strip(), reply.order))
        return {"type": "done", "payload": payload}
