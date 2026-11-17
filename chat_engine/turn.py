"""One question, from the seeded lookups to the final answer.

The turn is a generator of events. It raises a typed error when the model or
the archive fails. It never switches to another way of answering by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Callable, Generator, Iterator, Optional

from .cancel import CancelToken
from .citations import CitationStream, stored_form
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
from .render import Limits, ResultWriter, add_block, time_key
from .result import build_payload
from .scope import Scope
from .seed import plan_seed
from .session import ChatSession, Turn
from .tool_context import ToolOutcome
from .tools import ToolBox

Event = dict[str, Any]
MIN_RESULT_TOKENS = 300  # with less left, a lookup has no room for an answer worth a step
_CARRIED_TOKENS = 1200
_REPEATED = "Same lookup as before: its result is above. Try a different lookup, or answer."
_SKIPPED = "Lookup skipped: the limit on lookups for this question is reached."
_LOOKING = "Looking at more messages"
_LEFT_OUT = "[This earlier answer is left out: it was written for another scope than this question has.]"
_TOO_MANY = (
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
        self._call_ids: set[str] = set()
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

    def events(self) -> Generator[Event, None, None]:
        """The events of the question.

        A question that does not finish (cancelled, failed, or closed by the
        caller) leaves the session as it was: the messages and the aliases
        that it added are forgotten, so a later answer cannot cite them.
        """
        session = self._session
        question = session.begin()
        keep = False
        try:
            reply = yield from self._answer_steps()
            done = self._done(reply)
            keep = not reply.cancelled
            session.finish(question, keep)  # before the last event goes out: the caller can leave the stream open
            yield done
        finally:
            session.finish(question, keep)

    def _answer_steps(self) -> Generator[Event, None, Reply]:
        """The lookups and the model calls. After a cancel, nothing more is read from the archive."""
        if self._cancel.cancelled:
            return Reply(cancelled=True)
        yield status("Reading your archive")
        if self._cancel.cancelled:
            return Reply(cancelled=True)
        summary = self._box.guard(self._archive.summary)
        lookups = yield from self._seed()
        if self._cancel.cancelled:
            return Reply(cancelled=True)
        self._transcript = self._box.guard(lambda: self._opening(summary, lookups))
        return (yield from self._loop())

    def _seed(self) -> Generator[Event, None, list[tuple[str, str]]]:
        lookups: list[tuple[str, str]] = []
        for seed in plan_seed(self._box, self._question):
            yield status(seed.prepared.status)
            if self._cancel.cancelled:
                break
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
                return replace(reply, forced=forced)
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
        renumber = CitationStream(self._citable)
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
        unique, renamed = self._unique(calls)
        return Reply(
            text="".join(raw),
            shown="".join(shown),
            calls=unique,
            order=renumber.order,
            unknown=renumber.unknown,
            stop_reason=end.stop_reason,
            provider_items=() if renamed else tuple(end.provider_items),
            streamed=bool(shown),
            cancelled=self._cancel.cancelled,
        )

    def _citable(self, ref: str) -> bool:
        """A reference the answer may cite: a shown message inside the scope of this question."""
        message = self._session.registry.get(ref)
        return message is not None and self._box.context.allows(message)

    def _unique(self, calls: list[ToolCall]) -> tuple[tuple[ToolCall, ...], bool]:
        """Every call of the question with an id of its own. True when an id had to change.

        A changed id no longer matches the items of the provider, so the
        caller drops those and the call goes back in its plain form.
        """
        unique: list[ToolCall] = []
        renamed = False
        for call in calls:
            call_id = call.id
            number = len(self._call_ids)
            while not call_id or call_id in self._call_ids:
                number += 1
                call_id = f"call_{number}"
            self._call_ids.add(call_id)
            renamed = renamed or call_id != call.id
            unique.append(call if call_id == call.id else replace(call, id=call_id))
        return tuple(unique), renamed

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
        limit = self._settings.max_calls_per_step
        calls = reply.calls[:limit]
        extra = len(reply.calls) - len(calls)
        # calls past the cap leave the transcript: so do the items of the provider, which name every call
        items = () if extra else reply.provider_items
        self._transcript.append(ChatMessage("assistant", reply.text, tool_calls=calls, provider_items=items))
        if reply.streamed:
            yield status(_LOOKING, reset=True)  # the text shown so far was not the answer
        yield from self._run_calls(calls)
        if extra and self._transcript[-1].role == "tool":
            note = _TOO_MANY.format(count=extra, limit=limit)
            last = self._transcript[-1]
            self._transcript[-1] = replace(last, content=f"{last.content}\n{note}")
            self._left -= self._settings.count_tokens(note)

    def _run_calls(self, calls: tuple[ToolCall, ...]) -> Generator[Event, None, None]:
        for call in calls:
            if self._cancel.cancelled:
                return
            key = (call.name, call.arguments)
            if key in self._ran:
                self._skip(call, _REPEATED)
                continue
            if self._left < MIN_RESULT_TOKENS:
                self._skip(call, _SKIPPED)
                continue
            prepared = self._box.prepare(call.name, call.arguments)
            if prepared.status:
                yield status(prepared.status)
            if self._cancel.cancelled:
                return
            outcome = prepared.run(self._allowance())
            self._ran.add(key)
            self._totals["tool_calls"] += 1
            self._record(call.name, outcome, seeded=False)
            self._answer(call, outcome.text)

    def _answer(self, call: ToolCall, text: str) -> None:
        self._transcript.append(ChatMessage("tool", text, tool_call_id=call.id))

    def _skip(self, call: ToolCall, text: str) -> None:
        """Answer a call that is not run. The note counts as tool output too."""
        self._answer(call, text)
        self._left -= self._settings.count_tokens(text)

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
            messages.append(ChatMessage("assistant", self._earlier_answer(turn)))
        block = question_block(self._question, lookups, self._carried(turns), self._note)
        return [*messages, ChatMessage("user", block)]

    def _earlier_answer(self, turn: Turn) -> str:
        """An earlier answer, when everything it could read is inside the scope of this question."""
        return turn.answer if turn.scope.within(self._scope) else _LEFT_OUT

    def _recent_turns(self) -> list[Turn]:
        """The newest turns that fit the history budget. With a budget, the last one always fits."""
        if self._settings.history_tokens <= 0:
            return []
        count = self._settings.count_tokens
        kept: list[Turn] = []
        used = 0
        for turn in reversed(self._session.turns):
            cost = count(turn.question) + count(self._earlier_answer(turn))
            if kept and used + cost > self._settings.history_tokens:
                break
            kept.append(turn)
            used += cost
        return list(reversed(kept))

    def _carried(self, turns: list[Turn]) -> str:
        """Messages the earlier answers cited, still inside the scope. The newest turn comes first.

        A block that does not fit is cut shorter. A block that still does not
        fit is left out, and the next one is tried.
        """
        context = self._box.context
        registry = self._session.registry
        writer = ResultWriter(_CARRIED_TOKENS, self._settings.count_tokens)
        seen: set[str] = set()
        for turn in reversed(turns):
            by_conversation: dict[str, list[Message]] = {}
            for ref in turn.cited:
                message = registry.get(ref)
                if message is None or ref in seen or not context.allows(message):
                    continue
                seen.add(ref)
                by_conversation.setdefault(str(message["conversation"]), []).append(message)
            for messages in by_conversation.values():
                add_block(
                    writer, sorted(messages, key=time_key), book=context.book, registry=registry,
                    limits=self._limits, marks={},
                )
        return self._labeler.redact(writer.text().strip())

    # --- the result ---

    def _done(self, reply: Reply) -> Event:
        if not reply.cancelled and not reply.shown:
            raise _no_answer(reply)
        context = self._box.context
        payload = build_payload(
            answer=reply.shown,
            raw_answer=reply.text,
            order=reply.order,
            unknown=reply.unknown,
            registry=self._session.registry,
            book=context.book,
            redact=self._labeler.redact,
            allowed=context.allows,
            text_cap=self._settings.max_message_chars,
            steps=self._steps,
            usage=self._totals,
            cancelled=reply.cancelled,
            truncated=reply.stop_reason == "length",
        )
        if not reply.cancelled:
            cited = frozenset(reply.order)
            answer = stored_form(reply.text, cited.__contains__)
            self._session.add_turn(Turn(self._question, answer, reply.order, self._scope))
        return {"type": "done", "payload": payload}


def _no_answer(reply: Reply) -> ModelError:
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
