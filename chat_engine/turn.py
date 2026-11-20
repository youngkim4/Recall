"""One question, from the seeded lookups to the final answer.

The turn is a generator of events. It raises a typed error when the model or
the archive fails. It never switches to another way of answering by itself.
"""

from __future__ import annotations

import hashlib
import json

from dataclasses import replace
from datetime import datetime
from typing import Any, Generator, Iterator, Optional, Sequence

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
from . import condense as condensing
from .prompt import EARLIER_LIMITED, LIMIT_NOTE, STALL_NOTE, question_block, scope_line, system_prompt, timezone_label
from .question_time import time_line
from .render import Limits, ResultWriter, add_block, time_key
from .result import build_payload
from .scope import Scope
from .seed import SeedCall, plan_seed
from .session import ChatSession, Turn
from .tool_context import Memory, ToolOutcome
from .tools import ToolBox
from .turn_fetch import run_beside
from .turn_parts import TOO_MANY, Event, Reply, TurnSettings, no_answer, status, unique_calls

_CARRIED_TOKENS = 6000
_REPEATED = "Same lookup as before: its result is above. Try a different lookup, or answer."
_LOOKING = "Looking at more messages"
_THINKING = "Thinking it over"
_CONDENSING = "Condensing what I have read"
_LEFT_OUT = "[This earlier answer is left out: it was written for another scope than this question has.]"


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
        memory: Optional[Memory] = None,
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
        self._box = ToolBox(archive, labeler, session.registry, session.aliases, scope, limits, memory)
        self._limits = limits
        self._transcript: list[ChatMessage] = []
        self._steps: list[Event] = []
        self._ran: set[tuple[str, str]] = set()
        self._call_ids: set[str] = set()
        self._seen_results: set[str] = set()  # the text of every lookup result: a result seen before is nothing new
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
        plan = plan_seed(self._box, self._question, summary, self._now)
        lookups = yield from self._seed(plan.calls)
        if self._cancel.cancelled:
            return Reply(cancelled=True)
        when = time_line(plan.period)
        self._transcript = self._box.guard(lambda: self._opening(summary, lookups, when))
        yield status(_THINKING)  # the line of the last lookup does not stay while the model works
        return (yield from self._loop())

    def _seed(self, calls: Sequence[SeedCall]) -> Generator[Event, None, list[tuple[str, str]]]:
        lookups: list[tuple[str, str]] = []
        for seed in calls:
            yield status(seed.prepared.status)
            if self._cancel.cancelled:
                break
            if seed.prepared.fetch is not None:  # slow work, such as a ranking: its events reach the stream
                yield from run_beside(seed.prepared.fetch, self._cancel)
            outcome = seed.prepared.run(self._allowance())
            self._record(seed.tool, outcome, seeded=True)
            self._is_new(outcome.text)  # a later lookup that gives the same result finds nothing new
            lookups.append((seed.label, outcome.text))
        return lookups

    def _loop(self) -> Generator[Event, None, Reply]:
        """Rounds of lookups until the model answers.

        The model answers by itself when it has what it needs. It must answer
        when `stall_rounds` rounds in a row found nothing new, or at the safety
        cap `max_steps`; only the cap marks the answer limited.
        """
        stale = 0
        for round_number in range(self._settings.max_steps + 1):
            capped = round_number >= self._settings.max_steps
            forced = capped or stale >= self._settings.stall_rounds
            if forced:
                self._transcript.append(ChatMessage("user", LIMIT_NOTE if capped else STALL_NOTE))
            else:
                yield from self._condense_if_needed()
                if self._cancel.cancelled:
                    return Reply(cancelled=True)
            reply = yield from self._call_model(allow_tools=not forced)
            if reply.cancelled or forced or not reply.calls:
                return replace(reply, forced=forced, capped=capped)
            fresh = yield from self._run_tools(reply)
            stale = 0 if fresh else stale + 1
            if self._cancel.cancelled:
                return Reply(cancelled=True)
        raise ModelError(detail="the tool loop ended without a forced answer")

    def _condense_if_needed(self) -> Generator[Event, None, None]:
        """Condense the results before the newest round when the held results pass the mark."""
        count = self._settings.count_tokens
        if not condensing.needs_condensing(self._transcript, self._settings.held_tool_tokens, count):
            return
        yield status(_CONDENSING)
        outbound = self._outbound()
        asked = condensing.request(
            self._settings.model, self._question, outbound, self._settings.max_output_tokens, self._cancel
        )
        try:
            done = condensing.condense(self._llm, asked, outbound)
        except ChatEngineError:
            raise
        except Exception as error:
            raise ModelError(detail=f"{type(error).__name__}: {error}") from error
        self._count_usage(done.usage)
        kept = condensing.older_results(self._transcript)
        self._transcript = [
            replace(message, content=done.transcript[place].content) if place in kept else message
            for place, message in enumerate(self._transcript)
        ]

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
        unique, renamed = unique_calls(calls, self._call_ids)
        self._call_ids.update(call.id for call in unique)
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

    def _run_tools(self, reply: Reply) -> Generator[Event, None, bool]:
        """Run the calls of a reply. True when a lookup found something new: a new message or a new result."""
        limit = self._settings.max_calls_per_step
        calls = reply.calls[:limit]
        extra = len(reply.calls) - len(calls)
        # calls past the cap leave the transcript: so do the items of the provider, which name every call
        items = () if extra else reply.provider_items
        self._transcript.append(ChatMessage("assistant", reply.text, tool_calls=calls, provider_items=items))
        if reply.streamed:
            yield status(_LOOKING, reset=True)  # the text shown so far was not the answer
        known = self._session.registry.checkpoint()
        fresh = yield from self._run_calls(calls)
        if extra and self._transcript[-1].role == "tool":
            note = TOO_MANY.format(count=extra, limit=limit)
            last = self._transcript[-1]
            self._transcript[-1] = replace(last, content=f"{last.content}\n{note}")
        return fresh or self._session.registry.checkpoint() > known

    def _run_calls(self, calls: tuple[ToolCall, ...]) -> Generator[Event, None, bool]:
        fresh = False
        for call in calls:
            if self._cancel.cancelled:
                return fresh
            key = (call.name, same_lookup_key(call.arguments))
            if key in self._ran:
                self._answer(call, _REPEATED)
                continue
            prepared = self._box.prepare(call.name, call.arguments)
            if prepared.repeated:
                self._answer(call, _REPEATED)
                continue
            if prepared.status:
                yield status(prepared.status)
            if self._cancel.cancelled:
                return fresh
            if prepared.fetch is not None:  # slow work: its progress reaches the stream while it runs
                yield from run_beside(prepared.fetch, self._cancel)
            own = prepared.own_budget  # a lookup with a budget of its own, such as a digest
            outcome = prepared.run(self._allowance() if own is None else own)
            if not prepared.answers_repeats:
                self._ran.add(key)
            self._totals["tool_calls"] += 1
            self._record(call.name, outcome, seeded=False)
            fresh = self._is_new(outcome.text) or fresh
            self._answer(call, outcome.text)
        return fresh

    def _is_new(self, text: str) -> bool:
        digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
        if digest in self._seen_results:
            return False
        self._seen_results.add(digest)
        return True

    def _answer(self, call: ToolCall, text: str) -> None:
        self._transcript.append(ChatMessage("tool", text, tool_call_id=call.id))

    def _allowance(self) -> int:
        return self._settings.result_tokens

    def _record(self, tool: str, outcome: ToolOutcome, *, seeded: bool) -> None:
        self._steps.append({"tool": tool, "text": outcome.summary, "seeded": seeded})

    # --- the prompt ---

    def _opening(self, summary: Any, lookups: list[tuple[str, str]], when: str = "") -> list[ChatMessage]:
        """The system prompt, the earlier turns, and the question block. `when` names the days of the question."""
        context = self._box.context
        where = context.where(self._scope.conversations) if self._scope.conversations else None
        system = system_prompt(
            now=self._now,
            timezone=timezone_label(self._now, self._settings.timezone),
            summary=summary,
            scope=scope_line(where, self._scope.since, self._scope.until),
            offered=self._box.offered,
        )
        turns = self._recent_turns()
        messages = [ChatMessage("system", system)]
        for turn in turns:
            messages.append(ChatMessage("user", turn.question))
            messages.append(ChatMessage("assistant", self._earlier_answer(turn)))
        block = question_block(self._question, lookups, self._carried(turns), self._note, when)
        return [*messages, ChatMessage("user", block)]

    def _earlier_answer(self, turn: Turn) -> str:
        """An earlier answer, when everything it could read is inside the scope of this question."""
        if not turn.scope.within(self._scope):
            return _LEFT_OUT
        return f"{turn.answer}\n{EARLIER_LIMITED}" if turn.limited else turn.answer

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
            raise no_answer(reply)
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
            usage=self._box.usage(self._totals),
            cancelled=reply.cancelled,
            truncated=reply.stop_reason == "length",
        )
        # nothing is added with no digest source
        payload = {**payload, **self._box.memory_payload(), "limited": reply.capped}
        if not reply.cancelled:
            cited = frozenset(reply.order)
            answer = stored_form(reply.text, cited.__contains__)
            self._session.add_turn(Turn(self._question, answer, reply.order, self._scope, reply.capped))
        return {"type": "done", "payload": payload}


def same_lookup_key(arguments: str) -> str:
    """The arguments of a call, so that a near-same call (case, spaces, key order) counts as a repeat."""
    try:
        value = json.loads(arguments)
    except (TypeError, ValueError):
        return arguments
    return json.dumps(_folded(value), sort_keys=True, ensure_ascii=False)


def _folded(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, list):
        return [_folded(item) for item in value]
    if isinstance(value, dict):
        return {key: _folded(item) for key, item in value.items()}
    return value
