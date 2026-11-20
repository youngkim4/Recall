"""Watch the model port of one question: the calls, their usage, their tool calls and their text.

`chat.stream` gives the usage of the whole question, not of each call. The
long-context price rule is judged on each call, so the cost is priced call
by call here. The text of the last call is the raw answer of the model,
before the engine renumbers its references: it gives the references that
named no message (`invalid_references`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from chat_engine.ports import LLM, Completed, LLMEvent, LLMRequest, TextDelta, ToolCall, Usage
from recall_services import models
from recall_services.models import ChatModelConfig
from report_engine.ports import LLMUsage
from report_engine.pricing import call_cost_usd

_MARKER_RE = re.compile(r"\[([^\[\]\n]{1,64})\]")
_REF_RE = re.compile(r"\bm(\d+)(?:\s*[-–—]\s*m?(\d+))?", re.IGNORECASE)
_MAX_RANGE = 20


@dataclass(frozen=True)
class CallRecord:
    """One model call."""

    usage: Usage
    tool_calls: int
    text: str


def price(model: str, usage: Usage) -> float:
    """USD of one call of an offered model, from models.json."""
    tokens = LLMUsage(usage.input_tokens, usage.cached_input_tokens, usage.output_tokens)
    return call_cost_usd(models.model_info(model), tokens)


def references(text: str) -> tuple[str, ...]:
    """The message references ("m3") in the markers of a raw answer, in order, once each."""
    found: list[str] = []
    for marker in _MARKER_RE.finditer(text or ""):
        for piece in _REF_RE.finditer(marker.group(1)):
            start = int(piece.group(1))
            end = int(piece.group(2)) if piece.group(2) else start
            numbers = range(start, end + 1) if 0 < end - start <= _MAX_RANGE else (start, end)
            found.extend(f"m{number}" for number in numbers)
    return tuple(dict.fromkeys(found))


class RecordingLLM:
    """A model port that passes every event on and keeps a record of each call."""

    def __init__(self, inner: LLM) -> None:
        self._inner = inner
        self._calls: list[CallRecord] = []

    @property
    def calls(self) -> tuple[CallRecord, ...]:
        return tuple(self._calls)

    def stream(self, request: LLMRequest) -> Iterator[LLMEvent]:
        return self._watch(self._inner.stream(request))

    def _watch(self, events: Iterator[LLMEvent]) -> Iterator[LLMEvent]:
        parts: list[str] = []
        tools = 0
        usage = Usage()
        try:
            for event in events:
                if isinstance(event, TextDelta) and not event.commentary:
                    parts.append(event.text)
                elif isinstance(event, ToolCall):
                    tools += 1
                elif isinstance(event, Completed):
                    usage = event.usage
                yield event
        finally:
            self._calls.append(CallRecord(usage, tools, "".join(parts)))


class Recorder:
    """The model factory of one question. Each port it makes is recorded."""

    def __init__(self, factory: Callable[[ChatModelConfig], LLM]) -> None:
        self._factory = factory
        self._ports: list[RecordingLLM] = []

    def __call__(self, config: ChatModelConfig) -> LLM:
        port = RecordingLLM(self._factory(config))
        self._ports.append(port)
        return port

    @property
    def calls(self) -> tuple[CallRecord, ...]:
        return tuple(call for port in self._ports for call in port.calls)
