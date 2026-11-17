"""A scripted model for tests and for the evaluation set. It never uses the network.

Each step of the script answers one model call. A step is a list of events,
or a function that gets the request and returns the events: so a script can
read a tool result and answer from it, as a model would. An exception in the
list is raised at that point of the stream.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Iterator, Sequence, Union

from .ports import Completed, LLMEvent, LLMRequest, TextDelta, ToolCall, Usage

Item = Union[LLMEvent, BaseException]
Step = Union[Sequence[Item], Callable[[LLMRequest], Iterable[Item]]]


def say(text: str, *, chunk: int = 0, commentary: bool = False) -> list[TextDelta]:
    """Text deltas for `text`, cut into pieces of `chunk` characters."""
    if chunk <= 0:
        return [TextDelta(text, commentary)]
    return [TextDelta(text[i:i + chunk], commentary) for i in range(0, len(text), chunk)]


def call(tool: str, call_id: str = "", /, **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id or f"call_{tool}", name=tool, arguments=json.dumps(arguments))


def done(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
    stop_reason: str = "stop",
) -> Completed:
    usage = Usage(input_tokens, output_tokens, cached_input_tokens, reasoning_tokens)
    return Completed(usage=usage, stop_reason=stop_reason)


class FakeLLM:
    def __init__(self, script: Sequence[Step]) -> None:
        self._script = list(script)
        self.requests: list[LLMRequest] = []
        self.closed_early = 0
        self.finished = 0

    def stream(self, request: LLMRequest) -> Iterator[LLMEvent]:
        index = len(self.requests)
        self.requests.append(request)
        if index >= len(self._script):
            raise AssertionError(f"the script has {len(self._script)} steps; call {index + 1} was made")
        step = self._script[index]
        items = list(step(request) if callable(step) else step)
        return self._play(items)

    def _play(self, items: list[Item]) -> Iterator[LLMEvent]:
        try:
            for item in items:
                if isinstance(item, BaseException):
                    raise item
                yield item
            if not any(isinstance(item, Completed) for item in items):
                yield Completed()
        except GeneratorExit:
            self.closed_early += 1
            raise
        self.finished += 1
