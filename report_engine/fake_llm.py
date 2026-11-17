"""A model for tests. It opens no connection and reads no key.

With no responder it answers both steps of the engine from the prompt alone:
a map call gets a digest built from the transcript it was shown, and a reduce
call gets four short sections. A responder can return a parsed reply, a full
LLMResult, or None for the default, and can raise to make a call fail.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from report_engine.models import SECTION_KEYS
from report_engine.ports import ChatMessage, LLMResult, LLMStatus, LLMUsage
from report_engine.prompts import MAP_SCHEMA_NAME, REDUCE_SCHEMA_NAME
from report_engine.transcript import TRANSCRIPT_CLOSE, TRANSCRIPT_OPEN

_DAY = re.compile(r"^--- (\d{4}-\d{2}-\d{2}) \w+ ---$")
_LINE = re.compile(r"^(m\d+) (\d{2}:\d{2}) ([^:]+): (.*)$")
NOTES_OPEN = "<notes>"
NOTES_CLOSE = "</notes>"


@dataclass(frozen=True)
class FakeLine:
    ref: str
    day: str
    clock: str
    label: str
    text: str


@dataclass(frozen=True)
class FakeCall:
    """One request as the fake received it."""

    messages: tuple[ChatMessage, ...]
    schema: Mapping[str, Any] | None
    max_output_tokens: int
    reasoning: str | None

    @property
    def kind(self) -> Literal["map", "reduce", "other"]:
        name = (self.schema or {}).get("title")
        if name == MAP_SCHEMA_NAME:
            return "map"
        if name == REDUCE_SCHEMA_NAME:
            return "reduce"
        return "other"

    @property
    def text(self) -> str:
        return "\n".join(message["content"] for message in self.messages)

    @property
    def transcript(self) -> str:
        text = self.text
        start = text.find(TRANSCRIPT_OPEN)
        end = text.rfind(TRANSCRIPT_CLOSE)
        if start < 0 or end < start:
            return ""
        return text[start + len(TRANSCRIPT_OPEN) : end].strip("\n")

    @property
    def lines(self) -> tuple[FakeLine, ...]:
        found: list[FakeLine] = []
        day = ""
        for row in self.transcript.splitlines():
            header = _DAY.match(row)
            if header:
                day = header.group(1)
                continue
            line = _LINE.match(row)
            if line:
                found.append(
                    FakeLine(line.group(1), day, line.group(2), line.group(3), line.group(4))
                )
        return tuple(found)

    @property
    def notes(self) -> tuple[tuple[str, str], ...]:
        """Notes from earlier periods as (period label, text), oldest first."""
        user = self.messages[-1]["content"] if self.messages else ""
        start, end = user.find(NOTES_OPEN), user.find(NOTES_CLOSE)
        if start < 0 or end < start:
            return ()
        rows = (
            row.partition(": ") for row in user[start + len(NOTES_OPEN) : end].strip().splitlines()
        )
        return tuple((label, text) for label, found, text in rows if found)

    @property
    def first_day(self) -> str:
        lines = self.lines
        return lines[0].day if lines else ""

    @property
    def last_day(self) -> str:
        lines = self.lines
        return lines[-1].day if lines else ""


Responder = Callable[[FakeCall], "LLMResult | Mapping[str, Any] | None"]


def words(text: str) -> int:
    return len(text.split())


def fake_usage(call: FakeCall, reply_text: str) -> LLMUsage:
    """One token per word, in and out. Matches a word counter exactly."""
    return LLMUsage(
        input_tokens=words(call.text), cached_input_tokens=0, output_tokens=words(reply_text)
    )


def completed(
    call: FakeCall,
    parsed: Mapping[str, Any] | None,
    *,
    text: str | None = None,
    status: LLMStatus = "completed",
    usage: LLMUsage | None = None,
) -> LLMResult:
    """Wrap a reply the way a real adapter does."""
    body = text if text is not None else json.dumps(parsed, ensure_ascii=False)
    return LLMResult(
        text=body,
        parsed=parsed,
        usage=usage if usage is not None else fake_usage(call, body),
        status=status,
    )


def default_map_reply(call: FakeCall) -> dict[str, Any]:
    lines = call.lines
    if not lines:
        return {"summary": "Fake summary of an empty period.", "carry_over": "", "moments": []}
    pick = max(lines, key=lambda line: len(line.text))
    title = " ".join(pick.text.split()[:5]) or "A line"
    return {
        "summary": f"Fake summary: {len(lines)} lines from {call.first_day} to {call.last_day}.",
        "carry_over": f"Fake notes: the talk ran until {call.last_day}.",
        "moments": [
            {
                "date": pick.day,
                "title": title,
                "detail": f"{pick.label} wrote the longest line of the period.",
                "category": "milestone",
                "score": 0.8,
                "quote": pick.text,
                "refs": [pick.ref],
            }
        ],
    }


def default_reduce_reply() -> dict[str, Any]:
    return {key: f"Fake {key.replace('_', ' ')} text." for key in SECTION_KEYS}


def default_reply(call: FakeCall) -> dict[str, Any]:
    if call.kind == "map":
        return default_map_reply(call)
    if call.kind == "reduce":
        return default_reduce_reply()
    return {}


class FakeLLM:
    """Implements the LLM port. `calls` records every request in order."""

    def __init__(self, responder: Responder | None = None) -> None:
        self._responder = responder
        self.calls: list[FakeCall] = []

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: Mapping[str, Any] | None = None,
        max_output_tokens: int,
        reasoning: str | None = None,
    ) -> LLMResult:
        call = FakeCall(
            messages=tuple({"role": m["role"], "content": m["content"]} for m in messages),
            schema=schema,
            max_output_tokens=max_output_tokens,
            reasoning=reasoning,
        )
        self.calls.append(call)
        answer = self._responder(call) if self._responder is not None else None
        if isinstance(answer, LLMResult):
            return answer
        return completed(call, answer if answer is not None else default_reply(call))

    @property
    def map_calls(self) -> list[FakeCall]:
        return [call for call in self.calls if call.kind == "map"]

    @property
    def reduce_calls(self) -> list[FakeCall]:
        return [call for call in self.calls if call.kind == "reduce"]
