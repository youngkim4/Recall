"""Shared helpers for the engine tests."""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from chat_engine.engine import ChatEngine, EngineConfig
from chat_engine.fake_archive import FakeArchive
from chat_engine.fake_llm import FakeLLM, Step
from chat_engine.ports import ChatMessage, LLMRequest
from tests.chat_engine.conftest import FIXED_NOW
from tests.chat_engine.tiny import tiny_archive, tiny_labeler

RAW_HANDLES = ["7205550199", "720555", "555-0199", "+1720", "+1 720", "rk.unsaved", "example.org"]


def make_engine(
    script: Sequence[Step],
    *,
    archive: Optional[FakeArchive] = None,
    labeler: Any = None,
    **config: Any,
) -> tuple[ChatEngine, FakeLLM, FakeArchive]:
    archive = archive or tiny_archive()
    fake = FakeLLM(script)
    settings = {"model": "test-model", "max_output_tokens": 800, "timezone": "Test/Zone", **config}
    engine = ChatEngine(
        archive, fake, labeler or tiny_labeler(), EngineConfig(**settings), clock=lambda: FIXED_NOW
    )
    return engine, fake, archive


def system_text(request: LLMRequest) -> str:
    return next(m.content for m in request.messages if m.role == "system")


def last_user(request: LLMRequest) -> str:
    return next(m.content for m in reversed(request.messages) if m.role == "user")


def tool_results(request: LLMRequest) -> list[str]:
    return [m.content for m in request.messages if m.role == "tool"]


def all_text(request: LLMRequest) -> str:
    """Everything one model call receives, as one string."""
    parts: list[str] = []
    for message in request.messages:
        parts.append(message.content)
        parts.extend(call.arguments for call in message.tool_calls)
    return "\n".join(parts)


def ref_of_line(text: str, needle: str) -> str:
    """The reference at the start of the line that holds `needle`."""
    for line in text.splitlines():
        if needle in line:
            match = re.match(r"(m\d+) ", line)
            if match:
                return match.group(1)
    raise AssertionError(f"no line with {needle!r} in:\n{text}")


def alias_of_line(text: str, needle: str) -> str:
    """The conversation alias on the line that holds `needle`."""
    for line in text.splitlines():
        if needle in line:
            match = re.search(r"\b(c\d+) ·", line)
            if match:
                return match.group(1)
    raise AssertionError(f"no conversation line with {needle!r} in:\n{text}")


def events_of(kind: str, events: Sequence[dict]) -> list[dict]:
    return [event for event in events if event["type"] == kind]


def message_texts(messages: Sequence[ChatMessage]) -> list[str]:
    return [message.content for message in messages]
