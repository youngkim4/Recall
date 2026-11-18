"""Helpers that run a conversation digest with a fake model, a fake clock and invented data."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from report_engine.chat_digest import (
    ConversationDigest,
    DigestEstimate,
    conversation_digest,
    estimate_digest,
)
from report_engine.fake_llm import FakeCall, FakeLLM, default_map_reply
from report_engine.model_info import ModelInfo
from report_engine.options import ReportOptions
from report_engine.ports import SourceMessage

from .support import SMALL_OPTIONS, FakeTime, runtime
from .synth import CONVERSATION, ListSource, WordCounter, bounds, labeler, model_info

ROW_COLUMNS = "key, conversation, label, model, prompt_version, payload"


def digest(
    messages: Sequence[SourceMessage],
    *,
    llm: Any = None,
    info: ModelInfo | None = None,
    options: ReportOptions = SMALL_OPTIONS,
    time: FakeTime | None = None,
    conversation_id: str = CONVERSATION,
    **kwargs: Any,
) -> ConversationDigest:
    """One digest with the model, the options and the counter of `support.run`."""
    return conversation_digest(
        source=kwargs.pop("source", None) or ListSource(messages, conversation_id),
        conversation=conversation_id,
        llm=llm if llm is not None else FakeLLM(),
        model_info=info or model_info(),
        labeler=kwargs.pop("labeler", labeler),
        options=options,
        counter=WordCounter(),
        runtime=runtime(time),
        **kwargs,
    )


def guess(
    messages: Sequence[SourceMessage],
    cache: Any = None,
    *,
    info: ModelInfo | None = None,
    options: ReportOptions = SMALL_OPTIONS,
    conversation_id: str = CONVERSATION,
    **kwargs: Any,
) -> DigestEstimate:
    """The estimate of the digest that `digest` would build."""
    return estimate_digest(
        source=kwargs.pop("source", None) or ListSource(messages, conversation_id),
        conversation=conversation_id,
        model_info=info or model_info(),
        labeler=kwargs.pop("labeler", labeler),
        cache=cache,
        options=options,
        counter=WordCounter(),
        **kwargs,
    )


def cache_rows(path: Path) -> list[tuple[Any, ...]]:
    """Every entry of a period cache file, without the times of its use."""
    with closing(sqlite3.connect(path)) as conn:
        return list(conn.execute(f"SELECT {ROW_COLUMNS} FROM period_results ORDER BY key"))


def tagged(messages: Sequence[SourceMessage], prefix: str = "guid-") -> list[SourceMessage]:
    """The same messages with ids that no short reference looks like."""
    return [
        {**message, "id": f"{prefix}{message['id']}"}  # type: ignore[misc]
        for message in messages
    ]


def months(llm: FakeLLM) -> list[str]:
    """The month of each period that the fake model read, in order."""
    return [call.first_day[:7] for call in llm.map_calls]


def notes_seen(llm: FakeLLM) -> list[tuple[str, list[str]]]:
    """For each period that the fake model read: its month and the labels of its notes."""
    return [(call.first_day[:7], [label for label, _ in call.notes]) for call in llm.map_calls]


def echo(call: FakeCall) -> Any:
    """A reply whose summary names the notes that the prompt held, so a payload
    shows which prompt made it."""
    if call.kind != "map":
        return None
    seen = [label for label, _ in call.notes]
    return {**default_map_reply(call), "summary": f"{call.first_day[:7]} saw notes={seen}"}


def versions(path: Path) -> list[tuple[str, str]]:
    """The label and the prompt version of every entry of a cache file, oldest period first."""
    return sorted((row[2], row[4]) for row in cache_rows(path))


def fail_month(month: str, error: Exception) -> Any:
    def responder(call: FakeCall) -> Any:
        if call.kind == "map" and call.first_day.startswith(month):
            raise error
        return None

    return responder


class Talks:
    """A source that holds several conversations. It gives only the one that is asked for."""

    def __init__(self, talks: Mapping[str, Sequence[SourceMessage]]) -> None:
        self._talks = {name: tuple(messages) for name, messages in talks.items()}

    def messages(
        self, conversation: str, since: str | None, until: str | None
    ) -> Iterator[SourceMessage]:
        low, high = bounds(since, until)
        for item in self._talks.get(conversation, ()):
            if low <= item["timestamp"] <= high:
                yield dict(item)  # type: ignore[misc]
