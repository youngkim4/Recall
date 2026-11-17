"""A question to the chat engine, in the shapes that the app reads.

`stream` gives the events that `askStream` in app/src/lib/api.ts reads:

    {"type": "status", "text": ...}           optional "reset": true
    {"type": "delta", "text": ...}
    {"type": "done", "payload": ...}
    {"type": "error", "code": ..., "error": ...}   optional "retry_after" in seconds

The last event is `done` or `error`. When a `status` event has `reset`, the
text that the deltas gave so far is a note of the model and not the answer.

The payload of `done` has the keys of `AskResponse` (`question`, `contact`,
`terms`, `answer`, `mode`, `citations`). It also has what the engine knows
beyond that: `unverifiedQuotes`, `steps`, `cancelled`, `truncated`, `usage`.
`terms` is always empty: the engine has no list of search words.

Rules:
- A citation has the shape of a search result, with the names of the screen
  from `names.py`. The labels inside the engine are masks for the model.
- `contact` is the scope of the question. Nothing widens it.
- The model comes from the registry in `models.py`. An id that is not there,
  or that is not offered, is an error before any call.
- Each failure is one plain sentence and a `code`. What the provider said
  never goes to the person.
- There is no silent keyword answer. `local_answer` runs only when the caller
  sets `use_ai=False`, and the answer then says that no AI model was used.
- The history of the browser goes through `ChatSession.from_history`.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from archive import Archive
from archive import ArchiveError as ArchiveFault
from chat_engine import errors as engine_errors
from chat_engine.archive_adapter import ArchiveAdapter
from chat_engine.cancel import CancelToken
from chat_engine.engine import ChatEngine, EngineConfig
from chat_engine.local_answer import local_answer
from chat_engine.ports import LLM
from chat_engine.scope import EVERYTHING, Scope
from chat_engine.session import ChatSession
from recall_services import inputs, messages, models
from recall_services.models import ChatModelConfig, ModelChoiceError
from recall_services.names import Names

logger = logging.getLogger(__name__)

Event = dict[str, Any]
LLMFactory = Callable[[ChatModelConfig], LLM]

CHAT_TIMEOUT_SECONDS = 60.0
# the browser sends the last few turns; more than this is no chat that a person had
HISTORY_ROWS = 40
EMPTY_QUESTION = "Type a question first."
ASK_AGAIN = "then ask again."
_ENGINE_WORDS = {
    kind.code: kind.default_message
    for kind in (engine_errors.ChatEngineError, engine_errors.ArchiveError, *engine_errors.MODEL_ERRORS)
}


class ChatFailure(Exception):
    """A question that `ask` could not answer. The message is the sentence for the person."""

    def __init__(self, code: str, sentence: str, retry_after: float | None = None) -> None:
        super().__init__(sentence)
        self.code = code
        self.retry_after = retry_after


@dataclass(frozen=True)
class _Asked:
    """One question as the route received it."""

    question: str
    conversation: str
    scope: Scope


def default_llm(config: ChatModelConfig) -> LLM:
    """The model port of the app. The SDK is loaded here, when the first question is asked.

    The client goes to the address of the provider only: an OPENAI_BASE_URL in
    the environment can not send message text elsewhere.
    """
    from chat_engine.openai_llm import OpenAIResponsesLLM
    from report_engine.openai_llm import build_client

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise engine_errors.NoApiKeyError()
    return OpenAIResponsesLLM(
        build_client(key, timeout_s=CHAT_TIMEOUT_SECONDS),
        api_key=key,
        timeout_seconds=CHAT_TIMEOUT_SECONDS,
        reasoning_effort=config.reasoning_effort,
    )


def _wait(retry_after: float | None) -> str:
    seconds = 0 if retry_after is None else math.ceil(retry_after)
    if seconds <= 0:
        return "Wait a moment"
    return f"Wait {seconds} second{'' if seconds == 1 else 's'}"


def _sentence(code: str, said: str, config: ChatModelConfig | None, retry_after: float | None) -> str:
    """One plain sentence for a failure. Words that the engine chose for a special case are kept."""
    if said and said != _ENGINE_WORDS.get(code):
        return said
    provider = models.provider_name(config.provider) if config else models.SOME_PROVIDER
    model = config.model if config else "that was asked for"
    sentences = {
        "no_key": f"No API key is set. Add your {provider} API key in Settings, {ASK_AGAIN}",
        "key_rejected": f"{provider} rejected the API key. Check the key in Settings, {ASK_AGAIN}",
        "unknown_model": f"{provider} does not know the model {model}. Choose another level in Settings.",
        "rate_limit": f"{provider} is limiting requests. {_wait(retry_after)}, {ASK_AGAIN}",
        "offline": f"{provider} could not be reached. Check the internet connection, {ASK_AGAIN}",
        "timeout": "The model took too long to answer. Ask again.",
        "model_error": "The model call failed. Ask again.",
        "archive_error": f"The message archive could not be read. Update the archive, {ASK_AGAIN}",
    }
    return sentences.get(code, _ENGINE_WORDS[engine_errors.ChatEngineError.code])


def _failure(event: Mapping[str, Any], config: ChatModelConfig | None) -> Event:
    """An error event of the engine as the event for the app."""
    code = str(event.get("code") or engine_errors.ChatEngineError.code)
    retry_after = event.get("retry_after")
    failure: Event = {
        "type": "error",
        "code": code,
        "error": _sentence(code, str(event.get("error") or ""), config, retry_after),
    }
    return failure if retry_after is None else {**failure, "retry_after": retry_after}


def _payload(result: Mapping[str, Any], asked: _Asked, archive: Archive, names: Names) -> dict[str, Any]:
    return {
        "question": asked.question,
        "contact": asked.conversation,
        "terms": [],
        "answer": result["answer"],
        "mode": result["mode"],
        "citations": messages.payloads(archive, names, result["citations"]),
        "unverifiedQuotes": list(result["unverified_quotes"]),
        "steps": [dict(step) for step in result["steps"]],
        "cancelled": bool(result["cancelled"]),
        "truncated": bool(result["truncated"]),
        "usage": dict(result["usage"]),
    }


def _session(history: object, scope: Scope) -> ChatSession:
    rows = history if isinstance(history, (list, tuple)) else []
    return ChatSession.from_history([row for row in rows if isinstance(row, Mapping)][-HISTORY_ROWS:], scope=scope)


def _events(
    asked: _Asked,
    model: str,
    history: object,
    cancel: CancelToken | None,
    archive: Archive,
    names: Names,
    use_ai: bool,
    llm_factory: LLMFactory,
    clock: Callable[[], datetime],
) -> Iterator[Event]:
    labeler = names.chat_labeler(archive.conversations())
    adapter = ArchiveAdapter(archive)
    config: ChatModelConfig | None = None
    try:
        if not use_ai:
            found = local_answer(adapter, asked.question, asked.scope, labeler=labeler)
            yield {"type": "done", "payload": _payload(found, asked, archive, names)}
            return
        config = models.chat_config(model.strip() or models.default_id())
        settings = EngineConfig(model=config.model, max_output_tokens=config.max_output_tokens)
        engine = ChatEngine(adapter, llm_factory(config), labeler, settings, clock=clock)
        answering = engine.stream(
            asked.question, scope=asked.scope, session=_session(history, asked.scope), cancel=cancel
        )
    except engine_errors.ChatEngineError as error:
        yield _failure(error.to_event(), config)
        return
    try:
        for event in answering:
            if event["type"] == "done":
                yield {"type": "done", "payload": _payload(event["payload"], asked, archive, names)}
            elif event["type"] == "error":
                yield _failure(event, config)
            else:
                yield dict(event)
    finally:
        answering.close()  # type: ignore[attr-defined]


def stream(
    question: str,
    contact: str = "",
    model: str = "",
    history: object = None,
    cancel: CancelToken | None = None,
    *,
    archive: Archive,
    names: Names,
    use_ai: bool = True,
    llm_factory: LLMFactory | None = None,
    clock: Callable[[], datetime] = datetime.now,
) -> Iterator[Event]:
    """Answer one question. See the module docstring for the events.

    `contact` is a conversation id, or "" for the whole archive. `model` is an
    id of the registry, or "" for its default. `history` is the list of
    `{role, content}` rows that the browser keeps. `cancel` stops the model
    call and the lookups: the question then ends with `done` and `cancelled`.
    Closing the iterator stops the model call too.

    Nothing is raised for a failure of the model, of the archive or of the
    request: the last event says it.
    """
    conversation = inputs.text(contact)
    scope = Scope(conversations=(conversation,)) if conversation else EVERYTHING
    asked = _Asked(inputs.text(question).strip(), conversation, scope)
    if not asked.question:
        yield {"type": "error", "code": "empty_question", "error": EMPTY_QUESTION}
        return
    answering = _events(
        asked, inputs.text(model), history, cancel, archive, names, use_ai, llm_factory or default_llm, clock
    )
    try:
        yield from answering
    except ModelChoiceError as error:
        yield {"type": "error", "code": error.code, "error": str(error)}
    except ArchiveFault:
        logger.exception("the archive could not be read for a chat question")
        yield _failure({"code": engine_errors.ArchiveError.code}, None)
    except Exception:
        logger.exception("a chat question failed")
        yield _failure({"code": engine_errors.ChatEngineError.code}, None)
    finally:
        answering.close()


def ask(
    question: str,
    contact: str = "",
    model: str = "",
    history: object = None,
    cancel: CancelToken | None = None,
    **more: Any,
) -> dict[str, Any]:
    """The payload of one answer, with no stream. Raises ChatFailure when the last event is an error."""
    last: Event = {}
    for event in stream(question, contact, model, history, cancel, **more):
        last = event
    if last.get("type") != "done":
        raise ChatFailure(str(last.get("code") or "chat_error"), str(last.get("error") or ""), last.get("retry_after"))
    payload: dict[str, Any] = last["payload"]
    return payload
