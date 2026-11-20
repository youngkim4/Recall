"""The model port over the OpenAI Responses API.

Function calling, streaming, `store=False`, an explicit timeout, a hard output
cap, and bounded retries that stop once the first token is out. The model id
and its limits come from the request: none is written here.

Give `reasoning_effort` for a model that reasons. The adapter then asks for
the encrypted reasoning and hands it back on the next call of the question,
because nothing is stored at the provider. A model that does not reason
refuses that request, so without `reasoning_effort` it is not made.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterator, Optional

import openai

from .cancel import CancelToken
from .errors import ModelError, NoApiKeyError
from .openai_errors import Failure, map_error, stream_failure
from .ports import ChatMessage, Completed, LLMEvent, LLMRequest, TextDelta, ToolCall, Usage

logger = logging.getLogger(__name__)

_MAX_WAIT_SECONDS = 20.0
_BACKOFF_CAP_SECONDS = 8.0
_STOP_REASONS = {"max_output_tokens": "length", "content_filter": "filtered"}


def _plain(value: Any) -> Any:
    """SDK objects as plain data, for items that go back to the provider."""
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {key: _plain(item) for key, item in vars(value).items() if item is not None}
    return value


def _usage(response: Any) -> Usage:
    usage = getattr(response, "usage", None)
    return Usage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cached_input_tokens=int(getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0),
        reasoning_tokens=int(getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0),
    )


def _carried(items: list[Any], whole: bool = True) -> tuple[dict[str, Any], ...]:
    """Output items to hand back on the next call of the same question.

    Reasoning goes back only as encrypted content, because nothing is stored.
    Without it, item ids are left out: an id without its reasoning is refused.
    A response that was cut (`whole=False`) can miss the item its reasoning
    belongs to, so its reasoning does not go back either.
    """
    reasoning = [item for item in items if getattr(item, "type", "") == "reasoning"]
    linked = whole and bool(reasoning) and all(getattr(item, "encrypted_content", None) for item in reasoning)
    carried: list[dict[str, Any]] = []
    for item in items:
        kind = getattr(item, "type", "")
        if kind == "reasoning" and linked:
            carried.append(
                {
                    "type": "reasoning",
                    "id": item.id,
                    "summary": _plain(getattr(item, "summary", None) or []),
                    "encrypted_content": item.encrypted_content,
                }
            )
        elif kind == "message":
            parts = getattr(item, "content", None) or []
            text = "".join(getattr(p, "text", "") or "" for p in parts if getattr(p, "type", "") == "output_text")
            if text:
                entry: dict[str, Any] = {"role": "assistant", "content": text}
                if getattr(item, "phase", None):
                    entry["phase"] = item.phase
                carried.append(entry)
        elif kind == "function_call":
            call = {
                "type": "function_call",
                "call_id": item.call_id,
                "name": item.name,
                "arguments": getattr(item, "arguments", None) or "{}",
            }
            if linked and getattr(item, "id", None):
                call["id"] = item.id
            carried.append(call)
    return tuple(carried)


def _input_items(messages: tuple[ChatMessage, ...]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "user":
            items.append({"role": "user", "content": message.content})
        elif message.role == "tool":
            items.append(
                {"type": "function_call_output", "call_id": message.tool_call_id, "output": message.content}
            )
        elif message.role == "assistant" and message.provider_items:
            items.extend(dict(item) for item in message.provider_items)
        elif message.role == "assistant":
            if message.content:
                items.append({"role": "assistant", "content": message.content})
            items.extend(
                {"type": "function_call", "call_id": call.id, "name": call.name, "arguments": call.arguments}
                for call in message.tool_calls
            )
    return items


OFFICIAL_BASE_URL = "https://api.openai.com/v1"


class OpenAIResponsesLLM:
    def __init__(
        self,
        client: Any = None,
        *,
        api_key: Optional[str] = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        carry_reasoning: Optional[bool] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        """`carry_reasoning=None` carries the reasoning when `reasoning_effort` is given."""
        self._client = client
        self._api_key = api_key
        self._timeout = float(timeout_seconds)
        self._max_retries = max(0, int(max_retries))
        self._reasoning_effort = reasoning_effort
        self._verbosity = verbosity
        self._carry_reasoning = bool(reasoning_effort) if carry_reasoning is None else bool(carry_reasoning)
        self._sleep = sleep

    def stream(self, request: LLMRequest) -> Iterator[LLMEvent]:
        client = self._ready_client()
        arguments = self._arguments(request)
        attempt = 0
        while not _cancelled(request.cancel):
            emitted = False
            try:
                for event in self._once(client, arguments, request.cancel):
                    emitted = emitted or isinstance(event, (TextDelta, ToolCall))
                    yield event
                return
            except Failure as failure:
                if _cancelled(request.cancel):
                    return
                wait = self._retry_wait(failure, attempt)
                if emitted or wait is None:
                    raise failure.error from failure.__cause__
                attempt += 1
                logger.warning("model call failed before the first token (%s); retrying", failure.error.code)
                self._pause(wait, request.cancel)

    # --- one attempt ---

    def _once(self, client: Any, arguments: dict[str, Any], cancel: Optional[CancelToken]) -> Iterator[LLMEvent]:
        stream: Any = None
        forget: Callable[[], None] = _nothing
        try:
            stream = client.responses.create(**arguments)
            if cancel is not None:
                forget = cancel.on_cancel(lambda: _close(stream))
            yield from self._read(stream)
        except (Failure, GeneratorExit):
            raise
        except Exception as error:
            if _cancelled(cancel):
                return
            raise map_error(error, self._secret()) from error
        finally:
            forget()
            _close(stream)

    def _read(self, stream: Any) -> Iterator[LLMEvent]:
        phases: dict[Any, Optional[str]] = {}
        items: list[Any] = []
        for event in stream:
            kind = getattr(event, "type", "")
            if kind == "response.output_item.added":
                phases[getattr(event, "output_index", None)] = getattr(event.item, "phase", None)
            elif kind == "response.output_text.delta":
                phase = phases.get(getattr(event, "output_index", None))
                yield TextDelta(event.delta, commentary=phase == "commentary")
            elif kind == "response.refusal.delta":
                yield TextDelta(event.delta)
            elif kind == "response.output_item.done":
                item = event.item
                if getattr(item, "status", None) == "incomplete":
                    continue  # cut by the output cap: a call with half its arguments must not run
                items.append(item)
                if getattr(item, "type", "") == "function_call":
                    yield ToolCall(item.call_id, item.name, item.arguments or "{}")
            elif kind == "response.completed":
                yield Completed(_usage(event.response), "stop", _carried(items))
                return
            elif kind == "response.incomplete":
                reason = getattr(getattr(event.response, "incomplete_details", None), "reason", None)
                stop = _STOP_REASONS.get(reason or "", "length")
                yield Completed(_usage(event.response), stop, _carried(items, whole=False))
                return
            elif kind == "response.failed":
                problem = getattr(event.response, "error", None)
                code, message = getattr(problem, "code", None), getattr(problem, "message", "")
                raise stream_failure(code, message, self._secret())
            elif kind == "error":
                raise stream_failure(getattr(event, "code", None), getattr(event, "message", ""), self._secret())
        raise Failure(ModelError(detail="the stream ended without a completion"), False)

    # --- request ---

    def _arguments(self, request: LLMRequest) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "model": request.model,
            "input": _input_items(request.messages),
            "max_output_tokens": request.max_output_tokens,
            "store": False,
            "stream": True,
            "timeout": self._timeout,
        }
        instructions = "\n\n".join(m.content for m in request.messages if m.role == "system" and m.content)
        if instructions:
            arguments["instructions"] = instructions
        if request.tools:
            arguments["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.parameters),
                    "strict": True,
                }
                for tool in request.tools
            ]
            arguments["tool_choice"] = "auto" if request.allow_tools else "none"
        if self._reasoning_effort:
            arguments["reasoning"] = {"effort": self._reasoning_effort}
        if self._verbosity:
            arguments["text"] = {"verbosity": self._verbosity}
        if self._carry_reasoning:
            arguments["include"] = ["reasoning.encrypted_content"]
        return arguments

    def _secret(self) -> str:
        """The key it was given, else the key of the app: the environment, else the file .env of the data folder."""
        from recall_services import keys  # read at call time: a key saved in the app is used with no new start

        return (self._api_key or keys.api_key()).strip()

    def _ready_client(self) -> Any:
        if self._client is None:
            key = self._secret()
            if not key:
                raise NoApiKeyError()
            # The address is always given here, and no proxy is taken from the
            # environment: neither can send message text to another place.
            self._client = openai.OpenAI(
                api_key=key,
                base_url=OFFICIAL_BASE_URL,
                max_retries=0,
                timeout=self._timeout,
                http_client=openai.DefaultHttpxClient(trust_env=False),
            )
        return self._client

    # --- retries ---

    def _retry_wait(self, failure: Failure, attempt: int) -> Optional[float]:
        """Seconds to wait before a new attempt, or None when there is none."""
        if not failure.retryable or attempt >= self._max_retries:
            return None
        asked = failure.error.retry_after
        if asked is not None:
            return asked if asked <= _MAX_WAIT_SECONDS else None
        return min(_BACKOFF_CAP_SECONDS, float(2 ** attempt))

    def _pause(self, seconds: float, cancel: Optional[CancelToken]) -> None:
        if self._sleep is not None:
            self._sleep(seconds)
        elif cancel is not None:
            cancel.wait(seconds)
        else:
            time.sleep(seconds)


def _nothing() -> None:
    return None


def _cancelled(cancel: Optional[CancelToken]) -> bool:
    return cancel is not None and cancel.cancelled


def _close(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # the stream is going away; nothing to recover
            logger.debug("closing the model stream failed", exc_info=True)

