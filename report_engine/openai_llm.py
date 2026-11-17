"""LLM port over the OpenAI Responses API.

Every request is sent with `store=False`, an explicit timeout and a hard
`max_output_tokens`. A schema becomes a strict JSON-schema format. Usage is
read from the reply.

Failures that pass (rate limit, timeout, connection, 5xx) are retried a fixed
number of times. The wait is the one the service names in `retry-after`; when
that wait is longer than the cap, the adapter gives up instead of asking
earlier. A reply that is cut off is returned with status "incomplete" and is
never sent again: the same request would be cut off again.

No prompt or reply text is logged or put in an error.
"""

from __future__ import annotations

import email.utils
import json
import logging
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import openai

from report_engine.errors import LLMError
from report_engine.ports import ChatMessage, LLMResult, LLMStatus, LLMUsage

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_RETRY_WAIT_SECONDS = 60.0
BACKOFF_BASE_SECONDS = 2.0
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
QUOTA_ERROR_CODE = "insufficient_quota"
DEFAULT_SCHEMA_NAME = "reply"
MAX_SCHEMA_NAME_CHARS = 64
MAX_ERROR_CHARS = 300

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")
_SECRETS = re.compile(r"\b(sk|org|proj|user)-[A-Za-z0-9_*-]+")


def build_client(
    api_key: str | None = None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    base_url: str | None = None,
) -> openai.OpenAI:
    """An SDK client with its own retries off, so the adapter's are the only ones.
    With no key given the SDK reads OPENAI_API_KEY."""
    return openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout_s)


def _scrub(text: str) -> str:
    flat = " ".join(str(text).split())
    return _SECRETS.sub(lambda match: f"{match.group(1)}-...", flat)[:MAX_ERROR_CHARS]


def _schema_format(schema: Mapping[str, Any]) -> dict[str, Any]:
    name = _UNSAFE_NAME.sub("_", str(schema.get("title") or "")).strip("_")[:MAX_SCHEMA_NAME_CHARS]
    body = {key: value for key, value in schema.items() if key != "title"}
    return {
        "type": "json_schema",
        "name": name or DEFAULT_SCHEMA_NAME,
        "schema": body,
        "strict": True,
    }


def _request(
    model: str,
    messages: Sequence[ChatMessage],
    schema: Mapping[str, Any] | None,
    max_output_tokens: int,
    reasoning: str | None,
    timeout_s: float,
) -> dict[str, Any]:
    unknown = sorted({message["role"] for message in messages} - {"system", "user"})
    if unknown:
        raise ValueError(f"unsupported message role: {', '.join(unknown)}")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    request: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "user", "content": message["content"]}
            for message in messages
            if message["role"] == "user"
        ],
        "store": False,
        "max_output_tokens": max_output_tokens,
        "timeout": timeout_s,
    }
    instructions = "\n\n".join(
        message["content"] for message in messages if message["role"] == "system"
    )
    if instructions:
        request["instructions"] = instructions
    if reasoning:
        request["reasoning"] = {"effort": reasoning}
    if schema is not None:
        request["text"] = {"format": _schema_format(schema)}
    return request


def _usage(response: Any) -> LLMUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        logger.warning("the model reply carried no usage; its tokens are counted as zero")
        return LLMUsage()
    details = getattr(usage, "input_tokens_details", None)
    return LLMUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        cached_input_tokens=int(getattr(details, "cached_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )


def _refusal(response: Any) -> str:
    parts = [
        str(getattr(content, "refusal", "") or "")
        for item in getattr(response, "output", None) or []
        for content in getattr(item, "content", None) or []
        if getattr(content, "type", None) == "refusal"
    ]
    return " ".join(part for part in parts if part)


def _parsed(text: str) -> Mapping[str, Any] | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _result(response: Any, wants_json: bool, attempts: int) -> LLMResult:
    text = str(getattr(response, "output_text", "") or "").strip()
    raw_status = getattr(response, "status", None) or "completed"
    status: LLMStatus
    if raw_status == "incomplete":
        status = "incomplete"
    elif raw_status != "completed":
        status = "failed"
    elif not text and _refusal(response):
        status, text = "refused", _refusal(response)
    else:
        status = "completed"
    parsed = _parsed(text) if wants_json and status == "completed" and text else None
    return LLMResult(
        text=text, parsed=parsed, usage=_usage(response), status=status, attempts=attempts
    )


def _retry_after(error: Exception) -> float | None:
    """Seconds the service asks us to wait, from its headers. None when it names none."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is None:
        return None
    millis = headers.get("retry-after-ms")
    seconds = headers.get("retry-after")
    try:
        if millis is not None:
            return max(0.0, float(millis) / 1000.0)
        if seconds is not None:
            return max(0.0, float(seconds))
    except ValueError:
        pass
    if seconds is None:
        return None
    try:
        moment = email.utils.parsedate_to_datetime(seconds)
    except (TypeError, ValueError):
        return None
    return max(0.0, moment.timestamp() - time.time())


def _status_code(error: Exception) -> int | None:
    code = getattr(error, "status_code", None)
    return code if isinstance(code, int) else None


def _passing(error: Exception) -> bool:
    """True for a failure that a later attempt can get past."""
    if isinstance(error, openai.APIConnectionError):
        return True
    code = _status_code(error)
    if code is None:
        return False
    if getattr(error, "code", None) == QUOTA_ERROR_CODE:
        return False
    return code in RETRYABLE_STATUS_CODES or code >= 500


def _describe(error: Exception) -> str:
    code = _status_code(error)
    api_code = getattr(error, "code", None)
    where = ", ".join(str(part) for part in (f"HTTP {code}" if code else None, api_code) if part)
    message = _scrub(getattr(error, "message", "") or str(error))
    head = (
        f"the model service failed ({where})"
        if where
        else f"the model service failed ({type(error).__name__})"
    )
    return f"{head}: {message}" if message else head


class OpenAIResponsesLLM:
    """One model behind the LLM port. Give it a client made by `build_client`."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_retry_wait_s: float = DEFAULT_MAX_RETRY_WAIT_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries can not be negative")
        self._client = client
        self._model = model
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._max_retry_wait_s = max_retry_wait_s
        self._sleep = sleep
        self._jitter = jitter

    def _wait_before_retry(self, error: Exception, attempt: int) -> float | None:
        """How long to wait before the next attempt. None means give up now."""
        named = _retry_after(error)
        if named is not None:
            return named if named <= self._max_retry_wait_s else None
        backoff = min(self._max_retry_wait_s, BACKOFF_BASE_SECONDS * 2.0 ** (attempt - 1))
        return backoff * (0.5 + 0.5 * self._jitter())

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: Mapping[str, Any] | None = None,
        max_output_tokens: int,
        reasoning: str | None = None,
    ) -> LLMResult:
        request = _request(
            self._model, messages, schema, max_output_tokens, reasoning, self._timeout_s
        )
        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self._client.responses.create(**request)
            except openai.OpenAIError as error:
                wait = self._wait_before_retry(error, attempt) if _passing(error) else None
                if wait is None or attempt == attempts:
                    raise LLMError(
                        _describe(error),
                        status_code=_status_code(error),
                        retry_after_s=_retry_after(error),
                        attempts=attempt,
                    ) from error
                logger.warning(
                    "model call failed (%s); attempt %d of %d, next in %.1f s",
                    type(error).__name__,
                    attempt,
                    attempts,
                    wait,
                )
                self._sleep(wait)
                continue
            return _result(response, schema is not None, attempt)
        raise LLMError("the model service gave no reply", attempts=attempts)
