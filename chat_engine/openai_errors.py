"""Every way the OpenAI call can fail, as one typed error each.

`map_error` reads an exception. `stream_failure` reads a failure event of the
stream. Both say whether a new attempt may help.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional

import httpx
import openai

from .errors import (
    ChatEngineError,
    KeyRejectedError,
    ModelError,
    ModelTimeoutError,
    OfflineError,
    RateLimitedError,
    UnknownModelError,
)

_SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]+")
_MIN_SECRET_CHARS = 8
_NO_CREDIT = "The API account has no credit left, or its spending limit is reached."
_NO_REGION = "The model provider does not serve this region."
_NOT_ALLOWED = "The model provider does not allow this API key to make the request."
_TIMEOUT_STATUS = 408
_CONFLICT_STATUS = 409
_SERVER_STATUS = 500
_KEY_CODES = {"invalid_api_key", "incorrect_api_key", "account_deactivated"}


class Failure(Exception):
    """A typed error on its way out, and whether a new attempt may help."""

    def __init__(self, error: ChatEngineError, retryable: bool) -> None:
        super().__init__(error.message)
        self.error = error
        self.retryable = retryable


def scrub(text: str, secret: str = "") -> str:
    """Text for the log, without the API key."""
    cleaned = text.replace(secret, "***") if len(secret) >= _MIN_SECRET_CHARS else text
    return _SECRET_RE.sub("sk-***", cleaned)


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _seconds_until(value: Any) -> Optional[float]:
    """A wait that the provider gives as an HTTP date."""
    try:
        moment = parsedate_to_datetime(str(value))
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (moment - datetime.now(timezone.utc)).total_seconds()


def _retry_after(error: BaseException) -> Optional[float]:
    """Seconds the provider asks to wait: "retry-after-ms", then "retry-after". Never negative."""
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    millis = _finite(headers.get("retry-after-ms"))
    if millis is not None:
        return max(0.0, millis / 1000.0)
    value = headers.get("retry-after")
    seconds = _finite(value)
    if seconds is None and value is not None:
        seconds = _seconds_until(value)
    return None if seconds is None else max(0.0, seconds)


def _from_code(code: Optional[str], detail: str) -> Optional[Failure]:
    if code == "model_not_found":
        return Failure(UnknownModelError(detail=detail), False)
    if code in _KEY_CODES:
        return Failure(KeyRejectedError(detail=detail), False)
    if code == "insufficient_quota":
        return Failure(RateLimitedError(_NO_CREDIT, detail=detail), False)
    if code == "rate_limit_exceeded":
        return Failure(RateLimitedError(detail=detail), True)
    if code == "server_error":
        return Failure(ModelError(detail=detail), True)
    if code == "unsupported_country_region_territory":
        return Failure(ModelError(_NO_REGION, detail=detail), False)
    return None


def _from_status(error: BaseException, detail: str) -> Failure:
    """A failure the provider answered with a status and no code this module knows."""
    status = getattr(error, "status_code", None)
    if status == _TIMEOUT_STATUS:
        return Failure(ModelTimeoutError(detail=detail), True)
    if status == _CONFLICT_STATUS:
        return Failure(ModelError(detail=detail), True)
    if isinstance(error, openai.AuthenticationError):
        return Failure(KeyRejectedError(detail=detail), False)
    if isinstance(error, openai.PermissionDeniedError):
        return Failure(KeyRejectedError(_NOT_ALLOWED, detail=detail), False)
    if isinstance(error, openai.NotFoundError):
        # a wrong address also gives 404: only a 404 about the model is an unknown model
        about_model = "model" in str(error).lower()
        return Failure(UnknownModelError(detail=detail) if about_model else ModelError(detail=detail), False)
    server_side = isinstance(status, int) and status >= _SERVER_STATUS
    return Failure(ModelError(detail=detail), server_side or isinstance(error, openai.InternalServerError))


def map_error(error: BaseException, secret: str = "") -> Failure:
    """One typed error for each way the provider can fail.

    The SDK does not wrap a failure of the connection while the stream is
    read, so the errors of the transport are mapped here too.
    """
    detail = scrub(f"{type(error).__name__}: {error}", secret)
    if isinstance(error, (openai.APITimeoutError, httpx.TimeoutException)):
        return Failure(ModelTimeoutError(detail=detail), True)
    if isinstance(error, (openai.APIConnectionError, httpx.TransportError)):
        return Failure(OfflineError(detail=detail), True)
    known = _from_code(getattr(error, "code", None), detail)
    if isinstance(error, openai.RateLimitError):
        if known is not None and not known.retryable:
            return known
        return Failure(RateLimitedError(detail=detail, retry_after=_retry_after(error)), True)
    return known if known is not None else _from_status(error, detail)


def stream_failure(code: Optional[str], message: str, secret: str = "") -> Failure:
    """The failure that a "response.failed" or "error" event of the stream reports."""
    detail = scrub(f"{code or 'error'}: {message}", secret)
    known = _from_code(code, detail)
    return known if known is not None else Failure(ModelError(detail=detail), False)
