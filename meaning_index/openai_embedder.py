"""Embedder port over the OpenAI embeddings API.

    OpenAIEmbedder(client, "text-embedding-3-large", 1024)

The caller gives the client. The app builds it with
`report_engine.openai_llm.build_client`, which pins the address of the
provider, takes no proxy from the environment and switches the SDK's own
retries off. One call of `embed` is one request. The build does the waiting.

No text is logged or put in an error. An error names the HTTP status, the
service's error code and its request id, and nothing else. The SDK error holds
the request, and with it the texts: it is never chained.
"""

from __future__ import annotations

import email.utils
import math
import re
import time
from collections.abc import Sequence
from datetime import UTC
from typing import Any

import openai

from meaning_index.errors import (
    EmbeddingError,
    EmbeddingKeyError,
    EmbeddingNetworkError,
    EmbeddingRateLimited,
    EmbeddingReplyError,
    EmbeddingServiceError,
    EmbeddingTooLarge,
)

KEY_STATUS_CODES = frozenset({401, 403})
RATE_LIMIT_STATUS = 429
TOO_LARGE_STATUS = 413
BAD_REQUEST_STATUS = 400
QUOTA_ERROR_CODE = "insufficient_quota"
# Not read from the provider's pages: see the README.
TOO_LARGE_CODES = frozenset(
    {"context_length_exceeded", "max_tokens_per_request", "string_above_max_length", "array_above_max_length"}
)
TOO_LARGE_WORDS = ("maximum context length", "tokens per request", "too large", "too many inputs")
MAX_FACT_CHARS = 60

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _fact(value: object) -> str:
    """A code or an id of the service, cut to plain characters."""
    return _UNSAFE.sub("", str(value or ""))[:MAX_FACT_CHARS]


def _status_code(error: Exception) -> int | None:
    code = getattr(error, "status_code", None)
    return code if isinstance(code, int) else None


def _describe(error: Exception) -> str:
    """What failed, from facts that can hold no text of a message. The service's own message is left out."""
    code = _status_code(error)
    request_id = _fact(getattr(error, "request_id", None))
    facts = [
        f"HTTP {code}" if code else type(error).__name__,
        _fact(getattr(error, "code", None)),
        f"request {request_id}" if request_id else "",
    ]
    return f"the embedding service failed ({', '.join(fact for fact in facts if fact)})"


def _seconds(value: str | None, scale: float) -> float | None:
    """A header value as seconds. None when it is not a plain, finite number."""
    try:
        number = float(value) / scale if value is not None else math.nan
    except ValueError:
        return None
    return max(0.0, number) if math.isfinite(number) else None


def _seconds_until(value: str | None) -> float | None:
    """An HTTP date as seconds from now. A date with no zone is read as UTC."""
    if value is None:
        return None
    try:
        moment = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return max(0.0, moment.timestamp() - time.time())


def _retry_after(error: Exception) -> float | None:
    """Seconds that the service asks the caller to wait, from its headers. None when it names none."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is None:
        return None
    plain = headers.get("retry-after")
    for found in (_seconds(headers.get("retry-after-ms"), 1000.0), _seconds(plain, 1.0)):
        if found is not None:
            return found
    return _seconds_until(plain)


def _too_large(error: Exception, status: int | None, code: str) -> bool:
    if status == TOO_LARGE_STATUS:
        return True
    if status != BAD_REQUEST_STATUS:
        return False
    said = str(getattr(error, "message", "")).lower()  # read to sort the failure; never kept
    return code in TOO_LARGE_CODES or any(words in said for words in TOO_LARGE_WORDS)


def _typed(error: Exception) -> EmbeddingError:
    """The one typed failure that stands for an SDK error."""
    status, code, text = _status_code(error), _fact(getattr(error, "code", None)), _describe(error)
    if isinstance(error, openai.APIConnectionError):
        return EmbeddingNetworkError(text)
    if status in KEY_STATUS_CODES:
        return EmbeddingKeyError(text, status_code=status, code=code or None)
    if status == RATE_LIMIT_STATUS and code != QUOTA_ERROR_CODE:
        return EmbeddingRateLimited(text, retry_after_s=_retry_after(error), status_code=status, code=code or None)
    if _too_large(error, status, code):
        return EmbeddingTooLarge(text, status_code=status, code=code or None)
    return EmbeddingServiceError(text, status_code=status, code=code or None)


def _vectors(response: Any, count: int) -> list[Sequence[float]]:
    """The vectors of a reply in the order of the texts. The reply names the place of each."""
    try:
        by_place = {int(item.index): item.embedding for item in response.data}
        size = len(response.data)
    except (AttributeError, TypeError, ValueError):
        raise EmbeddingReplyError("the embedding service gave a reply with no vectors") from None
    if size != count or sorted(by_place) != list(range(count)):
        raise EmbeddingReplyError(f"the embedding service gave {size} vectors for {count} texts")
    return [by_place[place] for place in range(count)]


class OpenAIEmbedder:
    """One embedding model at one size, behind the Embedder port."""

    def __init__(self, client: Any, model: str, dimensions: int) -> None:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("dimensions must be a whole number, 1 or more")
        self._client = client
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(model=self.model, input=list(texts), dimensions=self.dimensions)
        except openai.OpenAIError as error:
            raise _typed(error) from None
        except Exception as error:  # noqa: BLE001 - the SDK also raises plain errors for a reply that it can not read
            raise EmbeddingServiceError(f"the embedding service failed ({type(error).__name__})") from None
        return _vectors(response, len(texts))
