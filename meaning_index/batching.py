"""Put texts into requests, and get one unit vector for each text.

A request holds at most `max_texts` texts and at most `max_tokens` counted
tokens. A request that the service calls too large is sent again as two
halves. A rate limit waits the time that the service names, and the request is
tried again, `max_tries` times at most. With no named wait the waits double:
2, 4, 8, 16 and 32 seconds, which is longer than a limit for one minute lasts.
A wait is slept in parts of one second at most, and the caller can stop the
build between two parts. No other failure is tried again.

No text is logged or put in an error.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from meaning_index.errors import (
    EmbeddingError,
    EmbeddingRateLimited,
    EmbeddingReplyError,
    EmbeddingServiceError,
    EmbeddingTooLarge,
)
from meaning_index.ports import Embedder, Sleep

logger = logging.getLogger(__name__)

VECTOR_TYPE = np.dtype("<f2")
SLEEP_PART_S = 1.0

Check = Callable[[], None]


@dataclass(frozen=True)
class BatchLimits:
    """The limits of one request. They are Recall's own: the provider's limits were not read."""

    max_texts: int = 96
    max_tokens: int = 100_000
    max_tries: int = 6
    max_wait_s: float = 300.0
    backoff_s: float = 2.0

    def __post_init__(self) -> None:
        if self.max_texts < 1 or self.max_tokens < 1 or self.max_tries < 1:
            raise ValueError("a request needs at least one text, one token and one try")
        if self.max_wait_s < 0 or self.backoff_s < 0:
            raise ValueError("a wait can not be negative")


DEFAULT_LIMITS = BatchLimits()


def plan_batches(token_counts: Sequence[int], limits: BatchLimits = DEFAULT_LIMITS) -> list[tuple[int, int]]:
    """Cut the texts into requests, in order: `[(start, stop), ...]`.

    A text above the token limit is a request of its own.
    """
    batches: list[tuple[int, int]] = []
    start, held = 0, 0
    for place, count in enumerate(token_counts):
        full = place - start >= limits.max_texts or held + count > limits.max_tokens
        if place > start and full:
            batches.append((start, place))
            start, held = place, 0
        held += count
    if len(token_counts) > start:
        batches.append((start, len(token_counts)))
    return batches


def unit_rows(vectors: object, count: int, dimensions: int) -> np.ndarray:
    """The vectors of one reply as `count` rows of length 1, float16."""
    try:
        rows = np.asarray(vectors, dtype=np.float64)
    except (TypeError, ValueError):
        raise EmbeddingReplyError("the embedder gave vectors that are not rows of numbers") from None
    if rows.shape != (count, dimensions):
        raise EmbeddingReplyError(
            f"the embedder gave an answer of shape {rows.shape} for {count} texts of {dimensions} dimensions"
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        lengths = np.linalg.norm(rows, axis=1, keepdims=True)
        usable = np.isfinite(rows).all() and np.isfinite(lengths).all() and (lengths > 0).all()
    if not usable:
        raise EmbeddingReplyError("the embedder gave a vector with no length or with a value that is no number")
    return (rows / lengths).astype(VECTOR_TYPE)


def _wait(error: EmbeddingRateLimited, attempt: int, limits: BatchLimits) -> float | None:
    """How long to wait before the next try. None means: give up now."""
    if error.retry_after_s is not None:
        return error.retry_after_s if error.retry_after_s <= limits.max_wait_s else None
    return min(limits.max_wait_s, limits.backoff_s * 2.0 ** (attempt - 1))


def _sleep_in_parts(wait: float, sleep: Sleep, check: Check | None) -> None:
    """Wait `wait` seconds, one second at most at a time. `check` can raise between two parts."""
    left = wait
    while left > 0:
        if check is not None:
            check()
        part = min(SLEEP_PART_S, left)
        sleep(part)
        left -= part
    if check is not None:
        check()


def _ask(embedder: Embedder, texts: Sequence[str], limits: BatchLimits, sleep: Sleep, check: Check | None) -> object:
    """One request, tried again only when the service says to wait."""
    for attempt in range(1, limits.max_tries + 1):
        try:
            return embedder.embed(texts)
        except EmbeddingRateLimited as error:
            wait = _wait(error, attempt, limits)
            if wait is None or attempt == limits.max_tries:
                raise
            logger.warning(
                "the embedding service asks to wait; try %d of %d, next in %.1f s", attempt, limits.max_tries, wait
            )
            _sleep_in_parts(wait, sleep, check)
        except EmbeddingError:
            raise
        except Exception as error:  # noqa: BLE001 - what an embedder raises can hold the texts
            raise EmbeddingServiceError(f"the embedder failed ({type(error).__name__})") from None
    raise EmbeddingServiceError("the embedder gave no answer")  # pragma: no cover - the loop returns or raises


def embed_in_batches(
    embedder: Embedder,
    texts: Sequence[str],
    token_counts: Sequence[int],
    *,
    limits: BatchLimits = DEFAULT_LIMITS,
    sleep: Sleep = time.sleep,
    check: Check | None = None,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Give `(start, stop, rows)` for each request that the service answered, in order.

    The stream is lazy: the next request is made when the caller asks for the
    next item. A caller can keep the rows, or stop, between two requests.
    `check` is called during a wait, one time a second or more often. What it
    raises ends the wait and the stream.
    """
    if len(texts) != len(token_counts):
        raise ValueError("each text needs one token count")
    waiting = deque(plan_batches(token_counts, limits))
    while waiting:
        start, stop = waiting.popleft()
        try:
            answer = _ask(embedder, texts[start:stop], limits, sleep, check)
        except EmbeddingTooLarge:
            if stop - start < 2:
                raise
            middle = (start + stop) // 2
            logger.warning("a request of %d texts was too large; it is sent again as two halves", stop - start)
            waiting.extendleft([(middle, stop), (start, middle)])
            continue
        yield start, stop, unit_rows(answer, stop - start, embedder.dimensions)
