"""How texts are put into requests, and what a request does when the service says no."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from meaning_index.batching import DEFAULT_LIMITS, BatchLimits, embed_in_batches, plan_batches, unit_rows
from meaning_index.errors import (
    EmbeddingKeyError,
    EmbeddingRateLimited,
    EmbeddingReplyError,
    EmbeddingServiceError,
    EmbeddingTooLarge,
)
from tests.meaning_index.support import ScriptedEmbedder

MARK = "MARKED-TEXT-7781"


def texts_of(count: int) -> list[str]:
    return [f"text number {n}" for n in range(count)]


def embed_all(embedder: object, texts: Sequence[str], counts: Sequence[int] | None = None, **named: object) -> tuple:
    slept: list[float] = []
    named.setdefault("sleep", slept.append)
    parts = list(embed_in_batches(embedder, texts, counts or [1] * len(texts), **named))  # type: ignore[arg-type]
    return parts, slept


def test_the_limits_are_the_ones_of_the_contract() -> None:
    assert (DEFAULT_LIMITS.max_texts, DEFAULT_LIMITS.max_tokens) == (96, 100_000)
    assert (DEFAULT_LIMITS.max_tries, DEFAULT_LIMITS.max_wait_s) == (6, 300.0)


def test_with_no_named_wait_the_default_waits_last_longer_than_one_minute() -> None:
    embedder = ScriptedEmbedder(lambda number, texts: EmbeddingRateLimited("wait"))
    slept: list[float] = []
    with pytest.raises(EmbeddingRateLimited):
        embed_all(embedder, texts_of(2), sleep=slept.append)
    assert slept == [2.0, 4.0, 8.0, 16.0, 32.0]
    assert sum(slept) > 60


def test_a_request_holds_96_texts_at_most() -> None:
    assert plan_batches([1] * 200) == [(0, 96), (96, 192), (192, 200)]
    assert plan_batches([1] * 96) == [(0, 96)]
    assert plan_batches([]) == []


def test_a_request_holds_100_000_counted_tokens_at_most() -> None:
    assert plan_batches([30_000] * 7) == [(0, 3), (3, 6), (6, 7)]
    assert plan_batches([50_000, 50_000, 1]) == [(0, 2), (2, 3)]


def test_a_text_above_the_token_limit_is_a_request_of_its_own() -> None:
    assert plan_batches([10, 150_000, 10]) == [(0, 1), (1, 2), (2, 3)]


def test_every_text_is_embedded_one_time_in_order() -> None:
    embedder = ScriptedEmbedder()
    texts = texts_of(200)
    parts, slept = embed_all(embedder, texts)
    assert embedder.sizes == [96, 96, 8]
    assert [(start, stop) for start, stop, _ in parts] == [(0, 96), (96, 192), (192, 200)]
    assert embedder.texts == texts
    assert slept == []
    rows = np.concatenate([rows for _, _, rows in parts])
    assert rows.shape == (200, 16)
    assert rows.dtype == np.float16
    assert np.allclose(np.linalg.norm(rows.astype(np.float64), axis=1), 1.0, atol=1e-3)


def test_a_request_that_is_too_large_is_sent_again_as_two_halves() -> None:
    embedder = ScriptedEmbedder(lambda number, texts: EmbeddingTooLarge("too large") if len(texts) > 30 else None)
    texts = texts_of(96)
    parts, _ = embed_all(embedder, texts)
    assert embedder.sizes == [96, 48, 24, 24, 48, 24, 24]
    assert [(start, stop) for start, stop, _ in parts] == [(0, 24), (24, 48), (48, 72), (72, 96)]
    assert embedder.texts == texts


def test_one_text_that_is_too_large_is_the_typed_failure() -> None:
    embedder = ScriptedEmbedder(lambda number, texts: EmbeddingTooLarge("too large"))
    with pytest.raises(EmbeddingTooLarge):
        embed_all(embedder, texts_of(3))
    assert embedder.sizes == [3, 1]


def test_a_rate_limit_waits_the_time_that_the_service_names_and_tries_again() -> None:
    embedder = ScriptedEmbedder({1: EmbeddingRateLimited("wait", retry_after_s=7.5)})
    parts, slept = embed_all(embedder, texts_of(4))
    assert slept == [7.5]
    assert embedder.sizes == [4, 4]
    assert len(parts) == 1


def test_a_rate_limit_that_names_no_wait_backs_off() -> None:
    embedder = ScriptedEmbedder({n: EmbeddingRateLimited("wait") for n in (1, 2, 3)})
    _, slept = embed_all(embedder, texts_of(2), limits=BatchLimits(backoff_s=2.0))
    assert slept == [2.0, 4.0, 8.0]


def test_the_tries_of_one_request_are_bounded() -> None:
    embedder = ScriptedEmbedder(lambda number, texts: EmbeddingRateLimited("wait", retry_after_s=1.0))
    with pytest.raises(EmbeddingRateLimited):
        embed_all(embedder, texts_of(2), limits=BatchLimits(max_tries=4))
    assert embedder.sizes == [2, 2, 2, 2]


def test_the_bound_is_for_each_request_not_for_the_whole_build() -> None:
    limited = {1: EmbeddingRateLimited("wait", retry_after_s=1.0), 3: EmbeddingRateLimited("wait", retry_after_s=2.0)}
    embedder = ScriptedEmbedder(limited)
    parts, slept = embed_all(embedder, texts_of(150), limits=BatchLimits(max_tries=2))
    assert slept == [1.0, 2.0]
    assert len(parts) == 2


def test_a_named_wait_above_the_cap_ends_the_request_with_no_wait() -> None:
    embedder = ScriptedEmbedder({1: EmbeddingRateLimited("wait", retry_after_s=3600.0)})
    slept: list[float] = []
    with pytest.raises(EmbeddingRateLimited) as caught:
        embed_all(embedder, texts_of(2), limits=BatchLimits(max_wait_s=300.0), sleep=slept.append)
    assert caught.value.retry_after_s == 3600.0
    assert slept == []


def test_other_typed_failures_are_not_tried_again() -> None:
    embedder = ScriptedEmbedder({1: EmbeddingKeyError("no key")})
    with pytest.raises(EmbeddingKeyError):
        embed_all(embedder, texts_of(2))
    assert embedder.sizes == [2]


def test_a_failure_that_is_not_typed_becomes_typed_and_holds_no_text() -> None:
    embedder = ScriptedEmbedder(lambda number, texts: RuntimeError(f"could not embed {texts[0]}"))
    with pytest.raises(EmbeddingServiceError) as caught:
        embed_all(embedder, [f"a line with {MARK}"])
    assert MARK not in str(caught.value)
    assert "RuntimeError" in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_the_next_request_is_made_only_when_the_caller_asks_for_it() -> None:
    embedder = ScriptedEmbedder()
    stream = embed_in_batches(embedder, texts_of(200), [1] * 200, sleep=lambda seconds: None)
    next(stream)
    assert embedder.sizes == [96]


class Fixed:
    model = "fixed"
    dimensions = 4

    def __init__(self, reply: object) -> None:
        self._reply = reply

    def embed(self, texts: Sequence[str]) -> object:
        return self._reply


@pytest.mark.parametrize(
    "reply",
    [
        [[1.0, 0.0, 0.0, 0.0]],  # one vector for two texts
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # three numbers, not four
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0]],  # sizes that differ
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # no length
        [[1.0, 0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0, 0.0], [float("inf"), 0.0, 0.0, 1.0]],
        [["a", "b", "c", "d"], ["a", "b", "c", "d"]],
        None,
    ],
)
def test_a_reply_that_breaks_the_port_is_a_typed_failure(reply: object) -> None:
    with pytest.raises(EmbeddingReplyError):
        embed_all(Fixed(reply), ["one", "two"])


def test_rows_are_made_unit_length_whatever_length_the_embedder_gave() -> None:
    rows = unit_rows([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.0]], 2, 4)
    assert rows.dtype == np.float16
    assert rows[0].tolist() == pytest.approx([0.6, 0.8, 0.0, 0.0], abs=1e-3)
    assert rows[1].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_counts_that_do_not_match_the_texts_are_refused() -> None:
    with pytest.raises(ValueError):
        embed_all(ScriptedEmbedder(), texts_of(3), [1, 1])
    with pytest.raises(ValueError):
        BatchLimits(max_texts=0)
    with pytest.raises(ValueError):
        BatchLimits(max_tries=0)
    with pytest.raises(ValueError):
        BatchLimits(max_wait_s=-1.0)
