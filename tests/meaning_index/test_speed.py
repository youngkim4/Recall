"""The speed of the contract: a search over 100,000 segments of 1024 dimensions takes less than one second.

The vectors are invented with a seeded generator. They go through the files of
the store, so the search reads them as the app does.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from meaning_index.search import MeaningIndex
from meaning_index.shapes import Segment
from meaning_index.store import IndexStore

SEGMENTS = 100_000
DIMENSIONS = 1024
SEED = 20261117
CHUNK = 10_000
PLANTED = 54_321
LIMIT_SECONDS = 1.0


def rows() -> Iterator[np.ndarray]:
    generator = np.random.default_rng(SEED)
    for _ in range(0, SEGMENTS, CHUNK):
        chunk = generator.standard_normal((CHUNK, DIMENSIONS), dtype=np.float32)
        chunk /= np.linalg.norm(chunk, axis=1, keepdims=True)
        yield chunk.astype("<f2")


def segments() -> list[Segment]:
    return [
        Segment(
            segment_id=f"{n:064x}",
            conversation_id=f"chat-{n % 1400}",
            first_message_id=f"first-{n}",
            last_message_id=f"last-{n}",
            first_timestamp=f"20{10 + n % 16:02d}-{1 + n % 12:02d}-{1 + n % 28:02d} 09:00:00",
            last_timestamp=f"20{10 + n % 16:02d}-{1 + n % 12:02d}-{1 + n % 28:02d} 09:30:00",
            message_count=8,
            content_hash=f"{n + SEGMENTS:064x}",
        )
        for n in range(SEGMENTS)
    ]


def timed(index: MeaningIndex, query: list[float], **filters: object) -> tuple[float, list]:
    started = time.perf_counter()
    hits = index.search(query, **filters)  # type: ignore[arg-type]
    return time.perf_counter() - started, hits


def test_a_search_over_100_000_segments_of_1024_dimensions_takes_less_than_one_second(folder: Path) -> None:
    store = IndexStore(folder)
    with store.building():
        store.replace("invented-model", DIMENSIONS, segments(), rows())
    index = MeaningIndex.load(store)
    assert index is not None
    assert (len(index), index.dimensions) == (SEGMENTS, DIMENSIONS)
    stored = store.read()
    assert stored is not None
    query = np.asarray(stored.vectors[PLANTED], dtype=np.float32).tolist()

    times = []
    for _ in range(3):
        seconds, hits = timed(index, query)
        times.append(seconds)
        assert hits[0].segment.segment_id == f"{PLANTED:064x}"
        assert hits[0].score > 0.99
        assert hits[1].score < 0.5
        assert len(hits) == 20
    assert max(times) < LIMIT_SECONDS, times

    # the planted segment is of 2011: half of the conversations, and the years 2011 to 2024
    half = [f"chat-{n}" for n in range(0, 1400, 2)] + [f"chat-{PLANTED % 1400}"]
    seconds, hits = timed(index, query, conversations=half, since="2011-01-01", until="2024-12-31")
    assert seconds < LIMIT_SECONDS, seconds
    assert hits[0].segment.segment_id == f"{PLANTED:064x}"
    assert all(hit.segment.conversation_id in set(half) for hit in hits)
    assert all("2011" <= hit.segment.first_timestamp[:4] <= "2024" for hit in hits)
    logging.getLogger(__name__).info("search times: %s; with filters: %.3f s", [round(t, 3) for t in times], seconds)
