"""Build the index from a message source, and say before what a build would cost.

A build cuts every conversation into segments. It embeds only the segments
whose content hash has no vector yet: not in the index, and not in the work
file of a build that stopped. It writes a new set of files and swaps them at
the end. A build that fails or is cancelled leaves the index as it was.

No text, no id of a conversation and no id of a message is logged or put in an error.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass

import numpy as np

from meaning_index.batching import DEFAULT_LIMITS, VECTOR_TYPE, BatchLimits, embed_in_batches
from meaning_index.errors import BuildCancelled, IndexFormatError, MeaningIndexError, SourceError, StoreError
from meaning_index.model_facts import EmbedderFacts
from meaning_index.ports import CountTokens, Embedder, MessageSource, Sleep
from meaning_index.segments import cut_segments, stored
from meaning_index.shapes import BuildEstimate, BuildProgress, BuildResult, CutSegment, Segment
from meaning_index.store import IndexStore, StoredIndex
from meaning_index.tokens import default_count_tokens
from meaning_index.workfile import WorkFile

logger = logging.getLogger(__name__)

WRITE_CHUNK = 1024
TOKENS_PER_PRICE = 1_000_000

ProgressCallback = Callable[[BuildProgress], None]
_Check = Callable[[int, int], None]


@dataclass(frozen=True)
class _Reading:
    """What the source holds now: every segment, and the text of each content hash with no vector yet."""

    segments: tuple[Segment, ...]
    texts: Mapping[str, str]


def _checker(cancel: object) -> _Check:
    """A function that raises BuildCancelled when the caller asked to stop."""
    if cancel is None:
        asked: Callable[[], object] = bool  # bool() is False: nobody asks
    elif callable(cancel):
        asked = cancel
    elif callable(getattr(cancel, "is_set", None)):
        asked = cancel.is_set  # type: ignore[attr-defined]
    elif hasattr(cancel, "cancelled"):
        asked = lambda: cancel.cancelled  # type: ignore[attr-defined]  # noqa: E731
    else:
        raise TypeError("cancel is a callable, or it has is_set() or cancelled")

    def check(texts_embedded: int = 0, tokens: int = 0) -> None:
        if asked():
            raise BuildCancelled("the build was cancelled", texts_embedded=texts_embedded, tokens=tokens)

    return check


def _tell(progress: ProgressCallback | None, phase: str, done: int, total: int) -> None:
    if progress is not None:
        progress(BuildProgress(phase=phase, done=done, total=total))  # type: ignore[arg-type]


def _conversations(source: MessageSource) -> list[str]:
    try:
        return list(dict.fromkeys(source.conversation_ids()))
    except MeaningIndexError:
        raise
    except Exception as exc:  # noqa: BLE001 - what a source raises can hold a handle
        raise SourceError(f"the message source failed ({type(exc).__name__})") from None


def _cuts(source: MessageSource, conversation_id: str) -> list[CutSegment]:
    try:
        return list(cut_segments(conversation_id, source.messages(conversation_id)))
    except MeaningIndexError:
        raise
    except Exception as exc:  # noqa: BLE001 - what a source raises can hold the text of a message
        raise SourceError(f"the message source failed ({type(exc).__name__})") from None


def _read(
    source: MessageSource,
    model: str,
    dimensions: int,
    known: Set[str],
    check: _Check,
    progress: ProgressCallback | None,
) -> _Reading:
    conversations = _conversations(source)
    segments: list[Segment] = []
    texts: dict[str, str] = {}
    _tell(progress, "read", 0, len(conversations))
    for number, conversation_id in enumerate(conversations, start=1):
        check()
        for cut in _cuts(source, conversation_id):
            segment = stored(cut, model, dimensions)
            segments.append(segment)
            if segment.content_hash not in known:
                texts.setdefault(segment.content_hash, cut.text)
        _tell(progress, "read", number, len(conversations))
    return _Reading(segments=tuple(segments), texts=texts)


def _old_index(store: IndexStore) -> StoredIndex | None:
    """The index that is there. One that can not be read is no index: the build makes it new."""
    try:
        return store.read()
    except IndexFormatError:
        raise
    except StoreError as exc:
        logger.warning("the index that is there can not be read (%s); every segment is new", type(exc).__name__)
        return None


def _old_rows(old: StoredIndex | None, dimensions: int) -> dict[str, int]:
    """The row of each content hash in the index that is there, when its vectors have this size."""
    if old is None or old.meta.dimensions != dimensions:
        return {}
    return {segment.content_hash: row for row, segment in enumerate(old.segments)}


def _embed(
    embedder: Embedder,
    texts: Mapping[str, str],
    work: WorkFile,
    count_tokens: CountTokens,
    limits: BatchLimits,
    sleep: Sleep,
    check: _Check,
    progress: ProgressCallback | None,
) -> int:
    """Get a vector for each text and keep it in the work file. Returns the tokens that were sent."""
    hashes = list(texts)
    bodies = [texts[key] for key in hashes]
    counts = [count_tokens(body) for body in bodies]
    done = tokens = 0
    _tell(progress, "embed", 0, len(hashes))
    answers = embed_in_batches(embedder, bodies, counts, limits=limits, sleep=sleep)
    while True:
        check(done, tokens)  # between two requests: nothing is asked for after a cancel
        answer = next(answers, None)
        if answer is None:
            return tokens
        start, stop, rows = answer
        work.put(hashes[start:stop], rows)
        done, tokens = stop, tokens + sum(counts[start:stop])
        _tell(progress, "embed", done, len(hashes))


def _rows(
    segments: Sequence[Segment],
    old: StoredIndex | None,
    old_rows: Mapping[str, int],
    work: WorkFile,
    dimensions: int,
    progress: ProgressCallback | None,
) -> Iterator[np.ndarray]:
    """The vectors in the order of the segments: from the old index, or from the work file."""
    _tell(progress, "write", 0, len(segments))
    for start in range(0, len(segments), WRITE_CHUNK):
        chunk = segments[start : start + WRITE_CHUNK]
        paid = work.get_many([s.content_hash for s in chunk if s.content_hash not in old_rows])
        rows = np.empty((len(chunk), dimensions), dtype=VECTOR_TYPE)
        for place, segment in enumerate(chunk):
            row = old_rows.get(segment.content_hash)
            if row is not None and old is not None:
                rows[place] = old.vectors[row]
            elif segment.content_hash in paid:
                rows[place] = paid[segment.content_hash]
            else:
                raise StoreError("a segment has no vector: the work file changed during the build")
        yield rows
        _tell(progress, "write", start + len(chunk), len(segments))


def build_index(
    source: MessageSource,
    embedder: Embedder,
    store: IndexStore,
    *,
    progress: ProgressCallback | None = None,
    cancel: object = None,
    count_tokens: CountTokens = default_count_tokens,
    sleep: Sleep = time.sleep,
    limits: BatchLimits = DEFAULT_LIMITS,
) -> BuildResult:
    """Make the index hold the segments of the source, with the vectors of this embedder.

    `cancel` is a callable that returns a bool, or an object with `is_set()` or
    `cancelled`. It is looked at before each conversation and between two requests.
    """
    check = _checker(cancel)
    model, dimensions = embedder.model, embedder.dimensions
    with store.building():
        old = _old_index(store)
        work = store.work_file(dimensions)
        work.repair()
        old_rows = _old_rows(old, dimensions)
        known = old_rows.keys() | work.hashes()
        reading = _read(source, model, dimensions, known, check, progress)
        tokens = _embed(embedder, reading.texts, work, count_tokens, limits, sleep, check, progress)
        rows = _rows(reading.segments, old, old_rows, work, dimensions, progress)
        store.replace(model, dimensions, reading.segments, rows)
        work.clear()
    reused = sum(1 for segment in reading.segments if segment.content_hash in known)
    now = {segment.segment_id for segment in reading.segments}
    removed = sum(1 for segment in old.segments if segment.segment_id not in now) if old is not None else 0
    result = BuildResult(
        segments=len(reading.segments),
        embedded=len(reading.segments) - reused,
        reused=reused,
        removed=removed,
        tokens=tokens,
    )
    logger.info(
        "index built: %d segments, %d embedded, %d reused, %d removed, %d tokens",
        result.segments, result.embedded, result.reused, result.removed, result.tokens,
    )
    return result


def estimate_build(
    source: MessageSource,
    store: IndexStore,
    embedder_facts: EmbedderFacts,
    count_tokens: CountTokens = default_count_tokens,
) -> BuildEstimate:
    """What a build with this model would embed now, and what that costs. It writes nothing."""
    dimensions = embedder_facts.dimensions
    known = _old_rows(_old_index(store), dimensions).keys() | store.work_file(dimensions).hashes()
    reading = _read(source, embedder_facts.model, dimensions, known, _checker(None), None)
    tokens = sum(count_tokens(text) for text in reading.texts.values())
    return BuildEstimate(
        segments=len(reading.segments),
        new_segments=sum(1 for segment in reading.segments if segment.content_hash not in known),
        tokens=tokens,
        usd=tokens * embedder_facts.usd_per_mtok / TOKENS_PER_PRICE,
    )
