"""Search by meaning over the segments of the archive. See meaning_index/README.md.

The OpenAI adapter is not imported here: `meaning_index.openai_embedder`
loads the SDK, and a caller that only searches does not need it.
"""

from __future__ import annotations

from meaning_index.archive_source import ArchiveSource
from meaning_index.batching import BatchLimits
from meaning_index.build import build_index, estimate_build
from meaning_index.errors import (
    BuildCancelled,
    EmbeddingError,
    EmbeddingKeyError,
    EmbeddingNetworkError,
    EmbeddingRateLimited,
    EmbeddingReplyError,
    EmbeddingServiceError,
    EmbeddingTooLarge,
    IndexBusyError,
    IndexFormatError,
    MeaningIndexError,
    ModelFactsError,
    SourceError,
    StoreError,
)
from meaning_index.fake_embedder import FakeEmbedder
from meaning_index.model_facts import EmbedderFacts, embedder_facts
from meaning_index.ports import Embedder, MessageSource, SourceMessage
from meaning_index.search import MeaningIndex
from meaning_index.segments import cut_segments
from meaning_index.shapes import (
    BuildEstimate,
    BuildProgress,
    BuildResult,
    CutSegment,
    IndexMeta,
    IndexStatus,
    SearchHit,
    Segment,
)
from meaning_index.store import IndexStore
from meaning_index.tokens import default_count_tokens

__all__ = [
    "ArchiveSource",
    "BatchLimits",
    "BuildCancelled",
    "BuildEstimate",
    "BuildProgress",
    "BuildResult",
    "CutSegment",
    "Embedder",
    "EmbedderFacts",
    "EmbeddingError",
    "EmbeddingKeyError",
    "EmbeddingNetworkError",
    "EmbeddingRateLimited",
    "EmbeddingReplyError",
    "EmbeddingServiceError",
    "EmbeddingTooLarge",
    "FakeEmbedder",
    "IndexBusyError",
    "IndexFormatError",
    "IndexMeta",
    "IndexStatus",
    "IndexStore",
    "MeaningIndex",
    "MeaningIndexError",
    "MessageSource",
    "ModelFactsError",
    "SearchHit",
    "Segment",
    "SourceError",
    "SourceMessage",
    "StoreError",
    "build_index",
    "cut_segments",
    "default_count_tokens",
    "embedder_facts",
    "estimate_build",
]
