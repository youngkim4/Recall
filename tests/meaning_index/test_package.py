"""What the package gives by name, and what its import costs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import meaning_index

PUBLIC = {
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
}


def test_the_package_gives_the_names_of_the_contract() -> None:
    assert set(meaning_index.__all__) == PUBLIC
    assert all(hasattr(meaning_index, name) for name in PUBLIC)


def test_every_error_is_a_meaning_index_error() -> None:
    endings = ("Error", "Cancelled", "Limited", "Large")
    errors = [getattr(meaning_index, name) for name in PUBLIC if name.endswith(endings)]
    assert len(errors) == 14
    assert all(issubclass(error, meaning_index.MeaningIndexError) for error in errors)


def test_the_import_loads_no_provider_and_no_tokenizer() -> None:
    root = Path(meaning_index.__file__).resolve().parent.parent
    code = (
        "import sys, meaning_index;"
        "print(sorted(name for name in ('openai', 'tiktoken', 'httpx', 'pandas', 'archive') if name in sys.modules))"
    )
    done = subprocess.run([sys.executable, "-B", "-c", code], cwd=root, capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"
