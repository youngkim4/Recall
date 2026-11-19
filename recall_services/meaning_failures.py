"""A status, an estimate or a build of the index that gave no result, as one plain sentence and a code.

`meaning_index` raises typed errors. `failure_of` turns each one into the
sentence for the person. The sentence holds nothing of what the error said.
"""

from __future__ import annotations

import logging

from archive import ArchiveError as ArchiveFault
from meaning_index import (
    BuildCancelled,
    EmbeddingError,
    EmbeddingKeyError,
    EmbeddingNetworkError,
    EmbeddingRateLimited,
    IndexBusyError,
    IndexFormatError,
    SourceError,
    StoreError,
)
from recall_services import keys, models
from report_engine import SourceError as ReportSourceError

logger = logging.getLogger(__name__)

# the embedder of the app is the adapter of this provider, named as every sentence names it
PROVIDER = models.provider_name("openai")
AGAIN = "then build the index again."
NO_KEY = f"{keys.NO_KEY}, {AGAIN}"
KEY_REJECTED = f"{PROVIDER} refused the API key. {keys.CHECK_KEY}, {AGAIN}"
RATE_LIMIT = f"{PROVIDER} is limiting requests, or the account has no credit left. Wait a moment, {AGAIN}"
OFFLINE = f"{PROVIDER} could not be reached. Check the internet connection, {AGAIN}"
MODEL_ERROR = "The embedding model did not answer. What was read is kept: build the index again."
BUSY = "The index is being built now. Wait until that ends."
CANCELLED = "The build was stopped. What was read is kept for the next build."
ARCHIVE_UNREADABLE = f"The message archive could not be read. Update the archive, {AGAIN}"
INDEX_NEWER = "A newer version of Recall made this index. Update Recall, then try again."
INDEX_ERROR = f"The index could not be written. Check the free space on the disk, {AGAIN}"
BUILD_FAILED = "The index could not be built. Build it again."
_TOO_MANY = 429


class MeaningFailure(Exception):
    """A status, an estimate or a build that gave no result. The message is the sentence for the person."""

    def __init__(self, code: str, sentence: str) -> None:
        super().__init__(sentence)
        self.code = code


class NoKey(EmbeddingKeyError):
    """No key is set. No request was made."""


def failure_of(error: BaseException) -> MeaningFailure:
    """The failure for the person. It holds nothing of what the error said."""
    if isinstance(error, NoKey):
        return MeaningFailure("no_key", NO_KEY)
    if isinstance(error, EmbeddingKeyError):
        return MeaningFailure("key_rejected", KEY_REJECTED)
    if isinstance(error, EmbeddingRateLimited) or getattr(error, "status_code", None) == _TOO_MANY:
        return MeaningFailure("rate_limit", RATE_LIMIT)
    if isinstance(error, EmbeddingNetworkError):
        return MeaningFailure("offline", OFFLINE)
    if isinstance(error, EmbeddingError):
        return MeaningFailure("model_error", MODEL_ERROR)
    if isinstance(error, IndexBusyError):
        return MeaningFailure("busy", BUSY)
    if isinstance(error, BuildCancelled):
        return MeaningFailure("cancelled", CANCELLED)
    if isinstance(error, (SourceError, ReportSourceError, ArchiveFault)):
        return MeaningFailure("archive_error", ARCHIVE_UNREADABLE)
    if isinstance(error, IndexFormatError):
        return MeaningFailure("index_newer", INDEX_NEWER)
    if isinstance(error, StoreError):
        return MeaningFailure("index_error", INDEX_ERROR)
    return MeaningFailure("build_failed", BUILD_FAILED)


def failed(error: BaseException, doing: str) -> MeaningFailure:
    """The failure for the person, and one line for the log: the type of the error and the facts of the service.

    The caller does not chain the error: what an archive raises can hold an id.
    """
    facts = ""
    if isinstance(error, EmbeddingError):
        facts = f", HTTP {error.status_code}, code {error.code}"
    logger.warning("%s failed: %s%s", doing, type(error).__name__, facts)
    return failure_of(error)
