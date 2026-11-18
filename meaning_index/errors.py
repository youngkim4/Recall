"""Errors of meaning_index. Nothing fails without one of these or a warning.

No error holds the text of a message, a name, a handle or a conversation id.
"""

from __future__ import annotations


class MeaningIndexError(Exception):
    """Base class for every error that meaning_index raises."""


class SourceError(MeaningIndexError):
    """The message source failed, or gave a row that the package can not use."""


class ModelFactsError(MeaningIndexError):
    """The facts file does not hold the model, or the dimensions do not fit it."""


class StoreError(MeaningIndexError):
    """The files of the index could not be read or written."""


class IndexBusyError(StoreError):
    """Another build holds the folder of the index."""


class BuildCancelled(MeaningIndexError):
    """The caller stopped the build. The index is as it was.

    `embedded` and `tokens` say what the build paid for before it stopped.
    The work file keeps those vectors for the next build.
    """

    def __init__(self, message: str, *, embedded: int = 0, tokens: int = 0) -> None:
        super().__init__(message)
        self.embedded = embedded
        self.tokens = tokens


class EmbeddingError(MeaningIndexError):
    """The embedder failed. `status_code` and `code` are those of the service, when it gave them."""

    def __init__(self, message: str, *, status_code: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class EmbeddingKeyError(EmbeddingError):
    """The service refused the key."""


class EmbeddingRateLimited(EmbeddingError):
    """The service asks the caller to wait. `retry_after_s` is the wait that it names, or None."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_s: float | None = None,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, code=code)
        self.retry_after_s = retry_after_s


class EmbeddingTooLarge(EmbeddingError):
    """One request held more than the service takes."""


class EmbeddingNetworkError(EmbeddingError):
    """The service could not be reached, or it did not answer in time."""


class EmbeddingServiceError(EmbeddingError):
    """Every other failure of the embedder."""


class EmbeddingReplyError(EmbeddingServiceError):
    """The embedder answered, and the answer breaks the port: a wrong count, a wrong size, a number that is none."""
