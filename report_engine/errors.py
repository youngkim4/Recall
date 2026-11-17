"""Errors the engine raises. Nothing fails without one of these or a warning."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from report_engine.models import Usage


class ReportEngineError(Exception):
    """Base class for every error raised by report_engine."""


class ConfigError(ReportEngineError):
    """Options and model facts do not fit together."""


class SourceError(ReportEngineError):
    """The source port returned data the engine can not use."""


class EmptyConversationError(ReportEngineError):
    """The conversation has no message with text in the given scope."""


class LLMError(ReportEngineError):
    """The model port failed after its own retries."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_s: float | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_s = retry_after_s
        self.attempts = attempts


class RunError(ReportEngineError):
    """A run ended without a report. `usage` holds what the run spent."""

    def __init__(self, message: str, usage: Usage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class ReportCancelled(RunError):
    """The cancel token stopped the run between two model calls."""


class AllPeriodsFailed(RunError):
    """No period produced a summary, so there is nothing to write from."""


class ReduceError(RunError):
    """The reduce reply was empty, cut off or not in the expected shape."""


class StoreError(ReportEngineError):
    """A report could not be written or read."""


class ReportNotFound(StoreError):
    """No report with this id exists for this conversation."""
