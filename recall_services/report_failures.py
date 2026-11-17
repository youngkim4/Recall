"""A report run that wrote no report, as one plain sentence and a code.

The report engine raises typed errors, and its model port says only the HTTP
status of a failure. `failure_of` puts the two together, so that the Jobs
screen can say why a run failed and what to do.
"""

from __future__ import annotations

from typing import Any

from report_engine import (
    AllPeriodsFailed,
    ConfigError,
    EmptyConversationError,
    LLMError,
    ReduceError,
    ReportCancelled,
    ReportEngineError,
    RunError,
    SourceError,
    StoreError,
)

AGAIN = "then run the report again."
NO_KEY = f"No API key is set. Add your OpenAI API key in Settings, {AGAIN}"
_UNAUTHORIZED, _FORBIDDEN, _NOT_FOUND, _TOO_MANY = 401, 403, 404, 429


class ReportFailure(Exception):
    """A run that wrote no report. The message is the sentence for the person.

    `code` is a fixed word for the client. `usage` is what the run spent until
    it stopped, when the engine could count it.
    """

    def __init__(self, code: str, sentence: str, usage: dict[str, Any] | None = None) -> None:
        super().__init__(sentence)
        self.code = code
        self.usage = usage


def _of_the_model(last: LLMError | None, provider: str, model: str) -> tuple[str, str] | None:
    """The code and the sentence for the last failure of the model port, when it says what to do."""
    if last is None:
        return None
    status, said = last.status_code, str(last)
    if status == _UNAUTHORIZED:
        return "key_rejected", f"{provider} rejected the API key. Check the key in Settings, {AGAIN}"
    if status == _FORBIDDEN:
        return (
            "key_rejected",
            f"{provider} does not allow this API key to use the model. Check the key in Settings, {AGAIN}",
        )
    if status == _NOT_FOUND:
        return "unknown_model", f"{provider} does not know the model {model}. Choose another level in Settings."
    if status == _TOO_MANY:
        return (
            "rate_limit",
            f"{provider} is limiting requests, or the account has no credit left. Wait a moment, {AGAIN}",
        )
    if status is None and "Timeout" in said:
        return "timeout", "The model took too long to answer. Run the report again."
    if status is None and "Connection" in said:
        return "offline", f"{provider} could not be reached. Check the internet connection, {AGAIN}"
    return None


def _usage_of(error: BaseException) -> dict[str, Any] | None:
    usage = getattr(error, "usage", None)
    return None if usage is None else usage.model_dump(mode="json")


def failure_of(error: ReportEngineError, last: LLMError | None, provider: str, model: str) -> ReportFailure:
    """The failure for the person, from the error of the engine and the last failure of the model port."""
    usage = _usage_of(error)
    if isinstance(error, EmptyConversationError):
        return ReportFailure(
            "empty", "This conversation has no messages with text in these dates. No report was written."
        )
    if isinstance(error, ConfigError):
        return ReportFailure(
            "too_large", "The model can not read this conversation with its limits. No report was written."
        )
    if isinstance(error, SourceError):
        return ReportFailure(
            "archive_error", "The messages of this conversation could not be read. No report was written."
        )
    if isinstance(error, StoreError):
        # the engine writes only its notes of the parts that it read: the report itself is saved by the caller
        return ReportFailure(
            "cache_unreadable",
            "What earlier report runs read could not be read or kept. No report was written.",
            usage,
        )
    if isinstance(error, ReportCancelled):
        return ReportFailure("cancelled", "The report was stopped. What was read is kept for the next run.", usage)
    told = _of_the_model(last, provider, model)
    if told is not None and isinstance(error, (AllPeriodsFailed, ReduceError)):
        return ReportFailure(told[0], told[1], usage)
    if isinstance(error, AllPeriodsFailed):
        return ReportFailure(
            "model_error", "The model could not read the conversation. No report was written. Run the report again.", usage
        )
    if isinstance(error, ReduceError):
        return ReportFailure(
            "report_not_written",
            "The model read the conversation but did not write the report. What it read is kept: run the report again.",
            usage,
        )
    if isinstance(error, RunError):
        return ReportFailure(
            "run_failed", "The report run failed. What was read is kept for the next run.", usage
        )
    return ReportFailure("run_failed", "The report run failed. No report was written.", usage)
