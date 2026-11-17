"""Typed failures. The engine raises them; the caller decides what to show.

`message` is safe to show. `detail` is for the local log only: it can hold
what the provider said.
"""

from __future__ import annotations

from typing import Any, Optional


class ChatEngineError(Exception):
    code = "chat_error"
    default_message = "The question could not be answered."

    def __init__(
        self,
        message: str = "",
        *,
        detail: str = "",
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message or self.default_message)
        self.message = message or self.default_message
        self.detail = detail
        self.retry_after = retry_after

    def to_event(self) -> dict[str, Any]:
        event: dict[str, Any] = {"type": "error", "code": self.code, "error": self.message}
        if self.retry_after is not None:
            event["retry_after"] = self.retry_after
        return event


class NoApiKeyError(ChatEngineError):
    code = "no_key"
    default_message = "No API key is set."


class KeyRejectedError(ChatEngineError):
    code = "key_rejected"
    default_message = "The model provider rejected the API key."


class UnknownModelError(ChatEngineError):
    code = "unknown_model"
    default_message = "The model provider does not know this model."


class RateLimitedError(ChatEngineError):
    code = "rate_limit"
    default_message = "The model provider is limiting requests. Try again soon."


class OfflineError(ChatEngineError):
    code = "offline"
    default_message = "The model provider could not be reached."


class ModelTimeoutError(ChatEngineError):
    code = "timeout"
    default_message = "The model took too long to answer."


class ModelError(ChatEngineError):
    """Any other failure of the model call."""

    code = "model_error"
    default_message = "The model call failed."


class ArchiveError(ChatEngineError):
    code = "archive_error"
    default_message = "The message archive could not be read."


MODEL_ERRORS: tuple[type[ChatEngineError], ...] = (
    NoApiKeyError,
    KeyRejectedError,
    UnknownModelError,
    RateLimitedError,
    OfflineError,
    ModelTimeoutError,
    ModelError,
)
