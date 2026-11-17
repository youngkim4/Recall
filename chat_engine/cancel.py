"""A cancel token: one thread asks to stop, the engine and the model stream obey."""

from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[], None]] = []

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks = list(self._callbacks)
            self._callbacks = []
        for callback in callbacks:
            _run(callback)

    def on_cancel(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Run `callback` when cancelled. Returns a function that removes it."""
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(callback)
                return lambda: self._remove(callback)
        _run(callback)
        return lambda: None

    def wait(self, seconds: float) -> bool:
        """Sleep up to `seconds`. True when cancelled during the wait."""
        return self._event.wait(max(0.0, seconds))

    def _remove(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)


def _run(callback: Callable[[], None]) -> None:
    try:
        callback()
    except Exception:  # a failing hook must not block the cancel
        logger.exception("cancel callback failed")
