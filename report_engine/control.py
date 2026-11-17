"""How a caller watches and stops a run, and the clock a run reads."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Literal

ProgressStep = Literal["prepare", "map", "reduce", "done"]
ProgressState = Literal["start", "cached", "waiting", "done", "failed"]


@dataclass(frozen=True)
class ProgressEvent:
    """One step of a run. `period` is set for map events; `index` counts from 1."""

    step: ProgressStep
    state: ProgressState
    period: str | None = None
    index: int = 0
    total: int = 0


ProgressCallback = Callable[[ProgressEvent], None]


class CancelToken:
    """Set from any thread. The run stops before its next model call."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_report_id(created: datetime) -> str:
    """Sorts by time; the random tail keeps two runs in one second apart."""
    return f"{created.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"


@dataclass(frozen=True)
class Runtime:
    """Time and identity sources. Tests replace them; production uses the defaults."""

    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], datetime] = _utc_now
    new_id: Callable[[datetime], str] = field(default=new_report_id)
