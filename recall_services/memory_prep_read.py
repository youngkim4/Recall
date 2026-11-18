"""The reading of the `memory` job: one task for each conversation, the progress of the parts, and the counts.

The tasks run in threads (`report_engine.run_in_parallel`). Each one builds
the digest of one conversation through the pool of the job. The progress
counts each part that ends, read or failed, over all the conversations.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from recall_services import memory_prep_words as words
from recall_services.report_failures import ReportFailure
from report_engine import CallPool, ProgressEvent, Usage
from report_engine.chat_digest import ConversationDigest

# the states of a part that end it
ENDED = frozenset({"done", "failed"})

Build = Callable[[str, Callable[[ProgressEvent], None]], ConversationDigest]


@dataclass(frozen=True)
class Planned:
    """One conversation of the memory, as the estimate found it."""

    conversation: str
    periods: int
    periods_cached: int
    usd: float

    @property
    def not_read(self) -> int:
        return self.periods - self.periods_cached


@dataclass(frozen=True)
class Read:
    """What the job got for one conversation: its digest, or the failure of `digests.build`."""

    planned: Planned
    digest: ConversationDigest | None = None
    failure: ReportFailure | None = None


class Reading:
    """The state that the tasks of one job share, under one lock."""

    def __init__(
        self, teller: words.Teller, total: int, conversations: int, cancel: Any, pool: CallPool, build: Build
    ) -> None:
        self.total = total
        self._teller = teller
        self._conversations = conversations
        self._cancel = cancel
        self._pool = pool
        self._build = build
        self._lock = threading.Lock()
        self._done = 0
        self._ended = 0

    def cancelled(self) -> bool:
        return bool(getattr(self._cancel, "cancelled", False))

    def stopped(self) -> bool:
        """True when no new conversation starts: the pool stopped or the job was cancelled."""
        return self._pool.stopped or self.cancelled()

    def _heard(self, event: ProgressEvent) -> None:
        if event.step != "map" or event.state not in ENDED:
            return
        with self._lock:
            self._done += 1
            self._teller.tell(step="read", done=min(self._done, self.total), total=self.total)

    def _told(self, read: Read) -> None:
        with self._lock:
            self._ended += 1
            if read.digest is None:
                line = words.one_failed(self._ended, self._conversations)
            else:
                found = read.digest
                line = words.one_read(
                    found.periods_built, found.periods_cached, found.periods_failed, self._ended, self._conversations
                )
            self._teller.tell(log=line)

    def task(self, planned: Planned) -> Callable[[], Read]:
        """The task that reads one conversation. A failure that every conversation would have stops the pool."""

        def work() -> Read:
            try:
                read = Read(planned, digest=self._build(planned.conversation, self._heard))
            except ReportFailure as failure:
                if failure.code in words.FATAL:
                    self._pool.stop()
                read = Read(planned, failure=failure)
            except Exception:
                self._pool.stop()
                raise
            self._told(read)
            return read

        return work


def _usage(read: Read) -> Usage:
    if read.digest is not None:
        return read.digest.usage
    spent = read.failure.usage if read.failure is not None else None
    return Usage.model_validate(spent) if spent else Usage()


def tally(planned: Sequence[Planned], reads: Sequence[Read], skipped: int = 0) -> dict[str, Any]:
    """The result of the job. A conversation that was read whole before counts its parts as cached.
    A conversation that could not be planned (`skipped`) counts as failed."""
    whole = [one for one in planned if one.not_read == 0]
    digests = [read.digest for read in reads if read.digest is not None]
    failed = [read for read in reads if read.digest is None]
    usage = Usage()
    for read in reads:
        usage = usage.plus(_usage(read))
    return {
        "conversations": len(planned) + skipped,
        "conversationsRead": len(whole) + sum(found.periods_failed == 0 for found in digests),
        "conversationsFailed": len(failed) + skipped,
        "periodsBuilt": sum(found.periods_built for found in digests),
        "periodsCached": sum(one.periods for one in whole) + sum(found.periods_cached for found in digests),
        "periodsFailed": sum(found.periods_failed for found in digests) + sum(read.planned.not_read for read in failed),
        "usd": float(usage.cost_usd),
        "usage": usage.model_dump(mode="json"),
    }
