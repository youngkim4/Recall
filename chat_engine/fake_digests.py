"""A scripted digest source for tests. It calls no model and uses no network.

It returns the periods of the script as they are. It does not apply the days
of the call, so a test can script a period that the engine must drop.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence, Union

from .cancel import CancelToken
from .ports import DigestCost, DigestMoment, DigestPeriod, DigestProgressCallback, DigestResult


class DigestStopped(RuntimeError):
    """The fake stopped a digest because its cancel token was cancelled."""


class DigestFailed(RuntimeError):
    """A digest that failed after it was paid for. Give it as `error`: `usd` is what the failed call cost."""

    def __init__(self, message: str = "the digest failed", *, usd: Any = 0.0) -> None:
        super().__init__(message)
        self.usd = usd


def moment(
    date: str, title: str, detail: str, quote: str = "", message_ids: Sequence[str] = ()
) -> DigestMoment:
    return {"date": date, "title": title, "detail": detail, "quote": quote, "message_ids": list(message_ids)}


def period(
    label: str,
    first_day: str,
    last_day: str,
    messages: int,
    summary: str,
    moments: Sequence[DigestMoment] = (),
) -> DigestPeriod:
    return {
        "label": label,
        "first_day": first_day,
        "last_day": last_day,
        "messages": int(messages),
        "summary": summary,
        "moments": [_copy_moment(row) for row in moments],
    }


def _copy_moment(row: DigestMoment) -> DigestMoment:
    return {**row, "message_ids": list(row["message_ids"])}


class FakeDigests:
    """Periods by conversation id. Every call is recorded in `calls`.

    usd: what `estimate` gives for the periods that are not cached.
    cached: True when every period is in the cache, or the number of periods that are.
    failed: the `periods_failed` of every digest.
    spent: the `usd` of every digest. None: the result has no `usd`.
    error: raised by `digest`. estimate_error: raised by `estimate`.
    during: called inside `digest`, before it returns. A test cancels there.
    honour_cancel: `digest` raises DigestStopped when its token is cancelled, before its next part.
    per_part: called with the number of each part that is read now, between its "start" and its "done".
    A test blocks there.

    `digest` reports each part to `progress`: "cached" for the cached parts,
    which come first, then "start" and "done" for each other part.
    """

    def __init__(
        self,
        periods: Mapping[str, Sequence[DigestPeriod]],
        *,
        usd: float = 0.0,
        cached: Union[bool, int] = False,
        failed: int = 0,
        spent: Any = None,
        error: Optional[Exception] = None,
        estimate_error: Optional[Exception] = None,
        during: Optional[Callable[[], None]] = None,
        honour_cancel: bool = True,
        per_part: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._periods = {str(key): list(rows) for key, rows in periods.items()}
        self._usd = float(usd)
        self._cached = cached
        self._failed = int(failed)
        self._spent = spent
        self._per_part = per_part
        self._error = error
        self._estimate_error = estimate_error
        self._during = during
        self._honour_cancel = honour_cancel
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def estimate(self, conversation_id: str, *, since: Optional[str], until: Optional[str]) -> DigestCost:
        self.calls.append(("estimate", {"conversation_id": conversation_id, "since": since, "until": until}))
        if self._estimate_error is not None:
            raise self._estimate_error
        count = len(self._periods.get(str(conversation_id), []))
        cached = self._cached_of(count)
        return {"periods": count, "periods_cached": cached, "usd": self._usd if cached < count else 0.0}

    def digest(
        self,
        conversation_id: str,
        *,
        since: Optional[str],
        until: Optional[str],
        cancel: CancelToken,
        progress: Optional[DigestProgressCallback] = None,
    ) -> DigestResult:
        self.calls.append(("digest", {"conversation_id": conversation_id, "since": since, "until": until}))
        rows = self._periods.get(str(conversation_id), [])
        self._read(rows, cancel, progress)
        if self._during is not None:
            self._during()
        if self._honour_cancel and cancel.cancelled:
            raise DigestStopped("the digest was cancelled")
        if self._error is not None:
            raise self._error
        copied = [period(**{**row, "moments": row["moments"]}) for row in rows]
        result: DigestResult = {"periods": copied, "periods_failed": self._failed if rows else 0}
        if self._spent is not None:
            result["usd"] = self._spent
        return result

    def _cached_of(self, count: int) -> int:
        return count if self._cached is True else min(count, int(self._cached))

    def _read(
        self, rows: Sequence[DigestPeriod], cancel: CancelToken, progress: Optional[DigestProgressCallback]
    ) -> None:
        """Report each part. A part that is read waits in `per_part`; a cancel stops before the next part."""
        total, cached = len(rows), self._cached_of(len(rows))
        for index, row in enumerate(rows, start=1):
            if self._honour_cancel and cancel.cancelled:
                raise DigestStopped("the digest was cancelled")
            states = ("cached",) if index <= cached else ("start", "done")
            for state in states:
                if state == "done" and self._per_part is not None:
                    self._per_part(index)
                if progress is not None:
                    progress({"part": index, "of": total, "label": row["label"], "state": state})
