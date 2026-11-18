"""A scripted digest source for tests. It calls no model and uses no network.

It returns the periods of the script as they are. It does not apply the days
of the call, so a test can script a period that the engine must drop.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from .cancel import CancelToken
from .ports import DigestCost, DigestMoment, DigestPeriod, DigestResult


class DigestStopped(RuntimeError):
    """The fake stopped a digest because its cancel token was cancelled."""


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
    cached: every period is in the cache, so the cost is 0.
    failed: the `periods_failed` of every digest.
    error: raised by `digest`. estimate_error: raised by `estimate`.
    during: called inside `digest`, before it returns. A test cancels there.
    honour_cancel: `digest` raises DigestStopped when its token is cancelled.
    """

    def __init__(
        self,
        periods: Mapping[str, Sequence[DigestPeriod]],
        *,
        usd: float = 0.0,
        cached: bool = False,
        failed: int = 0,
        error: Optional[Exception] = None,
        estimate_error: Optional[Exception] = None,
        during: Optional[Callable[[], None]] = None,
        honour_cancel: bool = True,
    ) -> None:
        self._periods = {str(key): list(rows) for key, rows in periods.items()}
        self._usd = float(usd)
        self._cached = cached
        self._failed = int(failed)
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
        if self._cached or not count:
            return {"periods": count, "periods_cached": count, "usd": 0.0}
        return {"periods": count, "periods_cached": 0, "usd": self._usd}

    def digest(
        self, conversation_id: str, *, since: Optional[str], until: Optional[str], cancel: CancelToken
    ) -> DigestResult:
        self.calls.append(("digest", {"conversation_id": conversation_id, "since": since, "until": until}))
        if self._during is not None:
            self._during()
        if self._honour_cancel and cancel.cancelled:
            raise DigestStopped("the digest was cancelled")
        if self._error is not None:
            raise self._error
        rows = self._periods.get(str(conversation_id), [])
        copied = [period(**{**row, "moments": row["moments"]}) for row in rows]
        return {"periods": copied, "periods_failed": self._failed if rows else 0}
