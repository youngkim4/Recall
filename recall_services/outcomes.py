"""What the two ways to fill the archive answer. Each answer is a type of its own.

`status` is a fixed word for the client. `reason` is one plain sentence for
the person, or "" when there is nothing to say. `changed` says whether the
archive holds other messages than before.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

ARCHIVE_KEPT = "The archive was not changed."
BUSY = "An import is already running."
EnsureStatus = Literal["current", "built", "kept", "no_export", "busy", "cancelled", "failed"]


@dataclass(frozen=True)
class EnsureResult:
    """The answer of `ensure_from_export`.

    current: the archive already holds this export. built: the export was loaded.
    kept: the archive holds a pull from Messages, which an export never replaces.
    no_export, busy, cancelled, failed: nothing was changed, and `reason` says why.
    """

    status: EnsureStatus
    reason: str = ""
    counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.status == "built"

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "changed": self.changed, "reason": self.reason, "counts": dict(self.counts)}


@dataclass(frozen=True)
class PullDone:
    """The pull ran: the archive holds what the database holds."""

    counts: Mapping[str, int]
    messages: int
    conversations: int
    status: ClassVar[str] = "done"
    changed: ClassVar[bool] = True
    reason: ClassVar[str] = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "changed": self.changed,
            "reason": self.reason,
            "messages": self.messages,
            "conversations": self.conversations,
            "counts": dict(self.counts),
        }


@dataclass(frozen=True)
class PullRefused:
    """The database lacks more of the stored messages than the caller allows. Nothing was changed.

    The caller can ask the person and pull again with a larger `max_removed_share`.
    """

    would_remove: int
    stored: int
    max_share: float
    status: ClassVar[str] = "removal_limit"
    changed: ClassVar[bool] = False

    @property
    def reason(self) -> str:
        return (
            f"This database lacks {self.would_remove:,} of the {self.stored:,} messages in the archive. "
            f"That is more than {self.max_share:.0%}. {ARCHIVE_KEPT}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "changed": self.changed,
            "reason": self.reason,
            "wouldRemove": self.would_remove,
            "stored": self.stored,
            "maxShare": self.max_share,
        }


@dataclass(frozen=True)
class _Unchanged:
    changed: ClassVar[bool] = False
    status: ClassVar[str] = ""

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "changed": self.changed, "reason": getattr(self, "reason", "")}


@dataclass(frozen=True)
class PullBusy(_Unchanged):
    """Another import writes the archive."""

    status: ClassVar[str] = "busy"
    reason: ClassVar[str] = BUSY


@dataclass(frozen=True)
class PullCancelled(_Unchanged):
    """The caller stopped the pull. The archive is as it was."""

    status: ClassVar[str] = "cancelled"
    reason: ClassVar[str] = f"The update was stopped. {ARCHIVE_KEPT}"


@dataclass(frozen=True)
class PullFailed(_Unchanged):
    """The pull did not run or did not end. `reason` is for the person, `detail` for the log."""

    reason: str = ""
    detail: str = ""
    status: ClassVar[str] = "failed"


PullResult = PullDone | PullRefused | PullBusy | PullCancelled | PullFailed
