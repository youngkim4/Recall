"""The progress of a digest as events of the stream.

A digest source reports each change of a part (`ports.DigestProgress`).
`progress_events` makes one `progress` event of each report in the right form:

    {"type": "progress", "tool": "conversation_digest", "conversation": <label as shown>,
     "part": i, "of": n, "label": <label of the part>, "state": "start"|"done"|"cached"|"failed"}

A report in another form is dropped, and so is a repeat of a report. The
tool box masks `conversation` and `label` before the event goes out.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

STATES = frozenset({"start", "done", "cached", "failed"})
TOOL = "conversation_digest"
Report = tuple[int, int, str, str]


def progress_events(conversation: str) -> Callable[[object], Optional[dict[str, Any]]]:
    """The event of each report of one digest. `conversation` is its label as the reader sees it."""
    seen: set[Report] = set()

    def event(raw: object) -> Optional[dict[str, Any]]:
        report = _checked(raw)
        if report is None or report in seen:
            return None
        seen.add(report)
        part, of, label, state = report
        return {
            "type": "progress", "tool": TOOL, "conversation": conversation,
            "part": part, "of": of, "label": label, "state": state,
        }  # fmt: skip

    return event


def _whole(value: object) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _checked(raw: object) -> Optional[Report]:
    """The report as (part, of, label, state). None when it has another form."""
    if not isinstance(raw, Mapping):
        return None
    part, of = _whole(raw.get("part")), _whole(raw.get("of"))
    label, state = raw.get("label"), raw.get("state")
    if part is None or of is None or not 1 <= part <= of:
        return None
    if not isinstance(label, str) or state not in STATES:
        return None
    return part, of, label, str(state)
