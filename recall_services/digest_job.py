"""The `digest` job: one conversation read for chat, outside a question.

A question reads at most a few parts of a conversation (`DIGEST_PARTS_LIMIT`
in `memory.py`). A longer conversation is read here, by a job. It uses the
model of a digest, so a later question finds the parts in the notes. It has
no limit of parts, and the model port of a report.

`run` gives the counts and the cost of the run. A listener gets the progress
in the shape of a report job: `{"progress": {"step", "of", "label"}}`, with a
line for the job under `log`. A line holds the label of a part and counts. It
holds no text of a message, no name and no handle.
"""

from __future__ import annotations

from typing import Any

from archive import Archive
from recall_services import digests, memory
from recall_services.names import Names
from recall_services.report_progress import Listener, Teller
from report_engine import ProgressEvent


class _Teller(Teller):
    """The events of a digest as the updates of a job: the conversation is read, then each part."""

    def engine_event(self, event: ProgressEvent) -> None:
        if event.step == "prepare" and event.state == "start":
            self.tell(step=(1, 1, "Reading the conversation"), log="Reading the conversation")
        elif event.step == "map":
            self._part(event, event.total + 1)


def _parts(count: int) -> str:
    return f"{count:,} part{'' if count == 1 else 's'}"


def run(
    conversation_id: str,
    since: str | None = None,
    until: str | None = None,
    on_progress: Listener | None = None,
    cancel: Any = None,
    *,
    archive: Archive,
    names: Names,
    **more: Any,
) -> dict[str, Any]:
    """Read one conversation for chat. Gives `conversation`, `periods`, `periodsBuilt`,
    `periodsCached`, `periodsFailed`, `usd` and `usage`.

    `more` goes to `digests.build`: tests give the places, the options, the
    counter and a model port. Raises what `digests.build` raises: a
    `ReportFailure` with one plain sentence, a ValueError for a missing
    conversation or a bad day.
    """
    teller = _Teller(on_progress)
    settings = {"model": memory.DIGEST_MODEL, **more}
    found = digests.build(
        conversation_id, since, until, cancel, teller.engine_event, archive=archive, names=names, **settings
    )
    teller.tell(
        log=f"The conversation is read for chat: {_parts(found.periods_built)} read now, "
        f"{found.periods_cached:,} from earlier runs, {found.periods_failed:,} not read."
    )
    return {
        "conversation": found.conversation_id,
        "periods": len(found.periods),
        "periodsBuilt": found.periods_built,
        "periodsCached": found.periods_cached,
        "periodsFailed": found.periods_failed,
        "usd": float(found.usage.cost_usd),
        "usage": found.usage.model_dump(mode="json"),
    }
