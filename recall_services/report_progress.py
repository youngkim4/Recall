"""The progress of a report run in the shape that the Jobs screen reads.

A listener gets small dicts. Each has `progress` (`step`, `of`, `label`), a
line for the log of the job under `log`, or both:

    {"progress": {"step": 3, "of": 7, "label": "Reading part 2 of 4"}, "log": "Reading Feb 2023 (part 2 of 4)"}

The steps of a run: the conversation is read, each part goes to the model,
the report is written, the report is saved. The number of parts is known
only after the conversation is read, so the first update counts three steps.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from report_engine import ProgressEvent

logger = logging.getLogger(__name__)

Listener = Callable[[dict[str, Any]], None]
# reading the conversation, writing the report, saving it
STEPS_BESIDE_THE_PARTS = 3
WAITING = "Waiting for the rate limit of the model"


class Teller:
    """Turns the events of the engine into updates for a listener. A listener that fails does not stop the run."""

    def __init__(self, listener: Listener | None) -> None:
        self._listener = listener
        self._warned = False

    def tell(self, *, step: tuple[int, int, str] | None = None, log: str = "") -> None:
        if self._listener is None:
            return
        update: dict[str, Any] = {}
        if step is not None:
            update["progress"] = {"step": step[0], "of": step[1], "label": step[2]}
        if log:
            update["log"] = log
        try:
            self._listener(update)
        except Exception:  # noqa: BLE001 - the run is worth more than its progress line
            if not self._warned:
                logger.exception("the progress listener of a report run failed")
                self._warned = True

    def engine_event(self, event: ProgressEvent) -> None:
        """One event of `run_report`."""
        of = event.total + STEPS_BESIDE_THE_PARTS
        if event.step == "prepare" and event.state == "start":
            self.tell(step=(1, STEPS_BESIDE_THE_PARTS, "Reading the conversation"), log="Reading the conversation")
        elif event.step == "map":
            self._part(event, of)
        elif event.step == "reduce" and event.state == "start":
            self.tell(step=(of - 1, of, "Writing the report"), log="Writing the report")
        elif event.step == "reduce" and event.state == "waiting":
            self.tell(log=WAITING)
        elif event.step == "reduce" and event.state == "failed":
            self.tell(log="The report could not be written")
        elif event.step == "done":
            self.tell(step=(of, of, "Saving the report"))

    def _part(self, event: ProgressEvent, of: int) -> None:
        place = f"part {event.index} of {event.total}"
        step = (1 + event.index, of, f"Reading {place}")
        if event.state == "start":
            self.tell(step=step, log=f"Reading {event.period} ({place})")
        elif event.state == "cached":
            self.tell(step=step, log=f"{event.period} was read in an earlier run ({place})")
        elif event.state == "waiting":
            self.tell(log=WAITING)
        elif event.state == "failed":
            self.tell(log=f"{event.period} could not be read ({place})")
