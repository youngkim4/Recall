"""Reduce step: one call that writes the four sections.

It reads period summaries, the selected moments and exact figures. It never
reads a transcript. A reply that is empty, cut off or missing a section is an
error: the run ends with no report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from report_engine.calls import Caller
from report_engine.control import ProgressEvent
from report_engine.errors import LLMError, ReduceError
from report_engine.map_step import STATUS_REASONS
from report_engine.models import SECTION_KEYS, SECTION_TITLES, Moment, PeriodSummary, Section
from report_engine.plan import Plan
from report_engine.ports import LLMResult
from report_engine.prompts import REDUCE_SCHEMA, reduce_messages


def parse_sections(result: LLMResult) -> tuple[Section, ...]:
    """The four sections of a reduce reply. Raises ReduceError when one is missing."""
    if result.status != "completed":
        reason = STATUS_REASONS.get(result.status, f"the reply has status {result.status}")
        raise ReduceError(f"the reduce step gave no report: {reason}")
    parsed = result.parsed
    if not isinstance(parsed, Mapping):
        raise ReduceError("the reduce step gave no report: the reply is empty or not a JSON object")
    sections: list[Section] = []
    for key in SECTION_KEYS:
        text = parsed.get(key)
        if not isinstance(text, str) or not text.strip():
            raise ReduceError(f"the reduce step gave no report: section {key!r} is empty")
        sections.append(Section(key=key, title=SECTION_TITLES[key], markdown=text.strip()))
    return tuple(sections)


def write_sections(
    *,
    plan: Plan,
    caller: Caller,
    labels: Sequence[str],
    figures: str,
    periods: Sequence[PeriodSummary],
    moments: Sequence[Moment],
) -> tuple[Section, ...]:
    """Send the reduce request and check the reply. Raises ReduceError or ReportCancelled."""
    messages = reduce_messages(labels=labels, figures=figures, periods=periods, moments=moments)
    estimate = plan.meter.count("\n".join(message["content"] for message in messages))
    cap = plan.limits.reduce_input_cap
    if estimate > cap:
        raise ReduceError(
            f"the reduce request is too large for the model: {estimate} tokens, the limit is {cap}"
        )
    total = len(plan.periods)
    try:
        result = caller.send(
            messages,
            schema=REDUCE_SCHEMA,
            max_output_tokens=plan.limits.reduce_max_output,
            input_estimate=estimate,
            waiting=ProgressEvent(step="reduce", state="waiting", total=total),
            starting=ProgressEvent(step="reduce", state="start", total=total),
        )
    except LLMError as exc:
        raise ReduceError(f"the reduce call failed: {exc}") from exc
    return parse_sections(result)
