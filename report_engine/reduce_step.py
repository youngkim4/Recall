"""Reduce step: one call that writes the four sections.

It reads period summaries, the selected moments and exact figures. It never
reads a transcript. A reply that is empty, cut off or missing a section is an
error: the run ends with no report.

When the summaries and moments are too large for the model, they are cut to
fit before the call, and the report gets a warning that says so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from report_engine.calls import Caller
from report_engine.control import ProgressEvent
from report_engine.errors import LLMError, ReduceError
from report_engine.map_step import STATUS_REASONS
from report_engine.markdown import clean_section
from report_engine.models import SECTION_KEYS, SECTION_TITLES, Moment, PeriodSummary, Section
from report_engine.plan import Plan
from report_engine.ports import ChatMessage, LLMResult
from report_engine.prompts import REDUCE_SCHEMA, reduce_fixed_text, reduce_messages
from report_engine.reduce_fit import Fitted, fit_reduce
from report_engine.transcript import scrub

MAX_FIT_ROUNDS = 6
FIT_SLACK_TOKENS = 8


@dataclass(frozen=True)
class Reduced:
    sections: tuple[Section, ...]
    warnings: tuple[str, ...] = ()


def parse_sections(result: LLMResult) -> tuple[Section, ...]:
    """The four sections of a reduce reply. Raises ReduceError when one is missing.
    Links, images and HTML are taken out of the text: a report loads nothing
    from the network when it is shown."""
    if result.status != "completed":
        reason = STATUS_REASONS.get(result.status, f"the reply has status {result.status}")
        raise ReduceError(f"the reduce step gave no report: {reason}")
    parsed = result.parsed
    if not isinstance(parsed, Mapping):
        raise ReduceError("the reduce step gave no report: the reply is empty or not a JSON object")
    sections: list[Section] = []
    for key in SECTION_KEYS:
        text = parsed.get(key)
        cleaned = clean_section(scrub(text)) if isinstance(text, str) else ""
        if not cleaned:
            raise ReduceError(f"the reduce step gave no report: section {key!r} is empty")
        sections.append(Section(key=key, title=SECTION_TITLES[key], markdown=cleaned))
    return tuple(sections)


def _fit_warnings(fitted: Fitted) -> tuple[str, ...]:
    warnings: list[str] = []
    if fitted.shortened:
        many = fitted.shortened != 1
        warnings.append(
            f"{fitted.shortened} period summar{'ies were' if many else 'y was'} shortened "
            "to fit the model's input limit."
        )
    if fitted.left_out:
        many = fitted.left_out != 1
        warnings.append(
            f"{fitted.left_out} moment{'s were' if many else ' was'} not shown to the model "
            "when it wrote the sections, to fit its input limit."
        )
    return tuple(warnings)


def _room(plan: Plan, with_moments: bool) -> int:
    """Tokens left for period and moment lines once the fixed text is counted."""
    fixed = plan.meter.count(reduce_fixed_text(plan.names, plan.figures, with_moments))
    return plan.limits.reduce_input_cap - fixed


def _fit_request(
    plan: Plan, periods: Sequence[PeriodSummary], moments: Sequence[Moment]
) -> tuple[Fitted, list[ChatMessage], int]:
    """Lines cut to fit, the request made from them, and its size. The size is
    measured on the request itself. A tokenizer can count a joined text higher
    than its lines: when the request is still too large, the fit runs again
    with the tokens it used last time, less the excess."""
    cap = plan.limits.reduce_input_cap
    room = _room(plan, bool(moments))
    estimate = cap + 1
    for _ in range(MAX_FIT_ROUNDS):
        fitted = fit_reduce(periods, moments, room, plan.meter)
        messages = reduce_messages(
            labels=plan.names, figures=plan.figures, periods=fitted.periods, moments=fitted.moments
        )
        estimate = plan.meter.count("\n".join(message["content"] for message in messages))
        if estimate <= cap:
            return fitted, messages, estimate
        room = fitted.tokens - (estimate - cap) - FIT_SLACK_TOKENS
    raise ReduceError(
        f"the reduce request is too large for the model: {estimate} tokens, the limit is {cap}"
    )


def write_sections(
    *,
    plan: Plan,
    caller: Caller,
    periods: Sequence[PeriodSummary],
    moments: Sequence[Moment],
) -> Reduced:
    """Send the reduce request and check the reply. Raises ReduceError or ReportCancelled."""
    fitted, messages, estimate = _fit_request(plan, periods, moments)
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
    return Reduced(sections=parse_sections(result), warnings=_fit_warnings(fitted))
