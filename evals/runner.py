"""Ask each question through `recall_services.chat.stream`, the way the app asks, and score it.

A follow-up first asks its `prior` question. The second question then gets
the history rows that the app sends (`historyRows` in
app/src/lib/chatMessages.ts): the user row with its `contact`, then the
answer. The scores are of the second answer; the tokens and the money count
both turns, since both are needed.

The model factory is a port: the run by hand gives `chat.default_llm`, a
test gives a scripted model. No question is judged when it ended in an
error: the error is recorded and the judge is not paid for.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial
from typing import Any

from archive import Archive
from chat_engine.labeler import MaskingLabeler
from chat_engine.ports import LLM
from evals.judge import Judge, JudgeCase, Verdict
from evals.question_types import Question
from evals.recorder import CallRecord, Recorder, price, references
from evals.score import Metrics, citation_scores, cited_guids
from evals.world.people import NOW, unsaved_handles
from recall_services import chat
from recall_services.models import ChatModelConfig
from recall_services.names import Names

Factory = Callable[[ChatModelConfig], LLM]
# the start of the error of a question whose answer came and whose judge failed
JUDGE_ERROR = "judge: "

@dataclass(frozen=True)
class Settings:
    model: str
    effort: str = ""
    clock: datetime = NOW
    unsaved: tuple[str, ...] = unsaved_handles()  # handles an answer must never hold


@dataclass(frozen=True)
class Turn:
    """One question as the app asked it: the payload of `done`, or the error."""

    payload: Mapping[str, Any] | None
    error_code: str
    calls: tuple[CallRecord, ...]
    wall_seconds: float

    @property
    def answer(self) -> str:
        return str(self.payload.get("answer") or "") if self.payload else ""


@dataclass(frozen=True)
class Result:
    question: Question
    answer: str
    prior_answer: str
    cited: tuple[str, ...]
    verdict: Verdict
    metrics: Metrics
    round: int = 0  # with --repeat, which pass over the questions this is, from 0


def ask(
    text: str,
    history: Sequence[Mapping[str, Any]],
    *,
    archive: Archive,
    names: Names,
    settings: Settings,
    factory: Factory,
    contact: str = "",
) -> Turn:
    recorder = Recorder(factory)
    started = time.monotonic()
    last: Mapping[str, Any] = {}
    for event in chat.stream(
        text, contact, settings.model, list(history), archive=archive, names=names,
        reasoning_effort=settings.effort, llm_factory=recorder, clock=lambda: settings.clock,
    ):  # fmt: skip
        last = event
    wall = time.monotonic() - started
    if last.get("type") == "done":
        return Turn(dict(last["payload"]), "", recorder.calls, wall)
    return Turn(None, str(last.get("code") or "no_answer"), recorder.calls, wall)


def history_rows(question: str, contact: str, answer: str) -> tuple[dict[str, str], ...]:
    """The rows the app sends after one question and its answer."""
    return ({"role": "user", "content": question, "contact": contact}, {"role": "assistant", "content": answer})


def _cost(model: str, turns: Iterable[Turn]) -> float:
    total = 0.0
    for turn in turns:
        total += sum(price(model, call.usage) for call in turn.calls)
        usage = turn.payload.get("usage") if turn.payload else None
        total += float((usage or {}).get("digest_usd") or 0.0)
    return total


def _leaked(answer: str, unsaved: tuple[str, ...]) -> bool:
    return MaskingLabeler(known_handles=unsaved).redact(answer) != answer


def _metrics(settings: Settings, turns: tuple[Turn, ...], cited: tuple[str, ...], question: Question, verdict: Verdict) -> Metrics:
    last = turns[-1]
    recall, precision = citation_scores(question.kind, question.gold_guids, cited)
    payload = last.payload or {}
    calls = [call for turn in turns for call in turn.calls]
    raw = last.calls[-1].text if last.calls else ""
    error = last.error_code or (f"{JUDGE_ERROR}{verdict.error}" if verdict.error else "")
    return Metrics(
        correct=verdict.correct,
        score=verdict.score,
        citation_recall=recall,
        citation_precision=precision,
        citations=len(payload.get("citations") or []),
        unverified_quotes=len(payload.get("unverifiedQuotes") or []),
        invalid_references=max(0, len(references(raw)) - len(payload.get("citations") or [])),
        leaked_handle=_leaked(last.answer, settings.unsaved),
        limited=bool(payload.get("limited")),
        truncated=bool(payload.get("truncated")),
        model_calls=len(calls),
        tool_calls=sum(call.tool_calls for call in calls),
        input_tokens=sum(call.usage.input_tokens for call in calls),
        cached_input_tokens=sum(call.usage.cached_input_tokens for call in calls),
        output_tokens=sum(call.usage.output_tokens for call in calls),
        reasoning_tokens=sum(call.usage.reasoning_tokens for call in calls),
        chat_usd=_cost(settings.model, turns),
        judge_usd=verdict.cost_usd,
        wall_seconds=sum(turn.wall_seconds for turn in turns),
        error=error,
    )


def run_question(
    question: Question, *, archive: Archive, names: Names, settings: Settings, factory: Factory, judge: Judge
) -> Result:
    asking = partial(ask, archive=archive, names=names, settings=settings, factory=factory, contact=question.contact)
    turns: tuple[Turn, ...] = ()
    history: tuple[dict[str, str], ...] = ()
    if question.prior:
        first = asking(question.prior, ())
        turns = (first,)
        history = history_rows(question.prior, question.contact, first.answer) if first.payload else ()
    turn = asking(question.text, history)
    turns = (*turns, turn)
    cited = cited_guids((turn.payload or {}).get("citations") or [], archive)
    if turn.payload is None:
        verdict = Verdict(False, 0.0, error="")
    else:
        case = JudgeCase(question.text, question.kind, question.gold, question.key_facts, turn.answer, question.prior)
        verdict = judge.judge(case)
    prior_answer = turns[0].answer if question.prior else ""
    metrics = _metrics(settings, turns, cited, question, verdict)
    return Result(question, turn.answer, prior_answer, cited, verdict, metrics)


def run_all(
    questions: Sequence[Question],
    *,
    archive: Archive,
    names: Names,
    settings: Settings,
    factory: Factory,
    judge: Judge,
    on_result: Callable[[Result], None] | None = None,
    repeat: int = 1,
) -> tuple[Result, ...]:
    """Ask every question, `repeat` times over. Each result says its round."""
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    results: list[Result] = []
    for round_number in range(repeat):
        for question in questions:
            asked = run_question(question, archive=archive, names=names, settings=settings, factory=factory, judge=judge)
            result = replace(asked, round=round_number)
            if on_result is not None:
                on_result(result)
            results.append(result)
    return tuple(results)
