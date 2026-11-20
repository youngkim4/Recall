"""The cost of a run, estimated with no model call.

Each question goes through `chat.stream` with a probe model: the probe keeps
the first request (the system prompt, the tools, the seeded search results
and the question) and answers at once. Its size is counted with the cautious
estimate of the engine. The rest is assumed:

- a question takes `ASSUMED_CALLS` model calls; each later call adds one tool
  result of `ASSUMED_RESULT_TOKENS` and one tool call;
- the provider caches the start that a call shares with the call before it;
- each call writes the output tokens of `OUTPUT_BY_EFFORT` for the effort;
- the judge reads the rubric, the gold and an answer of `ASSUMED_ANSWER_TOKENS`.

Digests read inside a question are not counted: the engine refuses one that
costs more than its limit (`EngineConfig.digest_usd_limit`). Prices come
from models.json.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from archive import Archive
from chat_engine.ports import Completed, LLMEvent, LLMRequest, TextDelta, Usage
from chat_engine.render import estimate_tokens
from evals.judge import RUBRIC, JudgeCase, judge_prompt
from evals.question_types import Question
from evals.recorder import price
from evals.runner import Settings
from recall_services import chat
from recall_services.models import ChatModelConfig
from recall_services.names import Names

ASSUMED_CALLS = 4
ASSUMED_RESULT_TOKENS = 2500
ASSUMED_TOOL_CALL_TOKENS = 80
ASSUMED_ANSWER_TOKENS = 400
JUDGE_OUTPUT_TOKENS = 250
OUTPUT_BY_EFFORT = {"": 600, "none": 300, "low": 600, "medium": 1200, "high": 2500, "xhigh": 5000, "max": 9000}


@dataclass(frozen=True)
class Estimate:
    questions: int
    turns: int
    model: str
    effort: str
    judge_model: str
    first_call_tokens: int  # all turns, first calls only
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    chat_usd: float
    judge_usd: float

    @property
    def usd(self) -> float:
        return self.chat_usd + self.judge_usd


class ProbeLLM:
    """A model port that keeps the first request and answers at once. It makes no network call."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def stream(self, request: LLMRequest) -> Iterator[LLMEvent]:
        self.requests.append(request)
        return iter((TextDelta("probe"), Completed()))


def request_tokens(request: LLMRequest) -> int:
    """The cautious size of one request: its messages and its tool specs."""
    text = sum(estimate_tokens(message.content) for message in request.messages)
    tools = estimate_tokens(json.dumps([[tool.name, tool.description, tool.parameters] for tool in request.tools]))
    return text + tools


def _first_call(text: str, *, archive: Archive, names: Names, settings: Settings) -> int:
    probe = ProbeLLM()

    def factory(_config: ChatModelConfig) -> ProbeLLM:
        return probe

    for _event in chat.stream(
        text, "", settings.model, [], archive=archive, names=names, reasoning_effort=settings.effort,
        llm_factory=factory, clock=lambda: settings.clock,
    ):  # fmt: skip
        pass
    return request_tokens(probe.requests[0]) if probe.requests else 0


def _turn_usage(first: int, output: int) -> tuple[Usage, ...]:
    calls = []
    previous = 0
    for step in range(ASSUMED_CALLS):
        size = first + step * (ASSUMED_RESULT_TOKENS + ASSUMED_TOOL_CALL_TOKENS)
        calls.append(Usage(input_tokens=size, output_tokens=output, cached_input_tokens=previous))
        previous = size
    return tuple(calls)


def _judge_usage(question: Question) -> Usage:
    case = JudgeCase(question.text, question.kind, question.gold, question.key_facts, "", question.prior)
    size = estimate_tokens(RUBRIC) + estimate_tokens(judge_prompt(case)) + ASSUMED_ANSWER_TOKENS
    return Usage(input_tokens=size, output_tokens=JUDGE_OUTPUT_TOKENS)


def estimate(
    questions: Sequence[Question],
    *,
    archive: Archive,
    names: Names,
    settings: Settings,
    resolved_effort: str,
    judge_model: str,
) -> Estimate:
    """The estimate of a run of `questions`. No call leaves this machine."""
    output = OUTPUT_BY_EFFORT.get(resolved_effort, OUTPUT_BY_EFFORT[""])
    firsts = [
        _first_call(text, archive=archive, names=names, settings=settings)
        for question in questions
        for text in ((question.prior, question.text) if question.prior else (question.text,))
    ]
    calls = [usage for first in firsts for usage in _turn_usage(first, output)]
    judged = [_judge_usage(question) for question in questions]
    return Estimate(
        questions=len(questions),
        turns=len(firsts),
        model=settings.model,
        effort=resolved_effort or "none on file",
        judge_model=judge_model,
        first_call_tokens=sum(firsts),
        input_tokens=sum(usage.input_tokens for usage in calls),
        cached_input_tokens=sum(usage.cached_input_tokens for usage in calls),
        output_tokens=sum(usage.output_tokens for usage in calls),
        chat_usd=sum(price(settings.model, usage) for usage in calls),
        judge_usd=sum(price(judge_model, usage) for usage in judged),
    )


def describe(found: Estimate) -> str:
    """The estimate as a few lines of text."""
    return "\n".join(
        (
            f"dry run: {found.questions} questions, {found.turns} turns, model {found.model}, effort {found.effort}",
            f"first calls: {found.first_call_tokens:,} tokens (system prompt, tools, seeded results, question)",
            f"assumed {ASSUMED_CALLS} calls a turn: {found.input_tokens:,} input tokens "
            f"({found.cached_input_tokens:,} cached), {found.output_tokens:,} output tokens",
            f"chat: ${found.chat_usd:.4f}  judge ({found.judge_model}): ${found.judge_usd:.4f}  "
            f"total: ${found.usd:.4f}",
            "not counted: digests read inside a question (each refused above its limit), --with-memory builds",
        )
    )
