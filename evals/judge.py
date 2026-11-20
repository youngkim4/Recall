"""The judge: one model call that grades one answer against the gold answer.

`Judge` is a port. The run by hand uses `ModelJudge` over the chat model
port of the app; tests give a fake. The judge reads the question, the gold
answer, the key facts and the answer. It never reads the messages: it grades
the answer against the gold, not against the archive.

The reply must be JSON: {"correct": bool, "score": 0..1, "missing": [...],
"wrong": [...]}. A reply that is not that gives a verdict with `error` set
and a score of 0, so a broken judge never counts as a pass.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from chat_engine.errors import ChatEngineError
from chat_engine.ports import LLM, ChatMessage, Completed, LLMRequest, TextDelta, Usage

JUDGE_OUTPUT_TOKENS = 1200
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_MAX_ITEMS = 12
_MAX_ITEM_CHARS = 300

RUBRIC = """You grade one answer of an assistant that searches a person's text messages.
You get the question, the gold answer, the key facts and the answer. Grade strictly.

Rules:
- "correct" is true only when the answer states every key fact and says nothing that contradicts the
  gold answer. An invented fact or a contradiction makes the answer wrong.
- Equivalent descriptions count. Judge whether the answer conveys the same fact or the same relationship,
  not whether it uses the same words. Example: for "close friend, still in touch", "a friend from your
  Oakland days; you are still in touch and the warmth stayed" counts.
- A key fact counts when the answer gives it as its main reading, even with light hedging ("appears to
  be your landlord", "likely your manager"). It does not count when the answer offers two different
  readings as equally likely.
- A number must match exactly when the key fact is a number. "About" in a key fact allows a near number.
- A figure stated another way counts when it gives the same values. Example: "285 in the 9 months before
  against 47 in the 9 months after" counts for "about 32 a month against about 5 a month". Check the
  arithmetic yourself; a figure that does not work out does not count.
- A date must have the right day, month and year. The answer may leave out the year when the year is
  clear from the rest of the answer, from the earlier question, or from the question itself.
- When the kind is "absent", the archive does not hold the answer. The answer is correct when it says that
  it could not find it or that the messages do not mention it. Mentioning an unrelated message that it
  did find is fine. An answer that claims the thing happened is wrong, however plausible.
- Detail beyond the gold answer is neutral. You cannot check it against the messages, so never list it as
  wrong or as invented unless it contradicts the gold answer or a key fact. Hedging that leaves out a key fact
  still fails.
- "score" is the share of key facts stated, from 0 to 1, and 0 when the answer contradicts the gold.
- "missing" lists the key facts the answer does not state. "wrong" lists the claims that contradict the gold.

Reply with JSON only, no other text:
{"correct": true or false, "score": number, "missing": [strings], "wrong": [strings]}"""


@dataclass(frozen=True)
class JudgeCase:
    """What the judge reads for one question."""

    question: str
    kind: str
    gold: str
    key_facts: tuple[str, ...]
    answer: str
    prior: str = ""


@dataclass(frozen=True)
class Verdict:
    correct: bool
    score: float
    missing: tuple[str, ...] = ()
    wrong: tuple[str, ...] = ()
    error: str = ""
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0


class Judge(Protocol):
    def judge(self, case: JudgeCase) -> Verdict: ...


def judge_prompt(case: JudgeCase) -> str:
    """The user message of the judge call, as JSON."""
    body: dict[str, Any] = {
        "kind": case.kind,
        "question": case.question,
        "gold_answer": case.gold,
        "key_facts": list(case.key_facts),
        "answer": case.answer,
    }
    if case.prior:
        body = {"earlier_question_in_the_chat": case.prior, **body}
    return json.dumps(body, ensure_ascii=False, indent=1)


def _texts(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item)[:_MAX_ITEM_CHARS] for item in value[:_MAX_ITEMS])


def parse_verdict(text: str) -> Verdict:
    """The verdict in a reply of the judge. A reply that is not the JSON of the rubric gives `error`."""
    found = _OBJECT_RE.search(text or "")
    if found is None:
        return Verdict(False, 0.0, error="the judge gave no JSON")
    try:
        raw = json.loads(found.group(0))
    except ValueError:
        return Verdict(False, 0.0, error="the judge gave JSON that can not be read")
    if not isinstance(raw, dict) or not isinstance(raw.get("correct"), bool):
        return Verdict(False, 0.0, error="the judge gave no true or false for correct")
    score = raw.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return Verdict(False, 0.0, error="the judge gave no number for score")
    return Verdict(
        correct=raw["correct"],
        score=min(1.0, max(0.0, float(score))),
        missing=_texts(raw.get("missing")),
        wrong=_texts(raw.get("wrong")),
    )


class ModelJudge:
    """A judge over one model port. `price` gives the cost of one call from its usage."""

    def __init__(
        self,
        llm: LLM,
        model: str,
        price: Callable[[Usage], float],
        max_output_tokens: int = JUDGE_OUTPUT_TOKENS,
    ) -> None:
        self._llm = llm
        self._model = model
        self._price = price
        self._max_output_tokens = max_output_tokens

    def _request(self, case: JudgeCase) -> LLMRequest:
        messages = (ChatMessage("system", RUBRIC), ChatMessage("user", judge_prompt(case)))
        return LLMRequest(self._model, messages, (), self._max_output_tokens, allow_tools=False)

    def judge(self, case: JudgeCase) -> Verdict:
        parts: list[str] = []
        usage = Usage()
        try:
            for event in self._llm.stream(self._request(case)):
                if isinstance(event, TextDelta) and not event.commentary:
                    parts.append(event.text)
                elif isinstance(event, Completed):
                    usage = event.usage
        except ChatEngineError as error:
            return Verdict(False, 0.0, error=f"the judge call failed: {error.code}")
        verdict = parse_verdict("".join(parts))
        return Verdict(
            verdict.correct, verdict.score, verdict.missing, verdict.wrong, verdict.error,
            usage, self._price(usage),
        )  # fmt: skip
