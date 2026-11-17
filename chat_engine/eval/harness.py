"""Run the questions through the engine and score each result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from ..engine import ChatEngine, EngineConfig
from ..errors import ChatEngineError
from ..labeler import looks_like_handle
from ..ports import LLM
from ..scope import Scope
from ..session import ChatSession
from .questions import Question
from .scoring import passed, score
from .synthetic import SyntheticArchive


@dataclass(frozen=True)
class Row:
    """The score of one question. It holds no message text and no answer text."""

    id: str
    kind: str
    passed: bool
    checks: Mapping[str, bool]
    tools: tuple[str, ...]
    model_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    error: str = ""


def unsaved_handles(synthetic: SyntheticArchive) -> list[str]:
    """Every handle of the archive that has no saved name."""
    handles = []
    for conversation in synthetic.conversations:
        for handle in [conversation["id"], *conversation["members"]]:
            if looks_like_handle(handle) and handle not in synthetic.saved_names and handle not in handles:
                handles.append(handle)
    return handles


def with_roots(questions: Sequence[Question], only: Sequence[str]) -> list[Question]:
    """The chosen questions, each follow-up after the question it continues."""
    if not only:
        return list(questions)
    wanted = set(only)
    wanted |= {question.follows for question in questions if question.id in wanted and question.follows}
    return [question for question in questions if question.id in wanted]


def evaluate(
    synthetic: SyntheticArchive,
    questions: Sequence[Question],
    llm_for: Callable[[Question], LLM],
    config: EngineConfig,
    on_result: Optional[Callable[[Question, Mapping[str, Any]], None]] = None,
) -> list[Row]:
    archive = synthetic.archive()
    labeler = synthetic.labeler()
    forbidden = unsaved_handles(synthetic)
    sessions: dict[str, ChatSession] = {}
    table: list[Row] = []
    for question in questions:
        session = sessions.get(question.follows) or ChatSession()
        sessions[question.id] = session
        engine = ChatEngine(archive, llm_for(question), labeler, config, clock=lambda: synthetic.today)
        scope = Scope(conversations=tuple(synthetic.keys[key] for key in question.scope))
        try:
            result = engine.ask(question.text, scope=scope, session=session)
        except ChatEngineError as error:
            table.append(Row(question.id, question.kind, False, {}, (), 0, 0, 0, 0, error=error.code))
            continue
        if on_result is not None:
            on_result(question, result)
        checks = score(result, question, forbidden)
        usage = result["usage"]
        table.append(
            Row(
                id=question.id,
                kind=question.kind,
                passed=passed(checks),
                checks=checks,
                tools=tuple(step["tool"] for step in result["steps"] if not step["seeded"]),
                model_calls=usage["model_calls"],
                tool_calls=usage["tool_calls"],
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
            )
        )
    return table
