"""Run the chat eval: python -m evals.run

It builds the eval world in a temp folder, asks each question through
`recall_services.chat.stream` as the app does, has a judge model grade each
answer, prints a table per kind and writes the full JSON.

WARNING: without --dry-run this calls the provider with the key of Recall
and costs money. Run --dry-run first: it prints an estimate and calls nothing.

    python -m evals.run --dry-run
    python -m evals.run --limit 5 --kinds fact,count
    python -m evals.run --model gpt-6.1-terra --effort low --out /tmp/terra.json
    python -m evals.run --with-memory     # builds the index of meanings, memory and people first

Exit codes: 0 done, 2 a bad option or no key.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from archive import Archive
from evals import places, report
from evals.estimate import describe, estimate
from evals.judge import Judge, ModelJudge
from evals.memory import build_memory, estimate_memory
from evals.questions import KINDS, QuestionError, build_questions, select
from evals.recorder import price
from evals.runner import Factory, Result, Settings, run_all
from evals.score import table
from evals.world import build_world, world, zone
from evals.world.people import NOW
from recall_services import chat, keys, models
from recall_services.models import ModelChoiceError

Say = Callable[[str], None]
JUDGE_EFFORT = "low"


class UsageProblem(Exception):
    """An option that can not be used. The message is for the person who ran it."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="", help="chat model id of models.json (default: the default model)")
    parser.add_argument("--effort", default="", help="reasoning effort (default: the effort of models.json)")
    parser.add_argument("--judge-model", default="", help="judge model id (default: the default model)")
    parser.add_argument("--judge-effort", default=JUDGE_EFFORT, help=f"judge effort (default: {JUDGE_EFFORT})")
    parser.add_argument("--limit", type=int, default=None, help="ask at most N questions")
    parser.add_argument("--kinds", default="", help=f"comma-separated kinds: {', '.join(KINDS)}")
    parser.add_argument("--out", default="", help="JSON path (default: <data folder>/evals/chat-eval-<time>.json)")
    parser.add_argument("--with-memory", action="store_true", help="build meaning, memory and people first (costs)")
    parser.add_argument("--dry-run", action="store_true", help="estimate tokens and cost; call nothing")
    return parser


def _checked_model(model: str, effort: str) -> tuple[str, str]:
    """The model id and the effort it resolves to. Raises UsageProblem for a model or effort that can not run."""
    chosen = model.strip() or models.default_id()
    try:
        config = models.chat_config(chosen, effort)
    except ModelChoiceError as error:
        raise UsageProblem(str(error)) from None
    return config.model, config.reasoning_effort or ""


def _kinds(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _model_judge(model: str, effort: str) -> Judge:
    config = models.chat_config(model, effort)
    return ModelJudge(chat.default_llm(config), config.model, lambda usage: price(config.model, usage))


def _line(result: Result) -> str:
    metrics = result.metrics
    recall = "-" if metrics.citation_recall is None else f"{metrics.citation_recall:.2f}"
    mark = "ok" if metrics.correct else "no"
    tail = f"  {metrics.error}" if metrics.error else ""
    return (
        f"{result.question.id:<24}{result.question.kind:<13}{mark:<4}score {metrics.score:.2f}  "
        f"recall {recall:<5} ${metrics.usd:.4f}  {metrics.wall_seconds:.1f}s{tail}"
    )


def _meta(arguments: argparse.Namespace, settings: Settings, judge_model: str, count: int) -> dict[str, Any]:
    made = world()
    return {
        "started": datetime.now().isoformat(timespec="seconds"),
        "model": settings.model,
        "effort": settings.effort,
        "judge_model": judge_model,
        "judge_effort": arguments.judge_effort,
        "kinds": list(_kinds(arguments.kinds)),
        "limit": arguments.limit,
        "questions": count,
        "with_memory": bool(arguments.with_memory),
        "clock": NOW.isoformat(),
        "world": {"messages": len(made.lines), "tapbacks": len(made.tapbacks)},
    }


def _dry_run(arguments: argparse.Namespace, archive: Archive, built: Any, settings: Settings, judge_model: str,
             questions: Sequence[Any], say: Say) -> int:  # fmt: skip
    found = estimate(
        questions, archive=archive, names=built.names, settings=settings,
        resolved_effort=settings.effort, judge_model=judge_model,
    )  # fmt: skip
    say(describe(found))
    if arguments.with_memory:
        try:
            spent = estimate_memory(archive, built.names)
            say(f"with memory: about ${spent['usd']:.4f} more to build the meaning, memory and people indexes")
        except Exception as error:  # noqa: BLE001 - an estimate that fails must not stop the dry run
            say(f"with memory: no estimate ({type(error).__name__})")
    return 0


def _run(
    arguments: argparse.Namespace, settings: Settings, judge_model: str, factory: Factory | None,
    judge: Judge | None, say: Say,
) -> int:  # fmt: skip
    questions = select(build_questions(world()), _kinds(arguments.kinds), arguments.limit)
    with tempfile.TemporaryDirectory(prefix="recall-eval-") as folder, zone(), places.isolated_saves(Path(folder)):
        built = build_world(Path(folder) / "world")
        with Archive.open(built.archive_path) as archive:
            if arguments.dry_run:
                return _dry_run(arguments, archive, built, settings, judge_model, questions, say)
            if factory is None and not keys.api_key():
                say(f"{keys.NO_KEY}. A run needs the key; --dry-run does not.")
                return 2
            meta = _meta(arguments, settings, judge_model, len(questions))
            if arguments.with_memory:
                meta = {**meta, "memory": build_memory(archive, built.names, say)}
            results = run_all(
                questions, archive=archive, names=built.names, settings=settings,
                factory=factory or chat.default_llm,
                judge=judge or _model_judge(judge_model, arguments.judge_effort),
                on_result=lambda result: say(_line(result)),
            )  # fmt: skip
    out = Path(arguments.out) if arguments.out else places.default_out(datetime.now())
    written = report.write(out, report.document(meta, results))
    say(table(report.summaries(results)))
    say(f"results: {written}")
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    factory: Factory | None = None,
    judge: Judge | None = None,
    say: Say = print,
) -> int:
    """Run the eval. A test gives `factory` (the chat model) and `judge`, so nothing leaves the machine."""
    arguments = _parser().parse_args(argv)
    try:
        model, effort = _checked_model(arguments.model, arguments.effort)
        judge_model, _judge_effort = _checked_model(arguments.judge_model, arguments.judge_effort)
        select((), _kinds(arguments.kinds))
        if arguments.limit is not None and arguments.limit < 1:
            raise UsageProblem("--limit must be at least 1")
    except (UsageProblem, QuestionError) as problem:
        say(str(problem))
        return 2
    return _run(arguments, Settings(model, effort), judge_model, factory, judge, say)


if __name__ == "__main__":
    raise SystemExit(main())
