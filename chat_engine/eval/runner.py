"""Run the evaluation set and print the scores.

With a real model (your own key; this sends the synthetic archive's text to
the provider and costs money):

    python -m chat_engine.eval.runner --model MODEL --max-output-tokens 2000

With the scripted model (no key, no network):

    python -m chat_engine.eval.runner --scripted

It prints scores only: no question, no answer and no message text.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Sequence

from ..engine import EngineConfig
from ..fake_llm import FakeLLM
from ..openai_llm import OpenAIResponsesLLM
from ..ports import LLM
from .harness import Row, evaluate, with_roots
from .oracle import SCRIPTS
from .questions import Question, build_questions
from .synthetic import build_synthetic_archive

_KEY_NAME = "OPENAI_API_KEY"


def read_key(path: str) -> str:
    """The API key from one named env file. No other file is looked for."""
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        name, _, value = line.strip().removeprefix("export ").partition("=")
        if name.strip() == _KEY_NAME:
            return value.strip().strip("'\"")
    return ""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chat_engine.eval.runner", description=__doc__.split("\n\n")[0])
    parser.add_argument("--scripted", action="store_true", help="use the scripted model: no key, no network")
    parser.add_argument("--model", help="model id for the real run")
    parser.add_argument("--max-output-tokens", type=int, help="hard output cap for each model call")
    parser.add_argument("--timeout", type=float, default=90.0, help="seconds before a model call times out")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--verbosity", default=None)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--only", default="", help="comma-separated question ids")
    parser.add_argument("--env-file", default="", help=f"file to read {_KEY_NAME} from")
    return parser


def _real_llm(arguments: argparse.Namespace) -> LLM:
    key = read_key(arguments.env_file) if arguments.env_file else os.environ.get(_KEY_NAME, "")
    return OpenAIResponsesLLM(
        api_key=key or None,
        timeout_seconds=arguments.timeout,
        reasoning_effort=arguments.reasoning_effort,
        verbosity=arguments.verbosity,
    )


def _failed(row: Row) -> str:
    if row.error:
        return f"error: {row.error}"
    return ", ".join(name for name, ok in row.checks.items() if not ok)


def report(rows: Sequence[Row]) -> list[str]:
    lines = [f"{'id':24s} {'kind':16s} {'pass':5s} {'calls':>5s} {'tools':>5s} {'in_tok':>9s} {'out_tok':>8s}  failed"]
    for row in rows:
        lines.append(
            f"{row.id:24s} {row.kind:16s} {'ok' if row.passed else 'FAIL':5s} {row.model_calls:5d} "
            f"{row.tool_calls:5d} {row.input_tokens:9,d} {row.output_tokens:8,d}  {_failed(row)}"
        )
    total = Counter(row.kind for row in rows)
    good = Counter(row.kind for row in rows if row.passed)
    lines.append("")
    lines.append(
        f"{sum(good.values())} of {len(rows)} passed. Model calls {sum(r.model_calls for r in rows)}, "
        f"tool calls {sum(r.tool_calls for r in rows)}, input tokens {sum(r.input_tokens for r in rows):,}, "
        f"output tokens {sum(r.output_tokens for r in rows):,}."
    )
    lines.append("By kind: " + ", ".join(f"{kind} {good[kind]}/{count}" for kind, count in sorted(total.items())))
    return lines


def main(argv: Optional[Sequence[str]] = None, out: Callable[[str], None] = print) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if not arguments.scripted and not (arguments.model and arguments.max_output_tokens):
        parser.error("give --model and --max-output-tokens, or --scripted")
    synthetic = build_synthetic_archive()
    only = [name.strip() for name in arguments.only.split(",") if name.strip()]
    questions = with_roots(build_questions(synthetic), only)
    if arguments.scripted:
        config = EngineConfig(model="scripted", max_output_tokens=600, max_steps=arguments.max_steps)
        llm_for: Callable[[Question], LLM] = lambda question: FakeLLM(SCRIPTS[question.id]())  # noqa: E731
    else:
        config = EngineConfig(
            model=arguments.model, max_output_tokens=arguments.max_output_tokens, max_steps=arguments.max_steps
        )
        shared = _real_llm(arguments)
        llm_for = lambda question: shared  # noqa: E731
    rows = evaluate(synthetic, questions, llm_for, config)
    for line in report(rows):
        out(line)
    return 1 if any(row.error for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
