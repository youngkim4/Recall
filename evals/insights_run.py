"""Run the insight eval: python -m evals.insights_run

It builds the insight world in a temp folder, makes the threads from the
archive as the app would, runs the engine and scores it against the gold list.

WARNING: without --no-model this calls the provider with the key of Recall
and costs money (about one cent a run with the cheap model).

    python -m evals.insights_run --no-model          # counted insights only, no call
    python -m evals.insights_run                     # the model of the memory
    python -m evals.insights_run --model gpt-6.1-terra --effort low --repeat 3
    python -m evals.insights_run --out /tmp/run.json

Exit codes: 0 done, 2 no key.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from archive import Archive
from evals.insight_world import NOW, insight_world
from evals.insights_score import by_kind, person, score
from evals.world import build_world, zone
from insights import engine
from recall_services import keys, memory_prep, models
from recall_services.insight_threads import threads
from report_engine.openai_llm import OpenAIResponsesLLM, build_client
from report_engine.ports import LLMUsage
from report_engine.pricing import call_cost_usd


def _cost(model: str, usage: Sequence[LLMUsage]) -> float:
    info = models.model_info(model)
    return sum(call_cost_usd(info, item) for item in usage)


def _row(result, report, model: str, seconds: float = 0.0) -> dict:
    return {
        "recall": round(result.recall, 3),
        "by_kind": by_kind(result),
        "missed": [f"{want.kind} {want.person}: {want.why}" for want in result.missed],
        "sprung": [f"{trap.kind} {trap.person}: {trap.why}" for trap in result.sprung],
        "extras": [f"{item.kind} {person(item)}: {item.summary}" for item in result.extras],
        "picks": [f"{pick.name} ({pick.score}): " + " | ".join(item.summary for item in pick.reasons) for pick in report.picks],
        "calls": report.calls,
        "failed": report.failed,
        "dropped": report.dropped,
        "usd": round(_cost(model, report.usage), 4) if model else 0.0,
    }


def _noise(args: argparse.Namespace) -> int:
    """The busy chat world has no gold list: print every insight, for a person to read."""
    from evals.world.hard import hard_world
    from evals.world.people import NOW as HARD_NOW

    model = "" if args.no_model else (args.model or memory_prep.model())
    with tempfile.TemporaryDirectory() as folder, zone():
        built = build_world(Path(folder), hard_world())
        archive = Archive.open(built.archive_path)
        try:
            made = threads(archive, built.names)
        finally:
            archive.close()
        llm = OpenAIResponsesLLM(build_client(), model) if model else None
        report = engine.run(made, HARD_NOW, llm, reasoning=args.effort or None)
    print(f"== hard world ({model or 'no model'}): {len(made)} threads, {report.calls} calls, {report.failed} failed, usd {_cost(model, report.usage) if model else 0:.4f}")
    for item in report.insights:
        print(f"  {item.kind:<10} {item.name}: {item.summary}")
    for pick in report.picks:
        print(f"  pick: {pick.name} ({pick.score})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--model", default="")
    parser.add_argument("--effort", default="")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", default="")
    parser.add_argument("--world", choices=("insights", "hard"), default="insights")
    args = parser.parse_args(argv)
    if args.world == "hard":
        return _noise(args)
    model = "" if args.no_model else (args.model or memory_prep.model())
    if model and not keys.api_key():
        print(f"{keys.NO_KEY}. Use --no-model to run with no call.")
        return 2
    world = insight_world()
    rows = []
    with tempfile.TemporaryDirectory() as folder, zone():
        built = build_world(Path(folder), world)
        archive = Archive.open(built.archive_path)
        try:
            made = threads(archive, built.names)
        finally:
            archive.close()
        llm = OpenAIResponsesLLM(build_client(), model) if model else None
        for _ in range(max(1, args.repeat)):
            report = engine.run(made, NOW, llm, reasoning=args.effort or None)
            rows.append(_row(score(world, report.insights), report, model))
    for place, row in enumerate(rows, start=1):
        print(f"== run {place} ({model or 'no model'}{', ' + args.effort if args.effort else ''})")
        for kind, found, wanted in row["by_kind"]:
            print(f"  {kind:<11} {found}/{wanted}")
        print(f"  recall {row['recall']:.2f}  traps sprung {len(row['sprung'])}  extras {len(row['extras'])}  calls {row['calls']}  failed {row['failed']}  usd {row['usd']}")
        for key, label in (("missed", "missed"), ("sprung", "trap"), ("extras", "extra"), ("picks", "pick")):
            for line in row[key]:
                print(f"  {label}: {line}")
    if args.out:
        Path(args.out).write_text(json.dumps({"model": model, "effort": args.effort, "runs": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
