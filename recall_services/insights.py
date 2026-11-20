"""The insights of the app: `GET /api/insights` and the `insights` job.

- `payload` is the answer of the route. It counts drift, balance,
  unanswered questions and birthdays now, with no model and no key. It adds
  the plans, promises and news of the last reading, when one is saved. The
  digest (`picks`) is worked out from both, now. It also says what a new
  reading would cost (`toRead`, `estimatedCost`).
- `run` is the `insights` job. It reads each thread worth a call with the
  model of the memory, saves what it found, and writes one line to the
  spend log. A reading where every call failed saves nothing and fails.

The saved reading is one JSON file for each archive, mode 0600, in
`saves/insights/`. It holds names, summaries and message ids; it never holds
message text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from archive import Archive
from insights import digest, engine, prompt, reading
from insights.thread import Insight, Thread
from recall_services import keys, memory, memory_prep, models, reports
from recall_services.insight_threads import threads
from recall_services.names import Names
from recall_services.report_failures import ReportFailure
from report_engine.ports import LLMUsage
from report_engine.pricing import call_cost_usd

logger = logging.getLogger(__name__)
spend = logging.getLogger(memory.SPEND_LOG)

VERSION = 1
STORE_LEAF = "insights"
NAME_CHARS = 24
READ_KINDS = frozenset(prompt.READ_KINDS)
CHARS_PER_TOKEN = 4
OUTPUT_TOKENS_GUESS = 400
NOT_READ = "Recall could not read your conversations for insights. Try again later."
NO_KEY = f"{keys.NO_KEY}, then look for things to follow up on again."

Teller = Callable[[dict[str, Any]], None]


class InsightFailure(Exception):
    """A reading that ended with nothing saved. `code` is a fixed word; the text is one sentence."""

    def __init__(self, code: str, sentence: str, usd: float = 0.0) -> None:
        super().__init__(sentence)
        self.code = code
        self.usd = usd


def store_file(archive_file: str | os.PathLike[str]) -> Path:
    """The file of the saved reading of one archive, read at each call."""
    import recall_paths

    resolved = str(Path(archive_file).expanduser().resolve())
    name = hashlib.sha256(resolved.encode("utf-8", "surrogatepass")).hexdigest()[:NAME_CHARS]
    return Path(recall_paths.SAVES_DIR) / STORE_LEAF / f"{name}.json"


def model() -> str:
    return memory_prep.model()


def as_json(insight: Insight) -> dict[str, Any]:
    return {
        "kind": insight.kind,
        "thread": insight.thread,
        "name": insight.name,
        "summary": insight.summary,
        "when": insight.when.isoformat(timespec="seconds"),
        "messageIds": list(insight.message_ids),
        "figures": dict(insight.figures),
    }


def _from_json(raw: Mapping[str, Any]) -> Insight | None:
    try:
        kind = str(raw["kind"])
        if kind not in READ_KINDS:
            return None
        return Insight(
            kind=kind,  # type: ignore[arg-type]
            thread=str(raw["thread"]),
            name=str(raw["name"]),
            summary=str(raw["summary"]),
            when=datetime.fromisoformat(str(raw["when"])),
            message_ids=tuple(str(item) for item in raw.get("messageIds") or ()),
        )
    except (KeyError, TypeError, ValueError):
        return None


def read_saved(path: Path) -> dict[str, Any] | None:
    """The saved reading, or None when there is none or it can not be read."""
    try:
        held = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        logger.warning("the saved insights could not be read (%s)", type(error).__name__)
        return None
    return held if isinstance(held, dict) and held.get("version") == VERSION else None


def _save(path: Path, found: Sequence[Insight], read_at: datetime, model_id: str, usd: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    body = json.dumps(
        {"version": VERSION, "readAt": read_at.isoformat(timespec="seconds"), "model": model_id, "usd": usd,
         "items": [as_json(item) for item in found if item.kind in READ_KINDS]},
        ensure_ascii=False,
    )  # fmt: skip
    partial = path.with_suffix(".part")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(body)
    os.replace(partial, path)


def _saved_items(held: Mapping[str, Any] | None, made: Sequence[Thread]) -> list[Insight]:
    """The model insights of the saved reading, for threads that are still there, under their name now."""
    names = {thread.key: thread.name for thread in made}
    items = [_from_json(raw) for raw in (held or {}).get("items") or () if isinstance(raw, Mapping)]
    return [
        Insight(item.kind, item.thread, names[item.thread], item.summary, item.when, item.message_ids)
        for item in items
        if item is not None and item.thread in names
    ]


def estimate(made: Sequence[Thread], now: datetime, model_id: str) -> tuple[int, float]:
    """How many threads a reading would read now, and about what it would cost."""
    info = models.model_info(model_id)
    chosen = [thread for thread in made if reading.worth_reading(thread, now)]
    cost = 0.0
    for thread in chosen:
        text = prompt.SYSTEM + prompt.user_text(thread.name, reading.window(thread, now), now)
        cost += call_cost_usd(info, LLMUsage(len(text) // CHARS_PER_TOKEN, 0, OUTPUT_TOKENS_GUESS))
    return len(chosen), round(cost, 4)


def payload(archive: Archive, names: Names, *, now: datetime | None = None, path: Path | None = None) -> dict[str, Any]:
    """The answer of `GET /api/insights`. No model call, no key."""
    now = now or datetime.now()
    model_id = model()
    made = threads(archive, names)
    held = read_saved(path or store_file(archive.path))
    found = [item for thread in made for item in engine.counted(thread, now)] + _saved_items(held, made)
    found.sort(key=lambda item: (-item.when.timestamp(), item.name))
    try:
        to_read, cost = estimate(made, now, model_id)
    except Exception as error:  # noqa: BLE001 - a model with no facts: the answer still holds what is found
        logger.warning("insights could not be priced (%s)", type(error).__name__)
        to_read, cost = 0, 0.0
    picks = digest.picks(found, now)
    return {
        "state": "ready" if held else "none",
        "readAt": (held or {}).get("readAt"),
        "model": model_id,
        "people": len(made),
        "toRead": to_read,
        "estimatedCost": cost,
        "picks": [
            {"thread": pick.thread, "name": pick.name, "score": pick.score, "reasons": [as_json(item) for item in pick.reasons]}
            for pick in picks
        ],
        "insights": [as_json(item) for item in found],
    }


def _port(llm_factory: Callable[[Any], Any] | None, model_id: str) -> Any:
    try:
        return reports._port(llm_factory or reports.default_llm, models.model_info(model_id))
    except ReportFailure as told:
        raise InsightFailure(told.code, str(told)) from told


def run(
    archive: Archive,
    names: Names,
    told: Teller | None = None,
    *,
    now: datetime | None = None,
    llm_factory: Callable[[Any], Any] | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """The `insights` job. Gives `read`, `failed`, `found`, `usd` and `usage`. Raises InsightFailure."""
    tell = told or (lambda _update: None)
    now = now or datetime.now()
    model_id = model()
    made = threads(archive, names)
    to_read = [thread for thread in made if reading.worth_reading(thread, now)]
    tell({"log": f"Reading {len(to_read)} recent conversation{'' if len(to_read) == 1 else 's'}"})
    llm = _port(llm_factory, model_id) if to_read else None
    report = engine.run(to_read, now, llm)
    info = models.model_info(model_id)
    usd = round(sum(call_cost_usd(info, item) for item in report.usage), 6)
    usage = {
        "calls": report.calls - report.failed,
        "input_tokens": sum(item.input_tokens for item in report.usage),
        "output_tokens": sum(item.output_tokens for item in report.usage),
        "cost_usd": usd,
    }
    tell({"spent": {"usd": usd, "usage": usage}})
    found = [item for item in report.insights if item.kind in READ_KINDS]
    stopped = report.calls > 0 and report.failed == report.calls
    spend.info(
        "%s: %d conversations read, %d failed, %d found, %.4f USD",
        "the insights job" if not stopped else "the insights job stopped (run_failed)",
        report.calls - report.failed, report.failed, len(found), usd,
    )  # fmt: skip
    if stopped:
        raise InsightFailure("run_failed", NOT_READ, usd)
    _save(path or store_file(archive.path), found, now, model_id, usd)
    tell({"log": f"Found {len(found)} thing{'' if len(found) == 1 else 's'} to follow up on"})
    return {"read": report.calls - report.failed, "failed": report.failed, "found": len(found), "usd": usd, "usage": usage}
