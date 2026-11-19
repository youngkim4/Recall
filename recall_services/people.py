"""The people index: one profile for each person and each group, searchable by meaning.

A profile holds figures of the archive (no model) and what a cheap model
(the model of the memory, `memory_prep.model()`) made of the digests of the
person's own conversations: a kind, a status, a summary and turning points.
No message is read again by a model. See "People" in `README.md`.

- `status` reads the store only: `none` (no profile) or `ready`.
- `estimate` and `payload` (the answer of `GET /api/people`) plan every
  profile: `none`, `partial` or `ready`, the counts, and `estimatedCost` of
  the profiles that a build would write now. No model call, no key, no file.
- `build` is the `people` job. A profile whose inputs did not change costs
  no call. Each end of a build writes one line to the spend log.
- `search` ranks the profiles by meaning. It never fails a question.
- `worth_starting` says whether the server builds by itself after a
  `memory` job or an import: only when a build of this archive completed
  before, and the profiles to write cost at most `PEOPLE_AUTO_USD`.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import sqlite3
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote

from archive import Archive
from meaning_index import EmbedderFacts
from recall_services import meaning, memory, memory_prep, models, reports
from recall_services.models import ModelChoiceError
from recall_services.names import Names
from recall_services.people_build import Look, Progress, assemble, embed, look, now, prepare, read_store, save, usage_of
from recall_services.people_calls import TRIES, Calls
from recall_services.people_prompt import CHARS_PER_TOKEN, KINDS, MAX_OUTPUT_TOKENS, OUTPUT_TOKENS, STATUSES
from recall_services.people_search import Found, search
from recall_services.people_store import Held, PeopleStore, StoreMeta, people_folder, profile_text
from recall_services.people_words import PeopleFailure, Teller, all_written, failure, looking, to_write
from recall_services.report_failures import ReportFailure
from report_engine import LLMUsage, Usage
from report_engine.pricing import call_cost_usd
from report_engine.tokens import meter_for

__all__ = [
    "KINDS", "MAX_OUTPUT_TOKENS", "STATUSES", "TRIES", "Found", "PeopleFailure", "build", "completed_before",
    "estimate", "payload", "people_folder", "search", "status", "worth_starting",
]  # fmt: skip

logger = logging.getLogger(__name__)
spend = logging.getLogger(memory.SPEND_LOG)

PEOPLE_PARALLEL = 4
PEOPLE_AUTO_USD = 0.10
# what a look takes of the settings of a build
LOOK_KEYS = ("places", "options", "counter")


def model() -> str:
    """The model of the profiles: the model of the memory, read at each call."""
    return memory_prep.model()


def store_for(archive: Archive, store: PeopleStore | None = None) -> PeopleStore:
    return store or PeopleStore(people_folder(archive.path))


def status(archive: Archive, names: Names, *, store: PeopleStore | None = None) -> dict[str, Any]:
    """What the store holds: `state` (`none` or `ready`), `profiles`, `builtAt`, `model`, `embedding`.

    It reads no digest and no message. A store that can not be read holds nothing here.
    """
    try:
        held = read_store(store_for(archive, store))
    except PeopleFailure:
        held = None
    if held is None or not held.profiles:
        return {"state": "none", "profiles": 0, "builtAt": None, "model": None, "embedding": None}
    meta = held.meta
    return {
        "state": "ready",
        "profiles": len(held.profiles),
        "builtAt": meta.built_at,
        "model": meta.model,
        "embedding": {"model": meta.embed_model, "dimensions": meta.dimensions},
    }


def _cost(seen: Look, facts: EmbedderFacts) -> float:
    """What a build would pay now, for one try of each profile. The input of a profile is an upper bound
    from the lengths of its notes (`input_bound`); the output is the fullest reply that the limits allow
    (`OUTPUT_TOKENS`); the embedding of a new profile counts that many tokens too, and a row that is
    only embedded again counts its text."""
    info = seen.planner.info
    writing = seen.to_write
    model_usd = sum(
        call_cost_usd(info, LLMUsage(input_tokens=one.input_tokens, output_tokens=OUTPUT_TOKENS)) for one in writing
    )
    embed_tokens = OUTPUT_TOKENS * len(writing) + sum(
        math.ceil(len(profile_text(row)) / CHARS_PER_TOKEN) for row in seen.to_embed(facts)
    )
    return float(model_usd + embed_tokens * facts.usd_per_mtok / 1_000_000)


def _counts(seen: Look, cost: float, facts: EmbedderFacts) -> dict[str, Any]:
    profiles = len(seen.held.profiles) if seen.held else 0
    writing, embedding = len(seen.to_write), len(seen.to_embed(facts))
    waiting = writing + embedding + seen.not_read
    state = "none" if profiles == 0 else "ready" if waiting == 0 else "partial"
    return {
        "state": state,
        "model": seen.planner.info.id,
        "people": seen.people,
        "profiles": profiles,
        "current": seen.people - seen.not_read - writing,
        "toWrite": writing,
        "toEmbed": embedding,
        "notRead": seen.not_read,
        "estimatedCost": cost,
        "builtAt": seen.held.meta.built_at if seen.held else None,
        "error": None,
        "code": None,
    }


def estimate(archive: Archive, names: Names, **more: Any) -> dict[str, Any]:
    """What a build that starts now would write, embed and cost. No model call and no key. It reads no
    message of a conversation whose plan is kept, and writes only the plans it made.

    Raises PeopleFailure and ModelChoiceError. `more` gives the store, the
    embedding facts and the settings of the notes (tests).
    """
    kept = {name: more[name] for name in LOOK_KEYS if name in more}
    facts = more.get("facts") or meaning.default_facts()
    seen = look(archive, names, store_for(archive, more.get("store")), model=model(), **kept)
    return _counts(seen, _cost(seen, facts), facts)


def _nothing(error: str, code: str) -> dict[str, Any]:
    return {
        "state": "none", "model": model(), "people": 0, "profiles": 0, "current": 0, "toWrite": 0, "toEmbed": 0,
        "notRead": 0,
        "estimatedCost": 0.0, "builtAt": None, "error": error, "code": code,
    }  # fmt: skip


def payload(archive: Archive, names: Names, **more: Any) -> dict[str, Any]:
    """The answer of `GET /api/people`: `estimate`, or zeros with `error` and `code`."""
    try:
        return estimate(archive, names, **more)
    except ModelChoiceError as error:
        return _nothing(failure("model_not_run").args[0], str(getattr(error, "code", "model_not_run")))
    except PeopleFailure as told:
        return _nothing(str(told), told.code)


def notes_mark(places: reports.ReportPlaces | None = None) -> int | None:
    """The count of parts in the notes, or None with no notes file. A mark of use does not change it,
    a new part does: the key of the answer of the route. It writes nothing."""
    path = (places or reports.default_places()).cache_path
    if not path.is_file():
        return None
    try:
        with contextlib.closing(sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True)) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM period_results").fetchone()[0])
    except sqlite3.Error as error:
        logger.warning("the notes could not be counted (%s)", type(error).__name__)
        return None


def completed_before(archive_file: str | os.PathLike[str], *, store: PeopleStore | None = None) -> bool:
    """True when a build of this archive completed before. It reads no archive."""
    try:
        held = read_store(store or PeopleStore(people_folder(archive_file)))
    except PeopleFailure:
        return False
    return held is not None and held.meta.completed_at is not None


def worth_starting(found: Mapping[str, Any]) -> bool:
    """True when the server builds by itself: `found` is a `payload`, a profile is to write or to embed,
    and it is cheap. A profile can take `TRIES` calls, so `TRIES` times the estimate is compared."""
    cost = found.get("estimatedCost")
    if found.get("error") or not isinstance(cost, (int, float)) or isinstance(cost, bool):
        return False
    waiting = int(found.get("toWrite") or 0) + int(found.get("toEmbed") or 0)
    return waiting > 0 and TRIES * cost <= PEOPLE_AUTO_USD


def _port(llm_factory: Callable[[Any], Any] | None, info: Any) -> Any:
    try:
        return reports._port(llm_factory or reports.default_llm, info)
    except ReportFailure as told:
        raise failure(told.code, provider=models.provider_name(models.facts(info.id).provider), model=info.id) from told


def build(
    archive: Archive,
    names: Names,
    cancel: Any = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    **more: Any,
) -> dict[str, Any]:
    """Write the profiles whose inputs changed, embed, and save. Gives `people`, `built`, `kept`, `failed`,
    `notRead`, `removed`, `embedded`, `embedTokens`, `usd` and `usage`.

    `cancel` is a token with `cancelled`, or a callable. `more` gives the
    store, the model port, the embedder, the embedding facts, the token
    counters and the settings of the notes (tests), and `parallel`.
    Raises PeopleFailure with a code and one sentence; `usage` is what it spent.
    """
    teller = Teller(on_progress)
    store = store_for(archive, more.get("store"))
    facts: EmbedderFacts = more.get("facts") or meaning.default_facts()
    seen = look(archive, names, store, teller, model=model(), **{k: more[k] for k in LOOK_KEYS if k in more})
    teller.tell(log=looking(seen.people))
    writing = seen.to_write
    teller.tell(step="write", total=len(writing), log=to_write(len(writing), seen.people - seen.not_read - len(writing), seen.not_read))
    info = seen.planner.info
    calls = Calls(_port(more.get("llm_factory"), info) if writing else None, info, seen.planner.mask, cancel,
                  Progress(teller, len(writing)).heard)  # fmt: skip
    written, unplanned = calls.run(
        writing, int(more.get("parallel") or PEOPLE_PARALLEL), lambda item: prepare(seen, item)
    )
    usage = usage_of(written)
    stop = "run_failed" if unplanned is not None else calls.stop_code
    profiles = assemble(seen, written)
    embedded = tokens = 0
    teller.tell(step="save", done=0, total=1)
    if stop is None:
        factory = more.get("embedder_factory") or meaning.embedder_for
        profiles, embedded, tokens, stop = embed(profiles, facts, factory, more.get("count_tokens") or _count_tokens())
    completed = now() if stop is None else (seen.held.meta.completed_at if seen.held else None)
    save(store, Held(StoreMeta(info.id, facts.model, facts.dimensions, now(), completed), profiles), usage)
    result = _result(seen, written, profiles, embedded, tokens, usage, facts)
    _spent(result, stop)
    if stop is not None:
        raise failure(stop, usage.model_dump(mode="json"), model=info.id)
    teller.tell(step="save", done=1, total=1, log=all_written(result))
    return result


def _count_tokens() -> Callable[[str], int]:
    from meaning_index import default_count_tokens

    return default_count_tokens


def _result(
    seen: Look, written: Any, profiles: Any, embedded: int, tokens: int, usage: Usage, facts: EmbedderFacts
) -> dict[str, Any]:
    built = sum(one.text is not None for one in written)
    kept_keys = {profile.key for profile in profiles}
    return {
        "people": seen.people,
        "built": built,
        "kept": seen.people - seen.not_read - len(seen.to_write),
        "failed": sum(one.text is None for one in written),
        "notRead": seen.not_read,
        "removed": sum(key not in kept_keys for key in seen.rows),
        "embedded": embedded,
        "embedTokens": tokens,
        "usd": float(usage.cost_usd + tokens * facts.usd_per_mtok / 1_000_000),
        "usage": usage.model_dump(mode="json"),
    }


def _spent(result: Mapping[str, Any], stop: str | None) -> None:
    """One line of the spend log for each end of a build: counts and the cost, never an id or a name."""
    how = "the people job" if stop is None else f"the people job stopped ({stop})"
    spend.info(
        "%s: %d profiles written, %d kept, %d failed, %d model calls, %.4f USD",
        how, result["built"], result["kept"], result["failed"], result["usage"]["calls"], result["usd"],
    )  # fmt: skip

