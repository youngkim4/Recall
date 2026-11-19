"""The first run of a new person: what getting chat ready costs, and what is ready (`GET /api/first-run`).

Chat is ready when three things are built, in this order:

1. `meaning`: the index of meanings (`meaning.py`);
2. `memory`: the memory of every conversation, read with
   `MEMORY_MIN_MESSAGES` (1): a first run reads every conversation;
3. `people`: the people index (`people.py`), written from the memory.

`estimate` says the state and the cost of each step. It makes no model
call and no request, and needs no key:

- `meaning` and `memory` are the estimates of `GET /api/meaning` and
  `GET /api/memory`.
- `people` before the memory is read has nothing to measure: its cost is an
  upper bound from the plan of the memory (`people_bound`). After the memory
  is read it is `TRIES` times the estimate of `GET /api/people`.
- `estimatedCost` is the sum of the steps that are not ready. `ready` is
  true when all three are ready.

The job that builds the steps is in `first_run_job.py`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from archive import Archive
from archive import ArchiveError as ArchiveFault
from meaning_index import EmbedderFacts, StoreError
from recall_services import keys, meaning, memory_prep, models, people
from recall_services.memory_prep_read import Planned
from recall_services.models import ModelChoiceError
from recall_services.names import Names
from recall_services.people_inputs import subjects_of
from recall_services.people_plans import AROUND_NOTES_CHARS, INPUT_CHARS_PER_TOKEN, MOMENT_HEAD_CHARS, PART_HEAD_CHARS
from recall_services.people_prompt import NOTES_INTRO, OUTPUT_TOKENS, PROFILE_SYSTEM, PROFILE_TASK
from report_engine import ReportOptions, digest
from report_engine.ports import LLMUsage
from report_engine.pricing import call_cost_usd

STEP_NAMES = ("meaning", "memory", "people")
MEMORY_MIN_MESSAGES = 1

NO_KEY = f"{keys.NO_KEY}, then get chat ready again."
NO_MESSAGES = "The archive holds no message yet. Read your messages first, then get chat ready."
SKIPPED = {
    "meaning": "The index for the search by meaning is ready: nothing to build.",
    "memory": "The memory of chat is read: nothing to read.",
    "people": "The people index is ready: nothing to write.",
}
PEOPLE_NOT_PLANNED = "The cost of the people index could not be worked out."


def _step(name: str, state: str, cost: float, error: str | None = None, code: str | None = None) -> dict[str, Any]:
    return {"name": name, "state": state, "estimatedCost": float(cost), "error": error, "code": code}


def _shown(archive_held: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    waiting = [float(step["estimatedCost"]) for step in steps if step["state"] != "ready"]
    return {
        "archive": dict(archive_held),
        "steps": [dict(step) for step in steps],
        "estimatedCost": float(sum(waiting)),
        "ready": bool(steps) and not waiting,
    }


def nothing(conversations: int = 0, messages: int = 0) -> dict[str, Any]:
    """The answer for an archive that holds no message, or no archive: each step `none`, at no cost."""
    held = {"holdsMessages": False, "conversations": conversations, "messages": messages}
    return _shown(held, [_step(name, "none", 0.0) for name in STEP_NAMES])


def index_mark(archive_file: Any) -> str | None:
    """The id of the build of the index of meanings on disk, or None: a part of the key of the cached answer."""
    store = meaning.index_store(archive_file)
    if not store.folder.is_dir():
        return None
    try:
        meta = store.meta()
    except StoreError:
        return "unreadable"
    return None if meta is None else meta.build_id


def _meaning_step(archive: Archive, names: Names) -> dict[str, Any]:
    found = meaning.payload(archive, names)
    cost = float((found["estimate"] or {}).get("estimatedCost") or 0.0)
    return _step("meaning", found["status"]["state"], cost, found["error"], found["code"])


def _memory_step(archive: Archive, names: Names) -> tuple[dict[str, Any], tuple[Planned, ...]]:
    found, planned = memory_prep.plan(archive, names, min_messages=MEMORY_MIN_MESSAGES)
    return _step("memory", found["state"], found["estimatedCost"], found["error"], found["code"]), planned


def notes_tokens_per_part(info: Any, options: ReportOptions | None = None) -> int:
    """The most input tokens that the notes of one part can add to a profile request.

    `input_bound` counts a part as its summary and, for each moment, a head,
    a title and a detail. The digest cuts each of those (`report_engine.digest`
    limits), so that sum is a bound. The reply of a part is a bound too: its
    output limit (`map_max_output_tokens`, at most the model's own). The
    smaller one is taken.
    """
    most_chars = digest.MAX_SUMMARY_CHARS + digest.MAX_MOMENTS_PER_PERIOD * (
        MOMENT_HEAD_CHARS + digest.MAX_TITLE_CHARS + digest.MAX_DETAIL_CHARS
    )
    reply_tokens = min((options or ReportOptions()).map_max_output_tokens, info.max_output_tokens)
    return min(reply_tokens, math.ceil(most_chars / INPUT_CHARS_PER_TOKEN))


def people_bound(
    archive: Archive,
    names: Names,
    planned: Sequence[Planned],
    *,
    options: ReportOptions | None = None,
    facts: EmbedderFacts | None = None,
) -> float:
    """An upper bound of the first build of the people index, with every try, before the memory is read.

    For each profile, its parts are the parts that the plan of the memory
    gives its conversations. The notes of a part are at most
    `notes_tokens_per_part`; the frame of the request and the head of each
    part are counted as `input_bound` counts them. The reply is the fullest
    one that the limits of a profile allow (`OUTPUT_TOKENS`), and its
    embedding counts as many tokens. A profile with no part has no text and
    is not written.
    """
    info = models.model_info(people.model())
    part_tokens = notes_tokens_per_part(info, options)
    embed_usd = OUTPUT_TOKENS * (facts or meaning.default_facts()).usd_per_mtok / 1_000_000
    parts = {one.conversation: one.periods for one in planned}
    frame = len(PROFILE_SYSTEM) + len(PROFILE_TASK) + len(NOTES_INTRO) + AROUND_NOTES_CHARS
    usd = 0.0
    for subject in subjects_of(archive, names):
        count = sum(parts.get(conversation, 0) for conversation in subject.conversations)
        if count == 0:
            continue
        around = math.ceil((frame + len(subject.label) + count * PART_HEAD_CHARS) / INPUT_CHARS_PER_TOKEN)
        usage = LLMUsage(input_tokens=around + count * part_tokens, output_tokens=OUTPUT_TOKENS)
        usd += call_cost_usd(info, usage) + embed_usd
    return float(people.TRIES * usd)


def _people_step(archive: Archive, names: Names, memory_ready: bool, planned: Sequence[Planned]) -> dict[str, Any]:
    """Before the memory is read: `none` or `partial`, and the bound. After: the estimate of the route.

    The step is `ready` only after a build of this archive completed, and
    when nothing is left to write or to embed. A profile whose memory can not
    be read (a conversation that can not be planned) does not hold it back.
    """
    if not memory_ready:
        held = people.status(archive, names)["state"]
        state = "none" if held == "none" else "partial"
        try:
            return _step("people", state, people_bound(archive, names, planned))
        except ModelChoiceError as error:
            return _step("people", state, 0.0, PEOPLE_NOT_PLANNED, getattr(error, "code", "model_not_run"))
        except ArchiveFault:
            return _step("people", state, 0.0, PEOPLE_NOT_PLANNED, "archive_error")
    found = people.payload(archive, names)
    if found["error"]:
        return _step("people", "none", 0.0, found["error"], found["code"])
    nothing_waits = int(found["toWrite"]) + int(found["toEmbed"]) == 0
    done = people.completed_before(archive.path) and (found["state"] == "ready" or nothing_waits)
    state = "ready" if done else "partial" if found["state"] == "ready" else found["state"]
    return _step("people", state, people.TRIES * float(found["estimatedCost"]))


def estimate(archive: Archive, names: Names) -> dict[str, Any]:
    """`archive`, `steps`, `estimatedCost` and `ready`. No model call, no request, no key."""
    summary = archive.summary()
    conversations, messages = int(summary["conversations"]), int(summary["messages"])
    if messages == 0:
        return nothing(conversations, messages)
    held = {"holdsMessages": True, "conversations": conversations, "messages": messages}
    meaning_step = _meaning_step(archive, names)
    memory_step, planned = _memory_step(archive, names)
    people_step = _people_step(archive, names, memory_step["state"] == "ready", planned)
    return _shown(held, [meaning_step, memory_step, people_step])
