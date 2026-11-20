"""`--with-memory`: build the memory of the world with the service functions the server uses.

The order is the order of the first run of the app (`first_run_job.py`):
the index of meanings, then the memory (the digest of each conversation),
then the people index. Each step calls the provider and costs money, so a
run does this only when asked. Call it inside `places.isolated_saves`, so
every file lands in the temp folder of the world.

`estimate_memory` gives what the three steps would cost, with no call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from archive import Archive
from recall_services import meaning, memory_prep, people
from recall_services.first_run import MEMORY_MIN_MESSAGES
from recall_services.names import Names

PER_MTOK = 1_000_000

Say = Callable[[str], None]


def _meaning_usd(tokens: int) -> float:
    return tokens * meaning.default_facts().usd_per_mtok / PER_MTOK


def build_memory(archive: Archive, names: Names, say: Say = print) -> dict[str, Any]:
    """Build the three parts. Gives the USD of each. A failure of a step is raised as the service raises it."""
    say("memory: building the index of meanings")
    found = meaning.build(archive, names)
    meaning_usd = _meaning_usd(int(found.get("tokens") or 0))
    say("memory: reading each conversation")
    read = memory_prep.run(archive=archive, names=names, min_messages=MEMORY_MIN_MESSAGES)
    say("memory: building the people index")
    profiles = people.build(archive, names)
    spent = {
        "meaning_usd": meaning_usd,
        "memory_usd": float(read.get("usd") or 0.0),
        "people_usd": float(profiles.get("usd") or 0.0),
    }
    return {**spent, "usd": sum(spent.values())}


def estimate_memory(archive: Archive, names: Names) -> dict[str, Any]:
    """What the three builds would cost now. No model call and no key."""
    costs = {
        "meaning_usd": float(meaning.estimate(archive, names)["estimatedCost"]),
        "memory_usd": float(memory_prep.payload(archive, names, min_messages=MEMORY_MIN_MESSAGES)["estimatedCost"]),
        "people_usd": float(people.estimate(archive, names)["estimatedCost"]),
    }
    return {**costs, "usd": sum(costs.values())}
