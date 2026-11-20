"""`--with-memory`: build the memory of the world with the service functions the server uses.

The order is the order of the first run of the app (`first_run_job.py`):
the index of meanings, then the memory (the digest of each conversation),
then the people index. Each step calls the provider and costs money, so a
run does this only when asked. Call it inside `places.isolated_saves`, so
every file lands in the temp folder of the world.

`estimate_memory` gives what the three steps would cost, with no call. The
people index can be priced only for people whose memory is read; before the
memory step the others are priced at an assumed profile size.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from archive import Archive
from chat_engine.ports import Usage
from evals.recorder import price
from recall_services import meaning, memory_prep, people
from recall_services.first_run import MEMORY_MIN_MESSAGES
from recall_services.names import Names

PER_MTOK = 1_000_000
# the assumed size of one profile call of the people index, for a person whose memory is not read yet
PROFILE_INPUT_TOKENS = 4000
PROFILE_OUTPUT_TOKENS = 600

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
    if meaning.RECIPE.context:
        # the context line of a segment holds the summary of its period: build again now that the memory is read
        say("memory: building the index of meanings again with the summaries of the memory")
        again = meaning.build(archive, names)
        meaning_usd += _meaning_usd(int(again.get("tokens") or 0))
    say("memory: building the people index")
    profiles = people.build(archive, names)
    spent = {
        "meaning_usd": meaning_usd,
        "memory_usd": float(read.get("usd") or 0.0),
        "people_usd": float(profiles.get("usd") or 0.0),
    }
    return {**spent, "usd": sum(spent.values())}


def _people_usd(found: dict[str, Any]) -> tuple[float, bool]:
    """The cost of the people index, and whether it is assumed.

    `people.estimate` prices only the people whose memory is read already.
    Before the memory step it knows the others only by number (`notRead`):
    each of them is priced as one profile call of the assumed size.
    """
    waiting = int(found.get("notRead") or 0)
    if not waiting:
        return float(found["estimatedCost"]), False
    one = price(people.model(), Usage(input_tokens=PROFILE_INPUT_TOKENS, output_tokens=PROFILE_OUTPUT_TOKENS))
    return float(found["estimatedCost"]) + waiting * one, True


def estimate_memory(archive: Archive, names: Names) -> dict[str, Any]:
    """What the three builds would cost now, with the estimates the server uses. No model call and no key."""
    people_usd, assumed = _people_usd(people.estimate(archive, names))
    costs = {
        "meaning_usd": float(meaning.estimate(archive, names)["estimatedCost"]),
        "memory_usd": float(memory_prep.payload(archive, names, min_messages=MEMORY_MIN_MESSAGES)["estimatedCost"]),
        "people_usd": people_usd,
    }
    return {**costs, "usd": sum(costs.values()), "people_assumed": assumed}
