"""The `digest` job: one conversation read for chat, outside a question.

A question reads at most `DIGEST_PARTS_LIMIT` parts that are not read yet
(`memory.py`). The `memory` job (`memory_prep.py`) reads every conversation
of the memory ahead of time. This job reads one conversation, with optional
days, when the app asks for it: for example a conversation that is too long
to read inside a question. It uses the model of a digest, so the parts it
reads are in the notes for each later question. It has no limit of parts,
and the model port of a report.

`run` gives the counts and the cost of the run. A listener gets the progress
in the shape of a report job: `{"progress": {"step", "of", "label"}}`, with a
line for the job under `log`. A line holds the label of a part and counts. It
holds no text of a message, no name and no handle.

A failure is a `DigestFailure`: the code of the failure of `digests.build`,
and a sentence of this job. No sentence names a report: this job writes none.
"""

from __future__ import annotations

from typing import Any

from archive import Archive
from recall_services import digests, keys, memory, models
from recall_services.names import Names
from recall_services.report_failures import ReportFailure
from recall_services.report_progress import Listener, Teller
from report_engine import ProgressEvent

AGAIN = "then ask again."
NOT_READ = "The conversation was not read."
COULD_NOT_BE_READ = "The conversation could not be read for chat. Ask again."


class DigestFailure(ReportFailure):
    """A `digest` job that read no conversation. The message is the sentence for the person."""


def sentence_of(code: str, provider: str, model: str) -> str:
    """One plain sentence for a failure code of a digest. A code with no sentence of its own gets a general one."""
    sentences = {
        "no_key": f"{keys.NO_KEY}, {AGAIN}",
        "key_rejected": f"{provider} refused the API key. {keys.CHECK_KEY}, {AGAIN}",
        "unknown_model": f"{provider} does not know the model that reads conversations. {NOT_READ}",
        "rate_limit": f"{provider} is limiting requests, or the account has no credit left. Wait a moment, {AGAIN}",
        "offline": f"{provider} could not be reached. Check the internet connection, {AGAIN}",
        "timeout": "The model took too long to answer. Ask again.",
        "model_error": "The model could not read the conversation. Ask again.",
        "empty": "This conversation has no messages with text in these dates.",
        "too_large": f"The model can not read this conversation with its limits. {NOT_READ}",
        "archive_error": f"The messages of this conversation could not be read. Update the archive, {AGAIN}",
        "cache_unreadable": f"What earlier runs read could not be read or kept. {NOT_READ}",
        "cancelled": "The reading was stopped. What was read is kept.",
        "run_failed": "The reading failed. What was read is kept: ask again.",
    }
    return sentences.get(code, COULD_NOT_BE_READ)


def _provider(model: str) -> str:
    """The name of the provider of a model for a sentence. `digests.build` refused a model that is not on file before."""
    return models.provider_name(models.facts(model).provider)


class _Teller(Teller):
    """The events of a digest as the updates of a job: the conversation is read, then each part."""

    def engine_event(self, event: ProgressEvent) -> None:
        if event.step == "prepare" and event.state == "start":
            self.tell(step=(1, 1, "Reading the conversation"), log="Reading the conversation")
        elif event.step == "map":
            self._part(event, event.total + 1)


def _parts(count: int) -> str:
    return f"{count:,} part{'' if count == 1 else 's'}"


def run(
    conversation_id: str,
    since: str | None = None,
    until: str | None = None,
    on_progress: Listener | None = None,
    cancel: Any = None,
    *,
    archive: Archive,
    names: Names,
    **more: Any,
) -> dict[str, Any]:
    """Read one conversation for chat. Gives `conversation`, `periods`, `periodsBuilt`,
    `periodsCached`, `periodsFailed`, `usd` and `usage`.

    `more` goes to `digests.build`: tests give the places, the options, the
    counter and a model port. Raises `DigestFailure`, with a code and one
    plain sentence, a ValueError for a missing conversation or a bad day, and
    a ModelChoiceError for a model that can not be run.
    """
    teller = _Teller(on_progress)
    settings = {"model": memory.DIGEST_MODEL, **more}
    model = str(settings["model"] or "") or models.default_id()
    try:
        found = digests.build(
            conversation_id, since, until, cancel, teller.engine_event, archive=archive, names=names, **settings
        )
    except ReportFailure as failure:
        said = sentence_of(failure.code, _provider(model), model)
        raise DigestFailure(failure.code, said, failure.usage) from failure
    teller.tell(
        log=f"The conversation is read for chat: {_parts(found.periods_built)} read now, "
        f"{found.periods_cached:,} from earlier runs, {found.periods_failed:,} not read."
    )
    return {
        "conversation": found.conversation_id,
        "periods": len(found.periods),
        "periodsBuilt": found.periods_built,
        "periodsCached": found.periods_cached,
        "periodsFailed": found.periods_failed,
        "usd": float(found.usage.cost_usd),
        "usage": found.usage.model_dump(mode="json"),
    }
