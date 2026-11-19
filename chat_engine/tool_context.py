"""What every tool needs: the archive, the labels, the references, the scope.

The scope is enforced here, in one place. A tool gets conversation ids and
days only through `conversations()` and `days()`, so it cannot widen them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from .cancel import CancelToken
from .directory import Directory
from .ports import ChatArchive, Conversation, DigestSource, Labeler, MeaningSearch, Message, PeopleSearch
from .registry import ConversationAliases, EvidenceRegistry
from .render import ConversationBook, Limits
from .scope import Scope
from .textnorm import fold
from .tool_args import ToolInputError

OUTSIDE_SCOPE = "that is outside the scope the user chose. Stay inside the scope."
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_NAMED_IN_WHERE = 3
_ECHO_CHARS = 40


@dataclass(frozen=True)
class ToolOutcome:
    """The text the model reads, and one plain line on what the lookup did."""

    text: str
    summary: str
    ok: bool = True


Report = Callable[[object], None]


@dataclass(frozen=True)
class Fetch:
    """Slow work of a lookup: a call to a port that touches nothing of the engine. The turn runs it beside the stream.

    work: does the work. It gives each raw report of progress to the function it gets, from any thread.
    event: one raw report as an event of the stream, or None to drop it. The turn calls it on its own thread.
    """

    work: Callable[[Report], None]
    event: Callable[[object], Optional[dict[str, Any]]]


@dataclass(frozen=True)
class Prepared:
    """A checked tool call: a status line to show, and the lookup to run.

    own_budget: the tokens the result may hold when the lookup has a budget of
    its own. None: it spends from the budget of the question.
    repeated: the question already ran this lookup, so it is not run again.
    answers_repeats: the tool itself knows what a repeat of this call gets, so
    the turn sends a call with the same arguments to the tool again.
    fetch: slow work that the turn runs before `run`, beside the stream, so
    its progress reaches the reader. `run` does the work itself when nobody
    ran `fetch`.
    """

    status: str
    run: Callable[[int], ToolOutcome]
    own_budget: Optional[int] = None
    repeated: bool = False
    answers_repeats: bool = False
    fetch: Optional[Fetch] = None


@dataclass(frozen=True)
class Memory:
    """The optional ports of one question, and the limits of a digest.

    A port that is None is not offered: its tool is not in the tool list.
    `cancel` is the token of the question. It goes to the digest source.
    """

    meaning: Optional[MeaningSearch] = None
    digests: Optional[DigestSource] = None
    digest_tokens: int = 24_000
    digest_usd_limit: float = 0.5
    cancel: CancelToken = field(default_factory=CancelToken)
    digest_parts_limit: int = 4  # the most periods not yet read that one digest reads inside a question
    people: Optional[PeopleSearch] = None


def month_year(timestamp: str) -> str:
    text = str(timestamp or "")
    try:
        return f"{_MONTHS[int(text[5:7]) - 1]} {int(text[:4])}"
    except (ValueError, IndexError):
        return "an unknown date"


def plural(count: int, word: str, many: str = "") -> str:
    return f"{count:,} {word if count == 1 else (many or word + 's')}"


class ToolContext:
    def __init__(
        self,
        archive: ChatArchive,
        labeler: Labeler,
        registry: EvidenceRegistry,
        aliases: ConversationAliases,
        scope: Scope,
        limits: Limits,
        memory: Optional[Memory] = None,
    ) -> None:
        self.archive = archive
        self.labeler = labeler
        self.registry = registry
        self.scope = scope
        self.limits = limits
        self.memory = memory or Memory()
        # the digests of this question that gave an answer: "shown" or "failed"
        self.digests_asked: dict[tuple[str, Optional[str], Optional[str]], str] = {}
        # the digests too long to read inside this question, and the line each got
        self.digests_long: dict[tuple[str, Optional[str], Optional[str]], ToolOutcome] = {}
        self.digests_built = 0
        self.digest_estimates = 0
        self.digest_usd = 0.0  # what the digests of this question cost, as far as the source said
        self.book = ConversationBook(archive, labeler, aliases)
        self._directory: Optional[Directory] = None
        self._listed: Optional[list[Conversation]] = None

    def listed(self) -> list[Conversation]:
        """Every conversation of the archive, whatever the scope.

        The labeler sees each conversation and each member here. So it can
        mask the handle of any person of the archive in any text, also a
        person outside the scope. Nothing of this goes to the model.
        """
        if self._listed is None:
            listed = self.archive.conversations(limit=None, offset=0, order="count")
            for conversation in listed:
                self.book.put(conversation)
                self.book.label(str(conversation["id"]))
                for handle in conversation.get("members") or []:
                    self.labeler.sender(str(handle))
            self._listed = listed
        return self._listed

    def directory(self) -> Directory:
        if self._directory is None:
            self._directory = Directory.load(self.archive, self.book, self.labeler, self.scope, self.listed())
        return self._directory

    # --- scope ---

    def conversations(self, values: Optional[Sequence[str]]) -> Optional[tuple[str, ...]]:
        """Aliases or exact names, as ids the scope allows. None means all."""
        requested = None if values is None else [self._resolve(value) for value in values]
        allowed = self.scope.narrow(requested)
        if requested is not None and len(allowed or ()) < len(set(requested)):
            raise ToolInputError(OUTSIDE_SCOPE)
        return allowed

    def days(self, since: Optional[str], until: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        low, high = self.scope.clamp(since, until)
        if low and high and low > high:
            if since and until and since > until:
                raise ToolInputError("since is after until.")
            raise ToolInputError(OUTSIDE_SCOPE)
        return low, high

    def allows(self, message: Message) -> bool:
        return self.scope.allows(str(message["conversation"])) and self.scope.allows_time(
            str(message.get("timestamp") or "")
        )

    def _resolve(self, value: str) -> str:
        found = self.book.resolve(value)
        if found:
            return found
        wanted = fold(value).strip()
        matches = [e for e in self.directory().entries if wanted and fold(e.label) == wanted]
        if len(matches) == 1:
            return str(matches[0].conversation["id"])
        shown = value if len(value) <= _ECHO_CHARS else value[:_ECHO_CHARS] + "..."
        if matches:
            raise ToolInputError(
                f'more than one conversation is named "{shown}". '
                "Call list_conversations and use the c-number."
            )
        raise ToolInputError(
            f'"{shown}" is not a known conversation. Call list_conversations to get its c-number.'
        )

    # --- words ---

    def where(self, conversation_ids: Optional[Sequence[str]]) -> str:
        if conversation_ids is None:
            return "all conversations"
        if len(conversation_ids) > _NAMED_IN_WHERE:
            return f"{len(conversation_ids)} conversations"
        return ", ".join(self.book.short(conversation_id) for conversation_id in conversation_ids)

    def names(self, conversation_ids: Optional[Sequence[str]]) -> str:
        """Labels without aliases, for the status line the user reads."""
        if conversation_ids is None:
            return ""
        if len(conversation_ids) > _NAMED_IN_WHERE:
            return f"{len(conversation_ids)} conversations"
        return ", ".join(dict.fromkeys(self.book.label(c) for c in conversation_ids))

    @staticmethod
    def when(since: Optional[str], until: Optional[str]) -> str:
        if since and until:
            return f", {since} to {until}"
        if since:
            return f", from {since}"
        return f", up to {until}" if until else ""

    @staticmethod
    def who(from_me: Optional[bool]) -> str:
        if from_me is None:
            return ""
        return ", sent by you" if from_me else ", sent by others"
