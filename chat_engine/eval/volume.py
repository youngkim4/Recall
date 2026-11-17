"""Run the chat tools on a message export at full volume and print numbers.

    python -m chat_engine.eval.volume PATH_TO_messages.csv
    python -m chat_engine.eval.volume --invented 565821

It reads the file (or makes an invented export of that many rows), builds a
`FrameArchive`, runs every tool and three whole questions with the scripted
model, and prints latency and result sizes.

It prints numbers and fixed labels only: no message text, no name, no handle.
It writes no file. It blocks the network before it reads the file. It calls
no AI model: the answers of the "model" are a fixed script.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import socket
import statistics
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from ..browse import browse
from ..engine import ChatEngine, EngineConfig
from ..fake_llm import FakeLLM, Step, call, say
from ..frame_archive import FrameArchive
from ..frame_load import read_export
from ..labeler import MaskingLabeler
from ..ports import LLMRequest
from ..registry import ConversationAliases, EvidenceRegistry
from ..render import Limits, estimate_tokens
from ..scope import Scope
from ..session import ChatSession
from ..tools import ToolBox
from .volume_cases import Case, Picks, browse_cases, figure_cases, pick, search_cases
from .volume_data import invented_export

_REF_LINE_RE = re.compile(r"^(m\d+) ", re.MULTILINE)
_REACTION_START_RE = r'(?:Loved|Liked|Disliked|Laughed at|Emphasized|Questioned) [“"]'
_LONG_TEXT = 10_000
_VERY_LONG_TEXT = 100_000
Out = Callable[[str], None]


class _NetworkBlocked(RuntimeError):
    pass


def _refuse(*_args: object, **_kwargs: object) -> None:
    raise _NetworkBlocked("the volume run must not use the network")


def block_network() -> None:
    """After this call no code of this process can open a connection."""
    targets = (
        (socket.socket, "connect"),
        (socket.socket, "connect_ex"),
        (socket, "create_connection"),
        (socket, "getaddrinfo"),
    )
    for owner, name in targets:
        setattr(owner, name, _refuse)


@dataclass(frozen=True)
class Measured:
    label: str
    millis: tuple[float, ...]
    chars: int
    tokens: int
    lines: int
    failed: str = ""  # the type of the error, never its text


def _timed(action: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    value = action()
    return value, (time.perf_counter() - started) * 1000.0


def _peak_megabytes() -> float:
    """The most memory this process has used. 0 where the system cannot tell."""
    try:
        import resource  # noqa: PLC0415 - not on every system
    except ImportError:
        return 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def measure(label: str, action: Callable[[], str], repeat: int) -> Measured:
    """Run one lookup `repeat` times. A failure is reported by the type of its error only."""
    millis: list[float] = []
    text = ""
    try:
        for _ in range(max(1, repeat)):
            text, spent = _timed(action)
            millis.append(spent)
    except Exception as error:  # the report must not show what the error says: it can hold data
        return Measured(label, tuple(millis), 0, 0, 0, failed=type(error).__name__)
    return Measured(label, tuple(millis), len(text), estimate_tokens(text), len(_REF_LINE_RE.findall(text)))


def _row(found: Measured, budget: int) -> str:
    if found.failed:
        return f"  {found.label:44s} failed: {found.failed}"
    over = "  OVER THE BUDGET" if found.tokens > budget else ""
    return (
        f"  {found.label:44s} {statistics.median(found.millis):9.1f} ms  (max {max(found.millis):8.1f})"
        f"  {found.chars:7,d} chars  {found.tokens:6,d} tokens  {found.lines:3d} messages{over}"
    )


# --- the archive ------------------------------------------------------------


def _read(path: str, invented: int) -> tuple[FrameArchive, int, int, float, float]:
    """The archive, the rows of the export, the rows that start as a reaction phrase, and two timings."""
    frame, read_ms = _timed(lambda: invented_export(invented) if invented else read_export(path))
    starts = int(frame["text"].str.match(_REACTION_START_RE).sum()) if "text" in frame.columns else 0
    archive, build_ms = _timed(lambda: FrameArchive(frame))
    return archive, len(frame), starts, read_ms, build_ms


def load(path: str, invented: int, out: Out) -> FrameArchive:
    out("export: invented, made-up text only" if invented else f"file: {os.path.getsize(path):,} bytes")
    archive, rows, starts, read_ms, build_ms = _read(path, invented)
    dropped = archive.dropped
    lengths = archive.text_lengths()
    summary = archive.summary()
    out(f"rows in the file: {rows:,}")
    out(f"read (or make) the export: {read_ms:,.0f} ms; build the archive: {build_ms:,.0f} ms")
    out(f"messages: {summary['messages']:,}; conversations: {summary['conversations']:,}")
    out(
        f"rows left out: reaction phrases {dropped['reactions']:,}, unreadable rows {dropped['bad_rows']:,}, "
        f"second rows of one id {dropped['duplicates']:,}"
    )
    out(f"rows that start as a reaction phrase but do not end as one (kept as messages): "
        f"{starts - dropped['reactions']:,}")
    out(f"messages that are only an attachment: {archive.attachment_only:,}")
    out(
        f"texts of {_LONG_TEXT:,} characters or more: {int((lengths >= _LONG_TEXT).sum()):,}; of "
        f"{_VERY_LONG_TEXT:,} or more: {int((lengths >= _VERY_LONG_TEXT).sum()):,}; "
        f"longest: {int(lengths.max()) if len(lengths) else 0:,} characters"
    )
    index, index_ms = _timed(archive.build_index)
    out(f"build the search index: {index_ms:,.0f} ms; words: {index.words:,}; entries: {index.entries:,}")
    out(f"memory at its peak so far: {_peak_megabytes():,.0f} MB")
    return archive


# --- the tools --------------------------------------------------------------


class _WithoutReader:
    """The same archive without `iter_messages`, to measure the page walk."""

    def __init__(self, archive: FrameArchive) -> None:
        self._archive = archive

    def __getattr__(self, name: str) -> Any:
        if name == "iter_messages":
            raise AttributeError(name)
        return getattr(self._archive, name)


def run_tools(archive: FrameArchive, picks: Picks, limits: Limits, repeat: int, out: Out) -> list[Measured]:
    box = ToolBox(archive, MaskingLabeler(), EvidenceRegistry(), ConversationAliases(), Scope(), limits)
    _, listed_ms = _timed(box.context.directory)
    out(f"read and label every conversation (once for each question): {listed_ms:,.0f} ms")
    ids = [picks.largest, picks.middle, *([picks.group] if picks.group else [])]
    alias = {conversation_id: box.context.book.alias(conversation_id) for conversation_id in ids}
    budget = limits.result_tokens

    def lookup(case: Case) -> Callable[[], str]:
        return lambda: box.prepare(case.tool, json.dumps(case.arguments)).run(budget).text

    found = [measure(c.label, lookup(c), repeat) for c in search_cases(picks, alias)]
    found += [measure(c.label, lookup(c), repeat) for c in browse_cases(picks, alias)]
    found += _walks(archive, picks, repeat)
    found += [measure(c.label, lookup(c), repeat) for c in figure_cases(picks, alias)]
    found += _reads(box, archive, picks, budget, repeat)
    out(f"tools (budget of one result: {budget:,} tokens; cap of one message: {limits.max_message_chars:,} characters)")
    for one in found:
        out(_row(one, budget))
    return found


def _walks(archive: FrameArchive, picks: Picks, repeat: int) -> list[Measured]:
    """The first and the last messages through pages only, as an archive without a reader needs."""
    plain: Any = _WithoutReader(archive)

    def walk(edge: str, since: Optional[str] = None, until: Optional[str] = None) -> Callable[[], str]:
        def run() -> str:
            got = browse(plain, picks.largest, edge=edge, limit=8, since=since, until=until)
            if not (got.anchored and got.reached_edge):
                raise LookupError("the walk stopped at its limit")
            return "".join(f"m{number} \n" for number in range(1, len(got.messages) + 1))
        return run

    year = {"since": picks.chat_since, "until": picks.chat_until}
    empty = {"since": picks.empty_since, "until": picks.empty_until}
    return [
        measure("page walk to the first messages", walk("oldest"), repeat),
        measure("page walk to the first messages of one year", walk("oldest", **year), repeat),
        measure("page walk to the latest messages up to a day", walk("newest", until=picks.chat_until), repeat),
        measure("page walk to a year with no message", walk("oldest", **empty), repeat),
        measure("page walk to the latest before the start", walk("newest", until=picks.empty_until), repeat),
    ]


def _reads(box: ToolBox, archive: FrameArchive, picks: Picks, budget: int, repeat: int) -> list[Measured]:
    """read_around on a message from a search, and on the message with the longest text."""
    registry = box.context.registry

    def read(reference: str, radius: int) -> Callable[[], str]:
        arguments = json.dumps({"reference": reference, "radius": radius})
        return lambda: box.prepare("read_around", arguments).run(budget).text

    found: list[Measured] = []
    if len(registry):
        found += [measure(f"read around a message, radius {radius}", read("m1", radius), repeat) for radius in (6, 15)]
    for message in archive.longest_messages(1):
        found.append(measure("read around the longest text", read(registry.add(message), 6), repeat))
    return found


# --- whole questions with the scripted model --------------------------------


def _script(picks: Picks, marks: dict[str, float]) -> list[Step]:
    """A fixed stand-in for a model: two lookups, one read, one answer with a quote check."""

    def first(_request: LLMRequest) -> list:
        marks.setdefault("first_model_call", time.perf_counter())
        return [
            call("search_messages", "v1", query=f"{picks.medium} {picks.rare} {picks.common}", match="any"),
            call("count_messages", "v2", text=picks.medium, group_by="year"),
        ]

    def second(request: LLMRequest) -> list:
        results = [m.content for m in request.messages if m.role == "tool"]
        ref = _REF_LINE_RE.search(results[0]) if results else None
        marks["ref"] = float(ref.group(1)[1:]) if ref else 0.0
        if not ref:
            return say("There is no evidence for this.")
        return [call("read_around", "v3", reference=ref.group(1), radius=6)]

    def third(_request: LLMRequest) -> list:
        ref = int(marks.get("ref", 0))
        return say(f'This is the answer [m{ref}] with "a quote that no message holds".' if ref else "No evidence.")

    return [first, second, third]


def run_questions(archive: FrameArchive, picks: Picks, config: EngineConfig, out: Out) -> None:
    out("whole questions with the scripted model (seeded lookups, 3 lookups of the model, quote check)")
    session = ChatSession()
    question = f"{picks.medium} {picks.rare} {picks.common}?"
    plans = [
        ("question on the whole archive", Scope(), session),
        ("follow-up question in the same chat", Scope(), session),
        ("question on the largest conversation", Scope(conversations=(picks.largest,)), ChatSession()),
    ]
    for label, scope, chat in plans:
        marks: dict[str, float] = {}
        fake = FakeLLM(_script(picks, marks))
        engine = ChatEngine(archive, fake, MaskingLabeler(), config)
        started = time.perf_counter()
        try:
            result = engine.ask(question, scope=scope, session=chat)
        except Exception as error:  # only the type: the text of an error can hold data
            out(f"  {label}: failed: {type(error).__name__}")
            continue
        total = (time.perf_counter() - started) * 1000.0
        seeded = (marks.get("first_model_call", started) - started) * 1000.0
        prompt = sum(len(m.content) for m in fake.requests[0].messages)
        largest = max(sum(estimate_tokens(m.content) for m in request.messages) for request in fake.requests)
        results = sum(estimate_tokens(m.content) for m in fake.requests[-1].messages if m.role == "tool")
        out(
            f"  {label}: {total:,.0f} ms in all, {seeded:,.0f} ms before the first model call; first prompt "
            f"{prompt:,} chars; largest request {largest:,} tokens; tool results {results:,} tokens; "
            f"result {len(json.dumps(result)):,} chars with {len(result['citations'])} citations, "
            f"{len(result['unverified_quotes'])} flagged quotes"
        )


# --- the run ----------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None, out: Out = print) -> int:
    parser = argparse.ArgumentParser(prog="chat_engine.eval.volume", description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("path", nargs="?", default="", help="the export file (messages.csv)")
    parser.add_argument("--invented", type=int, default=0, metavar="ROWS", help="use an invented export of ROWS rows")
    parser.add_argument("--repeat", type=int, default=3, help="runs of each lookup")
    arguments = parser.parse_args(argv)
    if bool(arguments.path) == (arguments.invented > 0):
        parser.error("give the export file, or --invented ROWS")
    block_network()
    warnings.simplefilter("ignore")  # a warning can quote a value of the file
    logged = logging.root.manager.disable
    logging.disable(logging.CRITICAL)  # a log line can hold the text of an error, and that text a row of the file
    try:
        return _run(arguments, out)
    finally:
        logging.disable(logged)


def _run(arguments: argparse.Namespace, out: Out) -> int:
    try:
        archive = load(arguments.path, arguments.invented, out)
        picks = pick(archive)
    except Exception as error:  # only the type: the text of an error can hold a row of the file
        out(f"could not read the export: {type(error).__name__}")
        return 1
    limits = Limits()
    try:
        found = run_tools(archive, picks, limits, arguments.repeat, out)
        run_questions(archive, picks, EngineConfig(model="scripted", max_output_tokens=800), out)
    except Exception as error:  # only the type, for the same reason
        out(f"the run stopped: {type(error).__name__}")
        return 1
    failed = [one for one in found if one.failed]
    over = [one for one in found if one.tokens > limits.result_tokens]
    slowest = max((max(one.millis) for one in found if one.millis), default=0.0)
    out(f"lookups: {len(found)}; failed: {len(failed)}; over the budget: {len(over)}; slowest: {slowest:,.0f} ms")
    out(f"memory at its peak: {_peak_megabytes():,.0f} MB")
    return 1 if failed or over else 0


if __name__ == "__main__":
    sys.exit(main())
