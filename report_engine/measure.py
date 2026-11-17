"""Numbers about an export file, to choose limits by.

    python -m report_engine.measure MESSAGES_CSV [--top 10] [--budgets 40000,80000,160000]

The file is read as the app reads it. No model is called and no network
connection is opened. The output holds numbers only: no text of a message,
no name and no handle. A conversation is named by its rank in size, and an
error by its type.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import time
import traceback
import warnings
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import pandas as pd

from report_engine.chunking import (
    HEADER_TOKENS,
    DaySpan,
    LineSizer,
    Period,
    fit_lines,
    line_sizer,
    measure_days,
    plan_periods,
)
from report_engine.estimate import estimate
from report_engine.frame_source import EXPORT_COLUMNS, FrameSource, local_moments
from report_engine.measure_file import (
    check_tapbacks,
    count_rows,
    measure_lengths,
    tapback_flags,
    texts_of,
)
from report_engine.model_info import ModelInfo
from report_engine.options import DEFAULT_PERIOD_TOKEN_BUDGET, ReportOptions
from report_engine.tokens import (
    ENGINE_ENCODING,
    HEURISTIC_FAMILY,
    HeuristicCounter,
    TokenCounter,
    TokenMeter,
    load_counter,
    meter_for,
)
from report_engine.transcript import MAX_MESSAGE_CHARS, Line, build_lines

Emit = Callable[[str], None]
NO_LIMIT = 10**12
DEFAULT_BUDGETS = (20_000, 40_000, DEFAULT_PERIOD_TOKEN_BUDGET, 120_000, 160_000)
PERIOD_GROUPS = ((1, 1), (2, 5), (6, 20), (21, 50), (51, NO_LIMIT))
PLAIN_NAME = "Contact"
LOOKUPS = ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo")


@dataclass(frozen=True)
class Sized:
    """One conversation as the engine reads it, with the size of each day."""

    lines: tuple[Line, ...]
    days: tuple[DaySpan, ...]
    seconds: float

    @property
    def tokens(self) -> int:
        return sum(day.tokens for day in self.days)


@dataclass(frozen=True)
class Total:
    """The size of one conversation, and its number of periods for each budget."""

    tokens: int
    lines: int
    cut: int
    periods: tuple[int, ...]


def _refuse(*args: object, **kwargs: object) -> None:
    raise OSError("this tool opens no network connection")


def block_network() -> None:
    """After this call nothing in the process can open a connection."""
    for name in ("connect", "connect_ex", "sendto", "sendmsg"):
        setattr(socket.socket, name, _refuse)
    for name in (*LOOKUPS, "create_connection"):
        setattr(socket, name, _refuse)


@contextmanager
def _quiet_engine() -> Iterator[None]:
    """The engine logs a count for each conversation. Here the totals say it."""
    logger = logging.getLogger("report_engine")
    before = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(before)


def load_frame(path: str) -> pd.DataFrame:
    """The export as the app loads it (ids and senders as text, timestamps
    parsed), with the timestamps as the engine then reads them. So a row with
    no timestamp here is a row that a report leaves out."""
    frame = pd.read_csv(path, dtype={"chat_id": "string", "sender": "string"})
    absent = [name for name in EXPORT_COLUMNS if name not in frame.columns]
    if absent:
        raise ValueError(f"the file has no column {', '.join(absent)}")
    parsed = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame.assign(timestamp=local_moments(parsed))


def open_model(family: str) -> ModelInfo:
    """A model with no limit and no price of its own: only the options decide."""
    return ModelInfo(
        id="measure",
        context_window=NO_LIMIT,
        max_input_tokens=NO_LIMIT,
        max_output_tokens=NO_LIMIT,
        input_usd_per_mtok=0.0,
        cached_input_usd_per_mtok=0.0,
        output_usd_per_mtok=0.0,
        long_context=None,
        tokens_per_minute=None,
        tokenizer_family=family,
    )


def _name(sender: str) -> str:
    return PLAIN_NAME


def read_sized(frame: pd.DataFrame, chat: str, meter: TokenMeter) -> Sized | None:
    """Lines and day sizes of one conversation. None when it has no message with text."""
    started = time.perf_counter()
    lines = build_lines(FrameSource(frame).messages(chat, None, None), _name)
    if not lines:
        return None
    return Sized(lines, measure_days(lines, meter), time.perf_counter() - started)


def plan(
    sized: Sized, meter: TokenMeter, budget: int
) -> tuple[tuple[Period, ...], tuple[DaySpan, ...], LineSizer]:
    """The periods for one budget, as `build_plan` makes them for a model with no limit."""
    lines = fit_lines(sized.lines, meter, max(1, budget - HEADER_TOKENS))
    days = sized.days if lines is sized.lines else measure_days(lines, meter)
    sizer = line_sizer(lines, meter)
    return plan_periods(days, budget, sizer), days, sizer


def unstable_cuts(days: Sequence[DaySpan], budget: int, sizer: LineSizer, cuts: int) -> list[int]:
    """Cut the history at `cuts` days and plan each start again. Every period
    but the last one of a shorter history must stand unchanged in the full plan.
    Returns the cut points where that does not hold."""
    full = plan_periods(days, budget, sizer)
    step = max(1, len(days) // (cuts + 1))
    found: list[int] = []
    for stop in range(step, len(days), step):
        before = plan_periods(days[:stop], budget, sizer)
        if before[:-1] != full[: len(before) - 1]:
            found.append(stop)
    return found


def _kinds(periods: Sequence[Period]) -> tuple[int, int, int]:
    """Periods of whole months, of days inside one month, and of parts of one day."""
    parts = sum(" part " in period.label for period in periods)
    days = sum(period.label[:1].isdigit() for period in periods) - parts
    return len(periods) - days - parts, days, parts


def _report_file(frame: pd.DataFrame, emit: Emit) -> None:
    texts = texts_of(frame)
    tapbacks = tapback_flags(texts)
    rows = count_rows(frame, texts, tapbacks)
    check = check_tapbacks(frame, texts, tapbacks)
    lengths = measure_lengths(texts)
    share = rows.tapbacks / rows.rows if rows.rows else 0.0
    emit(
        f"rows {rows.rows:,}; conversations {rows.conversations:,} "
        f"(groups {rows.group_conversations:,}); rows with no conversation {rows.no_conversation:,}"
    )
    emit(
        f"rows left out: no timestamp {rows.no_timestamp:,}; no direction {rows.no_direction:,}; "
        f"no text {rows.no_text:,}; only the attachment placeholder {rows.placeholder_only:,}; "
        f"tapbacks {rows.tapbacks:,} ({share:.2%})"
    )
    emit(f"direction: sent {rows.sent:,}; received {rows.received:,}")
    emit(f"ids: missing {rows.no_id:,}; repeated {rows.repeated_ids:,}")
    emit(
        f"tapbacks: rows that start like one and are kept {rows.tapback_lookalikes:,}; "
        f"with quoted text {check.with_quote:,}, of which the text is in an earlier row of the "
        f"same conversation {check.found_earlier:,}"
    )
    emit(f"characters {lengths.characters:,}; longest text {lengths.longest:,}")
    emit(
        "length quantiles: "
        + "; ".join(f"{level:g} {value:,}" for level, value in lengths.quantiles)
    )
    for step in lengths.steps:
        mark = "  <- the cap now" if step.limit == MAX_MESSAGE_CHARS else ""
        emit(
            f"  over {step.limit:>7,} characters: {step.rows_over:>7,} rows hold "
            f"{step.share_in_rows_over:6.2%} of all characters; a cap here keeps "
            f"{step.share_kept_by_cap:6.2%}{mark}"
        )


def _report_conversation(
    rank: int,
    rows: int,
    sized: Sized,
    meter: TokenMeter,
    budgets: Sequence[int],
    cuts: int,
    emit: Emit,
) -> None:
    months = len({day.day[:7] for day in sized.days})
    cut = sum(line.cut for line in sized.lines)
    largest_day = max(day.tokens for day in sized.days)
    emit(
        f"conversation {rank}: rows {rows:,}; lines {len(sized.lines):,}; cut {cut:,}; "
        f"days {len(sized.days):,}; months {months:,}; tokens {sized.tokens:,}; "
        f"largest day {largest_day:,}; read and count {sized.seconds:.1f} s"
    )
    for budget in budgets:
        periods, days, sizer = plan(sized, meter, budget)
        whole, ranges, parts = _kinds(periods)
        fill = sum(period.tokens for period in periods) / (len(periods) * budget)
        over = sum(period.tokens > budget for period in periods)
        unstable = len(unstable_cuts(days, budget, sizer, cuts))
        emit(
            f"  budget {budget:>7,}: periods {len(periods):>4,} (months {whole:,}, day ranges "
            f"{ranges:,}, parts of a day {parts:,}); mean fill {fill:.0%}; over the budget {over:,}; "
            f"cut points not stable {unstable:,}"
        )


def _report_estimate(
    rank: int, frame: pd.DataFrame, chat: str, model: ModelInfo, counter: TokenCounter, emit: Emit
) -> None:
    started = time.perf_counter()
    result = estimate(
        FrameSource(frame),
        chat,
        None,
        model,
        None,
        labeler=_name,
        options=ReportOptions(),
        counter=counter,
    )
    emit(
        f"estimate {rank}: periods {result['periods']:,}; input tokens {result['input_tokens']:,}; "
        f"output tokens {result['output_tokens']:,}; {time.perf_counter() - started:.1f} s"
    )


def _report_all(
    totals: Sequence[Total],
    budgets: Sequence[int],
    failures: Counter[str],
    seconds: float,
    emit: Emit,
) -> None:
    tokens = sorted((total.tokens for total in totals), reverse=True)
    whole = sum(tokens)
    emit(
        f"all conversations: read {len(totals):,}; lines {sum(total.lines for total in totals):,}; "
        f"cut {sum(total.cut for total in totals):,}; tokens {whole:,}; {seconds:.1f} s"
    )
    emit(
        "not read, by type of error: "
        + (", ".join(f"{name} {count:,}" for name, count in sorted(failures.items())) or "none")
    )
    for index, budget in enumerate(budgets):
        counts = [total.periods[index] for total in totals]
        groups = [sum(low <= count <= high for count in counts) for low, high in PERIOD_GROUPS]
        emit(
            f"  at budget {budget:>7,}: periods {sum(counts):,}; conversations with 1 period "
            "{:,}; 2 to 5: {:,}; 6 to 20: {:,}; 21 to 50: {:,}; over 50: {:,}".format(*groups)
        )
    if whole:
        emit(
            f"share of all tokens in the 10 largest conversations: {sum(tokens[:10]) / whole:.1%}; "
            f"in the largest: {tokens[0] / whole:.1%}"
        )


def run(args: argparse.Namespace, emit: Emit) -> None:
    started = time.perf_counter()
    frame = load_frame(args.path)
    emit(f"read the file in {time.perf_counter() - started:.1f} s")
    _report_file(frame, emit)

    model = open_model(args.tokenizer_family)
    counter: TokenCounter = (
        HeuristicCounter() if args.heuristic else load_counter(args.tokenizer_family)
    )
    meter = meter_for(model, counter)
    budgets = tuple(sorted(set(args.budgets)))
    emit(f"tokens counted with {meter.describe()}; message cap {MAX_MESSAGE_CHARS:,} characters")
    if counter.family == HEURISTIC_FAMILY and not args.heuristic:
        emit("the tokenizer did not load: the token counts are an estimate by class of character")

    totals: list[Total] = []
    failures: Counter[str] = Counter()
    chats = frame["chat_id"].astype("string")
    places = frame.groupby(chats, sort=False).indices
    started = time.perf_counter()
    for rank, (chat, rows) in enumerate(chats.value_counts().items(), start=1):
        part = frame.iloc[places[chat]]
        try:
            sized = read_sized(part, str(chat), meter)
            if sized is None:
                failures["no message with text"] += 1
                continue
            periods = tuple(len(plan(sized, meter, budget)[0]) for budget in budgets)
            if rank <= args.top:
                _report_conversation(rank, int(rows), sized, meter, budgets, args.cuts, emit)
            if rank <= args.estimates:
                _report_estimate(rank, part, str(chat), model, counter, emit)
        except Exception as exc:  # noqa: BLE001 - one conversation must not stop the others
            # the type of an error is code; its text could hold a row
            failures[type(exc).__name__] += 1
            continue
        totals.append(
            Total(
                tokens=sized.tokens,
                lines=len(sized.lines),
                cut=sum(line.cut for line in sized.lines),
                periods=periods,
            )
        )
    _report_all(totals, budgets, failures, time.perf_counter() - started, emit)


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m report_engine.measure", description=__doc__)
    parser.add_argument("path", help="the messages export (CSV)")
    parser.add_argument(
        "--top", type=int, default=10, help="how many of the largest conversations to show"
    )
    parser.add_argument(
        "--estimates", type=int, default=3, help="for how many of them to run the estimate"
    )
    parser.add_argument(
        "--cuts", type=int, default=12, help="how many cut points to check for stable periods"
    )
    parser.add_argument(
        "--budgets",
        type=lambda text: [int(part) for part in text.split(",") if part],
        default=list(DEFAULT_BUDGETS),
        help="period budgets in tokens, with commas between them",
    )
    parser.add_argument("--tokenizer-family", default=ENGINE_ENCODING)
    parser.add_argument(
        "--heuristic", action="store_true", help="count by class of character, with no tokenizer"
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str], emit: Emit = print) -> int:
    args = _arguments(argv)
    try:
        # a warning of a library can quote a cell of the file: none is shown
        with _quiet_engine(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run(args, emit)
    except Exception as exc:  # noqa: BLE001 - the text of an error could hold a row of the file
        frames = traceback.extract_tb(exc.__traceback__)
        where = ", ".join(f"{frame.name}:{frame.lineno}" for frame in frames[-3:])
        emit(f"stopped: {type(exc).__name__} at {where}")
        return 1
    return 0


if __name__ == "__main__":
    block_network()
    sys.exit(main(sys.argv[1:]))
