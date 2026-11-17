"""Invented data for the tests of the chat seam.

Two exports, both made up:

- The seam export: `chat_engine.eval.volume_data.invented_export`, with one
  more conversation whose times lie on clock changes. It is loaded two ways:
  into the chat package's own `FrameArchive`, and into the real archive
  through the legacy bridge (`import_legacy_csv`).
- The cast export: a few people and a few lines that a person can read, for
  whole questions.

The bridge gives a row the id `legacy:<ROWID>`, so each export has whole
numbers as message ids.

The last part runs the tools and the scripted question of the chat package's
volume run on an archive.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from tests.integration import guards

ROWS = 20_000
SEED = 20261117
FIRST_SECOND = 1_434_697_200  # 2015-06-19 07:00:00 UTC
LAST_SECOND = 1_794_816_000  # 2026-11-16 08:00:00 UTC
LEGACY_PREFIX = "legacy:"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
ZONE = ZoneInfo(guards.TEST_TIME_ZONE)
EXPORT_COLUMNS = (
    "message_id",
    "timestamp",
    "sender",
    "text",
    "chat_id",
    "chat_display_name",
    "is_from_me",
    "service",
    "attachment_types",
    "attachment_files",
)

# One conversation on the clock changes of 2020 and 2021 in America/Los_Angeles.
CLOCK_CHAT = "clock@example.org"
CLOCK_ROWS: tuple[tuple[str, str, str, str], ...] = (
    # (timestamp, sender, is_from_me, text)
    ("2020-01-10 12:00:00", "", "1", "winter noon"),
    ("2020-09-10 12:00:00", CLOCK_CHAT, "0", "summer noon"),
    ("2021-03-13 12:00:00", "", "1", "the day before the clocks go forward"),
    ("2021-03-14 01:59:59", CLOCK_CHAT, "0", "the last second before the gap"),
    ("2021-03-14 02:30:00", CLOCK_CHAT, "0", "a time that the clocks skip"),
    ("2021-03-14 03:10:00", "", "1", "ten past three in summer time"),
    ("2021-11-07 01:30:00", "", "1", "a time that comes twice"),
    ("2021-11-07 01:45:00", "", "0", "a received line with no sender"),
    ("2021-11-08 09:00:00", CLOCK_CHAT, "0", "the day after the clocks go back"),
)
SKIPPED_TIME = "2021-03-14 02:30:00"
DOUBLE_TIME = "2021-11-07 01:30:00"


def true_utc(stamp: str) -> int:
    """UTC unix seconds of a local time of the export, by the rule of the archive README.

    A time that comes twice is its first pass. A time that the clocks skip is
    the moment that much later.
    """
    return int(datetime.strptime(stamp, TIME_FORMAT).replace(tzinfo=ZONE).timestamp())


def local_time(unix_seconds: int) -> str:
    """UTC unix seconds as local time in the zone of the tests."""
    return datetime.fromtimestamp(unix_seconds, ZONE).strftime(TIME_FORMAT)


def as_archive_reads(stamp: str) -> str:
    """The local time that the archive gives back for a time of the export."""
    return local_time(true_utc(stamp))


def weekday_hour(stamp: str) -> str:
    """The key of `count(group_by="weekday_hour")` for a local time: Monday is 0."""
    moment = datetime.strptime(stamp, TIME_FORMAT)
    return f"{moment.weekday()}-{moment.hour:02d}"


def message_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows of an export that are messages: every row but a reaction phrase."""
    from archive.tapbacks import is_reaction_phrase

    return frame[~frame["text"].map(is_reaction_phrase)]


def plain_id(message_id: object) -> str:
    """An id of the bridge archive as the export has it."""
    return str(message_id).removeprefix(LEGACY_PREFIX)


def legacy_id(row_id: object) -> str:
    """An id of the export as the bridge archive has it."""
    return f"{LEGACY_PREFIX}{row_id}"


def _rows_frame(rows: Sequence[Mapping[str, str]], first_id: int) -> pd.DataFrame:
    """Export rows with every column, numbered from `first_id`."""
    filled = [
        {**dict.fromkeys(EXPORT_COLUMNS, ""), "service": "iMessage", **row, "message_id": str(first_id + place)}
        for place, row in enumerate(rows)
    ]
    return pd.DataFrame(filled, columns=list(EXPORT_COLUMNS))


def clock_frame(first_id: int) -> pd.DataFrame:
    rows = [
        {"timestamp": stamp, "sender": sender, "is_from_me": mine, "text": text, "chat_id": CLOCK_CHAT}
        for stamp, sender, mine, text in CLOCK_ROWS
    ]
    return _rows_frame(rows, first_id)


def _stamps(rows: int) -> list[str]:
    """Local times for the rows of the invented export, oldest first.

    `invented_export` takes its first and its last day from the zone that the
    process had when the module was loaded. These times do not, so the export
    is the same on every machine and in every order of the tests.
    """
    rng = np.random.default_rng(SEED)
    seconds = np.sort(rng.uniform(FIRST_SECOND, LAST_SECOND, size=rows)).astype(np.int64)
    return list(pd.to_datetime(seconds, unit="s").strftime(TIME_FORMAT))


def seam_frame(rows: int = ROWS) -> pd.DataFrame:
    """The invented export with whole numbers as ids, and the clock conversation after it."""
    from chat_engine.eval.volume_data import invented_export

    bulk = invented_export(rows)
    numbers = [str(number) for number in range(1, len(bulk) + 1)]
    bulk = bulk.assign(message_id=numbers, timestamp=_stamps(len(bulk)))
    return pd.concat([bulk, clock_frame(len(bulk) + 1)], ignore_index=True)


@dataclass(frozen=True)
class Seam:
    """One invented export, loaded into `FrameArchive` and into the real archive."""

    frame: pd.DataFrame
    stamps: Mapping[str, str]  # export id -> export time
    moments: Mapping[str, int]  # export id -> UTC unix seconds, by the rule of the archive
    frame_archive: Any  # chat_engine.frame_archive.FrameArchive
    adapter: Any  # chat_engine.archive_adapter.ArchiveAdapter on the real archive
    archive: Any  # archive.Archive on the same file
    picks: Any  # chat_engine.eval.volume_cases.Picks
    imported: Any  # archive.ImportResult

    def conversation_ids(self) -> list[str]:
        listed = self.archive.conversations(limit=None, offset=0, order="count")
        return [str(conversation["id"]) for conversation in listed]

    def close(self) -> None:
        self.adapter.close()
        self.archive.close()


def build_seam(folder: Path, rows: int = ROWS) -> Seam:
    """Make the export in `folder` and load it two ways.

    The bridge reads the times of the export in the zone of the process, so
    the load runs in the zone of the tests.
    """
    from archive import Archive, import_legacy_csv
    from chat_engine.archive_adapter import open_archive
    from chat_engine.eval.volume_cases import pick
    from chat_engine.frame_archive import FrameArchive

    with guards.time_zone():
        frame = seam_frame(rows)
        csv_path = folder / "invented.csv"
        frame.to_csv(csv_path, index=False)
        archive_path = folder / "archive.sqlite3"
        imported = import_legacy_csv(csv_path, archive_path)
        frame_archive = FrameArchive.from_csv(csv_path)
        picks = pick(frame_archive)
    stamps = dict(zip(frame["message_id"], frame["timestamp"], strict=True))
    return Seam(
        frame=frame,
        stamps=stamps,
        moments={row_id: true_utc(stamp) for row_id, stamp in stamps.items()},
        frame_archive=frame_archive,
        adapter=open_archive(archive_path),
        archive=Archive.open(archive_path),
        picks=picks,
        imported=imported,
    )


_shared: list[Seam] = []


def _forget_shared() -> None:
    while _shared:
        _shared.pop().close()


def shared_seam(request: pytest.FixtureRequest) -> Seam:
    """The seam export of this test run. It is made one time: the tests only read it.

    Its two archives are closed when the test run ends.
    """
    if not _shared:
        factory: pytest.TempPathFactory = request.getfixturevalue("tmp_path_factory")
        _shared.append(build_seam(factory.mktemp("seam")))
        request.config.add_cleanup(_forget_shared)
    return _shared[0]


def _members_by_handle(conversation: Mapping[str, Any]) -> dict[str, Any]:
    return {**conversation, "members": sorted(conversation["members"])}


class OneOrder:
    """An archive that lists members by handle, and conversations of one size by id.

    For a comparison only. The real archive and `FrameArchive` each have an
    order of their own for these two, by design. With one order in both, the
    text of a tool result is the same in both.
    """

    def __init__(self, archive: Any) -> None:
        self._archive = archive

    def __getattr__(self, name: str) -> Any:
        return getattr(self._archive, name)

    def conversations(
        self, *, limit: int | None = None, offset: int = 0, order: str = "count"
    ) -> list[dict[str, Any]]:
        every = self._archive.conversations(limit=None, offset=0, order=order)
        listed = [_members_by_handle(conversation) for conversation in every]
        if order == "count":
            listed = sorted(listed, key=lambda conversation: (-conversation["message_count"], conversation["id"]))
        return listed[offset : None if limit is None else offset + limit]

    def conversation(self, conversation_id: str) -> dict[str, Any] | None:
        found = self._archive.conversation(conversation_id)
        return _members_by_handle(found) if found else None


# --- the cast: a few people and lines that a person can read ---------------------

MINA = "+15550100101"  # saved as Mina Okafor
MINA_NAME = "Mina Okafor"
JONAS = "+15550100199"  # nobody saved this number
ROWAN = "rowan.b@example.org"  # nobody saved this address
TESS = "+15550100177"  # nobody saved this number; writes in the group only
GROUP = "chat900001"
GROUP_TITLE = "Ferry crew"
SAVED_NAMES = {MINA: MINA_NAME}
UNSAVED_HANDLES = (JONAS, ROWAN, TESS)
JONAS_IN_TEXT = "+1 (555) 010-0199"  # the number of Jonas as a person types it
JONAS_LOCAL = "010-0199"
STARTER_LINE = "I named the sourdough starter Clementine"
FERRY_LINE = f"the ferry leaves at seven, call me at {JONAS_IN_TEXT} if you are late"
TICKETS_LINE = "I have the ferry tickets for all of us"

CAST_ROWS: tuple[tuple[str, str, str, str, str], ...] = (
    # (timestamp, chat_id, sender, is_from_me, text)
    ("2024-03-01 09:00:00", MINA, MINA, "0", STARTER_LINE),
    ("2024-03-01 09:02:00", MINA, "", "1", "Clementine is a great name for a starter"),
    ("2024-03-02 18:30:00", MINA, MINA, "0", "Clementine doubled overnight"),
    ("2024-03-05 08:15:00", MINA, "", "1", "save me a loaf"),
    ("2024-04-10 07:40:00", JONAS, JONAS, "0", FERRY_LINE),
    ("2024-04-10 07:42:00", JONAS, "", "1", f"ok. is {ROWAN} still the address of Rowan?"),
    ("2024-04-10 07:45:00", JONAS, JONAS, "0", f"yes. my short number is {JONAS_LOCAL}"),
    ("2024-04-11 12:00:00", ROWAN, ROWAN, "0", "the harbor office closes at noon on the ferry days"),
    ("2024-04-12 10:00:00", GROUP, TESS, "0", "who has the ferry tickets"),
    ("2024-04-12 10:01:00", GROUP, JONAS, "0", TICKETS_LINE),
    ("2024-04-12 10:03:00", GROUP, MINA, "0", "see you at the ferry at seven"),
    ("2024-04-12 10:04:00", GROUP, "", "1", "bringing bread from the starter"),
    ("2024-04-12 10:05:00", GROUP, TESS, "0", f"Jonas, is {JONAS} your new number?"),
)


def cast_frame() -> pd.DataFrame:
    rows = [
        {
            "timestamp": stamp,
            "chat_id": chat_id,
            "sender": sender,
            "is_from_me": mine,
            "text": text,
            "chat_display_name": GROUP_TITLE if chat_id == GROUP else "",
        }
        for stamp, chat_id, sender, mine, text in CAST_ROWS
    ]
    return _rows_frame(rows, 1)


def cast_id(text: str) -> str:
    """The archive id of the cast line with this text."""
    places = [place for place, row in enumerate(CAST_ROWS, start=1) if row[4] == text]
    if len(places) != 1:
        raise LookupError(f"{len(places)} cast lines have this text")
    return legacy_id(places[0])


def build_cast_archive(folder: Path) -> Path:
    """Load the cast export into a new archive in `folder` and return the path of the archive."""
    from archive import import_legacy_csv

    with guards.time_zone():
        csv_path = folder / "cast.csv"
        cast_frame().to_csv(csv_path, index=False)
        archive_path = folder / "cast.sqlite3"
        import_legacy_csv(csv_path, archive_path)
    return archive_path


# --- the tools and the scripted question of the volume run -----------------------


class Shown(NamedTuple):
    """One tool result as the model gets it."""

    text: str
    ok: bool
    tokens: int


def tool_box(archive: Any) -> Any:
    """The tools of one question on the whole archive, after the engine has listed every conversation."""
    from chat_engine.labeler import MaskingLabeler
    from chat_engine.registry import ConversationAliases, EvidenceRegistry
    from chat_engine.render import Limits
    from chat_engine.scope import Scope
    from chat_engine.tools import ToolBox

    box = ToolBox(archive, MaskingLabeler(), EvidenceRegistry(), ConversationAliases(), Scope(), Limits())
    box.context.directory()
    return box


def group_label(archive: Any, group: str) -> str:
    """The name of a group as the model reads it. Nobody is saved."""
    return str(tool_box(archive).context.book.label(group))


def tool_results(archive: Any, picks: Any) -> dict[str, Shown]:
    """Each tool call of the volume run on `archive`, by the label of its case."""
    from chat_engine.eval.volume_cases import browse_cases, figure_cases, search_cases

    box = tool_box(archive)
    limits = box.context.limits
    wanted = [picks.largest, picks.middle, picks.group]
    alias = {conversation_id: box.context.book.alias(conversation_id) for conversation_id in wanted}
    shown: dict[str, Shown] = {}
    for case in [*search_cases(picks, alias), *browse_cases(picks, alias), *figure_cases(picks, alias)]:
        outcome = box.prepare(case.tool, json.dumps(case.arguments)).run(limits.result_tokens)
        shown[case.label] = Shown(outcome.text, outcome.ok, limits.count_tokens(outcome.text))
    return shown


def volume_script(picks: Any) -> list[Any]:
    """The scripted question of the volume run: a search, a count, a read, an answer with a quote check."""
    from chat_engine.fake_llm import call, say

    found: dict[str, str] = {}

    def look(_request: Any) -> list[Any]:
        return [
            call("search_messages", "v1", query=f"{picks.medium} {picks.rare} {picks.common}", match="any"),
            call("count_messages", "v2", text=picks.medium, group_by="year"),
        ]

    def read(request: Any) -> list[Any]:
        results = [message.content for message in request.messages if message.role == "tool"]
        line = re.search(r"^(m\d+) ", results[0], re.MULTILINE)
        if line is None:
            return say("There is no evidence for this.")
        found["reference"] = line.group(1)
        return [call("read_around", "v3", reference=line.group(1), radius=6)]

    def answer(_request: Any) -> list[Any]:
        return say(f'This is the answer [{found["reference"]}] with "a quote that no message holds".')

    return [look, read, answer]
