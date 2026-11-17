"""Source adapters: today's export frame, and the archive that comes later."""

import sys
import types
from typing import Any, Iterator

import pandas as pd
import pytest

from report_engine.archive_source import ArchiveSource
from report_engine.errors import SourceError
from report_engine.fake_llm import FakeLLM
from report_engine.frame_source import FrameSource

from .support import run
from .synth import CONVERSATION, GROUP, MINA, RAVI, labeler


def frame(**overrides: Any) -> pd.DataFrame:
    columns: dict[str, Any] = {
        "message_id": [11, 12, 13, 14, 15],
        "timestamp": pd.to_datetime(
            [
                "2024-06-02 10:00:00",
                "2024-06-01 09:00:00",
                "2024-06-01 09:05:30",
                "2024-06-03 23:59:59",
                "2024-06-02 08:00:00",
            ]
        ),
        "sender": [MINA, MINA, None, MINA, RAVI],
        "text": ["later", "first", "mine", None, "other chat"],
        "chat_id": [CONVERSATION, CONVERSATION, CONVERSATION, CONVERSATION, GROUP],
        "is_from_me": [0, 0, 1, 0, 0],
        "service": ["iMessage"] * 5,
    }
    columns.update(overrides)
    return pd.DataFrame(columns)


def rows(source: Any, since: str | None = None, until: str | None = None, conversation: str = CONVERSATION) -> list:
    return list(source.messages(conversation, since, until))


def test_frame_source_yields_one_conversation_oldest_first() -> None:
    assert rows(FrameSource(frame())) == [
        {"id": "12", "timestamp": "2024-06-01 09:00:00", "is_from_me": False, "sender": MINA, "text": "first"},
        {"id": "13", "timestamp": "2024-06-01 09:05:30", "is_from_me": True, "sender": "", "text": "mine"},
        {"id": "11", "timestamp": "2024-06-02 10:00:00", "is_from_me": False, "sender": MINA, "text": "later"},
        {"id": "14", "timestamp": "2024-06-03 23:59:59", "is_from_me": False, "sender": MINA, "text": ""},
    ]


def test_frame_source_values_are_plain_python() -> None:
    first = rows(FrameSource(frame()))[0]
    assert [type(first[key]) for key in ("id", "timestamp", "is_from_me", "sender", "text")] == [
        str, str, bool, str, str,
    ]


def test_frame_source_matches_the_conversation_exactly() -> None:
    source = FrameSource(frame())
    assert rows(source, conversation="") == []
    assert rows(source, conversation=CONVERSATION[:6]) == []
    assert [row["id"] for row in rows(source, conversation=GROUP)] == ["15"]


@pytest.mark.parametrize(
    "since, until, ids",
    [
        ("2024-06-02", None, ["11", "14"]),
        (None, "2024-06-01", ["12", "13"]),
        ("2024-06-01", "2024-06-03", ["12", "13", "11", "14"]),
        ("2024-06-03", "2024-06-03", ["14"]),
        ("2024-06-01 09:05:30", "2024-06-02 10:00:00", ["13", "11"]),
        ("2025-01-01", None, []),
    ],
)
def test_frame_source_dates_are_whole_days_and_both_ends_count(
    since: str | None, until: str | None, ids: list[str]
) -> None:
    assert [row["id"] for row in rows(FrameSource(frame()), since, until)] == ids


def test_frame_source_reads_text_timestamps_and_loose_flags() -> None:
    source = FrameSource(
        frame(
            timestamp=["2024-06-02 10:00:00", "2024-06-01 09:00:00", "2024-06-01 09:05:30", "bad", None],
            is_from_me=["1", "true", "0", None, 1.0],
            message_id=[11.0, 12.0, 13.0, 14.0, 15.0],
            chat_id=[CONVERSATION] * 5,
        )
    )
    found = rows(source)
    assert [(row["id"], row["is_from_me"]) for row in found] == [("12", True), ("13", False), ("11", True)]


def test_frame_source_says_how_many_rows_had_no_time(caplog: pytest.LogCaptureFixture) -> None:
    source = FrameSource(frame(timestamp=["2024-06-02 10:00:00", None, None, "2024-06-03 10:00:00", None]))
    with caplog.at_level("WARNING"):
        assert len(rows(source)) == 2
    assert "2 message" in caplog.text


def test_frame_source_keeps_text_that_looks_like_a_missing_value() -> None:
    source = FrameSource(frame(text=["NA", "null", "None", "nan", "x"]))
    assert [row["text"] for row in rows(source)] == ["null", "None", "NA", "nan"]


def test_frame_source_needs_the_export_columns() -> None:
    with pytest.raises(SourceError) as caught:
        FrameSource(frame().drop(columns=["chat_id", "text"]))
    assert "chat_id" in str(caught.value) and "text" in str(caught.value)


def test_frame_source_does_not_change_the_frame() -> None:
    data = frame()
    before = data.copy(deep=True)
    rows(FrameSource(data), "2024-06-01", "2024-06-02")
    pd.testing.assert_frame_equal(data, before)


def test_frame_source_drops_time_zones_but_keeps_wall_time() -> None:
    stamps = pd.to_datetime(["2024-06-01 09:00:00"] * 5).tz_localize("America/Los_Angeles")
    found = rows(FrameSource(frame(timestamp=stamps, chat_id=[CONVERSATION] * 5)))
    assert {row["timestamp"] for row in found} == {"2024-06-01 09:00:00"}


def test_a_run_reads_from_a_frame() -> None:
    llm = FakeLLM()
    report = run([], llm=llm, source=FrameSource(frame()))
    assert report.periods[0].message_count == 3
    assert "m1 09:00 Mina: first" in llm.map_calls[0].text


class StubArchive:
    """Stands in for archive.queries.Archive. The real package is not in this branch."""

    def __init__(self, items: list[dict[str, Any]], inclusive_until: bool = True) -> None:
        self.items = items
        self.inclusive_until = inclusive_until
        self.calls: list[tuple[str, str | None, str | None]] = []
        self.closed = False

    def iter_messages(
        self, conversation_id: str, *, since: str | None = None, until: str | None = None
    ) -> Iterator[dict[str, Any]]:
        for value in (since, until):
            if value is not None and len(value) != 10:
                raise ValueError("since and until must be YYYY-MM-DD")
        self.calls.append((conversation_id, since, until))
        for item in self.items:
            day = item["timestamp"][:10]
            if since is not None and day < since:
                continue
            if until is not None and (day > until if self.inclusive_until else day >= until):
                continue
            yield dict(item)

    def close(self) -> None:
        self.closed = True


ARCHIVE_ROWS = [
    {"id": 7, "timestamp": "2024-06-01 09:00:00", "is_from_me": 0, "sender": MINA, "text": "hi", "kind": "message"},
    {"id": 8, "timestamp": "2024-06-01 09:01:00", "is_from_me": 1, "sender": "", "text": "hello", "kind": "message"},
    {"id": 9, "timestamp": "2024-06-02 10:00:00", "is_from_me": 0, "sender": MINA, "text": "named the chat", "kind": "system"},
    {"id": 10, "timestamp": "2024-06-03 11:00:00", "is_from_me": 0, "sender": MINA, "text": "", "kind": "message"},
    {"id": 11, "timestamp": "2024-06-04 12:00:00", "is_from_me": 0, "sender": MINA, "text": "bye", "kind": "message"},
]


def test_archive_source_maps_rows_and_skips_system_rows() -> None:
    source = ArchiveSource(StubArchive(ARCHIVE_ROWS))
    assert rows(source) == [
        {"id": "7", "timestamp": "2024-06-01 09:00:00", "is_from_me": False, "sender": MINA, "text": "hi"},
        {"id": "8", "timestamp": "2024-06-01 09:01:00", "is_from_me": True, "sender": "", "text": "hello"},
        {"id": "10", "timestamp": "2024-06-03 11:00:00", "is_from_me": False, "sender": MINA, "text": ""},
        {"id": "11", "timestamp": "2024-06-04 12:00:00", "is_from_me": False, "sender": MINA, "text": "bye"},
    ]


@pytest.mark.parametrize("inclusive_until", [True, False])
def test_archive_source_covers_whole_days_whatever_the_archive_does_with_until(inclusive_until: bool) -> None:
    archive = StubArchive(ARCHIVE_ROWS, inclusive_until=inclusive_until)
    found = rows(ArchiveSource(archive), "2024-06-01", "2024-06-03")
    assert [row["id"] for row in found] == ["7", "8", "10"]
    assert archive.calls == [(CONVERSATION, "2024-06-01", "2024-06-04")]


def test_archive_source_sends_days_only_and_keeps_exact_bounds_itself() -> None:
    archive = StubArchive(ARCHIVE_ROWS)
    found = rows(ArchiveSource(archive), "2024-06-01 09:01:00", "2024-06-04 11:59:59")
    assert [row["id"] for row in found] == ["8", "10"]
    assert archive.calls == [(CONVERSATION, "2024-06-01", "2024-06-05")]


def test_archive_source_passes_no_bounds_when_there_are_none() -> None:
    archive = StubArchive(ARCHIVE_ROWS)
    rows(ArchiveSource(archive))
    assert archive.calls == [(CONVERSATION, None, None)]


def test_archive_source_reports_a_row_it_can_not_use() -> None:
    class Loose:
        def iter_messages(self, conversation_id: str, **bounds: Any) -> Iterator[dict[str, Any]]:
            yield {"id": 1, "is_from_me": 0, "text": "no time"}

    with pytest.raises(SourceError) as caught:
        rows(ArchiveSource(Loose()))
    assert "timestamp" in str(caught.value)


def test_archive_source_opens_the_archive_package_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[tuple[Any, bool]] = []

    class Archive:
        @classmethod
        def open(cls, path: Any = None, *, readonly: bool = True) -> StubArchive:
            opened.append((path, readonly))
            return StubArchive(ARCHIVE_ROWS)

    package = types.ModuleType("archive")
    queries = types.ModuleType("archive.queries")
    queries.Archive = Archive  # type: ignore[attr-defined]
    package.queries = queries  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "archive", package)
    monkeypatch.setitem(sys.modules, "archive.queries", queries)

    with ArchiveSource.open("/tmp/does-not-matter.sqlite3") as source:
        assert len(rows(source)) == 4
        inner = source.archive
    assert opened == [("/tmp/does-not-matter.sqlite3", True)]
    assert inner.closed is True


def test_archive_source_says_when_the_package_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "archive", None)
    monkeypatch.setitem(sys.modules, "archive.queries", None)
    with pytest.raises(SourceError) as caught:
        ArchiveSource.open(None)
    assert "archive" in str(caught.value)


def test_importing_the_adapter_does_not_import_the_archive_package() -> None:
    assert "archive" not in sys.modules or sys.modules["archive"] is None


def test_a_run_reads_from_an_archive() -> None:
    llm = FakeLLM()
    report = run([], llm=llm, source=ArchiveSource(StubArchive(ARCHIVE_ROWS)), labeler=labeler)
    assert report.periods[0].message_count == 3
    assert "named the chat" not in llm.map_calls[0].text
