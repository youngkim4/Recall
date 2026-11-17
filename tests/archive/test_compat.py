"""The pandas bridge: a DataFrame with the columns of today's CSV export."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from archive import Archive, import_chat_db
from archive.compat import EXPORT_COLUMNS, UNSENT_COLUMN, messages_frame
from tests.archive.builders import ChatDbBuilder
from tests.archive import worlds
from tests.archive.conftest import Harbor
from tests.archive.legacy import REPO_ROOT, load_legacy_conversation

A, B, G = worlds.ROWAN_PHONE, worlds.ROWAN_EMAIL, worlds.HARBOR_GROUP_ID


@pytest.fixture
def frame(harbor: Harbor) -> pd.DataFrame:
    return messages_frame(harbor.archive)


def row_of(frame: pd.DataFrame, harbor: Harbor, label: str) -> pd.Series:
    return frame[frame["message_id"] == harbor.id(label)].iloc[0]


class TestShape:
    def test_columns_are_those_of_the_export(self, frame: pd.DataFrame) -> None:
        assert list(frame.columns) == [
            "message_id", "timestamp", "sender", "text", "chat_id", "chat_display_name",
            "is_from_me", "service", "attachment_types", "attachment_files", "unsent",
        ]  # fmt: skip
        assert list(frame.columns) == [*EXPORT_COLUMNS, UNSENT_COLUMN]

    def test_a_message_that_was_unsent_is_a_row_with_no_text(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        """The lists and the days of the archive count such a message, so the frame holds it too."""
        lake = worlds.build_lake(builder)
        builder.close()
        import_chat_db(source_path, archive_path)
        with Archive.open(archive_path) as archive:
            made = messages_frame(archive)
        taken_back = made[made[UNSENT_COLUMN]]
        assert list(taken_back["message_id"]) == [lake.guids["unsent"]]
        assert not taken_back.iloc[0]["text"]

    def test_one_row_per_message_and_none_for_events_or_tapbacks(self, frame: pd.DataFrame, harbor: Harbor) -> None:
        assert len(frame) == 19
        assert frame["message_id"].is_unique
        hidden = {harbor.id(label) for label in ("g_renamed", "a_event", "g15_like", "a2_love")}
        assert hidden.isdisjoint(set(frame["message_id"]))

    def test_types_match_the_frame_that_load_messages_makes(self, frame: pd.DataFrame) -> None:
        assert str(frame["timestamp"].dtype) == "datetime64[ns]"
        assert str(frame["is_from_me"].dtype) == "int64"
        assert str(frame["chat_id"].dtype) == "string"
        assert str(frame["sender"].dtype) == "string"
        assert set(frame["is_from_me"]) == {0, 1}

    def test_rows_are_in_time_order(self, frame: pd.DataFrame, harbor: Harbor) -> None:
        assert frame["timestamp"].is_monotonic_increasing
        assert frame.index.tolist() == list(range(19))
        assert frame.iloc[0]["message_id"] == harbor.id("g13")
        assert frame.iloc[-1]["message_id"] == harbor.id("a10")
        same_second = frame[frame["timestamp"] == pd.Timestamp("2024-03-10 12:00:00")]["message_id"].tolist()
        assert same_second == [harbor.id("a7"), harbor.id("a8")]


class TestValues:
    def test_ids_are_guids_and_time_is_local(self, frame: pd.DataFrame, harbor: Harbor) -> None:
        row = row_of(frame, harbor, "g17")
        assert row["timestamp"] == pd.Timestamp("2024-03-10 03:00:00")
        assert row["chat_id"] == G and row["text"] == "First second after"

    def test_sender_is_the_writer_and_missing_for_own_messages(self, frame: pd.DataFrame, harbor: Harbor) -> None:
        assert row_of(frame, harbor, "a1")["sender"] == A
        assert row_of(frame, harbor, "g13")["sender"] == worlds.MIRA_PHONE
        assert pd.isna(row_of(frame, harbor, "a2")["sender"])
        assert frame["sender"].fillna("").astype(str).ne("").sum() == 11

    def test_group_title_only_on_group_rows(self, frame: pd.DataFrame, harbor: Harbor) -> None:
        assert row_of(frame, harbor, "g13")["chat_display_name"] == worlds.HARBOR_GROUP_TITLE
        assert pd.isna(row_of(frame, harbor, "a1")["chat_display_name"])

    def test_attachment_columns_and_placeholder_text(self, frame: pd.DataFrame, harbor: Harbor) -> None:
        row = row_of(frame, harbor, "a9")
        assert row["attachment_types"] == "image/heic|video/quicktime"
        files = row["attachment_files"].split("|")
        assert [Path(name).name for name in files] == ["IMG_0101.HEIC", "IMG_0102.MOV"]
        assert row["text"] == f"[Attachment] image/heic|video/quicktime | {row['attachment_files']}"
        plain = row_of(frame, harbor, "a1")
        assert pd.isna(plain["attachment_types"]) and pd.isna(plain["attachment_files"])

    def test_service(self, frame: pd.DataFrame) -> None:
        assert set(frame["service"]) == {"iMessage"}

    def test_no_text_is_missing(self, frame: pd.DataFrame) -> None:
        assert frame["text"].map(lambda value: isinstance(value, str) and value != "").all()


class TestLegacyCodeRunsOnTheFrame:
    def test_filter_conversation_and_compute_stats(
        self, frame: pd.DataFrame, harbor: Harbor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        legacy = load_legacy_conversation(monkeypatch)
        for conversation in (A, B, G):
            rows = legacy.filter_conversation(frame, conversation)
            stats = legacy.compute_stats(rows)
            mine = harbor.archive.stats(conversation)
            assert stats.chat_id == conversation
            assert (stats.total_messages, stats.sent_count, stats.received_count) == (
                mine["total_messages"], mine["sent_count"], mine["received_count"],
            )  # fmt: skip
            assert stats.unknown_direction_count == 0
            assert str(stats.first_timestamp) == mine["first_timestamp"]
            assert stats.active_days == mine["active_days"]

    def test_format_all_messages(self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        legacy = load_legacy_conversation(monkeypatch)
        text = legacy.format_all_messages(legacy.filter_conversation(frame, B))
        assert text.splitlines() == [
            "[2024-03-10 12:30] THEM: Email thread: sent you the café menu ☕",
            "[2024-03-11 07:45] ME: Thanks, déjà vu from last year",
        ]


class TestPandasStaysInTheBridge:
    def test_only_compat_imports_pandas(self) -> None:
        offenders = [
            path.name
            for path in sorted((REPO_ROOT / "archive").glob("*.py"))
            if path.name != "compat.py" and ("import pandas" in path.read_text() or "from pandas" in path.read_text())
        ]
        assert offenders == []

    def test_importing_the_package_does_not_load_pandas(self) -> None:
        code = "import sys, archive, archive.queries, archive.importer; print('pandas' in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, check=True
        )
        assert result.stdout.strip() == "False"
