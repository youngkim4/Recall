"""frame.py: the archive as the frames that the figures on pandas read."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from archive import compat
from conversation import (
    activity_series,
    compute_stats,
    filter_conversation,
    in_conversation,
    load_messages,
    within_dates,
)
from day_view import day_window
from message_text import ATTACHMENT_PLACEHOLDER, attachment_label, display_text
from recall_services import frame
from recall_services.archive_state import ArchiveState
from tests.recall_services import state_data as data
from tests.recall_services import world

INES = world.INES_PHONE


@pytest.fixture
def state(orchard: world.Orchard) -> Iterator[ArchiveState]:
    made = ArchiveState(orchard.archive_path)
    yield made
    made.close()


@pytest.fixture
def messages(state: ArchiveState) -> pd.DataFrame:
    return frame.messages_frame(state)


def test_the_frame_has_the_columns_of_an_export(messages: pd.DataFrame) -> None:
    assert tuple(messages.columns) == compat.EXPORT_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(messages["timestamp"])
    assert str(messages["chat_id"].dtype) == str(messages["sender"].dtype) == "string"
    assert set(messages["is_from_me"].unique()) == {0, 1}


def test_message_id_is_the_id_of_the_archive(messages: pd.DataFrame, orchard: world.Orchard) -> None:
    """The guid after a pull. The old export held the ROWID there."""
    assert orchard.id("i1") in set(messages["message_id"])
    assert not messages["message_id"].astype(str).str.fullmatch(r"\d+").any()


def test_message_id_after_the_bridge_names_the_row_of_the_export(tmp_path: Path) -> None:
    bridged = ArchiveState(tmp_path / "saves" / "archive.sqlite3")
    bridged.ensure_from_export(data.write_export(tmp_path / "messages.csv", ["one", "two"], first_id=41))
    try:
        assert list(frame.messages_frame(bridged)["message_id"]) == ["legacy:41", "legacy:42"]
    finally:
        bridged.close()


def test_the_frame_holds_every_message_and_no_other_row(messages: pd.DataFrame, state: ArchiveState) -> None:
    summary = state.archive().summary()
    # a message with no conversation is a row with no chat id, as in the export
    assert len(messages) == summary["messages"] + summary["messages_without_conversation"]
    assert int(messages["chat_id"].isna().sum()) == 1
    # a tapback and an event of a group are no rows
    assert not messages["text"].astype(str).str.contains("Loved an invented message").any()


def test_a_message_that_is_only_a_photo_reads_as_the_export_of_today_writes_it(
    messages: pd.DataFrame, orchard: world.Orchard
) -> None:
    """The exporter of today writes the placeholder U+FFFC, and the server shows no text for it."""
    photo = messages[messages["message_id"] == orchard.id("i3")].iloc[0]
    assert photo["text"] == ATTACHMENT_PLACEHOLDER
    assert display_text(photo["text"]) == ""
    assert attachment_label(photo["attachment_types"]) == "Photo"
    two = messages[messages["message_id"] == orchard.id("i9")].iloc[0]
    assert (display_text(two["text"]), attachment_label(two["attachment_types"])) == ("", "2 photos")
    assert not messages["text"].astype(str).str.startswith(compat.ATTACHMENT_LABEL).any()


def test_words_that_only_look_like_the_old_label_are_kept(tmp_path: Path) -> None:
    typed = "[Attachment] is the word my phone shows"
    bridged = ArchiveState(tmp_path / "saves" / "archive.sqlite3")
    bridged.ensure_from_export(data.write_export(tmp_path / "messages.csv", [typed]))
    try:
        assert list(frame.messages_frame(bridged)["text"]) == [typed]
    finally:
        bridged.close()


def test_text_is_kept_as_the_archive_has_it(messages: pd.DataFrame, orchard: world.Orchard) -> None:
    spaced = messages[messages["message_id"] == orchard.id("i4")].iloc[0]
    assert spaced["text"] == "  Picked the first lemons today  "


def test_an_own_message_has_no_sender(messages: pd.DataFrame) -> None:
    own = messages[messages["is_from_me"] == 1]
    assert own["sender"].isna().all()
    assert messages[messages["is_from_me"] == 0]["sender"].notna().all()


def test_the_figures_of_the_server_read_the_frame_without_change(messages: pd.DataFrame, state: ArchiveState) -> None:
    conv = filter_conversation(messages, INES)
    stats = compute_stats(conv)
    held = state.archive().stats(INES)
    assert (stats.total_messages, stats.sent_count, stats.received_count, stats.unknown_direction_count) == (
        held["total_messages"],
        held["sent_count"],
        held["received_count"],
        0,
    )
    assert str(stats.first_timestamp) == held["first_timestamp"]
    assert stats.active_days == held["active_days"]
    assert str(stats.busiest_day)[:10] == held["busiest_day"] == world.LONG_DAY
    assert activity_series(conv)["unit"] == "day"
    ranged = within_dates(conv, "2024-03-01", "2024-03-01")
    assert len(ranged) == state.archive().count(conversations=[INES], since="2024-03-01", until="2024-03-01")


def test_the_day_view_of_the_server_reads_the_frame(messages: pd.DataFrame, state: ArchiveState) -> None:
    window = day_window(messages[in_conversation(messages, INES)], world.LONG_DAY, limit=400)
    held = state.archive().day(INES, world.LONG_DAY, limit=400)
    assert list(window.rows["message_id"]) == [message["id"] for message in held["messages"]]
    assert (window.total, window.previous_day, window.next_day) == (held["total"], "2024-03-01", "2024-03-05")


def test_the_frame_is_what_the_loader_of_the_server_reads_from_a_file(messages: pd.DataFrame, tmp_path: Path) -> None:
    """Written as an export and read back by load_messages, the frame has the same rows."""
    path = tmp_path / "from_archive.csv"
    messages.to_csv(path, index=False)
    loaded = load_messages(str(path))
    assert list(loaded["message_id"]) == list(messages["message_id"])
    assert list(loaded["timestamp"]) == list(messages["timestamp"])


# --- the frame is made once for one state of the archive ---------------------


def test_the_frame_is_made_once(state: ArchiveState) -> None:
    assert frame.messages_frame(state) is frame.messages_frame(state)


def test_the_frame_is_made_again_after_the_archive_changed(tmp_path: Path) -> None:
    changing = ArchiveState(tmp_path / "saves" / "archive.sqlite3")
    try:
        export = data.write_export(tmp_path / "messages.csv", ["one"])
        changing.ensure_from_export(export)
        first = frame.messages_frame(changing)
        changing.ensure_from_export(data.write_export(export, ["one", "two"]))
        second = frame.messages_frame(changing)
        assert (len(first), len(second)) == (1, 2)
        changing.pull_from_messages(data.write_database(tmp_path / "chat.db", ["a", "b", "c"]), None)
        assert list(frame.messages_frame(changing)["text"]) == ["a", "b", "c"]
    finally:
        changing.close()


def test_two_archives_have_two_frames(state: ArchiveState, tmp_path: Path) -> None:
    other = ArchiveState(tmp_path / "saves" / "archive.sqlite3")
    try:
        other.ensure_from_export(data.write_export(tmp_path / "messages.csv", ["one"]))
        assert len(frame.messages_frame(other)) == 1
        assert len(frame.messages_frame(state)) > 1
    finally:
        other.close()


def test_the_default_state_is_the_shared_one(tmp_path: Path) -> None:
    from recall_services import archive_state

    archive_state.ensure_from_export(data.write_export(tmp_path / "messages.csv", ["one", "two"]))
    try:
        assert len(frame.messages_frame()) == 2
    finally:
        archive_state.shared().close()


# --- attachments and tapbacks, for the figures beside the messages ------------


def test_the_attachments_of_one_conversation(state: ArchiveState, orchard: world.Orchard) -> None:
    found = frame.attachments_frame(state, INES)
    assert list(found.columns) == ["message_id", "chat_id", "mime_type", "is_from_me", "timestamp", "category"]
    assert list(found["mime_type"]) == ["image/heic", "image/jpeg", "image/png"]
    assert list(found["category"]) == ["photo", "photo", "photo"]
    assert list(found["is_from_me"]) == [0, 1, 1]
    assert set(found["chat_id"]) == {INES}
    assert list(found["message_id"]) == [orchard.id("i3"), orchard.id("i9"), orchard.id("i9")]


def test_the_tapbacks_of_one_conversation(state: ArchiveState) -> None:
    found = frame.reactions_frame(state, INES)
    assert list(found.columns) == ["message_id", "chat_id", "reaction_type", "is_add", "is_from_me", "timestamp"]
    assert found.to_dict("records") == [
        {
            "message_id": found.iloc[0]["message_id"],
            "chat_id": INES,
            "reaction_type": "love",
            "is_add": True,
            "is_from_me": 0,
            # the time of the message that was tapped: the archive does not give the time of a tapback
            "timestamp": pd.Timestamp("2024-03-05 07:50:00"),
        }
    ]


def test_the_media_figures_of_the_server_read_the_two_frames(state: ArchiveState, messages: pd.DataFrame) -> None:
    stats = compute_stats(
        filter_conversation(messages, INES), frame.attachments_frame(state, INES), frame.reactions_frame(state, INES)
    )
    assert (stats.attachments.total, stats.attachments.photos) == (3, 3)
    assert (stats.attachments.sent_attachments, stats.attachments.received_attachments) == (2, 1)
    assert (stats.reactions.total, stats.reactions.loves) == (1, 1)
    assert (stats.reactions.sent_reactions, stats.reactions.received_reactions) == (0, 1)


@pytest.mark.parametrize(
    ("mime", "category"),
    [
        ("image/gif", "gif"),
        ("image/heic", "photo"),
        ("video/quicktime", "video"),
        ("audio/amr", "audio"),
        ("application/pdf", "document"),
        ("application/vnd.oasis.opendocument.text", "document"),
        ("text/vcard", "other"),
        ("application/octet-stream", "other"),
        ("", "other"),
    ],
)
def test_the_kind_of_an_attachment_is_the_one_the_exporter_gives(mime: str, category: str) -> None:
    assert frame.category_of(mime) == category


def test_a_conversation_with_nothing_gives_empty_frames_with_their_columns(state: ArchiveState) -> None:
    for chat in (world.UNSAVED, "+15550000000", ""):
        assert frame.attachments_frame(state, chat).empty
        assert frame.reactions_frame(state, chat).empty
        assert "category" in frame.attachments_frame(state, chat).columns
        assert "reaction_type" in frame.reactions_frame(state, chat).columns
        stats = compute_stats(
            filter_conversation(frame.messages_frame(state), chat),
            frame.attachments_frame(state, chat),
            frame.reactions_frame(state, chat),
        )
        assert (stats.attachments.total, stats.reactions.total) == (0, 0)


def test_the_figures_of_a_range_of_days_read_the_two_frames(state: ArchiveState) -> None:
    assert len(frame.attachments_frame(state, INES, since="2024-03-01", until="2024-03-01")) == 1
    assert len(frame.attachments_frame(state, INES, since="2024-03-05")) == 2
    assert frame.reactions_frame(state, INES, until="2024-03-04").empty
