"""The decoder fix and the legacy bridge together.

An invented Messages database holds texts that are only in attributedBody, as
a current macOS writes them. The archives come from Apple's NSArchiver
(tests/fixtures/typedstream). The export is written as an earlier build of the
app wrote it, with the decoder of parse_imessage.py, and import_legacy_csv loads it.
The archive must hold each text whole and untrimmed.

The same database also goes through import_chat_db, which uses the archive's
own decoder. The two ways give the same text, with the exceptions stated below.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from archive import INDEXED_TEXT_CHARS, Archive, ImportResult, import_chat_db, import_legacy_csv
from conversation import load_messages
from parse_imessage import extract_attachments, extract_messages, extract_reactions
from tests import guards
from tests.integration import bridge_data as data
from tests.integration.bridge_data import ROWAN, AppleFixture, BlobDatabase

# texts that must arrive whole. The old decoder kept 43 bytes of `ascii_300`.
WHOLE = (
    "ascii_300",
    "multibyte_300",
    "ascii_32768",
    "ascii_70000",
    "emoji_end_130",
    "zwj_emoji",
    "format_chars",
    "spaces_both_ends",
    "newline_ends",
    "whitespace_only",
)
OTHER = ("control_chars", "attachment_char", "empty", "not_text")
TAPBACK = f"Loved {data.CURLY_OPEN}See you at eight{data.CURLY_CLOSE}"
FIXTURES = data.apple_fixtures()


@dataclass(frozen=True)
class Bridged:
    database: BlobDatabase
    exported: dict[str, Any]  # what write_old_export returned
    csv_path: Path
    result: ImportResult
    archive: Archive  # database -> the export of an earlier build -> import_legacy_csv
    pulled: Archive  # database -> import_chat_db


def write_old_export(database: Path, csv_path: Path) -> dict[str, Any]:
    """Write the three files of an export, as the import of an earlier build of the app wrote them.

    The server no longer writes an export: an update is a pull into the
    archive. The readers of that import are still in parse_imessage.py.
    """
    frames = {
        "messages": (extract_messages(str(database)), csv_path),
        "attachments": (extract_attachments(str(database)), csv_path.with_name(f"{csv_path.stem}_attachments.csv")),
        "reactions": (extract_reactions(str(database)), csv_path.with_name(f"{csv_path.stem}_reactions.csv")),
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    for frame, target in frames.values():
        frame.to_csv(target, index=False)
    return {**{name: len(frame) for name, (frame, _) in frames.items()}, "messagesPath": str(csv_path)}


@pytest.fixture(scope="module")
def bridged(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Bridged]:
    folder = tmp_path_factory.mktemp("decoder-bridge")
    blobs = {name: FIXTURES[name].blob for name in (*WHOLE, *OTHER)}
    csv_path = folder / "export" / "messages.csv"
    # the exporter writes local times and the bridge reads them: both in the time zone of the tests
    with guards.time_zone():
        database = data.blob_database(folder / "source" / "chat.db", blobs, tapback_text=TAPBACK)
        exported = write_old_export(database.path, csv_path)
        result = import_legacy_csv(csv_path, folder / "bridge" / "archive.sqlite3")
        import_chat_db(database.path, folder / "pull" / "archive.sqlite3")
    archive = Archive.open(folder / "bridge" / "archive.sqlite3")
    pulled = Archive.open(folder / "pull" / "archive.sqlite3")
    yield Bridged(database, exported, csv_path, result, archive, pulled)
    archive.close()
    pulled.close()


def row_of(bridged: Bridged, label: str, *, pulled: bool = False) -> dict[str, Any] | None:
    archive = bridged.pulled if pulled else bridged.archive
    return archive.message_by_source_rowid(bridged.database.rowids[label])


def test_the_fixtures_hold_texts_of_300_bytes_and_more() -> None:
    """The old decoder read a length of 128 bytes or more wrong. These texts are past each of its limits."""
    sizes = {name: FIXTURES[name].utf8_bytes for name in WHOLE}
    assert sizes["ascii_300"] == sizes["multibyte_300"] == 300
    assert (sizes["ascii_32768"], sizes["ascii_70000"]) == (32_768, 70_000)


def test_the_exporter_writes_the_three_files_of_an_export(bridged: Bridged) -> None:
    stored = len(WHOLE) + 4  # the text column row, control_chars, attachment_char and the photo
    assert bridged.exported == {
        "messages": stored + 1,  # and the tapback, which the export holds as a row of text
        "attachments": 1,
        "reactions": 1,
        "messagesPath": str(bridged.csv_path),
    }
    names = sorted(path.name for path in bridged.csv_path.parent.iterdir())
    assert names == ["messages.csv", "messages_attachments.csv", "messages_reactions.csv"]
    header = bridged.csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert tuple(header.split(",")) == data.EXPORT_COLUMNS


def test_the_bridge_stores_every_row_of_the_export(bridged: Bridged) -> None:
    result = bridged.result
    assert result.skipped == {}
    assert (result.messages_added, result.system_added) == (len(WHOLE) + 4, 1)
    assert result.oversize_texts == sum(1 for name in WHOLE if FIXTURES[name].utf8_bytes > INDEXED_TEXT_CHARS) == 2
    assert bridged.archive.summary()["messages"] == bridged.pulled.summary()["messages"] == len(WHOLE) + 4


@pytest.mark.parametrize("name", WHOLE)
def test_the_archive_holds_the_whole_text_untrimmed(bridged: Bridged, name: str) -> None:
    fixture: AppleFixture = FIXTURES[name]
    stored = row_of(bridged, name)
    assert stored is not None and stored["kind"] == "text"
    raw = stored["text"].encode("utf-8")
    assert len(raw) == fixture.utf8_bytes
    assert hashlib.sha256(raw).hexdigest() == fixture.sha256
    if fixture.text is not None:  # the manifest gives the text itself up to 512 bytes
        assert stored["text"] == fixture.text
    assert (stored["conversation"], stored["sender"], stored["is_from_me"]) == (ROWAN, ROWAN, False)


@pytest.mark.parametrize("name", WHOLE)
def test_the_export_file_holds_the_whole_text(bridged: Bridged, name: str) -> None:
    frame = load_messages(str(bridged.csv_path))
    texts = dict(zip(frame["message_id"], frame["text"], strict=True))
    assert texts[bridged.database.rowids[name]] == row_of(bridged, name)["text"]


@pytest.mark.parametrize(
    ("name", "start", "end"),
    [
        ("spaces_both_ends", "  two", "after "),
        ("newline_ends", "\nstarts", "newline\n"),
        ("whitespace_only", "  \n ", "  \n "),
    ],
)
def test_outer_white_space_is_kept(bridged: Bridged, name: str, start: str, end: str) -> None:
    text = row_of(bridged, name)["text"]
    assert text.startswith(start) and text.endswith(end)
    assert text != text.strip()


def test_invisible_characters_are_kept(bridged: Bridged) -> None:
    assert chr(0x200D) in row_of(bridged, "zwj_emoji")["text"]  # the joiner inside an emoji
    assert chr(0x200B) in row_of(bridged, "format_chars")["text"]  # a zero-width space
    assert row_of(bridged, "emoji_end_130")["text"].endswith(chr(0x1F600))


@pytest.mark.parametrize("label", ["text_column", *WHOLE, "control_chars", "attachment_char", "photo"])
def test_time_and_direction_come_through_the_export(bridged: Bridged, label: str) -> None:
    stored, pulled = row_of(bridged, label), row_of(bridged, label, pulled=True)
    assert stored["sent_at"] == pulled["sent_at"] == bridged.database.sent_at[label]
    assert stored["timestamp"] == pulled["timestamp"]
    assert stored["is_from_me"] is pulled["is_from_me"] is (label == "text_column")


@pytest.mark.parametrize("name", [*WHOLE, "attachment_char"])
def test_the_bridge_and_a_pull_hold_the_same_text(bridged: Bridged, name: str) -> None:
    """The decoder of parse_imessage.py and the decoder of the archive read the same text."""
    stored, pulled = row_of(bridged, name), row_of(bridged, name, pulled=True)
    assert (stored["text"], stored["kind"]) == (pulled["text"], pulled["kind"])
    assert stored["id"] == data.legacy_id(bridged.database.rowids[name]) != pulled["id"]


def test_a_message_that_is_only_the_attachment_placeholder(bridged: Bridged) -> None:
    stored = row_of(bridged, "attachment_char")
    assert (stored["kind"], stored["text"], stored["attachments"]) == ("attachment", "", [])
    photo = row_of(bridged, "photo")
    assert (photo["kind"], photo["text"], photo["attachments"]) == ("attachment", "", ["image/jpeg"])


def test_control_characters_are_the_one_difference_in_text(bridged: Bridged) -> None:
    """A difference: the exporter drops C0 controls other than tab and newline. A pull keeps them.

    parse_imessage.py, `_DROPPED_CONTROLS`: "not part of a message, and NUL breaks
    the CSV". The archive stores a text as Messages has it and removes only U+FFFC
    and NUL (README, "How a source row is stored"). A carriage return is a C0 control.
    """
    assert FIXTURES["control_chars"].text == "a\x00b\x01c\x1fd\te\nf\rg\x7fh"
    assert row_of(bridged, "control_chars")["text"] == "abcd\te\nfg\x7fh"
    assert row_of(bridged, "control_chars", pulled=True)["text"] == "ab\x01c\x1fd\te\nf\rg\x7fh"


@pytest.mark.parametrize("name", ["empty", "not_text"])
def test_a_blob_with_no_text_is_no_row_on_either_way(bridged: Bridged, name: str) -> None:
    assert row_of(bridged, name) is None
    assert row_of(bridged, name, pulled=True) is None


def test_a_tapback_is_a_hidden_row_after_the_bridge_and_a_reaction_after_a_pull(bridged: Bridged) -> None:
    """A difference: the export holds a tapback as a row of text. Only a pull knows it as a reaction.

    README, "Legacy export bridge": "A bridge archive has no group events, no unsent
    messages, no edit marks, no reply links and no reactions."
    """
    stored = row_of(bridged, "tapback")
    assert (stored["kind"], stored["text"]) == ("system", TAPBACK)
    assert bridged.result.counts["reaction_phrases"] == 1
    target = row_of(bridged, "text_column")
    assert bridged.archive.reactions([target["id"]]) == {target["id"]: []}
    assert row_of(bridged, "tapback", pulled=True) is None
    pulled_target = row_of(bridged, "text_column", pulled=True)
    assert bridged.pulled.reactions([pulled_target["id"]]) == {
        pulled_target["id"]: [{"kind": "love", "is_from_me": False, "sender": ROWAN}]
    }
    assert data.archive_ids(bridged.archive.day(ROWAN, "2024-05-06")["messages"]) == [
        data.legacy_id(bridged.database.rowids[label])
        for label in ("text_column", *WHOLE, "control_chars", "attachment_char", "photo")
    ]
