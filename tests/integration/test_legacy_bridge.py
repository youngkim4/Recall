"""The legacy bridge on an invented export that holds the rows a real export holds.

Reaction phrases, rows with no conversation, one oversize text, rows that are
only an attachment, and rows that the bridge does not store. The tests state
what the archive holds for each, and what archive.compat.messages_frame gives:
that frame is what the figures on pandas read after the hook-up.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest

from archive import INDEXED_TEXT_CHARS, compat, legacy_csv
from archive.tapbacks import is_reaction_phrase
from conversation import compute_stats, filter_conversation, in_conversation, sent_mask
from tests import guards
from tests.integration import bridge_data as data
from tests.integration.bridge_data import GROUP, GROUP_TITLE, MIRA, PLACEHOLDER, ROWAN, Loaded, export_row

DAY = "2024-03-01"
PARTY = chr(0x1F389)
FOLDER = "~/Library/Messages/Attachments"


def quoted(text: str) -> str:
    return f"{data.CURLY_OPEN}{text}{data.CURLY_CLOSE}"


WORDS = (
    "The ferry leaves at nine",
    "ok then",
    "rain again",
    "the gull took my sandwich",
    "bring the tickets",
    "pier two?",
    "we made the ferry",
)
# every form of archive/README.md, "Legacy export bridge"
PHRASES = (
    f"Loved {quoted(WORDS[0])}",
    'Liked "ok then"',
    f"Disliked {quoted(WORDS[2])}",
    f"Laughed at {quoted(WORDS[3])}",
    f"Emphasized {quoted(WORDS[4])}",
    f"Questioned {quoted(WORDS[5])}",
    "Loved an image",
    "Liked a movie",
    "Laughed at an audio message",
    "Emphasized an attachment",
    "Questioned a contact",
    f"Removed a heart from {quoted(WORDS[0])}",
    'Removed a like from "ok then"',
    "Removed a dislike from an image",
    f"Removed a laugh from {quoted(WORDS[3])}",
    f"Removed an exclamation from {quoted(WORDS[4])}",
    f"Removed a question mark from {quoted(WORDS[5])}",
    f"Reacted {PARTY} to {quoted(WORDS[6])}",
    f"Removed {PARTY} from {quoted(WORDS[6])}",
)
# text that starts like a phrase and is a message
NOT_PHRASES = (
    "Loved it",
    'Loved "Dune" a lot',
    "Loved a sticker",
    'Reacted badly to "the news"',
    f"loved {quoted('lower case')}",
    "Liked an image yesterday",
    f"Loved {data.CURLY_OPEN}a quote that never closes",
)
OVERSIZE = "harborlight starts a long note. " + "the tide table says nothing new. " * 400 + "afterglowzz ends it"
PHOTO = f"{FOLDER}/aa/01/IMG_0001.jpeg"
PAIR = f"{FOLDER}/bb/02/IMG_0002.heic|{FOLDER}/bb/03/IMG_0003.mov"
OLD_LABEL = f"[Attachment] image/heic|video/quicktime | {PAIR}"
VIEW = f"{FOLDER}/cc/04/view.png"
TWO_FILES = f"{FOLDER}/dd/05/IMG_0005.jpeg|{FOLDER}/dd/06/note.bin"
UNKNOWN_TYPE = "application/octet-stream"
FIRST_PHRASE, FIRST_NOT_PHRASE = 20, 40
UNFILED = (50, 51)
NOT_STORED = {"bad_row": 2, "no_date": 1, "empty": 1}
STORED = 26  # the messages that the archive holds, with the two that have no conversation
IN_ROWAN, SENT, RECEIVED = 22, 6, 16  # the messages of the conversation with Rowan


def files(types: str, paths: str) -> dict[str, str]:
    """The two attachment lists of a row of an export."""
    return {"attachment_types": types, "attachment_files": paths}


def _rows() -> list[dict[str, Any]]:
    def at(hour: int, minute: int) -> str:
        return f"{DAY} {hour:02d}:{minute:02d}:00"

    said = [export_row(1 + place, at(9, place), text, from_me=place % 2 == 1) for place, text in enumerate(WORDS)]
    taps = [
        export_row(FIRST_PHRASE + place, at(10, place), text, from_me=place % 2 == 0)
        for place, text in enumerate(PHRASES)
    ]
    plain = [export_row(FIRST_NOT_PHRASE + place, at(11, place), text) for place, text in enumerate(NOT_PHRASES)]
    rest = [
        export_row(50, at(12, 0), "unfiled words from Mira", "", sender=MIRA),
        export_row(51, at(12, 1), "unfiled words of mine", "", from_me=True),
        export_row(60, at(13, 0), OVERSIZE),
        export_row(70, at(14, 0), PLACEHOLDER, **files("image/jpeg", PHOTO)),
        export_row(71, at(14, 1), OLD_LABEL, from_me=True, **files("image/heic|video/quicktime", PAIR)),
        export_row(72, at(14, 2), "[Attachment]"),
        export_row(73, at(14, 3), f"{PLACEHOLDER}Look at this view", **files("image/png", VIEW)),
        export_row(74, at(14, 4), PLACEHOLDER, from_me=True),
        export_row(75, at(14, 5), PLACEHOLDER * 2, **files("image/jpeg", TWO_FILES)),
        export_row(80, at(15, 0), "Dinner on Friday?", GROUP, sender=MIRA, chat_display_name=GROUP_TITLE),
        export_row(81, at(15, 1), "I can bring bread", GROUP, from_me=True, chat_display_name=GROUP_TITLE),
        export_row(90, at(16, 0), "  \n ", from_me=True),
        export_row(95, at(16, 1), "this row does not say who sent it", is_from_me=""),
        export_row("x96", at(16, 2), "this row has no usable id"),
        # 2001-01-01 00:00:00 UTC in the time zone of the tests: what the old exporter wrote for a row with no date
        export_row(97, "2000-12-31 16:00:00", "this row has no date"),
        export_row(98, at(16, 3), ""),
    ]
    return [*said, *taps, *plain, *rest]


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Loaded]:
    # a module fixture is made before the time zone guard of a test: the bridge reads local times
    with guards.time_zone():
        both = data.load_both(tmp_path_factory.mktemp("bridge"), _rows())
    yield both
    both.archive.close()


@pytest.fixture(scope="module")
def frame(loaded: Loaded) -> pd.DataFrame:
    # the frame holds local times: it is made in the time zone that the tests read it in
    with guards.time_zone():
        return compat.messages_frame(loaded.archive)


def stored(loaded: Loaded, number: int) -> dict[str, Any]:
    message = loaded.archive.message(data.legacy_id(number))
    assert message is not None
    return message


def numbers(messages: Any) -> list[int]:
    return [int(str(message["id"]).removeprefix(data.LEGACY_PREFIX)) for message in messages]


def test_the_rule_reads_each_phrase_and_no_other_text() -> None:
    assert all(is_reaction_phrase(text) for text in PHRASES)
    assert not any(is_reaction_phrase(text) for text in (*NOT_PHRASES, *WORDS))


def test_the_run_counts_each_kind_of_row(loaded: Loaded) -> None:
    result = loaded.result
    assert (result.status, result.mode) == ("ok", "legacy_csv")
    assert result.rows_seen == len(_rows()) == 49
    assert (result.messages_seen, result.messages_added) == (STORED, STORED)
    assert (result.system_seen, result.system_added) == (len(PHRASES), len(PHRASES))
    assert result.counts["reaction_phrases"] == len(PHRASES) == 19
    assert result.counts["without_conversation"] == len(UNFILED)
    assert result.oversize_texts == 1
    assert result.skipped == NOT_STORED
    assert (result.attachments_added, result.conversations_added, result.handles_added) == (6, 2, 2)
    assert (result.reactions_added, result.blob_texts, result.decode_failures) == (0, 0, 0)


def test_the_summary_leaves_out_what_is_no_message_of_a_conversation(loaded: Loaded) -> None:
    assert loaded.archive.summary() == {
        "messages": STORED - len(UNFILED),
        "conversations": 2,
        "first_timestamp": f"{DAY} 09:00:00",
        "last_timestamp": f"{DAY} 16:00:00",
        "messages_without_conversation": 2,
    }
    listed = {found["id"]: found for found in loaded.archive.conversations()}
    assert list(listed) == [ROWAN, GROUP]
    assert (listed[ROWAN]["message_count"], listed[ROWAN]["members"]) == (IN_ROWAN, [ROWAN])
    assert (listed[GROUP]["name"], listed[GROUP]["is_group"], listed[GROUP]["members"]) == (GROUP_TITLE, True, [MIRA])
    assert loaded.archive.count() == STORED - len(UNFILED) == 24


def test_reaction_rows_are_hidden_from_every_default_query(loaded: Loaded) -> None:
    archive = loaded.archive
    taps = set(range(FIRST_PHRASE, FIRST_PHRASE + len(PHRASES)))
    assert not taps & set(numbers(archive.messages(ROWAN, limit=200)["messages"]))
    assert not taps & set(numbers(archive.iter_messages(ROWAN)))
    assert not taps & set(numbers(archive.day(ROWAN, DAY)["messages"]))
    assert archive.count(conversations=ROWAN) == archive.stats(ROWAN)["total_messages"] == IN_ROWAN
    assert archive.day(ROWAN, DAY)["total"] == IN_ROWAN
    # the words of a message are also inside the tapbacks on it: only the message is found
    assert numbers(hit["message"] for hit in archive.search("gull")["hits"]) == [4]
    assert archive.count(text="gull") == archive.search("gull")["total"] == 1
    # the neighbours of a message are messages: 19 tapback rows lie between 7 and 40
    assert numbers(archive.context(data.legacy_id(7), radius=3)) == [4, 5, 6, 7, 40, 41, 42]


def test_reaction_rows_are_given_on_request(loaded: Loaded) -> None:
    archive = loaded.archive
    assert archive.count(conversations=ROWAN, include_system=True) == IN_ROWAN + len(PHRASES)
    assert sorted(numbers(hit["message"] for hit in archive.search("gull", include_system=True)["hits"])) == [4, 23, 34]
    shown = archive.day(ROWAN, DAY, include_system=True)
    assert [message["text"] for message in shown["messages"] if message["kind"] == "system"] == list(PHRASES)


@pytest.mark.parametrize("place", range(len(PHRASES)))
def test_a_reaction_row_resolves_by_its_rowid_as_a_system_row(loaded: Loaded, place: int) -> None:
    """A saved chat of the app holds ROWIDs. The row of a tapback phrase is found, with kind `system`."""
    rowid = FIRST_PHRASE + place
    found = loaded.archive.message_by_source_rowid(rowid)
    assert found == stored(loaded, rowid)
    assert (found["kind"], found["text"], found["conversation"]) == ("system", PHRASES[place], ROWAN)
    assert found["is_from_me"] is (place % 2 == 0)


@pytest.mark.parametrize("place", range(len(NOT_PHRASES)))
def test_text_that_only_starts_like_a_phrase_is_a_message(loaded: Loaded, place: int) -> None:
    found = stored(loaded, FIRST_NOT_PHRASE + place)
    assert (found["kind"], found["text"]) == ("text", NOT_PHRASES[place])
    assert found["id"] in data.archive_ids(loaded.archive.day(ROWAN, DAY)["messages"])


def test_rows_with_no_conversation_are_stored_and_belong_to_none(loaded: Loaded) -> None:
    archive = loaded.archive
    received, sent = (stored(loaded, number) for number in UNFILED)
    assert (received["conversation"], received["sender"], received["is_from_me"]) == ("", MIRA, False)
    assert (sent["conversation"], sent["sender"], sent["is_from_me"]) == ("", "", True)
    assert archive.conversation("") is None
    assert archive.search("unfiled")["total"] == 0
    assert archive.search("unfiled", no_conversation=True)["total"] == 2
    assert archive.count(no_conversation=True) == 2
    assert numbers(archive.iter_messages(None, no_conversation=True)) == list(UNFILED)
    assert archive.message_by_source_rowid(50) == received


def test_an_oversize_text_is_whole_and_found_by_its_start_only(loaded: Loaded) -> None:
    archive = loaded.archive
    assert len(OVERSIZE) > INDEXED_TEXT_CHARS == 10_000
    assert OVERSIZE.index("afterglowzz") > INDEXED_TEXT_CHARS
    assert stored(loaded, 60)["text"] == OVERSIZE
    found = archive.search("harborlight")
    assert (found["total"], numbers(hit["message"] for hit in found["hits"])) == (1, [60])
    assert found["hits"][0]["ranges"] == [[0, len("harborlight")]]
    assert archive.search("afterglowzz")["total"] == archive.count(text="afterglowzz") == 0


@pytest.mark.parametrize(
    ("number", "kind", "text", "types"),
    [
        (70, "attachment", "", ["image/jpeg"]),
        (71, "attachment", "", ["image/heic", "video/quicktime"]),
        (72, "attachment", "", []),
        (73, "text", "Look at this view", ["image/png"]),
        (74, "attachment", "", []),
        # README: "When their lengths differ, no attachment of that row gets a type."
        (75, "attachment", "", [UNKNOWN_TYPE, UNKNOWN_TYPE]),
    ],
    ids=[
        "placeholder",
        "old label with two files",
        "old label alone",
        "words and a placeholder",
        "bare placeholder",
        "one type for two files",
    ],
)
def test_a_row_that_is_only_an_attachment(loaded: Loaded, number: int, kind: str, text: str, types: list[str]) -> None:
    found = stored(loaded, number)
    assert (found["kind"], found["text"], found["attachments"]) == (kind, text, types)
    assert PLACEHOLDER not in found["text"]
    assert found["id"] in data.archive_ids(loaded.archive.day(ROWAN, DAY)["messages"])


def test_text_of_white_space_is_kept_as_it_is(loaded: Loaded) -> None:
    assert (stored(loaded, 90)["kind"], stored(loaded, 90)["text"]) == ("text", "  \n ")


@pytest.mark.parametrize("rowid", [95, 97, 98])
def test_a_row_that_is_not_stored_resolves_to_nothing(loaded: Loaded, rowid: int) -> None:
    assert loaded.archive.message_by_source_rowid(rowid) is None
    assert loaded.archive.message(data.legacy_id(rowid)) is None


def test_the_columns_have_one_list_of_names_in_three_places() -> None:
    assert data.EXPORT_COLUMNS == compat.EXPORT_COLUMNS == legacy_csv.COLUMNS


def test_the_frame_has_the_columns_and_the_types_of_an_export(loaded: Loaded, frame: pd.DataFrame) -> None:
    assert tuple(frame.columns) == data.EXPORT_COLUMNS == tuple(loaded.frame.columns)
    assert str(frame["timestamp"].dtype) == str(loaded.frame["timestamp"].dtype) == "datetime64[ns]"
    assert (str(frame["chat_id"].dtype), str(frame["sender"].dtype)) == ("string", "string")
    assert str(frame["is_from_me"].dtype) == "int64" and set(frame["is_from_me"]) == {0, 1}


def test_the_frame_holds_the_messages_and_nothing_else(loaded: Loaded, frame: pd.DataFrame) -> None:
    """`message_id` is the id of the archive: `legacy:<ROWID>` after the bridge, a guid after a pull."""
    plain = range(FIRST_NOT_PHRASE, FIRST_NOT_PHRASE + len(NOT_PHRASES))
    visible = [1, 2, 3, 4, 5, 6, 7, *plain, *UNFILED, 60, 70, 71, 72, 73, 74, 75, 80, 81, 90]
    assert len(visible) == STORED
    assert list(frame["message_id"]) == [data.legacy_id(number) for number in visible]
    assert all(loaded.archive.message(message_id) is not None for message_id in frame["message_id"])
    assert not frame["text"].map(is_reaction_phrase).any()
    assert int(frame["chat_id"].notna().sum()) == loaded.archive.summary()["messages"]


def test_the_frame_has_no_chat_for_a_message_with_no_conversation(frame: pd.DataFrame) -> None:
    unfiled = frame[frame["chat_id"].isna()]
    assert list(unfiled["message_id"]) == [data.legacy_id(number) for number in UNFILED]
    assert not in_conversation(frame, "").any()


def test_the_frame_writes_the_old_label_for_a_row_that_is_only_an_attachment(frame: pd.DataFrame) -> None:
    text = dict(zip(frame["message_id"], frame["text"], strict=True))
    assert text[data.legacy_id(70)] == f"[Attachment] image/jpeg | {PHOTO}"
    assert text[data.legacy_id(71)] == OLD_LABEL  # the same text as the old exporter wrote
    assert text[data.legacy_id(72)] == text[data.legacy_id(74)] == "[Attachment]"
    assert text[data.legacy_id(73)] == "Look at this view"  # the export has the placeholder before these words
    types = dict(zip(frame["message_id"], frame["attachment_types"], strict=True))
    assert types[data.legacy_id(71)] == "image/heic|video/quicktime"
    assert types[data.legacy_id(72)] is None
    # one item for each attachment: the export has one type for the two files of this row
    assert types[data.legacy_id(75)] == f"{UNKNOWN_TYPE}|{UNKNOWN_TYPE}"
    assert text[data.legacy_id(75)] == f"[Attachment] {UNKNOWN_TYPE}|{UNKNOWN_TYPE} | {TWO_FILES}"


def test_the_frame_differs_from_the_export_in_the_sender_of_a_sent_row(loaded: Loaded, frame: pd.DataFrame) -> None:
    """The export names the other person on a sent row of a one-to-one chat. The frame names nobody."""
    sent = frame[frame["is_from_me"] == 1]
    assert sent["sender"].isna().all()
    export_sent = loaded.frame[loaded.frame["message_id"].astype(str) == "2"]
    assert list(export_sent["sender"]) == [ROWAN]
    received = frame[(frame["is_from_me"] == 0) & frame["chat_id"].notna()]
    assert set(received["sender"]) == {ROWAN, MIRA}


def test_the_frame_keeps_the_time_and_the_text_of_each_message(loaded: Loaded, frame: pd.DataFrame) -> None:
    export = loaded.frame.assign(message_id=data.frame_ids(loaded.frame)).set_index("message_id")
    of_frame = frame.set_index("message_id")
    assert (of_frame["timestamp"] == export.loc[of_frame.index, "timestamp"]).all()
    worded = [message_id for message_id in of_frame.index if int(message_id.split(":")[1]) not in range(70, 76)]
    assert list(of_frame.loc[worded, "text"]) == list(export.loc[worded, "text"])
    assert of_frame.loc[data.legacy_id(60), "text"] == OVERSIZE


def test_the_figures_on_the_frame_are_the_figures_of_the_archive(loaded: Loaded, frame: pd.DataFrame) -> None:
    """Today the figures count a tapback row as a message. On the frame of the archive they do not."""
    listed = loaded.archive.conversation(ROWAN)
    of_rowan = filter_conversation(frame, ROWAN)
    stats = compute_stats(of_rowan)
    # sent: three of the seven first messages, two attachment rows and the row of white space
    assert (stats.total_messages, stats.sent_count, stats.received_count) == (IN_ROWAN, SENT, RECEIVED)
    assert (listed["message_count"], listed["sent_count"], listed["received_count"]) == (IN_ROWAN, SENT, RECEIVED)
    assert int(sent_mask(of_rowan["is_from_me"]).sum()) == SENT and stats.unknown_direction_count == 0
    today = compute_stats(filter_conversation(loaded.frame, ROWAN))
    assert today.total_messages == IN_ROWAN + len(PHRASES) + sum(NOT_STORED.values())
