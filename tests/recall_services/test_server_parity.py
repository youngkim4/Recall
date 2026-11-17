"""The services against the server's own payload builders at this commit.

`ui_server` is imported here and in no service module. It is imported when
this file is collected, as the base tests do: imported inside a test, it would
keep the output stream of that one test.

Each test gives one invented export to the server's builder and, through the
legacy bridge, to the service, and compares the two payloads. A difference
that is there on purpose is stated in the test that holds it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import ui_server
from archive import Archive, import_legacy_csv
from setup_status import DbProbe
from recall_services import conversations, day, messages, search, unreadable
from recall_services.names import Names

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache")

AVERY = "+15550180001"
BLAKE = "+15550180002"
UNSAVED = "+15550189999"
GROUP = "chat880011"
TITLED = "chat880022"
RECORDS = [
    {"name": "Avery Stone", "phones": [AVERY], "emails": []},
    {"name": "Blake Reyes", "phones": [BLAKE], "emails": []},
]
COLUMNS = [
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
]
PHOTO = "￼"


def _row(chat: str, stamp: str, text: str, *, mine: bool = False, sender: str = "", title: str = "", kind: str = "") -> tuple:
    # the old exporter wrote the handle of the other person on an own message of a one-to-one thread
    writer = sender or (chat if not chat.startswith("chat") else "")
    return (0, stamp, writer, text, chat, title, int(mine), "iMessage", kind, "~/Library/x.heic" if kind else "")


ROWS = [
    _row(AVERY, "2025-01-01 09:00:00", "pizza night at mine"),
    _row(AVERY, "2025-01-01 09:05:00", "bringing the pizza stone", mine=True),
    _row(AVERY, "2025-01-01 09:06:00", PHOTO, kind="image/heic"),
    _row(AVERY, "2025-01-02 19:30:00", "  that was a good pizza  "),
    _row(AVERY, "2025-01-04 08:00:00", "back to salad"),
    _row(UNSAVED, "2025-01-03 10:00:00", "your pizza order is ready"),
    _row(GROUP, "2025-01-05 12:00:00", "pizza on friday?", sender=AVERY),
    _row(GROUP, "2025-01-05 12:01:00", "yes to pizza", sender=BLAKE),
    _row(GROUP, "2025-01-05 12:02:00", "count me in for pizza", sender=UNSAVED),
    _row(GROUP, "2025-01-05 12:03:00", "booking a table", mine=True),
    _row(TITLED, "2025-01-06 12:00:00", "minutes of the meeting", sender=BLAKE, title="Allotment Board"),
]


@pytest.fixture
def export(tmp_path: Path) -> Path:
    frame = pd.DataFrame(ROWS, columns=COLUMNS)
    path = tmp_path / "messages.csv"
    frame.assign(message_id=range(1, len(frame) + 1)).to_csv(path, index=False)
    return path


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> Names:
    """The same names for both sides. The server asks its own function, which is replaced here."""
    names = Names.from_records(RECORDS)

    def lookup(handles: Any, **_: Any) -> dict[str, str]:
        found = {str(handle): names.saved(str(handle)) for handle in handles}
        return {handle: name for handle, name in found.items() if name}

    monkeypatch.setattr(ui_server, "resolve_contact_names", lookup)
    return names


@pytest.fixture
def bridged(export: Path, tmp_path: Path) -> Iterator[Archive]:
    path = tmp_path / "archive.sqlite3"
    import_legacy_csv(export, path)
    with Archive.open(path) as opened:
        yield opened


def as_json(value: Any) -> Any:
    """A payload as the route sends it."""
    return json.loads(json.dumps(value, default=ui_server.json_default))


def same_message(old: dict[str, Any], new: dict[str, Any]) -> None:
    """One message from both sides. Three keys differ on purpose."""
    # the bridge names a message legacy:<ROWID>; the export named it by the ROWID
    assert new["messageId"] == f"legacy:{old['messageId']}"
    # an own message has no sender in the archive; the old export held the other person's handle there
    assert new["sender"] == ("" if new["isFromMe"] else old["sender"])
    if new["chatId"].startswith("chat"):
        assert new["senderName"] == old["senderName"]
    else:
        # Outside a group the page already says who wrote it, and no screen reads this key there.
        # The server names the writer only when the same person also wrote a group message among the results.
        assert new["senderName"] == ""
        assert old["senderName"] in ("", new["displayName"])
    for key in ("chatId", "displayName", "timestamp", "text", "attachment", "isFromMe"):
        assert new[key] == old[key], key


def test_the_text_limit_is_the_one_of_the_server() -> None:
    assert messages.MESSAGE_TEXT_LIMIT == ui_server.MESSAGE_TEXT_LIMIT


@pytest.mark.parametrize(("query", "contact", "limit"), [("pizza", "", 60), ("pizza", AVERY, 2), ("pizza", GROUP, 60), ("salad", "", 60), ("zeppelin", "", 60)])
def test_a_search_for_one_word(export: Path, bridged: Archive, saved: Names, query: str, contact: str, limit: int) -> None:
    old = as_json(ui_server.search_messages_payload(export, query=query, contact=contact, limit=limit))
    new = search.payload(bridged, saved, query, contact, limit)
    assert set(new) == set(old) | {"offset"}
    for key in ("query", "contact", "count", "total"):
        assert new[key] == old[key], key
    assert len(new["results"]) == len(old["results"])
    for before, after in zip(old["results"], new["results"], strict=True):
        same_message(before, after)


def test_browsing_one_conversation_with_no_words(export: Path, bridged: Archive, saved: Names) -> None:
    old = as_json(ui_server.search_messages_payload(export, contact=AVERY, limit=3))
    new = search.payload(bridged, saved, "", AVERY, 3)
    assert (new["count"], new["total"]) == (old["count"], old["total"]) == (3, 5)
    for before, after in zip(old["results"], new["results"], strict=True):
        same_message(before, after)


def test_no_words_and_no_conversation_differ_on_purpose(export: Path, bridged: Archive, saved: Names) -> None:
    # the server lists the newest messages of the whole archive; no screen asks for that, and the service gives none
    assert ui_server.search_messages_payload(export, limit=3)["total"] == len(ROWS)
    assert search.payload(bridged, saved, "", "", 3)["total"] == 0


def test_a_part_of_a_word_differs_on_purpose(export: Path, bridged: Archive, saved: Names) -> None:
    # the server matches plain text anywhere; the archive matches whole words, and a star makes a prefix
    assert ui_server.search_messages_payload(export, query="piz")["total"] == 7
    assert search.payload(bridged, saved, "piz")["total"] == 0
    assert search.payload(bridged, saved, "piz*")["total"] == 7
    assert search.payload(bridged, saved, "piz", as_you_type=True)["total"] == 7


@pytest.mark.parametrize(
    ("contact", "when", "more"),
    [
        (AVERY, "2025-01-01", {}),
        (AVERY, "2025-01-02", {}),
        (AVERY, "2025-01-03", {}),
        (GROUP, "2025-01-05", {}),
        (GROUP, "2025-01-05", {"offset": 1, "limit": 2}),
        (GROUP, "2025-01-05", {"around": "2025-01-05 12:02:00", "limit": 2}),
        (TITLED, "2025-01-06", {}),
        ("+15550180000", "2025-01-01", {}),
    ],
)
def test_a_day(export: Path, bridged: Archive, saved: Names, contact: str, when: str, more: dict[str, Any]) -> None:
    old = as_json(ui_server.day_payload(export, contact, when, **more))
    new = day.payload(bridged, saved, contact, when, **more)
    assert set(new) == set(old)
    for key in ("contact", "day", "total", "offset", "previousDay", "nextDay"):
        assert new[key] == old[key], key
    assert len(new["messages"]) == len(old["messages"])
    for before, after in zip(old["messages"], new["messages"], strict=True):
        same_message(before, after)


def test_the_conversation_list(export: Path, bridged: Archive, saved: Names) -> None:
    frame = ui_server.contacts_from_messages(export, limit=40, name_lookup=lambda handles: {h: saved.saved(h) for h in handles})
    old = as_json(frame.to_dict(orient="records"))
    new = conversations.rows(bridged, saved, 40)
    assert [row["chat_id"] for row in new] == [row["chat_id"] for row in old]
    for before, after in zip(old, new, strict=True):
        assert set(after) == set(before)
        assert after == before
    assert conversations.name_count(new) == ui_server.count_display_names(frame)


def test_the_totals_of_the_archive(export: Path, bridged: Archive) -> None:
    assert conversations.summary(bridged) == as_json(ui_server.archive_summary(export))


def test_the_name_of_each_conversation(export: Path, bridged: Archive, saved: Names) -> None:
    _frame, _catalog, old = ui_server.cached_messages_bundle(export)
    new = saved.conversations(bridged.conversations())
    assert {key: name for key, name in new.items() if name} == {key: name for key, name in old.items() if name}
    assert new[GROUP] == "Avery Stone, Blake Reyes and 1 other"
    assert new[TITLED] == "Allotment Board"


STATUSES = ("readable", "fda_blocked", "missing", "empty", "invalid", "locked", "error")


@pytest.mark.parametrize("status", STATUSES)
@pytest.mark.parametrize("kind", ["live", "copy"])
@pytest.mark.parametrize("detail", ["", "not a SQLite database"])
def test_the_reason_a_database_can_not_be_read_has_the_words_of_the_server(status: str, kind: str, detail: str) -> None:
    probe = DbProbe(status=status, kind=kind, path="/invented/place/chat.db", detail=detail)
    stopped = ui_server.why_unreadable(probe)
    assert unreadable.why(probe) == (None if stopped is None else str(stopped))
    assert unreadable.closed_to_recall(probe) == ui_server.closed_to_recall(probe)


def test_the_last_words_say_that_the_archive_is_as_it_was() -> None:
    assert unreadable.ARCHIVE_KEPT == ui_server.ARCHIVE_KEPT
    probe = DbProbe(status="missing", kind="copy", path="/invented/place/chat.db")
    assert str(unreadable.why(probe)).endswith(unreadable.ARCHIVE_KEPT)
