"""The shape every message has in a payload, and the ids that the browser saved earlier."""

from __future__ import annotations

from archive import Archive
from recall_services import messages
from recall_services.names import Names
from tests.recall_services import world

# SearchResult in app/src/types.ts at 89ab504
RESULT_KEYS = {
    "messageId",
    "chatId",
    "displayName",
    "senderName",
    "timestamp",
    "sender",
    "text",
    "attachment",
    "isFromMe",
}


def shaped(archive: Archive, saved: Names, orchard: world.Orchard, label: str) -> dict[str, object]:
    message = archive.message(orchard.id(label))
    assert message is not None
    return messages.payloads(archive, saved, [message])[0]


def test_a_message_has_the_keys_of_a_search_result(archive: Archive, saved: Names, orchard: world.Orchard) -> None:
    assert set(shaped(archive, saved, orchard, "i1")) == RESULT_KEYS


def test_a_received_message_of_one_person(archive: Archive, saved: Names, orchard: world.Orchard) -> None:
    assert shaped(archive, saved, orchard, "i1") == {
        "messageId": orchard.id("i1"),
        "chatId": world.INES_PHONE,
        "displayName": "Ines Calder",
        # with one person the page already says who wrote it
        "senderName": "",
        "timestamp": "2024-03-01 09:00:00",
        "sender": world.INES_PHONE,
        "text": "Are you bringing the ladder on Saturday?",
        "attachment": "",
        "isFromMe": False,
    }


def test_an_own_message_has_no_sender(archive: Archive, saved: Names, orchard: world.Orchard) -> None:
    found = shaped(archive, saved, orchard, "i2")
    assert (found["isFromMe"], found["sender"], found["senderName"]) == (True, "", "")


def test_text_is_trimmed_for_the_screen_only(archive: Archive, saved: Names, orchard: world.Orchard) -> None:
    stored = archive.message(orchard.id("i4"))
    assert stored is not None and stored["text"] == "  Picked the first lemons today  "
    assert shaped(archive, saved, orchard, "i4")["text"] == "Picked the first lemons today"


def test_a_long_text_is_cut_at_a_word_and_marked(archive: Archive, saved: Names, orchard: world.Orchard) -> None:
    text = str(shaped(archive, saved, orchard, "i5")["text"])
    assert len(text) <= messages.MESSAGE_TEXT_LIMIT + 1
    assert text.endswith("…")
    assert world.LONG_TEXT.startswith(text[:-1])
    assert world.LONG_TEXT_WORD not in text


def test_a_message_that_is_only_a_photo(archive: Archive, saved: Names, orchard: world.Orchard) -> None:
    found = shaped(archive, saved, orchard, "i3")
    assert (found["text"], found["attachment"]) == ("", "Photo")
    assert shaped(archive, saved, orchard, "i9")["attachment"] == "2 photos"


def test_in_a_group_a_received_message_names_who_wrote_it(archive: Archive, saved: Names, orchard: world.Orchard) -> None:
    by_tomas = shaped(archive, saved, orchard, "g2")
    assert (by_tomas["displayName"], by_tomas["senderName"], by_tomas["sender"]) == (
        world.GARDEN_TITLE,
        "Tomas Reyes",
        world.TOMAS,
    )
    # someone not in Contacts has no name here; the interface labels them
    unsaved = shaped(archive, saved, orchard, "g3")
    assert (unsaved["senderName"], unsaved["sender"]) == ("", world.UNSAVED)
    mine = shaped(archive, saved, orchard, "g4")
    assert (mine["senderName"], mine["sender"], mine["isFromMe"]) == ("", "", True)


def test_a_group_with_no_title_goes_by_its_people(archive: Archive, saved: Names, orchard: world.Orchard) -> None:
    assert shaped(archive, saved, orchard, "u_s0")["displayName"] == world.UNTITLED_NAME


def test_a_person_who_is_not_saved_has_no_display_name(archive: Archive, saved: Names, orchard: world.Orchard) -> None:
    assert shaped(archive, saved, orchard, "s1")["displayName"] == ""


def test_many_messages_keep_their_order_and_read_each_conversation_once(
    archive: Archive, saved: Names, orchard: world.Orchard
) -> None:
    page = archive.messages(world.GARDEN, limit=10)["messages"]
    shown = messages.payloads(archive, saved, page)
    assert [item["messageId"] for item in shown] == [message["id"] for message in page]
    assert {item["displayName"] for item in shown} == {world.GARDEN_TITLE}


# --- ids that the browser saved earlier --------------------------------------


def test_an_id_of_today_finds_its_message(archive: Archive, orchard: world.Orchard) -> None:
    found = messages.find_saved(archive, orchard.id("i1"))
    assert found is not None and found["id"] == orchard.id("i1")


def test_a_row_number_that_the_browser_saved_finds_its_message(archive: Archive, orchard: world.Orchard) -> None:
    # the export named a message by its ROWID; saved chats and saved messages hold that number as text
    for form in (str(orchard.rowids["i1"]), f"legacy:{orchard.rowids['i1']}", f" {orchard.rowids['i1']} "):
        found = messages.find_saved(archive, form)
        assert found is not None and found["id"] == orchard.id("i1")


def test_a_row_number_is_trusted_only_when_the_saved_conversation_and_time_agree(
    archive: Archive, orchard: world.Orchard
) -> None:
    rowid = str(orchard.rowids["i1"])
    assert messages.find_saved(archive, rowid, chat_id=world.INES_PHONE, timestamp="2024-03-01 09:00:00") is not None
    # a rebuilt Messages database numbers its rows again: the number then names another message
    assert messages.find_saved(archive, rowid, chat_id=world.GARDEN) is None
    assert messages.find_saved(archive, rowid, timestamp="2024-03-01 09:00:01") is None


def test_an_id_that_names_nothing_finds_nothing(archive: Archive) -> None:
    for unknown in ("", "   ", "99999999", "legacy:", "legacy:abc", "no-such-guid", None, 12):
        assert messages.find_saved(archive, unknown) is None  # type: ignore[arg-type]


def test_a_tapback_row_is_not_a_saved_message(archive: Archive, orchard: world.Orchard) -> None:
    # the old export held a tapback as a row of text; the archive holds it as a reaction
    event = archive.message(orchard.id("g_event"))
    assert event is not None and event["kind"] == "system"
    assert messages.find_saved(archive, orchard.id("g_event")) is None
