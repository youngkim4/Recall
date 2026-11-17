"""names.py: the one place for names. Names come in as records of the Contacts cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from chat_engine.labeler import MaskingLabeler
from group_names import derive_group_names
from recall_services import names as names_module
from recall_services.names import Names

INES_PHONE = "+15550170001"
INES_EMAIL = "ines@example.org"
TOMAS = "+15550170002"
MARA = "+15550170003"
UNSAVED = "+15550179999"
UNSAVED_EMAIL = "stranger@example.net"
RECORDS: list[Any] = [
    {"name": "Ines Calder", "phones": ["(555) 017-0001"], "emails": ["Ines@Example.org"]},
    {"name": "Tomas Reyes", "phones": ["+1 555 017 0002"], "emails": []},
    {"name": "  Mara   Lindqvist ", "phones": ["5550170003"]},
    {"name": "", "phones": ["5550170009"], "emails": []},
    {"name": "Second Ines", "phones": ["+15550170001"], "emails": []},
    "not a record",
    {"name": "No Handles"},
]


def conversation(conversation_id: str, members: list[str], *, title: str = "") -> dict[str, Any]:
    return {
        "id": conversation_id,
        "is_group": conversation_id.startswith("chat"),
        "name": title,
        "message_count": 10,
        "sent_count": 4,
        "received_count": 6,
        "first_timestamp": "2024-01-01 09:00:00",
        "last_timestamp": "2024-02-01 09:00:00",
        "members": members,
        "person_id": None,
    }


@pytest.fixture
def saved() -> Names:
    return Names.from_records(RECORDS)


@pytest.mark.parametrize(
    ("handle", "name"),
    [
        (INES_PHONE, "Ines Calder"),
        ("5550170001", "Ines Calder"),
        ("555-017-0001", "Ines Calder"),
        ("+1 (555) 017-0001", "Ines Calder"),
        (INES_EMAIL, "Ines Calder"),
        ("INES@example.ORG", "Ines Calder"),
        (TOMAS, "Tomas Reyes"),
        (MARA, "Mara Lindqvist"),
        (UNSAVED, ""),
        (UNSAVED_EMAIL, ""),
        ("5550170009", ""),
        ("chat123456", ""),
        ("", ""),
        ("   ", ""),
    ],
)
def test_the_saved_name_of_a_handle(saved: Names, handle: str, name: str) -> None:
    assert saved.saved(handle) == name


@pytest.mark.parametrize("handle", [None, 15550170001, 3.5, b"+15550170001", ["+15550170001"]])
def test_a_handle_that_is_not_text_has_no_name(saved: Names, handle: object) -> None:
    assert saved.saved(handle) == ""  # type: ignore[arg-type]


def test_a_number_matches_by_its_last_ten_digits_as_the_archive_matches_people(saved: Names) -> None:
    # archive/README.md "People": the digits, the last 10 when there are more
    assert saved.saved("+44 15550170001") == "Ines Calder"
    assert saved.saved("0170001") == ""


def test_the_first_record_wins_a_handle_that_two_names_share(saved: Names) -> None:
    assert saved.saved(INES_PHONE) == "Ines Calder"


def test_no_records_give_no_names() -> None:
    for nothing in (None, [], ()):
        empty = Names.from_records(nothing)
        assert empty.saved(INES_PHONE) == ""
        assert len(empty) == 0


@pytest.mark.parametrize("bad", ["text", {"name": "x"}, 7])
def test_records_that_are_not_a_list_are_refused(bad: object) -> None:
    with pytest.raises(TypeError):
        Names.from_records(bad)  # type: ignore[arg-type]


def test_how_many_handles_have_a_name(saved: Names) -> None:
    # three numbers and one address
    assert len(saved) == 4


def test_names_from_the_cache_file(tmp_path: Path) -> None:
    path = tmp_path / "contact_names.json"
    path.write_text(json.dumps({"exportedAt": "2026-11-16T10:00:00", "contacts": RECORDS}), encoding="utf-8")
    assert Names.load(path).saved(TOMAS) == "Tomas Reyes"


@pytest.mark.parametrize("text", ["{not json", "[]", '"text"', '{"contacts": "none"}', '{"people": []}', ""])
def test_a_cache_file_that_can_not_be_used_gives_no_names(tmp_path: Path, text: str) -> None:
    path = tmp_path / "contact_names.json"
    path.write_text(text, encoding="utf-8")
    assert len(Names.load(path)) == 0


def test_a_missing_cache_file_gives_no_names(tmp_path: Path) -> None:
    assert len(Names.load(tmp_path / "nowhere.json")) == 0
    assert len(Names.load(tmp_path)) == 0  # a folder is no cache file


def test_the_cache_file_is_read_again_only_when_it_changes(tmp_path: Path) -> None:
    path = tmp_path / "contact_names.json"
    path.write_text(json.dumps({"contacts": RECORDS[:1]}), encoding="utf-8")
    first = names_module.cached(path)
    assert names_module.cached(path) is first
    path.write_text(json.dumps({"contacts": RECORDS[:2], "pad": "a longer file"}), encoding="utf-8")
    second = names_module.cached(path)
    assert second is not first
    assert second.saved(TOMAS) == "Tomas Reyes" and first.saved(TOMAS) == ""


# --- conversations -----------------------------------------------------------


def test_a_person_goes_by_the_saved_name(saved: Names) -> None:
    assert saved.conversation(conversation(INES_PHONE, [INES_PHONE])) == "Ines Calder"
    assert saved.conversation(conversation(INES_EMAIL, [INES_EMAIL])) == "Ines Calder"


def test_a_person_who_is_not_saved_has_no_name(saved: Names) -> None:
    # the interface labels the rest: CONTRACTS.md "Names"
    assert saved.conversation(conversation(UNSAVED, [UNSAVED])) == ""
    assert saved.conversation(conversation(UNSAVED_EMAIL, [UNSAVED_EMAIL])) == ""


def test_a_group_goes_by_its_title(saved: Names) -> None:
    group = conversation("chat100", [INES_PHONE, TOMAS], title="  Garden Club ")
    assert saved.conversation(group) == "Garden Club"


def test_a_group_with_no_title_is_named_after_its_people_busiest_first(saved: Names) -> None:
    assert saved.conversation(conversation("chat101", [TOMAS, INES_PHONE])) == "Tomas Reyes and Ines Calder"
    assert saved.conversation(conversation("chat102", [INES_PHONE, TOMAS, MARA])) == (
        "Ines Calder, Tomas Reyes and Mara Lindqvist"
    )


def test_a_larger_group_names_its_two_busiest_saved_people(saved: Names) -> None:
    members = [UNSAVED, TOMAS, INES_PHONE, MARA]
    assert saved.conversation(conversation("chat103", members)) == "Tomas Reyes, Ines Calder and 2 others"
    assert saved.conversation(conversation("chat104", [INES_PHONE, UNSAVED])) == "Ines Calder and 1 other"


def test_a_group_with_one_person_holds_more_people_than_its_one_voice(saved: Names) -> None:
    assert saved.conversation(conversation("chat105", [INES_PHONE])) == "Ines Calder and others"


def test_a_group_in_which_nobody_is_saved_has_no_name(saved: Names) -> None:
    assert saved.conversation(conversation("chat106", [UNSAVED, UNSAVED_EMAIL])) == ""
    assert saved.conversation(conversation("chat107", [])) == ""


def test_one_person_with_two_handles_is_one_person_of_a_group(saved: Names) -> None:
    members = [INES_PHONE, TOMAS, INES_EMAIL]
    assert saved.conversation(conversation("chat108", members)) == "Ines Calder and Tomas Reyes"


def test_a_group_name_is_the_one_the_conversation_list_makes_today(saved: Names) -> None:
    """The same people, counted from an export frame by group_names.derive_group_names."""
    writers = [TOMAS] * 5 + [INES_PHONE] * 3 + [UNSAVED] * 2 + [MARA]
    frame = pd.DataFrame({"chat_id": "chat109", "sender": writers, "is_from_me": 0})
    today = derive_group_names(frame, ["chat109"], lambda handles: {h: saved.saved(h) for h in handles})
    # the archive lists members with the busiest writer first
    members = [TOMAS, INES_PHONE, UNSAVED, MARA]
    assert saved.conversation(conversation("chat109", members)) == today["chat109"]
    assert today["chat109"] == "Tomas Reyes, Ines Calder and 2 others"


def test_a_member_who_never_wrote_is_a_person_of_the_group(saved: Names) -> None:
    """A difference from the list of today, on purpose: the archive knows every member, the export only the writers."""
    writers = [TOMAS] * 3 + [INES_PHONE] * 2
    frame = pd.DataFrame({"chat_id": "chat110", "sender": writers, "is_from_me": 0})
    today = derive_group_names(frame, ["chat110"], lambda handles: {h: saved.saved(h) for h in handles})
    assert today["chat110"] == "Tomas Reyes and Ines Calder"
    members = [TOMAS, INES_PHONE, UNSAVED]  # the last one is in the group and wrote nothing
    assert saved.conversation(conversation("chat110", members)) == "Tomas Reyes, Ines Calder and 1 other"


def test_a_person_with_two_handles_stands_where_the_busier_handle_stands(saved: Names) -> None:
    """A difference from the list of today: that rule adds the messages of both handles of one person."""
    writers = [TOMAS] * 3 + [INES_PHONE] * 2 + [INES_EMAIL] * 2
    frame = pd.DataFrame({"chat_id": "chat111", "sender": writers, "is_from_me": 0})
    today = derive_group_names(frame, ["chat111"], lambda handles: {h: saved.saved(h) for h in handles})
    assert today["chat111"] == "Ines Calder and Tomas Reyes"
    members = [TOMAS, INES_PHONE, INES_EMAIL]  # by the count of each handle, as the archive lists them
    assert saved.conversation(conversation("chat111", members)) == "Tomas Reyes and Ines Calder"


def test_names_of_many_conversations(saved: Names) -> None:
    listed = [
        conversation(INES_PHONE, [INES_PHONE]),
        conversation(UNSAVED, [UNSAVED]),
        conversation("chat100", [INES_PHONE, TOMAS], title="Garden Club"),
    ]
    assert saved.conversations(listed) == {INES_PHONE: "Ines Calder", UNSAVED: "", "chat100": "Garden Club"}


def test_the_name_of_a_sender_is_the_saved_name_or_nothing(saved: Names) -> None:
    assert saved.sender(TOMAS) == "Tomas Reyes"
    assert saved.sender(UNSAVED) == ""
    assert saved.sender("") == ""


# --- labelers ----------------------------------------------------------------


def test_the_chat_labeler_uses_saved_names_and_masks_everyone_else(saved: Names) -> None:
    listed = [
        conversation(INES_PHONE, [INES_PHONE]),
        conversation(UNSAVED, [UNSAVED]),
        conversation("chat100", [INES_PHONE, TOMAS, UNSAVED_EMAIL], title="Garden Club"),
    ]
    labeler = saved.chat_labeler(listed)
    assert isinstance(labeler, MaskingLabeler)
    assert labeler.sender(INES_PHONE) == "Ines Calder"
    assert labeler.sender(TOMAS) == "Tomas Reyes"
    assert labeler.sender(UNSAVED) == "a number ending in 9999"
    assert labeler.sender(UNSAVED_EMAIL) == "an unsaved email contact"
    assert labeler.conversation(listed[0]) == "Ines Calder"  # type: ignore[arg-type]
    assert labeler.conversation(listed[1]) == "a number ending in 9999"  # type: ignore[arg-type]
    assert labeler.conversation(listed[2]) == "Garden Club"  # type: ignore[arg-type]


def test_the_chat_labeler_masks_an_unsaved_handle_inside_text(saved: Names) -> None:
    """Before the labeler named anyone: text of one conversation can hold the number of another."""
    listed = [
        conversation(UNSAVED, [UNSAVED]),
        conversation(INES_PHONE, [INES_PHONE]),
        conversation("chat100", [INES_PHONE, UNSAVED_EMAIL]),
    ]
    labeler = saved.chat_labeler(listed)
    masked = labeler.redact(f"call +1 555 017 9999 or (555) 017-0001, or write to {UNSAVED_EMAIL}")
    assert "9999" in masked and "555 017 9999" not in masked
    assert UNSAVED_EMAIL not in masked
    assert "(555) 017-0001" in masked  # a saved number is not masked


def test_the_chat_labeler_gets_the_name_that_names_gives(saved: Names) -> None:
    """One rule for names: the labeler is given each handle of the archive with its saved name."""
    odd = "+44 15550170001"  # the last ten digits are Ines's; two numbers with a country code differ for the labeler
    labeler = saved.chat_labeler([conversation(odd, [odd])])
    assert labeler.sender(odd) == saved.saved(odd) == "Ines Calder"


def test_the_report_labeler_gives_no_words_for_a_sender_who_is_not_saved(saved: Names) -> None:
    label = saved.report_labeler(conversation("chat100", [INES_PHONE, UNSAVED, UNSAVED_EMAIL]))
    assert label(INES_PHONE) == "Ines Calder"
    # an empty string, never "an unsaved contact": the engine reads one name as one person
    assert label(UNSAVED) == ""
    assert label(UNSAVED_EMAIL) == ""
    assert label("") == ""


def test_the_report_labeler_names_the_empty_sender_of_a_one_to_one_conversation(saved: Names) -> None:
    label = saved.report_labeler(conversation(INES_PHONE, [INES_PHONE]))
    assert label("") == "Ines Calder"
    assert label("   ") == "Ines Calder"
    assert label(INES_PHONE) == "Ines Calder"
    assert label(INES_EMAIL) == "Ines Calder"


def test_the_report_labeler_of_an_unsaved_person_gives_no_words(saved: Names) -> None:
    label = saved.report_labeler(conversation(UNSAVED, [UNSAVED]))
    assert label("") == ""
    assert label(UNSAVED) == ""


def test_the_report_labeler_never_raises_for_an_odd_sender(saved: Names) -> None:
    label = saved.report_labeler(conversation(INES_PHONE, [INES_PHONE]))
    assert label(None) == "Ines Calder"  # type: ignore[arg-type]
    assert label(42) == ""  # type: ignore[arg-type]
