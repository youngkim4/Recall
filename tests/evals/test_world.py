"""The eval world: deterministic, imported whole, invented, and with no fact by chance in the filler."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from archive import Archive
from evals.question_static import STATIC
from evals.world import Built, WorldError, build_world, world
from evals.world.build import scenes
from evals.world.filler import filler
from evals.world.filler_lines import RESERVED
from evals.world.people import CHATS, CONTACTS, FAMILY_TITLE, PEOPLE, conversation_id, unsaved_handles
from evals.world.script import Line, ScriptError, scene


def _archive_rows(path: Path) -> list[tuple[str, str, str, str]]:
    with Archive.open(path) as opened:
        ids = [chat.identifier for chat in CHATS]
        return [
            (str(row["id"]), str(row["conversation"]), str(row["timestamp"]), str(row["text"]))
            for row in opened.iter_messages(ids)
        ]


def test_the_world_is_the_same_on_each_build(built: Built, tmp_path: Path) -> None:
    again = build_world(tmp_path)
    assert again.world == built.world
    assert _archive_rows(again.archive_path) == _archive_rows(built.archive_path)


def test_a_folder_that_holds_a_world_is_refused(built: Built) -> None:
    with pytest.raises(WorldError):
        build_world(built.archive_path.parent.parent)


def test_every_message_is_imported_with_its_guid(built: Built, archive: Archive) -> None:
    rows = _archive_rows(built.archive_path)
    assert {row[0] for row in rows} == {line.guid for line in built.world.lines}
    assert len(rows) == len(built.world.lines)


def test_guids_and_labels_are_unique() -> None:
    made = world()
    guids = [line.guid for line in made.lines] + [tap.guid for tap in made.tapbacks]
    assert len(guids) == len(set(guids))
    labels = [line.label for line in made.lines if line.label]
    assert not [label for label, seen in Counter(labels).items() if seen > 1]


def test_the_world_spans_three_years_with_a_few_thousand_messages() -> None:
    made = world()
    assert 1000 <= len(made.lines) <= 4000
    assert made.lines[0].day.startswith("2023-01") and made.lines[-1].day.startswith("2025-12")
    assert len(made.tapbacks) >= 5


def test_each_conversation_has_its_id_and_the_group_its_title(archive: Archive) -> None:
    held = {conversation["id"]: conversation for conversation in archive.conversations()}
    assert {conversation_id(chat.key) for chat in CHATS} <= set(held)
    assert held[conversation_id("family")]["is_group"]
    assert held[conversation_id("family")]["name"] == FAMILY_TITLE


def test_saved_and_unsaved_handles(built: Built) -> None:
    for person in PEOPLE:
        assert built.names.saved(person.handle) == person.saved_as
    assert len(unsaved_handles()) == 3
    assert len(CONTACTS) == 7


def test_numbers_are_in_the_fictional_range_and_addresses_at_example_org() -> None:
    for person in PEOPLE:
        if person.handle.startswith("+"):
            assert re.fullmatch(r"\+1\d{3}55501\d{2}", person.handle), person.handle
        else:
            assert person.handle.endswith("@example.org")


def test_no_filler_line_holds_a_reserved_word() -> None:
    made = filler(scenes(), {})
    assert len(made) > 1000
    for line in made:
        lowered = line.text.casefold()
        assert not [word for word in RESERVED if word in lowered], line.text


@pytest.mark.parametrize("spec", [spec for spec in STATIC if spec.kind == "absent"], ids=lambda spec: spec.id)
def test_no_message_holds_the_words_of_an_absent_question(spec) -> None:
    for line in world().lines:
        for term in spec.absent_terms:
            pattern = re.escape(term) if not term[0].isalnum() else rf"\b{re.escape(term)}\b"
            assert not re.search(pattern, line.text, re.IGNORECASE), (spec.id, line.text)


def test_a_scene_reads_clock_moves_messages_and_tapbacks() -> None:
    items = scene("wren", "t", "@2024-01-01 10:00\nwren: hi  #hello\n+5\nme: hey\nwren ^love\n")
    lines = [item for item in items if isinstance(item, Line)]
    assert [line.text for line in lines] == ["hi", "hey"]
    assert lines[0].label == "hello" and lines[1].from_me
    assert (lines[1].when - lines[0].when).total_seconds() == 300
    assert items[-1].target == lines[1].guid  # type: ignore[union-attr]


@pytest.mark.parametrize("script", ["wren: no clock", "@2024-01-01 10:00\nthis is not a line", "@2024-01-01 10:00\nme ^hug"])
def test_a_bad_scene_is_refused(script: str) -> None:
    with pytest.raises(ScriptError):
        scene("wren", "bad", script)
