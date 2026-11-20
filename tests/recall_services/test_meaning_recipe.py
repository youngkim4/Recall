"""meaning.py with a recipe of segments: RECALL_SEGMENTS, the context line, overlap, and the state.

The archives are real ones with invented lines. The embedder is `FakeEmbedder`
and the model of the memory is the fake of the report engine: no request
leaves this machine.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from meaning_index import FakeEmbedder, IndexStore
from meaning_index.recipe import DEFAULT_RECIPE, Recipe
from recall_services import digests, meaning, meaning_recipe
from recall_services.meaning_notes import MemoryNotes, shortened
from tests.integration.report_data import MINA_CONTACT
from tests.recall_services import memory_kit as kit
from tests.recall_services.memory_kit import FAKE_FACTS, Lake
from tests.recall_services.report_world import MODEL, OPTIONS, Desk
from tests.report_engine.synth import MINA, RAVI, THEO, WordCounter, message

CONTEXT = Recipe(context=True)
OVERLAP = Recipe(overlap=True)
ALL = Recipe(context=True, overlap=True, adaptive=True)
GROUP = "chat700001"
QUIET_GROUP = "chat700002"
HANDLES = ("2025550143", "2025550171", "202-555", THEO)


class Keeping(FakeEmbedder):
    """The fake embedder, and each text that it was given."""

    def __init__(self) -> None:
        super().__init__(dimensions=FAKE_FACTS.dimensions, model=FAKE_FACTS.model)
        self.seen: list[str] = []

    def embed(self, texts: Sequence[str]) -> Any:
        self.seen.extend(texts)
        return super().embed(texts)


def texts_of(lake: Lake, recipe: Recipe, **more: Any) -> list[str]:
    keeping = Keeping()
    kit.build(lake, keeping, recipe=recipe, **more)
    return keeping.seen


def payload(lake: Lake, **more: Any) -> dict[str, Any]:
    return meaning.payload(lake.archive, lake.names, store=lake.store, facts=FAKE_FACTS, count_tokens=kit.words, **more)


def add_groups(lake: Lake) -> None:
    """A group with a title that holds a number nobody saved, and a group with no title."""
    lake.pulled.chat(GROUP, members=[MINA, THEO, RAVI], title=f"Lake crew {RAVI}")
    lake.pulled.chat(QUIET_GROUP, members=[MINA, RAVI])
    lake.pulled.write(GROUP, [message(0, "2024-07-01 10:00:00", "Who brings the canoe", sender=MINA)])
    lake.pulled.write(QUIET_GROUP, [message(0, "2024-08-02 11:00:00", "Rain all day", sender=RAVI)])
    lake.archive.close()
    lake.pulled.pull([MINA_CONTACT])
    lake.archive = lake.pulled.open()


def first_line(texts: Sequence[str], day: str) -> str:
    (found,) = [text.splitlines()[0] for text in texts if text.splitlines()[1] == day]
    return found


# --- RECALL_SEGMENTS ------------------------------------------------------------------------


def test_the_switch_names_the_recipe_and_no_value_is_the_default() -> None:
    assert meaning_recipe.ENV_NAME == "RECALL_SEGMENTS"
    assert meaning_recipe.from_env({}) == DEFAULT_RECIPE
    assert meaning_recipe.from_env({"RECALL_SEGMENTS": " "}) == DEFAULT_RECIPE
    assert meaning_recipe.from_env({"RECALL_SEGMENTS": "context,overlap,adaptive"}) == ALL
    assert meaning_recipe.from_env({"RECALL_SEGMENTS": "OVERLAP"}) == OVERLAP


def test_a_word_that_names_no_switch_is_left_out_with_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="recall_services.meaning_recipe"):
        found = meaning_recipe.from_env({"RECALL_SEGMENTS": "context,contxt,names"})
    assert found == CONTEXT
    (record,) = caplog.records
    assert "contxt, names" in record.getMessage() and "context, overlap, adaptive" in record.getMessage()


def test_the_recipe_of_the_process_is_read_one_time(monkeypatch: pytest.MonkeyPatch) -> None:
    assert meaning.RECIPE == meaning_recipe.from_env(os.environ) == DEFAULT_RECIPE
    monkeypatch.setenv("RECALL_SEGMENTS", "overlap")
    assert meaning.RECIPE == DEFAULT_RECIPE


def test_the_functions_build_with_the_recipe_of_the_process(lake: Lake, monkeypatch: pytest.MonkeyPatch) -> None:
    kit.build(lake)
    monkeypatch.setattr(meaning, "RECIPE", OVERLAP)
    assert payload(lake)["status"]["state"] == "behind"
    kit.build(lake)
    assert lake.store.meta().recipe == "v1+overlap"
    assert payload(lake)["status"]["state"] == "ready"


# --- the state ------------------------------------------------------------------------------


def test_a_change_of_recipe_makes_the_index_behind_until_a_build_with_it(lake: Lake) -> None:
    kit.build(lake)
    before = payload(lake)
    assert (before["status"]["state"], before["estimate"]["recipeChanged"]) == ("ready", False)
    changed = payload(lake, recipe=OVERLAP)
    # the exchanges of the lake are short: overlap carries nothing, and nothing is new
    assert (changed["status"]["state"], changed["estimate"]["recipeChanged"]) == ("behind", True)
    assert changed["estimate"]["newSegments"] == 0
    assert meaning.status(lake.archive, lake.names, store=lake.store, recipe=OVERLAP)["state"] == "behind"
    assert meaning.status(lake.archive, lake.names, store=lake.store)["state"] == "ready"
    result = kit.build(lake, recipe=OVERLAP)
    assert (result["embedded"], result["reused"]) == (0, kit.SEGMENTS)
    after = payload(lake, recipe=OVERLAP)
    assert (after["status"]["state"], after["estimate"]["recipeChanged"]) == ("ready", False)
    assert payload(lake)["status"]["state"] == "behind"


def test_with_every_switch_a_second_build_embeds_nothing(lake: Lake) -> None:
    first = kit.build(lake, recipe=ALL)
    assert first["embedded"] == kit.SEGMENTS
    assert kit.build(lake, recipe=ALL)["embedded"] == 0
    assert payload(lake, recipe=ALL)["status"]["state"] == "ready"


# --- context --------------------------------------------------------------------------------


def test_the_context_line_holds_the_month_the_kind_and_the_saved_name_or_a_mask(lake: Lake) -> None:
    add_groups(lake)
    texts = texts_of(lake, CONTEXT)
    assert first_line(texts, kit.FERRY_DAY) == "Context: March 2023; direct chat with Mina"
    assert first_line(texts, kit.LATE_FERRY_DAY) == "Context: June 2024; direct chat with an unsaved email contact"
    assert first_line(texts, "2024-07-01") == "Context: July 2024; group chat of 4 people: Lake crew a number ending in 0171"
    quiet = first_line(texts, "2024-08-02")
    assert quiet.startswith("Context: August 2024; group chat of 3 people with ")
    assert "Mina" in quiet and "a number ending in 0171" in quiet
    assert not [text for text in texts for handle in HANDLES if handle in text]


def test_the_context_line_is_embedded_but_the_hit_names_the_same_messages(lake: Lake) -> None:
    kit.build(lake)
    plain = {(s.first_message_id, s.last_message_id, s.message_count) for s in lake.store.read().segments}
    kit.build(lake, recipe=CONTEXT)
    framed = {(s.first_message_id, s.last_message_id, s.message_count) for s in lake.store.read().segments}
    assert framed == plain


def test_with_no_notes_of_the_memory_the_context_line_holds_no_memory(lake: Lake, tmp_path: Path) -> None:
    notes = MemoryNotes(places=kit_places(tmp_path), min_messages=0)
    assert not [text for text in texts_of(lake, CONTEXT, notes=notes) if "memory:" in text]
    assert not notes.places.cache_path.exists()


def kit_places(folder: Path) -> Any:
    from recall_services import reports

    return reports.ReportPlaces(out_dir=folder / "out", store_dir=folder / "reports", cache_path=folder / "notes.sqlite3")


def test_the_context_line_holds_the_start_of_the_summary_of_the_part_that_the_memory_read(
    desk: Desk, tmp_path: Path
) -> None:
    digest = digests.build(MINA, model=MODEL, **desk.more())
    assert len(digest.periods) == 4
    notes = MemoryNotes(places=desk.places, model=MODEL, options=OPTIONS, counter=WordCounter(), min_messages=0)
    store = IndexStore(tmp_path / "index")
    keeping = Keeping()
    settings = {"store": store, "facts": FAKE_FACTS, "count_tokens": kit.words, "recipe": CONTEXT, "notes": notes}
    meaning.build(desk.archive, desk.names, embedder_factory=lambda *_: keeping, **settings)
    mina = [text for text in keeping.seen if "direct chat with Mina" in text.splitlines()[0]]
    assert mina and len(mina) < len(keeping.seen)
    for text in mina:
        head, day = text.splitlines()[0], text.splitlines()[1]
        (part,) = [one for one in digest.periods if one.first_day <= day <= one.last_day]
        assert head.endswith(f"; memory: {shortened(part.summary)}")
        assert len(shortened(part.summary)) <= 200 + 3
    others = [text for text in keeping.seen if text not in mina]
    assert not [text for text in others if "memory:" in text.splitlines()[0]]
    # the same notes give the same lines: nothing is new
    found = meaning.estimate(desk.archive, desk.names, **settings)
    assert (found["newSegments"], found["recipeChanged"]) == (0, False)
    # a conversation shorter than the memory reads gives no memory
    fewer = MemoryNotes(places=desk.places, model=MODEL, options=OPTIONS, counter=WordCounter(), min_messages=10**6)
    assert meaning.estimate(desk.archive, desk.names, **{**settings, "notes": fewer})["newSegments"] == len(mina)


def test_a_summary_is_cut_at_a_space_with_a_mark() -> None:
    assert shortened("  a  short\nsummary ") == "a short summary"
    long = "word " * 100
    cut = shortened(long)
    assert cut.endswith("...") and len(cut) <= 203 and not cut[:-3].endswith(" ")
    assert shortened("x" * 300) == "x" * 200 + "..."


# --- overlap --------------------------------------------------------------------------------


def test_with_overlap_each_segment_names_a_range_of_the_archive_that_holds_its_messages(lake: Lake) -> None:
    talk = [(MINA, f"2024-09-01 08:{n:02d}:00", f"step {n} of the long talk", n % 2 == 1) for n in range(20)]
    lake.add(talk)
    kit.build(lake, recipe=OVERLAP)
    segments = lake.store.read().segments
    for segment in segments:
        ids = [row["id"] for row in lake.archive.iter_messages(segment.conversation_id)]
        first, last = ids.index(segment.first_message_id), ids.index(segment.last_message_id)
        assert last - first + 1 == segment.message_count
    long = sorted((s for s in segments if s.first_timestamp.startswith("2024-09-01")), key=lambda s: s.first_timestamp)
    assert [s.message_count for s in long] == [16, 6]
    assert long[1].first_timestamp == "2024-09-01 08:14:00"
    assert len({s.segment_id for s in segments}) == len(segments)
