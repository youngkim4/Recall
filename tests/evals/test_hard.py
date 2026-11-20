"""The hard world: its size, its determinism, the small world inside it, and the answers of its questions."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Iterator

import pytest

from archive import Archive
from chat_engine.archive_adapter import ArchiveAdapter
from evals.question_static import STATIC
from evals.questions import HARD_KINDS, build_questions
from evals.world import Built, build_world, world, zone
from evals.world.filler_lines import RESERVED
from evals.world.hard import hard_items, hard_world
from evals.world.hard.cast import HARD_PEOPLE, first_name
from evals.world.hard.services import service_lines
from evals.world.hard.talk import HARD_RESERVED
from evals.world.hard.traffic import ALL_PROFILES, traffic
from evals.world.script import Line

BUILD_SECONDS = 60
HARD = build_questions(hard_world())
BY_ID = {question.id: question for question in HARD}
STOP = frozenset(
    "the and did what when where who which how was were with for my me i you a an of in on at to is it its "
    "ever get go".split()
)  # fmt: skip


@pytest.fixture(scope="module")
def hard_built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Built, float]:
    started = time.monotonic()
    made = build_world(tmp_path_factory.mktemp("hard-world"), hard_world())
    return made, time.monotonic() - started


@pytest.fixture
def hard_archive(hard_built: tuple[Built, float]) -> Iterator[Archive]:
    with zone(), Archive.open(hard_built[0].archive_path) as opened:
        yield opened


def test_the_hard_world_is_big_and_long() -> None:
    made = hard_world()
    assert 40_000 <= len(made.lines) <= 60_000
    assert len(made.chats) >= 60 and sum(chat.is_group for chat in made.chats) >= 10
    assert made.lines[0].day.startswith("2020-01") and made.lines[-1].day.startswith("2025-12")
    assert len(made.tapbacks) > 500


def test_the_hard_world_is_the_same_each_time() -> None:
    assert hard_items() == hard_items()
    guids = [line.guid for line in hard_world().lines] + [tap.guid for tap in hard_world().tapbacks]
    assert len(guids) == len(set(guids))


def test_the_small_world_is_inside_unchanged() -> None:
    small = world()
    assert set(small.lines) <= set(hard_world().lines)
    for chat in small.chats:
        assert hard_world().of(chat.key) == small.of(chat.key)
    assert build_questions(small) == tuple(question for question in HARD if question.kind not in HARD_KINDS)


def test_it_builds_in_time_and_imports_every_message(hard_built: tuple[Built, float], hard_archive: Archive) -> None:
    built, seconds = hard_built
    assert seconds < BUILD_SECONDS, f"the hard world took {seconds:.0f} s to build"
    adapter = ArchiveAdapter(hard_archive)
    for chat in built.world.chats:
        assert adapter.count(conversations=[chat.identifier]) == len(built.world.of(chat.key)), chat.key


def test_about_forty_hard_questions_in_six_kinds() -> None:
    kinds = Counter(question.kind for question in HARD if question.kind in HARD_KINDS)
    assert set(kinds) == set(HARD_KINDS) and 35 <= sum(kinds.values()) <= 45


def test_every_gold_message_is_in_the_hard_archive(hard_archive: Archive) -> None:
    for question in HARD:
        for guid in question.gold_guids:
            assert hard_archive.message(guid) is not None, (question.id, guid)


@pytest.mark.parametrize("question", [q for q in HARD if q.count_query and q.kind in HARD_KINDS], ids=lambda q: q.id)
def test_hard_counts_are_the_counts_of_the_archive(question, hard_archive: Archive) -> None:
    query = question.count_query
    counted = ArchiveAdapter(hard_archive).count(
        conversations=query.conversations, from_me=query.from_me, since=query.since, until=query.until, text=query.text
    )
    assert counted == question.expected_count


def _words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.casefold()) if len(word) >= 3 and word not in STOP}


@pytest.mark.parametrize("question", [q for q in HARD if q.kind == "paraphrase"], ids=lambda q: q.id)
def test_no_word_of_a_paraphrase_question_is_in_its_gold_messages(question) -> None:
    by_guid = {line.guid: line for line in hard_world().lines}
    for guid in question.gold_guids:
        shared = _words(question.text) & _words(by_guid[guid].text)
        assert not shared, (question.id, shared)


def test_generated_traffic_holds_no_answer_and_no_absent_word() -> None:
    absent = [term for spec in STATIC if spec.kind == "absent" for term in spec.absent_terms]
    generated = [*(item for profile in ALL_PROFILES for item in traffic(profile)), *service_lines()]
    assert len(generated) > 40_000
    for item in generated:
        if not isinstance(item, Line):
            continue
        lowered = f"{item.text.casefold()} "
        assert not [word for word in (*RESERVED, *HARD_RESERVED) if word in lowered], item.text
    for line in hard_world().lines:
        for term in absent:
            pattern = re.escape(term) if not term[0].isalnum() else rf"\b{re.escape(term)}\b"
            assert not re.search(pattern, line.text, re.IGNORECASE), (term, line.text)


def test_only_the_two_mayas_share_a_first_name() -> None:
    names = Counter(first_name(person) for person in HARD_PEOPLE if person.saved_as)
    names.subtract({"Wren": 1})  # the phone and the address of one person
    assert [name for name, seen in names.items() if seen > 1] == ["Maya"]


def test_handles_are_unique_and_fictional() -> None:
    handles = [person.handle for person in HARD_PEOPLE]
    assert len(handles) == len(set(handles))
    for handle in handles:
        assert handle.endswith("@example.org") or re.fullmatch(r"\+1\d{3}55501\d{2}", handle), handle
    assert len(hard_world().unsaved_handles()) >= 15


def test_computed_answers_follow_the_world() -> None:
    assert "2023: 0" in BY_ID["story_kip"].gold
    assert BY_ID["temp_move_month"].key_facts == ("Rafferty Dunmore",)
    for qid in ("temp_move_month", "temp_2021_top"):
        most, after = (int(number) for number in re.findall(r"(?:: |with )(\d+)", BY_ID[qid].gold))
        assert most > after, qid
    assert BY_ID["agg_codes_2025"].expected_count and BY_ID["agg_codes_2025"].expected_count > 10
    assert "930 days" in BY_ID["temp_fulton_span"].gold


def test_a_written_scene_in_the_hard_world_keeps_its_label() -> None:
    made = hard_world()
    assert made.labelled("plant_name").text == "gertrude, obviously"
    assert made.labelled("plant_ask").conversation == "odalys"
    assert made.chat_id("crew") == "chat520101"


def test_the_hard_world_saves_its_contacts(hard_built: tuple[Built, float]) -> None:
    built = hard_built[0]
    assert built.world.name == "hard"
    for person in HARD_PEOPLE:
        assert built.names.saved(person.handle) == person.saved_as, person.key


def test_a_hard_run_asks_through_the_app_path(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evals.run import main
    from recall_services import chat
    from tests.evals.test_runner import Exploding, Factory, FakeJudge

    out = tmp_path / "hard.json"
    argv = ["--world", "hard", "--kinds", "needle", "--limit", "2", "--out", str(out)]
    assert main(argv, factory=Factory(), judge=FakeJudge(), say=lambda _line: None) == 0
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["meta"]["world"]["name"] == "hard" and body["meta"]["world"]["conversations"] >= 60
    assert [row["kind"] for row in body["results"]] == ["needle", "needle"]
    monkeypatch.setattr(chat, "default_llm", Exploding())
    said: list[str] = []
    assert main(["--world", "hard", "--dry-run", "--limit", "2"], factory=Exploding(), judge=Exploding(), say=said.append) == 0
    assert said[0].startswith("dry run: 2 questions")
