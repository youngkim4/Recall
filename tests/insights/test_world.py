"""The insight world through a real archive: threads as the app makes them, and the counted insights scored."""

from __future__ import annotations

import pytest

from archive import Archive
from evals.insight_world import NOW, insight_world
from evals.insight_world.cast import NAME_OF
from evals.insights_score import score
from evals.world import build_world, zone
from insights import engine
from recall_services.insight_threads import threads


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    folder = tmp_path_factory.mktemp("insight-world")
    with zone():
        built = build_world(folder, insight_world())
        archive = Archive.open(built.archive_path)
        try:
            yield threads(archive, built.names)
        finally:
            archive.close()


def test_one_thread_for_each_saved_person_i_wrote_to(made):
    names = {thread.name for thread in made}
    assert names == {name for key, name in NAME_OF.items() if key != "fernbank"}


def test_threads_hold_no_handle(made):
    assert not any("+1707" in thread.name for thread in made)


def test_notes_are_oldest_first_with_archive_ids(made):
    world = insight_world()
    felix = next(thread for thread in made if thread.name == "Felix Arnaud")
    assert world.labelled("felix_job").guid in {note.id for note in felix.notes}


def test_the_counted_insights_find_every_want_and_spring_no_trap(made):
    with zone():
        report = engine.run(made, NOW)
    result = score(insight_world(), report.insights)
    found = {(want.kind, want.person) for want in result.found}
    assert found == {("drift", "nora"), ("balance", "hugo"), ("balance", "quinn"), ("birthday", "iris"), ("unanswered", "jonah")}
    assert result.sprung == ()
    assert result.extras == ()
