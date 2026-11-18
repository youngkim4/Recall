"""digests.build with dates: when it shares the notes of the reports and when it keeps notes of its own."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from recall_services import digests, reports
from report_engine.fake_llm import FakeLLM
from tests.recall_services.report_world import MODEL, PERIOD_LABELS, Desk
from tests.report_engine.chat_digest_support import echo, notes_seen
from tests.report_engine.synth import MINA


def estimate_more(desk: Desk) -> dict[str, object]:
    return {"model": MODEL, **desk.estimate_more()}


def test_dates_that_start_after_parts_nobody_read_keep_notes_of_their_own(desk: Desk, tmp_path: Path) -> None:
    """The model reads such a part with no notes of the parts before it. A report must not take that for its own."""
    model = FakeLLM(echo)
    found = digests.build(MINA, "2023-04-01", model=MODEL, **desk.more(model))
    assert notes_seen(model) == [("2023-04", [])]
    assert [period.summary for period in found.periods] == ["2023-04 saw notes=[]"]
    after = FakeLLM(echo)
    done = reports.run(MINA, MODEL, **desk.more(after))
    assert notes_seen(after)[-1] == ("2023-04", PERIOD_LABELS[:3])
    other = dataclasses.replace(desk.places, cache_path=tmp_path / "fresh" / "cache.sqlite3")
    fresh = reports.run(MINA, MODEL, **{**desk.more(FakeLLM(echo)), "places": other})
    assert [period.summary for period in done.report.periods] == [period.summary for period in fresh.report.periods]
    # asked again, the dates cost nothing: now every part before them is read, and the report read April
    again = FakeLLM(echo)
    shared = digests.build(MINA, "2023-04-01", model=MODEL, **desk.more(again))
    assert again.calls == [] and shared.periods[0].summary == done.report.periods[3].summary


def test_the_same_dates_asked_twice_cost_one_reading(desk: Desk) -> None:
    first = digests.build(MINA, "2023-03-01", model=MODEL, **desk.more(FakeLLM(echo)))
    assert digests.estimate(MINA, "2023-03-01", **estimate_more(desk)).periods_cached == 2
    assert reports.estimate(MINA, MODEL, **desk.estimate_more())["periods_cached"] == 0
    model = FakeLLM(echo)
    again = digests.build(MINA, "2023-03-01", model=MODEL, **desk.more(model))
    assert model.calls == [] and again.periods == first.periods
    assert (again.periods_built, again.periods_cached) == (0, 2)
    # dates that start at another part share nothing with them
    assert digests.estimate(MINA, "2023-04-01", **estimate_more(desk)).periods_cached == 0


def test_dates_after_parts_that_a_report_read_share_the_notes_of_the_reports(desk: Desk) -> None:
    reports.run(MINA, MODEL, "2023-01-01", "2023-02-28", **desk.more(FakeLLM(echo)))
    model = FakeLLM(echo)
    found = digests.build(MINA, "2023-03-01", model=MODEL, **desk.more(model))
    assert notes_seen(model) == [("2023-03", PERIOD_LABELS[:2]), ("2023-04", PERIOD_LABELS[:3])]
    assert found.periods_built == 2
    after = FakeLLM(echo)
    reports.run(MINA, MODEL, **desk.more(after))
    assert after.map_calls == [] and len(after.reduce_calls) == 1
