"""A conversation digest: the map step of a report for chat, on the cache of the reports."""

from __future__ import annotations

import dataclasses
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from report_engine.cache import PeriodCache
from report_engine.chat_digest import ConversationDigest, MomentNote, PeriodNote
from report_engine.errors import ConfigError, EmptyConversationError
from report_engine.fake_llm import FakeLLM
from report_engine.models import PeriodSummary, Usage

from .chat_digest_support import cache_rows, digest, months, tagged
from .support import LARGE, LARGE_LABELS, SMALL, run
from .synth import CONVERSATION, ListSource

SHORT_REF = re.compile(r"[mM]\d+")


def notes_of(period: PeriodSummary) -> tuple[MomentNote, ...]:
    """The moments of a report period in the shape of a digest."""
    return tuple(
        MomentNote(moment.date, moment.title, moment.detail, moment.quote, moment.message_ids)
        for moment in period.moments
    )


# --- the shape ------------------------------------------------------------------


def test_a_digest_gives_the_summary_and_the_moments_of_each_period() -> None:
    llm = FakeLLM()
    found = digest(LARGE, llm=llm)
    assert isinstance(found, ConversationDigest)
    assert found.conversation_id == CONVERSATION
    assert [period.label for period in found.periods] == LARGE_LABELS
    assert [(period.first_day, period.last_day) for period in found.periods] == [
        ("2023-01-01", "2023-01-31"),
        ("2023-02-01", "2023-02-28"),
        ("2023-03-01", "2023-03-31"),
        ("2023-04-01", "2023-04-30"),
    ]
    assert sum(period.messages for period in found.periods) == len(LARGE)
    assert all(isinstance(period, PeriodNote) and period.summary for period in found.periods)
    moments = [moment for period in found.periods for moment in period.moments]
    assert len(moments) == 4 and all(isinstance(moment, MomentNote) for moment in moments)
    assert all(len(period.moments) == 1 for period in found.periods)


def test_a_digest_runs_the_map_step_only() -> None:
    llm = FakeLLM()
    found = digest(LARGE, llm=llm)
    assert len(llm.map_calls) == 4 and llm.reduce_calls == []
    assert (found.periods_built, found.periods_cached, found.periods_failed) == (4, 0, 0)
    assert isinstance(found.usage, Usage)
    assert found.usage.calls == 4 and found.usage.input_tokens > 0 and found.usage.cost_usd > 0


def test_the_data_of_a_digest_can_not_be_changed() -> None:
    found = digest(SMALL)
    period = found.periods[0]
    for item, name in ((found, "periods"), (period, "summary"), (period.moments[0], "quote")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(item, name, None)
    assert isinstance(found.periods, tuple) and isinstance(period.moments, tuple)
    assert isinstance(period.moments[0].message_ids, tuple)


def test_a_digest_changes_none_of_its_inputs() -> None:
    messages = [dict(message) for message in SMALL]
    before = [dict(message) for message in messages]
    digest(messages)  # type: ignore[arg-type]
    assert messages == before


# --- one cache for reports and digests ---------------------------------------------


def test_a_report_first_makes_the_digest_cost_no_map_call(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    report = run(LARGE, cache=cache)
    rows = cache_rows(cache_path)
    llm = FakeLLM()
    found = digest(LARGE, llm=llm, cache=cache)
    assert llm.calls == []
    assert (found.periods_built, found.periods_cached, found.periods_failed) == (0, 4, 0)
    assert found.usage == Usage()
    assert cache_rows(cache_path) == rows
    told = report.periods
    assert [period.summary for period in found.periods] == [period.summary for period in told]
    assert [period.moments for period in found.periods] == [notes_of(period) for period in told]
    assert [period.messages for period in found.periods] == [one.message_count for one in told]


def test_a_digest_first_makes_the_map_step_of_a_report_cost_no_map_call(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    found = digest(LARGE, cache=cache)
    rows = cache_rows(cache_path)
    llm = FakeLLM()
    report = run(LARGE, llm=llm, cache=cache)
    assert llm.map_calls == [] and len(llm.reduce_calls) == 1
    assert all(period.from_cache for period in report.periods)
    assert cache_rows(cache_path) == rows
    assert {period.cache_key for period in report.periods} == {row[0] for row in rows}
    assert [notes_of(period) for period in report.periods] == [
        period.moments for period in found.periods
    ]


def test_a_digest_and_a_report_write_the_same_cache_entries(tmp_path: Path) -> None:
    """The key, the conversation, the label, the model, the prompt version and the reply."""
    of_report, of_digest = tmp_path / "report.sqlite3", tmp_path / "digest.sqlite3"
    report = run(LARGE, cache=PeriodCache(of_report))
    digest(LARGE, cache=PeriodCache(of_digest))
    assert cache_rows(of_digest) == cache_rows(of_report)
    keys = {period.cache_key for period in report.periods}
    assert {row[0] for row in cache_rows(of_digest)} == keys
    assert len(cache_rows(of_digest)) == 4


def test_a_second_digest_reads_only_what_changed(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    digest(LARGE[: 100 * 6], cache=cache)
    llm = FakeLLM()
    found = digest(LARGE, llm=llm, cache=cache)
    assert months(llm) == ["2023-04"]
    assert (found.periods_built, found.periods_cached) == (1, 3)
    assert found.usage.calls == 1


def test_a_digest_with_no_cache_reads_every_period() -> None:
    llm = FakeLLM()
    digest(LARGE, llm=llm)
    again = digest(LARGE, llm=llm)
    assert len(llm.map_calls) == 8 and again.periods_cached == 0


# --- the span ------------------------------------------------------------------------


def test_since_and_until_select_the_periods_that_overlap_the_span() -> None:
    llm = FakeLLM()
    found = digest(LARGE, llm=llm, since="2023-02-10", until="2023-03-05")
    assert [period.label for period in found.periods] == ["Feb 2023", "Mar 2023"]
    # a period is read whole: its text is the text that a report reads
    assert [(call.first_day, call.last_day) for call in llm.map_calls] == [
        ("2023-02-01", "2023-02-28"),
        ("2023-03-01", "2023-03-31"),
    ]
    assert (found.periods_built, found.periods_cached, found.periods_failed) == (2, 0, 0)


def test_the_periods_of_a_span_are_cut_from_the_whole_conversation() -> None:
    source = ListSource(LARGE)
    digest(LARGE, source=source, since="2023-02-10", until="2023-03-05")
    assert source.calls == [(CONVERSATION, None, None)]


@pytest.mark.parametrize(
    ("since", "until", "labels"),
    [
        (None, None, LARGE_LABELS),
        ("2023-03-31", None, ["Mar 2023", "Apr 2023"]),
        (None, "2023-02-01", ["Jan 2023", "Feb 2023"]),
        ("2023-02-14", "2023-02-14", ["Feb 2023"]),
        ("2023-01-31", "2023-02-01", ["Jan 2023", "Feb 2023"]),
        ("2022-01-01", "2023-01-01", ["Jan 2023"]),
        ("2023-04-30", "2030-01-01", ["Apr 2023"]),
        ("2023-02-10 08:00:00", "2023-02-10 09:00:00", ["Feb 2023"]),
    ],
)
def test_both_ends_of_a_span_are_included(
    since: str | None, until: str | None, labels: list[str]
) -> None:
    found = digest(LARGE, since=since, until=until)
    assert [period.label for period in found.periods] == labels


def test_a_span_after_a_report_costs_no_map_call(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    run(LARGE, cache=cache)
    llm = FakeLLM()
    found = digest(LARGE, llm=llm, cache=cache, since="2023-02-10", until="2023-03-05")
    assert llm.calls == []
    assert (found.periods_built, found.periods_cached) == (0, 2)


def test_a_report_after_a_span_reads_only_the_other_periods(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    digest(LARGE, cache=cache, since="2023-02-10", until="2023-03-05")
    llm = FakeLLM()
    report = run(LARGE, llm=llm, cache=cache)
    assert months(llm) == ["2023-01", "2023-04"]
    assert [period.from_cache for period in report.periods] == [False, True, True, False]


def test_a_span_that_holds_no_message_is_an_empty_conversation() -> None:
    llm = FakeLLM()
    for since, until in (("2024-01-01", None), (None, "2022-12-31"), ("2020-01-01", "2020-12-31")):
        with pytest.raises(EmptyConversationError):
            digest(LARGE, llm=llm, since=since, until=until)
    with pytest.raises(EmptyConversationError):
        digest([], llm=llm)
    assert llm.calls == []


@pytest.mark.parametrize(
    ("since", "until"),
    [("2023-03-01", "2023-02-01"), ("March", None), (None, "2023-02-30"), ("2023-2-1", None)],
)
def test_a_span_that_is_not_understood_is_refused_before_the_messages_are_read(
    since: str | None, until: str | None
) -> None:
    llm, source = FakeLLM(), ListSource(LARGE)
    with pytest.raises(ConfigError):
        digest(LARGE, llm=llm, source=source, since=since, until=until)
    assert llm.calls == [] and source.calls == []


# --- the notes of earlier periods ------------------------------------------------


def test_a_period_of_a_span_reads_the_notes_of_the_earlier_periods_that_the_cache_holds(
    cache_path: Path,
) -> None:
    cache = PeriodCache(cache_path)
    digest(LARGE, cache=cache, until="2023-01-31")
    llm = FakeLLM()
    digest(LARGE, llm=llm, cache=cache, since="2023-03-01", until="2023-04-30")
    # Feb was never read: it has no notes. Mar leaves its notes for Apr.
    assert [[label for label, _ in call.notes] for call in llm.map_calls] == [
        ["Jan 2023"],
        ["Jan 2023", "Mar 2023"],
    ]


def test_with_no_cache_a_span_starts_with_no_notes() -> None:
    llm = FakeLLM()
    digest(LARGE, llm=llm, since="2023-03-01")
    assert [[label for label, _ in call.notes] for call in llm.map_calls] == [[], ["Mar 2023"]]


def test_an_earlier_period_whose_cache_entry_does_not_parse_leaves_no_note_and_is_not_run(
    cache_path: Path,
) -> None:
    cache = PeriodCache(cache_path)
    digest(LARGE, cache=cache, until="2023-02-28")
    with closing(sqlite3.connect(cache_path)) as conn, conn:
        conn.execute("UPDATE period_results SET payload = '{}' WHERE label = 'Jan 2023'")
    llm = FakeLLM()
    digest(LARGE, llm=llm, cache=cache, since="2023-03-01", until="2023-03-31")
    assert months(llm) == ["2023-03"]
    assert [[label for label, _ in call.notes] for call in llm.map_calls] == [["Feb 2023"]]


def test_a_whole_digest_gives_each_period_the_notes_that_a_report_gives_it() -> None:
    of_report, of_digest = FakeLLM(), FakeLLM()
    run(LARGE, llm=of_report)
    digest(LARGE, llm=of_digest)
    asked = [call.text for call in of_report.map_calls]
    assert [call.text for call in of_digest.map_calls] == asked and len(asked) == 4


# --- the ids ---------------------------------------------------------------------------


def test_the_message_ids_of_a_moment_are_the_ids_of_the_archive() -> None:
    messages = tagged(LARGE)
    known = {message["id"] for message in messages}
    found = digest(messages)
    moments = [moment for period in found.periods for moment in period.moments]
    ids = [mid for moment in moments for mid in moment.message_ids]
    assert len(ids) == 4
    assert set(ids) <= known
    assert not [mid for mid in ids if SHORT_REF.fullmatch(mid)]


def test_a_moment_keeps_its_quote_and_the_message_that_holds_it() -> None:
    messages = tagged(SMALL)
    text_of = {message["id"]: message["text"] for message in messages}
    (period,) = digest(messages).periods
    (moment,) = period.moments
    assert moment.quote and moment.message_ids
    assert any(moment.quote in text_of[mid] for mid in moment.message_ids)
    assert moment.date in {message["timestamp"][:10] for message in messages}
