"""When a digest of a span shares the cache of the reports, and when it keeps entries of its own.

The key of a period does not hold the notes of earlier periods, but the
prompt does. A digest uses the keys of a report only when its prompts are the
prompts of a report. The fake model here writes the notes that it was sent
into its summary, so each payload shows which prompt made it.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from report_engine.cache import PeriodCache
from report_engine.chat_digest import DigestTooLarge
from report_engine.errors import StoreError
from report_engine.fake_llm import FakeLLM
from report_engine.prompts import MAP_PROMPT_VERSION

from .chat_digest_support import cache_rows, digest, echo, guess, months, notes_seen, versions
from .support import LARGE, LARGE_LABELS, run

FRESH = [
    "2023-01 saw notes=[]",
    "2023-02 saw notes=['Jan 2023']",
    "2023-03 saw notes=['Jan 2023', 'Feb 2023']",
    "2023-04 saw notes=['Jan 2023', 'Feb 2023', 'Mar 2023']",
]


def own_version(label: str) -> str:
    return f"{MAP_PROMPT_VERSION}/span-from/{label}"


def summaries(found: Any) -> list[str]:
    return [period.summary for period in found.periods]


# --- a span that a report could not have read: entries of its own -----------------------


def test_a_cold_digest_of_a_span_leaves_a_report_what_a_fresh_report_gets(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    of_digest = FakeLLM(echo)
    found = digest(LARGE, llm=of_digest, cache=cache, since="2023-04-01")
    assert notes_seen(of_digest) == [("2023-04", [])]
    assert summaries(found) == ["2023-04 saw notes=[]"]
    of_report = FakeLLM(echo)
    report = run(LARGE, llm=of_report, cache=cache)
    assert notes_seen(of_report)[-1] == ("2023-04", LARGE_LABELS[:3])
    assert months(of_report) == ["2023-01", "2023-02", "2023-03", "2023-04"]
    assert [period.from_cache for period in report.periods] == [False] * 4
    assert summaries(report) == summaries(run(LARGE, llm=FakeLLM(echo))) == FRESH


def test_a_report_never_reads_the_entries_of_a_cold_span(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    digest(LARGE, llm=FakeLLM(echo), cache=cache, since="2023-02-10", until="2023-03-05")
    llm = FakeLLM(echo)
    report = run(LARGE, llm=llm, cache=cache)
    assert len(llm.map_calls) == 4 and summaries(report) == FRESH
    # the report wrote its four entries beside the two of the span
    assert versions(cache_path) == sorted(
        [
            ("Feb 2023", own_version("Feb 2023")),
            ("Mar 2023", own_version("Feb 2023")),
            *((label, MAP_PROMPT_VERSION) for label in LARGE_LABELS),
        ]
    )


def test_a_span_gives_the_same_payload_whatever_was_read_before(tmp_path: Path) -> None:
    """The notes of an earlier period that happens to be in the cache are not sent."""
    warm, cold = tmp_path / "warm.sqlite3", tmp_path / "cold.sqlite3"
    digest(LARGE, llm=FakeLLM(echo), cache=PeriodCache(warm), until="2023-01-31")
    seen = []
    for path in (warm, cold):
        llm = FakeLLM(echo)
        found = digest(LARGE, llm=llm, cache=PeriodCache(path), since="2023-03-01")
        seen.append((notes_seen(llm), summaries(found)))
    assert seen[0] == seen[1]
    assert seen[0][0] == [("2023-03", []), ("2023-04", ["Mar 2023"])]
    own = [row for row in cache_rows(cold) if row[4] != MAP_PROMPT_VERSION]
    assert [row for row in cache_rows(warm) if row[4] != MAP_PROMPT_VERSION] == own
    assert len(own) == 2


def test_the_entries_of_a_cold_span_name_the_period_where_it_starts(tmp_path: Path) -> None:
    of_span, of_report = tmp_path / "span.sqlite3", tmp_path / "report.sqlite3"
    digest(LARGE, cache=PeriodCache(of_span), since="2023-03-01")
    run(LARGE, cache=PeriodCache(of_report))
    assert versions(of_span) == [
        ("Apr 2023", own_version("Mar 2023")),
        ("Mar 2023", own_version("Mar 2023")),
    ]
    keys_of_a_report = {row[0] for row in cache_rows(of_report)}
    assert keys_of_a_report.isdisjoint(row[0] for row in cache_rows(of_span))


def test_two_spans_with_different_starts_share_no_entry(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    from_march = digest(LARGE, llm=FakeLLM(echo), cache=cache, since="2023-03-01")
    llm = FakeLLM(echo)
    from_april = digest(LARGE, llm=llm, cache=cache, since="2023-04-01")
    assert months(llm) == ["2023-04"] and from_april.periods_cached == 0
    assert summaries(from_march)[1] == "2023-04 saw notes=['Mar 2023']"
    assert summaries(from_april) == ["2023-04 saw notes=[]"]
    assert versions(cache_path) == [
        ("Apr 2023", own_version("Apr 2023")),
        ("Apr 2023", own_version("Mar 2023")),
        ("Mar 2023", own_version("Mar 2023")),
    ]
    # the order of the questions changes nothing
    again = PeriodCache(cache_path.with_name("other-order.sqlite3"))
    first = digest(LARGE, llm=FakeLLM(echo), cache=again, since="2023-04-01")
    second = digest(LARGE, llm=FakeLLM(echo), cache=again, since="2023-03-01")
    assert (summaries(first), summaries(second)) == (summaries(from_april), summaries(from_march))


def test_a_cold_span_that_is_asked_again_costs_no_call(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    first = digest(LARGE, llm=FakeLLM(echo), cache=cache, since="2023-03-01")
    llm = FakeLLM(echo)
    again = digest(LARGE, llm=llm, cache=cache, since="2023-03-10")
    shorter = digest(LARGE, llm=llm, cache=cache, since="2023-03-01", until="2023-03-31")
    assert llm.calls == []
    assert (again.periods_built, again.periods_cached) == (0, 2)
    assert summaries(again) == summaries(first) and summaries(shorter) == summaries(first)[:1]


# --- a span that a report could have read: the entries of the reports -----------------


def test_a_span_after_a_report_reads_the_entries_of_the_report(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    run(LARGE, llm=FakeLLM(echo), cache=cache)
    rows = cache_rows(cache_path)
    llm = FakeLLM(echo)
    found = digest(LARGE, llm=llm, cache=cache, since="2023-03-01")
    assert llm.calls == [] and summaries(found) == FRESH[2:]
    assert cache_rows(cache_path) == rows


def test_a_span_whose_earlier_periods_are_all_in_the_cache_writes_what_a_report_writes(
    tmp_path: Path,
) -> None:
    """Then the notes of all of them are sent, as a report sends them."""
    path, of_report = tmp_path / "digests.sqlite3", tmp_path / "report.sqlite3"
    cache = PeriodCache(path)
    digest(LARGE, llm=FakeLLM(echo), cache=cache, until="2023-02-28")
    llm = FakeLLM(echo)
    found = digest(LARGE, llm=llm, cache=cache, since="2023-03-01")
    assert notes_seen(llm) == [("2023-03", LARGE_LABELS[:2]), ("2023-04", LARGE_LABELS[:3])]
    assert summaries(found) == FRESH[2:]
    fresh = run(LARGE, llm=FakeLLM(echo), cache=PeriodCache(of_report))
    assert cache_rows(path) == cache_rows(of_report)
    after = FakeLLM(echo)
    report = run(LARGE, llm=after, cache=cache)
    assert after.map_calls == [] and summaries(report) == summaries(fresh) == FRESH


def test_a_span_from_the_first_period_is_read_as_a_report_reads_it(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    digest(LARGE, llm=FakeLLM(echo), cache=cache, since="2022-01-01", until="2023-02-10")
    assert versions(cache_path) == [
        ("Feb 2023", MAP_PROMPT_VERSION),
        ("Jan 2023", MAP_PROMPT_VERSION),
    ]
    llm = FakeLLM(echo)
    report = run(LARGE, llm=llm, cache=cache)
    assert months(llm) == ["2023-03", "2023-04"] and summaries(report) == FRESH


def test_an_earlier_period_that_failed_makes_the_span_keep_entries_of_its_own(
    cache_path: Path,
) -> None:
    """An entry that does not parse is no entry: a report would run that period first."""
    cache = PeriodCache(cache_path)
    digest(LARGE, llm=FakeLLM(echo), cache=cache, until="2023-02-28")
    with closing(sqlite3.connect(cache_path)) as conn, conn:
        conn.execute("UPDATE period_results SET payload = '{}' WHERE label = 'Jan 2023'")
    llm = FakeLLM(echo)
    found = digest(LARGE, llm=llm, cache=cache, since="2023-03-01", until="2023-03-31")
    assert notes_seen(llm) == [("2023-03", [])]
    assert summaries(found) == ["2023-03 saw notes=[]"]
    assert ("Mar 2023", own_version("Mar 2023")) in versions(cache_path)


class NoReads:
    """A cache that can not be read. What is written to it is kept."""

    def __init__(self) -> None:
        self.put_versions: list[str] = []

    def get_many(self, keys: Any, touch: bool = False) -> Any:
        raise StoreError("the disk is gone")

    def put(self, entry: Any, *, prompt_version: str, **_: str) -> None:
        self.put_versions.append(prompt_version)


def test_a_cache_that_can_not_be_read_holds_no_earlier_period() -> None:
    cache = NoReads()
    llm = FakeLLM(echo)
    digest(LARGE, llm=llm, cache=cache, since="2023-03-01")
    assert notes_seen(llm) == [("2023-03", []), ("2023-04", ["Mar 2023"])]
    assert cache.put_versions == [own_version("Mar 2023")] * 2


def test_with_no_cache_a_span_sends_no_notes_from_before_it() -> None:
    llm = FakeLLM(echo)
    digest(LARGE, llm=llm, since="2023-03-01")
    assert notes_seen(llm) == [("2023-03", []), ("2023-04", ["Mar 2023"])]


# --- the estimate and the limit count with the keys of the build ----------------------


def nothing(_cache: PeriodCache) -> None:
    return None


def a_report(cache: PeriodCache) -> None:
    run(LARGE, llm=FakeLLM(echo), cache=cache)


def the_first_two_periods(cache: PeriodCache) -> None:
    digest(LARGE, llm=FakeLLM(echo), cache=cache, until="2023-02-28")


def the_same_span(cache: PeriodCache) -> None:
    digest(LARGE, llm=FakeLLM(echo), cache=cache, since="2023-03-01")


def a_later_span(cache: PeriodCache) -> None:
    digest(LARGE, llm=FakeLLM(echo), cache=cache, since="2023-04-01")


def january_only(cache: PeriodCache) -> None:
    digest(LARGE, llm=FakeLLM(echo), cache=cache, until="2023-01-31")


@pytest.mark.parametrize(
    ("before", "calls"),
    [
        (nothing, 2),
        (a_report, 0),
        (the_first_two_periods, 2),
        (the_same_span, 0),
        (a_later_span, 2),
        (january_only, 2),
    ],
)
def test_the_estimate_of_a_span_agrees_with_its_build(
    cache_path: Path, before: Any, calls: int
) -> None:
    cache = PeriodCache(cache_path)
    before(cache)
    planned = guess(LARGE, cache, since="2023-03-01")
    llm = FakeLLM(echo)
    built = digest(LARGE, llm=llm, cache=cache, since="2023-03-01")
    assert (planned.periods, planned.periods - planned.periods_cached) == (2, calls)
    assert (built.periods_built, built.periods_cached) == (calls, 2 - calls)
    assert len(llm.calls) == calls
    after = guess(LARGE, cache, since="2023-03-01")
    assert (after.periods_cached, after.input_tokens, after.usd) == (2, 0, 0.0)


def test_the_estimate_of_a_cold_span_counts_no_notes_from_before_it(cache_path: Path) -> None:
    """A span that reads with the notes of the earlier periods sends more than a cold one."""
    cold = guess(LARGE, since="2023-03-01")
    cache = PeriodCache(cache_path)
    the_first_two_periods(cache)
    warm = guess(LARGE, cache, since="2023-03-01")
    assert (cold.periods, warm.periods) == (2, 2)
    assert cold.input_tokens < warm.input_tokens
    built = digest(LARGE, since="2023-03-01")
    assert built.usage.input_tokens <= cold.input_tokens <= built.usage.input_tokens * 1.4


def test_the_limit_counts_with_the_keys_of_the_build(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    the_same_span(cache)
    llm = FakeLLM(echo)
    never_pay = {"llm": llm, "cache": cache, "max_new_periods": 0}
    assert digest(LARGE, since="2023-03-01", **never_pay).periods_cached == 2
    with pytest.raises(DigestTooLarge) as caught:
        digest(LARGE, since="2023-04-01", **never_pay)
    assert (caught.value.periods, caught.value.periods_cached) == (1, 0)
    with pytest.raises(DigestTooLarge) as whole:
        digest(LARGE, llm=llm, cache=cache, max_new_periods=3)
    assert (whole.value.periods, whole.value.periods_cached) == (4, 0)
    assert llm.calls == []
    a_report(cache)
    assert digest(LARGE, since="2023-04-01", **never_pay).periods_cached == 1
    assert llm.calls == []
