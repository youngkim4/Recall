"""The run from end to end with a fake model: shape, reuse, references, usage."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from report_engine.cache import PeriodCache
from report_engine.errors import EmptyConversationError
from report_engine.fake_llm import FakeCall, FakeLLM, completed, default_reply
from report_engine.models import SECTION_KEYS, SECTION_TITLES, Scope
from report_engine.ports import LLMUsage
from report_engine.pricing import call_cost_usd
from report_engine.transcript import squash

from .support import LARGE, LARGE_LABELS, SMALL, SMALL_OPTIONS, run
from .synth import CONVERSATION, ListSource, conversation, message, model_info


def one_moment(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "date": "2024-06-01",
        "title": "A thing happened",
        "detail": "It did.",
        "category": "milestone",
        "score": 0.9,
        "quote": "",
        "refs": ["m1"],
    }
    values.update(overrides)
    return values


def map_reply(*moments: dict[str, Any]) -> dict[str, Any]:
    return {"summary": "A summary.", "carry_over": "Some notes.", "moments": list(moments)}


def test_a_small_and_a_large_conversation_give_the_same_four_sections() -> None:
    small, large = run(SMALL), run(LARGE)
    assert len(small.periods) == 1
    assert [period.label for period in large.periods] == LARGE_LABELS
    for report in (small, large):
        assert [section.key for section in report.sections] == list(SECTION_KEYS)
        assert [section.title for section in report.sections] == [SECTION_TITLES[key] for key in SECTION_KEYS]
        assert all(section.markdown.strip() for section in report.sections)
        assert report.status == "complete"
        assert report.warnings == ()


def test_the_model_is_called_once_per_period_oldest_first_and_then_once_to_reduce() -> None:
    llm = FakeLLM()
    run(LARGE, llm=llm)
    assert [call.kind for call in llm.calls] == ["map", "map", "map", "map", "reduce"]
    assert [call.first_day[:7] for call in llm.map_calls] == ["2023-01", "2023-02", "2023-03", "2023-04"]


def test_a_second_run_with_no_new_messages_makes_no_period_call(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    first = run(LARGE, llm=FakeLLM(), cache=cache)
    llm = FakeLLM()
    second = run(LARGE, llm=llm, cache=cache)
    assert llm.map_calls == []
    assert len(llm.reduce_calls) == 1
    assert [period.from_cache for period in first.periods] == [False] * 4
    assert [period.from_cache for period in second.periods] == [True] * 4
    assert [period.summary for period in second.periods] == [period.summary for period in first.periods]
    assert second.moments == first.moments


def test_a_run_after_new_messages_makes_one_period_call_and_the_reduce_call(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    run(LARGE, llm=FakeLLM(), cache=cache)
    grown = [*LARGE, *conversation("2023-05-01", days=2, per_day=3, first_id=50_000)]
    llm = FakeLLM()
    report = run(grown, llm=llm, cache=cache)
    assert len(llm.map_calls) == 1
    assert len(llm.reduce_calls) == 1
    assert len(llm.calls) == 2
    assert (llm.map_calls[0].first_day, llm.map_calls[0].last_day) == ("2023-04-01", "2023-05-02")
    assert [period.from_cache for period in report.periods] == [True, True, True, False]
    assert report.periods[-1].label == "Apr 2023 – May 2023"


def test_new_messages_that_start_a_new_period_leave_the_old_ones_in_the_cache(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    run(LARGE, llm=FakeLLM(), cache=cache)
    grown = [*LARGE, *conversation("2023-05-01", days=31, per_day=6, words=8, first_id=50_000)]
    llm = FakeLLM()
    report = run(grown, llm=llm, cache=cache)
    assert [call.first_day[:7] for call in llm.map_calls] == ["2023-05"]
    assert [period.from_cache for period in report.periods] == [True, True, True, True, False]


def test_notes_from_cached_periods_still_reach_the_period_that_runs(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    run(LARGE, llm=FakeLLM(), cache=cache)
    grown = [*LARGE, *conversation("2023-05-01", days=2, per_day=3, first_id=50_000)]
    llm = FakeLLM()
    run(grown, llm=llm, cache=cache)
    assert [label for label, _ in llm.map_calls[0].notes] == LARGE_LABELS[:3]


def test_another_model_does_not_reuse_the_cache(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    run(LARGE, llm=FakeLLM(), cache=cache)
    llm = FakeLLM()
    run(LARGE, llm=llm, cache=cache, info=model_info(id="test-model-2"))
    assert len(llm.map_calls) == 4


def test_a_cache_entry_that_no_longer_parses_is_run_again(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    first = run(SMALL, llm=FakeLLM(), cache=cache)
    entry = cache.get(first.periods[0].cache_key)
    assert entry is not None
    cache.put(
        replace(entry, payload={"moments": []}),
        conversation=CONVERSATION,
        label="Jun 2024",
        model="test-model-1",
        prompt_version="v",
    )
    llm = FakeLLM()
    second = run(SMALL, llm=llm, cache=cache)
    assert len(llm.map_calls) == 1
    assert second.periods[0].from_cache is False


def test_a_run_needs_no_cache() -> None:
    llm = FakeLLM()
    report = run(SMALL, llm=llm, cache=None)
    assert report.status == "complete"
    assert len(llm.calls) == 2


def test_references_map_back_to_the_right_message_ids() -> None:
    messages = [
        message("alpha", "2024-06-01 09:00:00", "first words here"),
        message("beta", "2024-06-01 09:01:00", "the second line is longer", is_from_me=True),
        message("gamma", "2024-06-02 10:00:00", "third"),
    ]

    def responder(call: FakeCall) -> Any:
        if call.kind != "map":
            return None
        return map_reply(one_moment(date="2024-06-02", quote="third", refs=["m3", "m1"]))

    report = run(messages, llm=FakeLLM(responder))
    (moment,) = report.moments
    assert moment.message_ids == ("gamma", "alpha")
    assert (moment.quote, moment.quote_verified) == ("third", True)


def test_references_restart_in_every_period_and_still_map_back() -> None:
    llm = FakeLLM()
    report = run(LARGE, llm=llm)
    assert all(call.lines[0].ref == "m1" for call in llm.map_calls)
    by_id = {item["id"]: item for item in LARGE}
    assert len(report.moments) == 4
    for period in report.periods:
        (moment,) = period.moments
        source = by_id[moment.message_ids[0]]
        assert squash(source["text"]) == moment.quote
        assert moment.quote_verified is True
        assert period.first_day <= source["timestamp"][:10] <= period.last_day
        assert moment.date == source["timestamp"][:10]


def test_a_quote_that_is_not_in_the_messages_is_cleared_and_flagged() -> None:
    def responder(call: FakeCall) -> Any:
        if call.kind != "map":
            return None
        return map_reply(one_moment(quote="words that nobody wrote in this chat"))

    llm = FakeLLM(responder)
    report = run(SMALL, llm=llm)
    (moment,) = report.moments
    assert (moment.quote, moment.quote_verified) == ("", False)
    assert moment.message_ids == (SMALL[0]["id"],)
    assert report.warnings == ("1 quote was not found in the messages and was removed.",)
    assert "nobody wrote" not in llm.reduce_calls[0].text
    assert report.status == "complete"


def test_a_wrong_date_and_an_unusable_moment_are_reported() -> None:
    def responder(call: FakeCall) -> Any:
        if call.kind != "map":
            return None
        return map_reply(
            one_moment(date="1999-01-01", title="Wrong day"),
            one_moment(date="1999-01-01", title="No anchor", refs=[]),
        )

    report = run(SMALL, llm=FakeLLM(responder))
    assert [moment.title for moment in report.moments] == ["Wrong day"]
    assert report.moments[0].date == "2024-06-01"
    assert report.warnings == (
        "1 moment had a date outside its period; the date was set from the cited messages.",
        "1 moment from the model could not be used and was dropped.",
    )


def test_each_period_gets_the_notes_of_the_periods_before_it() -> None:
    llm = FakeLLM()
    run(LARGE, llm=llm)
    notes = [call.notes for call in llm.map_calls]
    assert notes[0] == ()
    assert notes[1] == (("Jan 2023", "Fake notes: the talk ran until 2023-01-31."),)
    assert [label for label, _ in notes[3]] == ["Jan 2023", "Feb 2023", "Mar 2023"]


def test_notes_over_their_budget_drop_the_oldest_first() -> None:
    llm = FakeLLM()
    run(LARGE, llm=llm, options=replace(SMALL_OPTIONS, carry_over_token_budget=12))
    assert [label for label, _ in llm.map_calls[3].notes] == ["Mar 2023"]
    silent = FakeLLM()
    run(LARGE, llm=silent, options=replace(SMALL_OPTIONS, carry_over_token_budget=0))
    assert all(call.notes == () for call in silent.map_calls)


def test_usage_and_cost_add_up() -> None:
    results = []

    def responder(call: FakeCall) -> Any:
        usage = LLMUsage(input_tokens=1_000 * (len(results) + 1), cached_input_tokens=100, output_tokens=50)
        results.append(completed(call, default_reply(call), usage=usage))
        return results[-1]

    info = model_info()
    report = run(LARGE, llm=FakeLLM(responder), info=info)
    costs = [call_cost_usd(info, result.usage) for result in results]
    assert report.usage.calls == 5
    assert report.usage.input_tokens == 15_000
    assert report.usage.cached_input_tokens == 500
    assert report.usage.output_tokens == 250
    assert report.usage.cost_usd == pytest.approx(sum(costs))
    assert [period.usage.input_tokens for period in report.periods] == [1_000, 2_000, 3_000, 4_000]
    assert [period.usage.cost_usd for period in report.periods] == pytest.approx(costs[:4])
    assert all(period.usage.calls == 1 for period in report.periods)


def test_a_cached_period_costs_nothing_now_but_keeps_what_it_cost(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    first = run(LARGE, llm=FakeLLM(), cache=cache)
    second = run(LARGE, llm=FakeLLM(), cache=cache)
    assert second.usage.calls == 1
    assert second.usage.input_tokens < first.periods[0].usage.input_tokens
    assert [period.usage for period in second.periods] == [period.usage for period in first.periods]
    assert second.periods[0].usage.cost_usd > 0


def test_the_report_records_who_what_and_when() -> None:
    report = run(SMALL, stats={"totalMessages": 40, "nested": {"ratio": 0.5}})
    assert report.id == "20240701T101500Z-000001"
    assert report.created_at == "2024-07-01T10:15:00+00:00"
    assert report.conversation == CONVERSATION
    assert report.model == "test-model-1"
    assert report.scope == Scope()
    assert report.stats == {"totalMessages": 40, "nested": {"ratio": 0.5}}
    (period,) = report.periods
    assert (period.label, period.first_day, period.last_day) == ("Jun 2024", "2024-06-01", "2024-06-10")
    assert period.message_count == 40
    assert len(period.cache_key) == 64
    assert period.carry_over == "Fake notes: the talk ran until 2024-06-10."


def test_the_scope_goes_to_the_source_and_into_the_report() -> None:
    source = ListSource(SMALL)
    llm = FakeLLM()
    report = run(SMALL, llm=llm, source=source, scope={"since": "2024-06-03", "until": "2024-06-05"})
    assert source.calls == [(CONVERSATION, "2024-06-03", "2024-06-05")]
    assert report.scope == Scope(since="2024-06-03", until="2024-06-05")
    assert (llm.map_calls[0].first_day, llm.map_calls[0].last_day) == ("2024-06-03", "2024-06-05")
    assert report.periods[0].message_count == 12


def test_no_messages_in_scope_is_an_error_before_any_call() -> None:
    llm = FakeLLM()
    with pytest.raises(EmptyConversationError):
        run(SMALL, llm=llm, scope=Scope(since="2030-01-01"))
    assert llm.calls == []


def test_output_limits_and_reasoning_reach_the_model() -> None:
    llm = FakeLLM()
    run(SMALL, llm=llm, options=replace(SMALL_OPTIONS, reasoning="low"))
    assert [(call.max_output_tokens, call.reasoning) for call in llm.calls] == [(500, "low"), (800, "low")]
    capped = FakeLLM()
    run(SMALL, llm=capped, info=model_info(max_output_tokens=300))
    assert [call.max_output_tokens for call in capped.calls] == [300, 300]


def test_the_reduce_call_sees_figures_summaries_and_moments_but_no_transcript() -> None:
    llm = FakeLLM()
    report = run(LARGE, llm=llm)
    text = llm.reduce_calls[0].text
    assert "Messages with text: 720" in text
    for period in report.periods:
        assert period.label in text
        assert period.summary in text
    for moment in report.moments:
        assert moment.title in text
        assert f'Quote: "{moment.quote}"' in text
    assert llm.reduce_calls[0].transcript == ""


def test_the_floor_and_the_cap_shape_the_report_moments() -> None:
    def responder(call: FakeCall) -> Any:
        if call.kind != "map":
            return None
        day = call.first_day
        return map_reply(
            one_moment(date=day, title="Strong harbor night", score=0.9, refs=["m1"]),
            one_moment(date=day, title="Weak kettle note", score=0.3, refs=["m2"]),
        )

    report = run(LARGE, llm=FakeLLM(responder))
    assert [moment.title for moment in report.moments] == ["Strong harbor night"] * 4
    assert all(len(period.moments) == 2 for period in report.periods)
    capped = run(LARGE, llm=FakeLLM(responder), options=replace(SMALL_OPTIONS, max_moments=2))
    assert len(capped.moments) == 2
