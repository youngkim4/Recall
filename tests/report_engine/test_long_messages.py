"""Very long messages. One row must not cost more than the rest of a conversation,
and no period may be larger than its budget because of one row."""

import random
import time
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import pytest

from report_engine import prompts
from report_engine.digest import MAX_QUOTE_CHARS
from report_engine.estimate import estimate
from report_engine.fake_llm import FakeCall, FakeLLM, default_reply
from report_engine.options import ReportOptions
from report_engine.pipeline import run_report
from report_engine.plan import Plan, build_plan
from report_engine.ports import SourceMessage
from report_engine.transcript import (
    ELISION_MARK,
    MAX_MESSAGE_CHARS,
    build_lines,
    cut_start,
    render_period,
)

from .support import SMALL, SMALL_OPTIONS, run
from .synth import CONVERSATION, ListSource, labeler, message, model_info

MARK = f" {ELISION_MARK}"
# three megabytes with no space: text that nobody typed
GIANT = "QUJD" * 750_000


class FourCharsAToken:
    """A counter that follows the size of a text, as a real tokenizer does for
    text with no spaces."""

    family = "words"

    def count_many(self, texts: Any) -> list[int]:
        return [-(-len(text) // 4) for text in texts]


class OneCharAToken:
    """The densest text there is: every character is a token."""

    family = "words"

    def count_many(self, texts: Any) -> list[int]:
        return [len(text) for text in texts]


def plan_of(
    messages: list[SourceMessage], counter: Any, options: ReportOptions = SMALL_OPTIONS
) -> Plan:
    return build_plan(
        source=ListSource(messages),
        conversation=CONVERSATION,
        scope=None,
        labeler=labeler,
        model_info=model_info(),
        options=options,
        counter=counter,
    )


def test_a_message_over_the_cap_keeps_its_start_and_gets_a_mark() -> None:
    text = "start " + "word " * 3_000 + "the end"
    (line,) = build_lines([message(1, "2024-06-01 09:00:00", text)], labeler)
    assert line.cut
    assert line.flat.startswith("start word word")
    assert line.flat.endswith(MARK)
    assert len(line.flat) <= MAX_MESSAGE_CHARS + len(MARK)
    assert "the end" not in line.flat
    assert line.text == text
    assert render_period([line]).text.endswith(MARK)


def test_a_message_at_the_cap_is_whole() -> None:
    text = "a" * MAX_MESSAGE_CHARS
    (line,) = build_lines([message(1, "2024-06-01 09:00:00", text)], labeler)
    assert not line.cut
    assert line.flat == text


def test_the_cap_is_the_callers_choice_and_has_a_default() -> None:
    assert ReportOptions().max_message_chars == MAX_MESSAGE_CHARS == 4_000
    (line,) = build_lines([message(1, "2024-06-01 09:00:00", "ab " * 100)], labeler, 50)
    assert line.cut
    assert line.flat == ("ab " * 100)[:50].rstrip() + MARK
    plan = plan_of(
        [message(1, "2024-06-01 09:00:00", "ab " * 100)],
        FourCharsAToken(),
        replace(SMALL_OPTIONS, max_message_chars=50),
    )
    assert plan.cut_messages == 1


def test_one_giant_row_does_not_change_the_periods_and_costs_no_more_than_the_cap() -> None:
    counter = FourCharsAToken()
    with_giant = [*SMALL, message(999, "2024-06-05 12:00:00", GIANT)]
    started = time.perf_counter()
    plain, giant = plan_of(SMALL, counter), plan_of(with_giant, counter)
    assert time.perf_counter() - started < 5.0
    assert len(giant.periods) == len(plain.periods) == 1
    extra = giant.periods[0].period.tokens - plain.periods[0].period.tokens
    assert 0 < extra <= MAX_MESSAGE_CHARS // 4 + 20
    assert giant.cut_messages == 1
    assert plain.cut_messages == 0


def test_the_estimate_is_not_thrown_by_a_giant_row() -> None:
    def tokens(messages: list[SourceMessage]) -> int:
        result = estimate(
            ListSource(messages),
            CONVERSATION,
            None,
            model_info(),
            None,
            labeler=labeler,
            options=SMALL_OPTIONS,
            counter=FourCharsAToken(),
        )
        return int(result["input_tokens"])

    with_giant = [*SMALL, message(999, "2024-06-05 12:00:00", GIANT)]
    assert 0 < tokens(with_giant) - tokens(SMALL) <= MAX_MESSAGE_CHARS // 4 + 20


@pytest.mark.parametrize(
    "count, warning",
    [
        (1, "1 long message was cut to fit: the model read only its start."),
        (2, "2 long messages were cut to fit: the model read only the start of each."),
    ],
)
def test_a_run_says_how_many_messages_were_cut(count: int, warning: str) -> None:
    long_ones = [
        message(900 + n, f"2024-06-05 12:0{n}:00", "letter " * 1_000) for n in range(count)
    ]
    report = run([*SMALL, *long_ones])
    assert warning in report.warnings
    assert not any("cut to fit" in item for item in run(SMALL).warnings)


def test_a_quote_is_checked_against_the_whole_stored_text() -> None:
    """The model reads the start of a long message. The check reads all of it:
    a quote from the part that was cut is still found."""
    ending = "these are the words at the very end of a long letter"
    letter = "the letter starts here " + "filler " * 800 + ending

    def responder(call: FakeCall) -> Any:
        if call.kind != "map":
            return None
        (line,) = [item for item in call.lines if item.text.startswith("the letter starts here")]
        assert line.text.endswith(ELISION_MARK)
        assert ending not in call.transcript
        moment = {
            "date": line.day,
            "title": "A long letter",
            "detail": "One of you wrote a long letter.",
            "category": "milestone",
            "score": 0.9,
            "quote": ending,
            "refs": [line.ref],
        }
        return {**default_reply(call), "moments": [moment]}

    report = run([message(7, "2024-06-01 09:00:00", letter)], llm=FakeLLM(responder))
    (moment,) = report.moments
    assert moment.quote == ending
    assert moment.quote_verified
    assert moment.quote_message_id == "7"


def test_a_line_that_is_larger_than_a_period_is_cut_until_it_fits() -> None:
    """The cap counts characters. Text can hold a token in every character, so
    a capped line can still be larger than a small period. Then it is cut more."""
    options = replace(SMALL_OPTIONS, period_token_budget=600)
    messages = [
        message(1, "2024-06-01 09:00:00", "a few short words"),
        message(2, "2024-06-01 09:01:00", "x" * 3_000),
        message(3, "2024-06-02 09:00:00", "and some more words"),
    ]
    plan = plan_of(messages, OneCharAToken(), options)
    budget = plan.limits.transcript_budget
    assert budget == 600
    assert all(planned.period.tokens <= budget for planned in plan.periods)
    assert sum(len(planned.lines) for planned in plan.periods) == 3
    long_line = plan.lines[1]
    assert long_line.cut
    assert long_line.flat.endswith(ELISION_MARK)
    assert len(long_line.flat) < 600
    assert long_line.text == "x" * 3_000
    assert plan.cut_messages == 1


def test_a_run_with_such_a_line_completes() -> None:
    options = replace(SMALL_OPTIONS, period_token_budget=600)
    messages = [
        message(1, "2024-06-01 09:00:00", "a few short words"),
        message(2, "2024-06-01 09:01:00", "x" * 3_000),
    ]
    llm = FakeLLM()
    report = run_report(
        source=ListSource(messages),
        conversation=CONVERSATION,
        llm=llm,
        model_info=model_info(),
        labeler=labeler,
        options=options,
        counter=OneCharAToken(),
    )
    assert report.status == "complete"
    assert not any("too large" in warning for warning in report.warnings)
    assert any("1 long message was cut" in warning for warning in report.warnings)


@pytest.mark.parametrize("seed", range(30))
def test_no_period_is_larger_than_its_budget_whatever_the_messages(seed: int) -> None:
    rng = random.Random(seed)
    start = datetime(2024, 1, 1, 8, 0, 0)
    messages = [
        message(
            number,
            start + timedelta(hours=rng.randint(0, 40) + 30 * number),
            rng.choice(["ok", "a few words here", "y" * 300, "z " * 2_500, "q" * 70_000]),
        )
        for number in range(1, rng.randint(2, 40))
    ]
    budget = rng.choice([200, 450, 2_000])
    options = ReportOptions(
        period_token_budget=budget,
        map_max_output_tokens=500,
        reduce_max_output_tokens=800,
        carry_over_token_budget=100,
    )
    counter = rng.choice([OneCharAToken(), FourCharsAToken()])
    plan = plan_of(sorted(messages, key=lambda item: item["timestamp"]), counter, options)
    assert plan.limits.transcript_budget == budget
    assert all(planned.period.tokens <= budget for planned in plan.periods)
    assert [line.id for planned in plan.periods for line in planned.lines] == [
        line.id for line in plan.lines
    ]
    assert plan.cut_messages == sum(line.cut for line in plan.lines)


@pytest.mark.parametrize(
    "text, keep, shown",
    [
        ("one two three four", 9, "one two [...]"),
        ("one two three four", 7, "one two [...]"),
        ("one two three four", 8, "one two [...]"),
        ("one two three four", 2, "[...]"),
        ("one, two", 4, "one, [...]"),
        ("abcdefghij", 5, "[...]"),
        ("abc-defghij", 6, "abc- [...]"),
        (chr(0x4E1C) * 10, 4, chr(0x4E1C) * 4 + " [...]"),
        ("short", 50, "short [...]"),
    ],
)
def test_a_cut_does_not_fall_inside_a_word(text: str, keep: int, shown: str) -> None:
    assert cut_start(text, keep) == shown


@pytest.mark.parametrize("cap", [40, 41, 42, 43, 44, 45, 46, 47])
def test_a_quote_of_what_a_cut_line_shows_is_found_in_the_message(cap: int) -> None:
    """The model can quote only what it reads. When it quotes the visible start
    of a cut message, with or without the mark, the quote must be verified."""
    letter = "the harbor lights were on when you came back and nothing else mattered that night"
    quotes: list[str] = []

    def responder(call: FakeCall) -> Any:
        if call.kind != "map":
            return None
        (line,) = call.lines
        quotes.append(line.text)
        moment = {
            "date": line.day,
            "title": "The harbor",
            "detail": "A long line.",
            "category": "milestone",
            "score": 0.9,
            "quote": line.text,
            "refs": [line.ref],
        }
        return {**default_reply(call), "moments": [moment]}

    report = run(
        [message(7, "2024-06-01 09:00:00", letter)],
        llm=FakeLLM(responder),
        options=replace(SMALL_OPTIONS, max_message_chars=cap),
    )
    (moment,) = report.moments
    assert quotes[0].endswith(ELISION_MARK)
    assert moment.quote_verified
    assert moment.quote == quotes[0]
    assert not any("quote" in warning for warning in report.warnings)


@pytest.mark.parametrize("pad", range(1, 9))
def test_a_quote_longer_than_the_limit_is_cut_between_words_and_is_still_found(pad: int) -> None:
    """A model that quotes a long line in full gives more than a report keeps.
    The cut must not fall inside a word, or a true quote would not be found."""
    words = "a" * pad + " " + " ".join(f"lantern{number}" for number in range(300))

    def responder(call: FakeCall) -> Any:
        if call.kind != "map":
            return None
        (line,) = call.lines
        moment = {
            "date": line.day,
            "title": "Many lanterns",
            "detail": "A long line.",
            "category": "milestone",
            "score": 0.9,
            "quote": line.text,
            "refs": [line.ref],
        }
        return {**default_reply(call), "moments": [moment]}

    report = run([message(7, "2024-06-01 09:00:00", words)], llm=FakeLLM(responder))
    (moment,) = report.moments
    assert moment.quote_verified
    assert 0 < len(moment.quote) <= MAX_QUOTE_CHARS
    assert words.startswith(moment.quote + " ")
    assert report.warnings == ()


def test_the_prompt_says_what_the_mark_at_the_end_of_a_line_means() -> None:
    assert ELISION_MARK in prompts.MAP_SYSTEM
