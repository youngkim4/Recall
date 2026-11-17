"""Seams between modules: cases found in the cross-file review."""

import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from report_engine import markdown, plan, prompts, reduce_step
from report_engine.cache import PeriodCache
from report_engine.errors import ConfigError, ReduceError, SourceError
from report_engine.fake_llm import FakeCall, FakeLLM, default_reply
from report_engine.figures import conversation_figures
from report_engine.markdown import clean_section, render_markdown
from report_engine.models import Moment, PeriodSummary, Report
from report_engine.pipeline import run_report
from report_engine.reduce_fit import fit_reduce
from report_engine.tokens import TokenMeter
from report_engine.transcript import build_lines
from report_engine.verify import holds, quote_text

from .support import LARGE, SMALL, SMALL_OPTIONS, run
from .synth import (
    CONVERSATION,
    MINA,
    RAVI,
    THEO,
    ListSource,
    WordCounter,
    labeler,
    message,
    model_info,
)


@pytest.mark.parametrize("bad_id", ["", "   "])
def test_a_message_with_no_id_stops_the_run_before_any_call(bad_id: str) -> None:
    """A verified quote must name its message. An empty id would only be found
    out when the report is put together, after every call was paid for."""
    llm = FakeLLM()
    messages = [message(bad_id, "2024-06-01 09:00:00", "a line long enough to be quoted")]
    with pytest.raises(SourceError) as caught:
        run(messages, llm=llm)
    assert "id" in str(caught.value)
    assert llm.calls == []


def test_a_run_says_in_the_log_which_tokenizer_it_counts_with(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="report_engine.plan"):
        run(SMALL)
    assert "words x1" in caplog.text
    assert "1 period" in caplog.text


def long_summaries(words: int) -> Any:
    def responder(call: FakeCall) -> Any:
        if call.kind != "map":
            return None
        reply = default_reply(call)
        return {**reply, "summary": "word " * words}

    return responder


@pytest.mark.parametrize("reduce_cap", [*range(420, 440), *range(440, 800, 15)])
def test_a_reduce_request_that_was_cut_to_fit_is_never_larger_than_the_limit(
    reduce_cap: int,
) -> None:
    """The sizes used to cut the summaries must be the sizes of what is sent.
    A run that paid for its periods must not end on a request that is a few
    tokens too large."""
    info = model_info(context_window=20_000, max_input_tokens=20_000, max_output_tokens=20_000)
    options = replace(SMALL_OPTIONS, reduce_max_output_tokens=20_000 - reduce_cap)
    llm = FakeLLM(long_summaries(500))
    report = run(LARGE, llm=llm, info=info, options=options)
    (reduce_call,) = llm.reduce_calls
    assert len(reduce_call.text.split()) <= reduce_cap
    assert len(report.moments) == 4
    assert any("shortened" in warning for warning in report.warnings)


def test_a_limit_below_the_smallest_reduce_request_is_refused_before_any_call() -> None:
    info = model_info(context_window=20_000, max_input_tokens=20_000, max_output_tokens=20_000)
    llm = FakeLLM(long_summaries(500))
    with pytest.raises(ConfigError):
        run(
            LARGE,
            llm=llm,
            info=info,
            options=replace(SMALL_OPTIONS, reduce_max_output_tokens=20_000 - 419),
        )
    assert llm.calls == []


class LinesCostExtra:
    """Counts like a real tokenizer in one way: every line break is a token, so
    a joined text is larger than the sum of its lines."""

    family = "words"

    def count_many(self, texts: Any) -> list[int]:
        return [len(text.split()) + text.count("\n") for text in texts]


def test_the_fit_holds_when_joined_text_counts_more_than_its_lines() -> None:
    info = model_info(context_window=20_000, max_input_tokens=20_000, max_output_tokens=20_000)
    options = replace(SMALL_OPTIONS, reduce_max_output_tokens=20_000 - 900)
    llm = FakeLLM(long_summaries(700))
    report = run_report(
        source=ListSource(LARGE),
        conversation=CONVERSATION,
        llm=llm,
        model_info=info,
        labeler=labeler,
        options=options,
        counter=LinesCostExtra(),
    )
    (reduce_call,) = llm.reduce_calls
    assert LinesCostExtra().count_many([reduce_call.text])[0] <= 900
    assert report.status == "complete"


@pytest.mark.parametrize(
    "hostile",
    [
        "<<b>img src=x onerror=alert(1)>",
        "<<<i>b>script>alert(1)<<<i>b>/script>",
        "[[click](x)](https://attacker.example/a)",
        "![![a](b)](https://attacker.example/p.png)",
        "[a]([b](https://attacker.example))",
        "<" * 80 + "b" + ">" * 80 + "img src=x>",
        "[" * 60 + "x" + "](y)" * 60,
        "text\n[ref]: https://attacker.example/def\n[[ref]]: https://attacker.example/def2",
    ],
)
def test_no_nesting_lets_a_tag_a_link_or_an_image_through(hostile: str) -> None:
    cleaned = clean_section(f"Before. {hostile} After.")
    assert not re.search(r"</?[A-Za-z][^<>]*>", cleaned)
    assert not re.search(r"\[[^\]]*\]\([^)]*\)", cleaned)
    assert "attacker.example" not in cleaned or "](" not in cleaned
    assert clean_section(cleaned) == cleaned
    assert cleaned.startswith("Before.") and cleaned.endswith("After.")


def test_nested_markup_in_a_reply_does_not_reach_the_stored_report_or_the_markdown() -> None:
    def responder(call: FakeCall) -> Any:
        if call.kind != "reduce":
            return None
        body = "Fine. <<b>img src=x onerror=alert(1)> [[go](x)](https://attacker.example/a) <<<i>b>script>x()"
        return {"arc": body, "dynamics": body, "defining_moments": body, "big_picture": body}

    report = run(SMALL, llm=FakeLLM(responder))
    stored = report.model_dump_json()
    rendered = render_markdown(report)
    for text in (stored, rendered):
        assert "<img" not in text and "<script" not in text
        assert "](https://attacker.example" not in text


def test_two_people_whose_long_names_start_the_same_keep_two_names() -> None:
    names = {
        MINA: "Alexandra Catherine Montgomery-Whitfield (work)",
        RAVI: "Alexandra Catherine Montgomery-Whitfield (sister)",
        THEO: "Alexandra Catherine Montgomery-Whitfield (work)",
    }
    lines = build_lines(
        [
            message(1, "2024-06-01 09:00:00", "a", sender=MINA),
            message(2, "2024-06-01 09:01:00", "b", sender=RAVI),
            message(3, "2024-06-01 09:02:00", "c", sender=THEO),
        ],
        lambda sender: names[sender],
    )
    first, second, third = (line.label for line in lines)
    assert first != second
    assert first == third
    assert first == "Alexandra Catherine Montgomery-Whitfield"
    assert second == "Alexandra Catherine Montgomery-Whitfie 2"
    assert all(len(label) <= 40 for label in (first, second))
    figures = conversation_figures(lines)
    assert "Alexandra Catherine Montgomery-Whitfield 2" in figures
    assert "Alexandra Catherine Montgomery-Whitfie 2 1" in figures


def test_a_name_keeps_its_form_when_a_like_name_arrives_later() -> None:
    names = {MINA: "A" * 45 + " one", RAVI: "A" * 45 + " two"}
    early = [message(1, "2024-06-01 09:00:00", "a", sender=MINA)]
    later = [*early, message(2, "2024-06-02 09:00:00", "b", sender=RAVI)]
    assert (
        build_lines(early, lambda s: names[s])[0].label
        == build_lines(later, lambda s: names[s])[0].label
    )


def test_the_prompt_version_follows_the_prompt_text(monkeypatch: pytest.MonkeyPatch) -> None:
    assert prompts.map_prompt_version() == prompts.MAP_PROMPT_VERSION
    before = prompts.map_prompt_version()
    monkeypatch.setattr(prompts, "MAP_TASK", prompts.MAP_TASK + " One more rule.")
    changed_task = prompts.map_prompt_version()
    assert changed_task != before
    monkeypatch.setattr(prompts, "MAP_SCHEMA", {**prompts.MAP_SCHEMA, "description": "changed"})
    assert prompts.map_prompt_version() not in {before, changed_task}


def test_a_changed_prompt_does_not_reuse_the_cache(
    cache_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = PeriodCache(cache_path)
    run(SMALL, cache=cache)
    monkeypatch.setattr(plan, "MAP_PROMPT_VERSION", "map-another-text")
    llm = FakeLLM()
    run(SMALL, llm=llm, cache=cache)
    assert len(llm.map_calls) == 1


def test_a_quote_with_marks_inside_it_is_not_found() -> None:
    text = "first part then the middle part and at last the final part"
    assert holds(text, quote_text("first part then the middle part"))
    assert not holds(text, quote_text("first part [...] the middle [...] final part"))
    assert not holds(text, quote_text("first part [...] final part"))


def test_when_only_the_moments_are_too_many_no_summary_is_cut() -> None:
    meter = TokenMeter(WordCounter())
    periods = [
        PeriodSummary(
            label="A", first_day="2024-01-01", last_day="2024-01-31", summary="word " * 30
        )
    ]
    moments = [
        Moment(
            date="2024-01-05",
            title=f"t{n}",
            detail="detail " * 20,
            category="milestone",
            score=n / 10,
        )
        for n in range(1, 9)
    ]
    fitted = fit_reduce(periods, moments, 150, meter)
    assert fitted.shortened == 0
    assert fitted.periods == tuple(periods)
    assert 0 < len(fitted.moments) < 8
    assert [item.title for item in fitted.moments] == [
        f"t{n}" for n in range(9 - len(fitted.moments), 9)
    ]


class JoinedCostsMore:
    """Counts a joined text higher than the sum of its lines: every line break
    after the first one is forty tokens. Line sizes added up are far too low
    for such a counter."""

    family = "words"

    def count_many(self, texts: Any) -> list[int]:
        return [len(text.split()) + 40 * max(0, text.count("\n") - 1) for text in texts]


class NeverFits:
    """Counts words, but every reduce request that holds a period line is one
    token over `limit`, whatever is cut from it."""

    family = "words"

    def __init__(self, limit: int) -> None:
        self.limit = limit

    def count_many(self, texts: Any) -> list[int]:
        return [self.limit + 1 if "<periods>\n[" in text else len(text.split()) for text in texts]


def run_counted(counter: Any, reduce_cap: int, llm: FakeLLM) -> Report:
    """LARGE with a model that lets a reduce request carry `reduce_cap` tokens."""
    info = model_info(context_window=20_000, max_input_tokens=20_000, max_output_tokens=20_000)
    return run_report(
        source=ListSource(LARGE),
        conversation=CONVERSATION,
        llm=llm,
        model_info=info,
        labeler=labeler,
        options=replace(SMALL_OPTIONS, reduce_max_output_tokens=20_000 - reduce_cap),
        counter=counter,
    )


def test_a_request_that_counts_more_once_joined_fits_after_one_more_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first fit adds up line sizes and can have room to spare. When the
    joined request is then over the limit, the next fit starts from what the
    first one used, not from the room it was given: one more round is enough."""
    counter = JoinedCostsMore()
    probe = FakeLLM(long_summaries(300))
    run_counted(counter, 19_000, probe)
    uncut = counter.count_many([probe.reduce_calls[0].text])[0]
    cap = uncut - 10

    rooms: list[int] = []
    fit = reduce_step.fit_reduce

    def watched(periods: Any, moments: Any, room: int, meter: Any) -> Any:
        rooms.append(room)
        return fit(periods, moments, room, meter)

    monkeypatch.setattr(reduce_step, "fit_reduce", watched)
    llm = FakeLLM(long_summaries(300))
    report = run_counted(counter, cap, llm)
    (reduce_call,) = llm.reduce_calls
    assert counter.count_many([reduce_call.text])[0] <= cap
    assert len(rooms) == 2
    assert rooms[1] < rooms[0]
    assert report.status == "complete"
    assert any("shortened" in warning for warning in report.warnings)


def test_a_request_that_stays_over_the_limit_is_never_sent() -> None:
    cap = 3_000
    llm = FakeLLM(long_summaries(300))
    with pytest.raises(ReduceError) as caught:
        run_counted(NeverFits(cap), cap, llm)
    assert "too large" in str(caught.value)
    assert str(cap) in str(caught.value)
    assert len(llm.map_calls) == 4
    assert llm.reduce_calls == []
    assert caught.value.usage is not None
    assert caught.value.usage.calls == 4


@pytest.mark.parametrize(
    "quote, text, found",
    [
        ("tok", "tokens are fine", False),
        ("are fin", "tokens are fine", False),
        ("tok", "tokens and a tok", True),
        ("fin", "fine, fin", True),
    ],
)
def test_a_quote_that_starts_at_a_word_can_not_end_inside_one(
    quote: str, text: str, found: bool
) -> None:
    assert holds(text, quote_text(quote)) is found


def test_when_the_cleaning_passes_run_out_every_bracket_goes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(markdown, "MAX_CLEAN_PASSES", 1)
    cleaned = clean_section(
        "# Title\nSee [[here](x)](https://attacker.example/a) and <<b>i>this</i>."
    )
    assert not set("<>[]") & set(cleaned)
    assert cleaned.startswith("**Title**")
    assert "here" in cleaned and "this" in cleaned
