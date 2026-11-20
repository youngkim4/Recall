"""The recipe of the segments: the default, and each switch alone and together."""

from __future__ import annotations

from datetime import timedelta

import pytest

from meaning_index.recipe import (
    ADAPTIVE_MAX,
    ADAPTIVE_MIN,
    DEFAULT_NAME,
    DEFAULT_RECIPE,
    OVERLAP_MESSAGES,
    Recipe,
    adaptive_gap,
    context_line,
    recipe_from_words,
)
from meaning_index.segments import LINE_CHARS, MAX_CHARS, MAX_MESSAGES, cut_segments, segment_id
from tests.meaning_index.support import START, msg, run, stamp

CHAT = "invented-chat-1"
ALL = Recipe(context=True, overlap=True, adaptive=True)


def cut(messages: list, recipe: Recipe = DEFAULT_RECIPE, context=None) -> list:
    return list(cut_segments(CHAT, messages, recipe, context))


def talks(count: int, *, every: float, within: float, size: int = 6) -> list:
    """`count` talks of `size` messages: `within` minutes apart inside a talk, `every` minutes between talks."""
    found = []
    for talk in range(count):
        for n in range(size):
            found.append(msg(f"t{talk}-{n}", stamp(talk * every + n * within), f"talk {talk} line {n}", n % 2 == 1))
    return found


# --- the names of the recipes ------------------------------------------------------


def test_the_default_recipe_has_no_switch_on_and_a_name_with_its_version() -> None:
    assert DEFAULT_RECIPE == Recipe(False, False, False)
    assert DEFAULT_NAME == DEFAULT_RECIPE.name == "v1"
    assert Recipe(context=True).name == "v1+context"
    assert ALL.name == "v1+context+overlap+adaptive"
    assert len({Recipe(*flags).name for flags in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]}) == 8


def test_words_name_the_switches_in_any_order_and_case_and_unknown_words_are_given_back() -> None:
    assert recipe_from_words("") == (DEFAULT_RECIPE, ())
    assert recipe_from_words("adaptive, Context ,overlap") == (ALL, ())
    assert recipe_from_words("overlap,overlap") == (Recipe(overlap=True), ())
    assert recipe_from_words("context,names,,fast") == (Recipe(context=True), ("names", "fast"))


# --- the default recipe is the one of segments.py ------------------------------------


def test_the_default_recipe_cuts_and_writes_as_before_byte_for_byte() -> None:
    messages = [*run(35), *run(3, prefix="late", start=600)]
    plain = list(cut_segments(CHAT, messages))
    assert cut(messages) == plain
    assert cut(messages, DEFAULT_RECIPE, lambda *_: "never read") == plain
    assert [s.message_count for s in plain] == [16, 16, 3, 3]
    assert plain[0].text.splitlines()[0] == "2024-06-03"
    assert plain[0].segment_id == segment_id(CHAT, [f"m{n}" for n in range(16)])


# --- context ------------------------------------------------------------------------


def test_the_context_line_holds_the_month_and_the_year_and_the_words_of_the_caller() -> None:
    assert context_line("2024-06-03 09:00:00", "") == "Context: June 2024"
    assert context_line("2019-12-31 23:59:59", "  direct chat\n with Ana  ") == "Context: December 2019; direct chat with Ana"
    assert len(context_line("2019-01-01 00:00:00", "x " * 500)) == 400


def test_with_context_each_segment_starts_with_its_context_line_and_keeps_its_cut() -> None:
    messages = [*run(20), *run(2, prefix="late", start=600)]
    asked: list[tuple[str, str]] = []

    def context(conversation: str, first: str) -> str:
        asked.append((conversation, first))
        return "direct chat with Ana"

    plain, framed = cut(messages), cut(messages, Recipe(context=True), context)
    assert [(s.segment_id, s.first_message_id, s.message_count) for s in framed] == [
        (s.segment_id, s.first_message_id, s.message_count) for s in plain
    ]
    for before, after in zip(plain, framed, strict=True):
        assert after.text == f"Context: June 2024; direct chat with Ana\n{before.text}"
    assert asked == [(CHAT, s.first_timestamp) for s in plain]


def test_with_context_and_no_words_the_line_holds_the_month_alone() -> None:
    (segment,) = cut([msg("a", stamp(0), "hello")], Recipe(context=True))
    assert segment.text == "Context: June 2024\n2024-06-03\nThem: hello"


# --- overlap ------------------------------------------------------------------------


def test_with_overlap_a_segment_cut_by_size_carries_the_last_two_messages_before_it() -> None:
    messages = run(35)
    plain, carried = cut(messages), cut(messages, Recipe(overlap=True))
    assert OVERLAP_MESSAGES == 2
    assert [s.message_count for s in carried] == [16, 18, 5]
    assert carried[0] == plain[0]
    second = carried[1]
    assert (second.first_message_id, second.last_message_id) == ("m14", "m31")
    assert second.first_timestamp == messages[14]["timestamp"]
    assert second.text.splitlines()[1:3] == ["Them: a line 14", "Me: a line 15"]
    assert second.text.splitlines()[3:] == plain[1].text.splitlines()[1:]
    assert second.segment_id == segment_id(CHAT, [f"m{n}" for n in range(14, 32)])


def test_with_overlap_a_segment_after_a_gap_carries_nothing() -> None:
    messages = [*run(4), *run(3, prefix="late", start=600)]
    assert cut(messages, Recipe(overlap=True)) == cut(messages)


def test_with_overlap_the_ids_stay_unique_and_each_message_is_read_back_in_the_range_of_its_segment() -> None:
    messages = run(100)
    order = {m["id"]: place for place, m in enumerate(messages)}
    segments = cut(messages, Recipe(overlap=True))
    assert len({s.segment_id for s in segments}) == len(segments)
    for segment in segments:
        first, last = order[segment.first_message_id], order[segment.last_message_id]
        assert last - first + 1 == segment.message_count
        lines = segment.text.splitlines()[1:]
        assert lines == [f"{'Me' if n % 2 else 'Them'}: a line {n}" for n in range(first, last + 1)]


# --- adaptive -----------------------------------------------------------------------


def test_the_adaptive_gap_is_the_85th_percentile_of_the_gaps_clamped_to_30_minutes_and_8_hours() -> None:
    def moments(minutes: list[float]) -> list:
        return [START + timedelta(minutes=m) for m in minutes]

    default = timedelta(minutes=120)
    assert (ADAPTIVE_MIN, ADAPTIVE_MAX) == (timedelta(minutes=30), timedelta(hours=8))
    # 20 gaps of 1..20 hours: the nearest rank of p85 is the 17th, 17 hours, clamped to 8
    slow = moments([sum(range(1, n + 1)) * 60 for n in range(21)])
    assert adaptive_gap(slow, default) == ADAPTIVE_MAX
    # 20 gaps of 1 minute: clamped up to 30 minutes
    assert adaptive_gap(moments(list(range(21))), default) == ADAPTIVE_MIN
    # 20 gaps of 10, 20, ... 200 minutes: the 17th is 170 minutes
    assert adaptive_gap(moments([5 * n * (n + 1) for n in range(21)]), default) == timedelta(minutes=170)
    # too few gaps to know the rhythm, and a time that goes back
    assert adaptive_gap(moments([0, 600, 1200]), default) == default
    assert adaptive_gap(moments([0, -5, 1, 2, 3, 4, 5, 6, 7, 8]), default) == ADAPTIVE_MIN


def test_the_adaptive_gap_differs_between_a_chatty_and_a_slow_conversation() -> None:
    # talks of 6 messages; the pause between two talks is 60 minutes
    chatty = talks(5, every=60, within=0.5)
    slow = talks(5, every=60 * 10, within=50)
    # the fixed rule keeps the chatty talks together and cuts the slow ones
    assert [s.message_count for s in cut(chatty)] == [16, 14]
    assert [s.message_count for s in cut(slow)] == [6] * 5
    # the rhythm cuts the chatty talks at each pause, and keeps the slow talks as talks
    assert [s.message_count for s in cut(chatty, Recipe(adaptive=True))] == [6] * 5
    assert [s.message_count for s in cut(slow, Recipe(adaptive=True))] == [6] * 5
    wider = talks(5, every=60 * 20, within=170)
    assert [s.message_count for s in cut(wider)] == [1] * 30
    assert [s.message_count for s in cut(wider, Recipe(adaptive=True))] == [6] * 5


def test_adaptive_keeps_the_limits_of_messages_and_characters() -> None:
    messages = [msg(f"m{n}", stamp(n), "x" * 300) for n in range(60)]
    for segment in cut(messages, Recipe(adaptive=True)):
        assert segment.message_count <= MAX_MESSAGES
        assert len(segment.text) <= MAX_CHARS + 400


# --- together -----------------------------------------------------------------------


@pytest.mark.parametrize("recipe", [Recipe(context=True), Recipe(overlap=True), Recipe(adaptive=True), ALL])
def test_each_recipe_is_deterministic(recipe: Recipe) -> None:
    messages = [*talks(4, every=45, within=1, size=12), *run(40, prefix="late", start=9000)]
    first = cut(messages, recipe, lambda *_: "group chat of 3 people: Lake")
    assert first == cut([dict(m) for m in messages], recipe, lambda *_: "group chat of 3 people: Lake")


def test_all_switches_together_cut_by_the_rhythm_carry_two_messages_and_open_with_the_context() -> None:
    messages = talks(2, every=60, within=0.5, size=20)
    segments = cut(messages, ALL, lambda *_: "direct chat with Ana")
    assert [s.message_count for s in segments] == [16, 6, 16, 6]
    assert segments[1].first_message_id == "t0-14"
    assert segments[2].first_message_id == "t1-0"
    assert all(s.text.startswith("Context: June 2024; direct chat with Ana\n2024-06-03\n") for s in segments)


@pytest.mark.parametrize("recipe", [Recipe(context=True), Recipe(overlap=True), ALL])
def test_no_text_of_any_recipe_is_longer_than_one_input_of_the_models_can_be(recipe: Recipe) -> None:
    messages = [msg(f"m{n}", stamp(n), "\U0001f6f6" * (120 + 37 * (n % 9)), n % 3 == 0) for n in range(400)]
    segments = cut(messages, recipe, lambda *_: "\U0001f6f6" * 600)
    assert max(len(segment.text) for segment in segments) <= MAX_CHARS + LINE_CHARS
    assert max(len(segment.text.encode("utf-8")) for segment in segments) < 8192
    assert sum(1 for segment in segments for line in segment.text.splitlines() if line.startswith(("Me:", "Them:"))) >= 400
