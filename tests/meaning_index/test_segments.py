"""How a conversation is cut into segments, and what text a segment gives to embed."""

from __future__ import annotations

import copy

import pytest

from meaning_index.errors import SourceError
from meaning_index.segments import (
    GAP_MINUTES,
    LINE_CHARS,
    MAX_CHARS,
    MAX_MESSAGES,
    content_hash,
    cut_segments,
    segment_id,
    stored,
)
from tests.meaning_index.support import msg, run, stamp

CHAT = "invented-chat-1"


def cut(messages: list, conversation: str = CHAT) -> list:
    return list(cut_segments(conversation, messages))


def test_the_rules_are_the_ones_of_the_contract() -> None:
    assert (GAP_MINUTES, MAX_MESSAGES, MAX_CHARS, LINE_CHARS) == (120, 16, 1600, 400)


def test_a_short_exchange_is_one_segment_with_the_day_and_one_line_for_each_message() -> None:
    messages = [
        msg("a", "2024-06-03 09:00:00", "are we on for the lake", False),
        msg("b", "2024-06-03 09:01:30", "yes, bring the canoe", True),
    ]
    (segment,) = cut(messages)
    assert segment.text == "2024-06-03\nThem: are we on for the lake\nMe: yes, bring the canoe"
    assert segment.conversation_id == CHAT
    assert (segment.first_message_id, segment.last_message_id) == ("a", "b")
    assert (segment.first_timestamp, segment.last_timestamp) == ("2024-06-03 09:00:00", "2024-06-03 09:01:30")
    assert segment.message_count == 2


def test_a_gap_of_120_minutes_keeps_the_segment_and_a_longer_gap_starts_a_new_one() -> None:
    kept = cut([msg("a", stamp(0), "one"), msg("b", stamp(120), "two")])
    assert [s.message_count for s in kept] == [2]
    split = cut([msg("a", stamp(0), "one"), msg("b", stamp(120, ), "two"), msg("c", stamp(240.02), "three")])
    assert [(s.first_message_id, s.last_message_id) for s in split] == [("a", "b"), ("c", "c")]


def test_the_gap_is_measured_from_the_message_before_not_from_the_start() -> None:
    messages = [msg(f"m{n}", stamp(n * 100), f"line {n}") for n in range(5)]
    assert [s.message_count for s in cut(messages)] == [5]


def test_a_segment_holds_16_messages_at_most() -> None:
    segments = cut(run(35))
    assert [s.message_count for s in segments] == [16, 16, 3]
    assert segments[1].first_message_id == "m16"
    assert segments[2].first_message_id == "m32"


def test_a_segment_ends_when_its_text_reaches_1600_characters() -> None:
    # the day is 10 characters; each message adds a line break and a line of 106
    messages = [msg(f"m{n}", stamp(n), "x" * 100) for n in range(20)]
    first, second = cut(messages)
    assert first.message_count == 15
    assert len(first.text) == 10 + 15 * 107 >= MAX_CHARS
    assert len(first.text) - 107 < MAX_CHARS
    assert second.first_message_id == "m15"


def test_a_line_is_cut_at_400_characters() -> None:
    (segment,) = cut([msg("a", stamp(0), "y" * 1000, True)])
    line = segment.text.split("\n")[1]
    assert len(line) == LINE_CHARS
    assert line.startswith("Me: yyy")


def test_a_text_of_several_lines_is_one_line() -> None:
    (segment,) = cut([msg("a", stamp(0), "first\nsecond\r\n\tthird   fourth")])
    assert segment.text.split("\n") == ["2024-06-03", "Them: first second third fourth"]


@pytest.mark.parametrize("empty", ["", "   ", "\n\t "])
def test_a_message_with_no_text_is_in_no_segment(empty: str) -> None:
    messages = [msg("a", stamp(0), "one"), msg("b", stamp(1), empty), msg("c", stamp(2), "three")]
    (segment,) = cut(messages)
    assert segment.message_count == 2
    assert segment.text.count("\n") == 2
    assert (segment.first_message_id, segment.last_message_id) == ("a", "c")


def test_a_conversation_with_no_text_gives_no_segment() -> None:
    assert cut([msg("a", stamp(0), ""), msg("b", stamp(1), " ")]) == []
    assert cut([]) == []


def test_empty_messages_do_not_count_toward_the_16() -> None:
    messages = [msg(f"m{n}", stamp(n), "" if n % 2 else "word") for n in range(32)]
    assert [s.message_count for s in cut(messages)] == [16]


def test_segments_come_in_the_order_of_the_messages() -> None:
    messages = run(4) + run(4, prefix="n", start=500) + run(4, prefix="o", start=1000)
    segments = cut(messages)
    assert [s.first_message_id for s in segments] == ["m0", "n0", "o0"]
    assert [s.first_timestamp for s in segments] == sorted(s.first_timestamp for s in segments)
    assert segments[0].text.split("\n")[1:] == ["Them: a line 0", "Me: a line 1", "Them: a line 2", "Me: a line 3"]


def test_the_day_is_the_day_of_the_first_message() -> None:
    (segment,) = cut([msg("a", "2024-06-03 23:59:00", "late"), msg("b", "2024-06-04 00:01:00", "later")])
    assert segment.text.startswith("2024-06-03\n")
    assert segment.last_timestamp == "2024-06-04 00:01:00"


def test_a_time_that_goes_back_is_no_gap() -> None:
    segments = cut([msg("a", stamp(300), "one"), msg("b", stamp(0), "two")])
    assert [s.message_count for s in segments] == [2]


def test_ids_are_stable_and_follow_the_conversation_and_the_messages() -> None:
    first, again = cut(run(20)), cut(run(20))
    assert [s.segment_id for s in first] == [s.segment_id for s in again]
    assert len({s.segment_id for s in first}) == 2
    assert all(len(s.segment_id) == 64 and int(s.segment_id, 16) >= 0 for s in first)
    assert cut(run(20), "another-chat")[0].segment_id != first[0].segment_id
    assert first[0].segment_id == segment_id(CHAT, [f"m{n}" for n in range(16)])
    assert segment_id(CHAT, ["a", "bc"]) != segment_id(CHAT, ["ab", "c"])


def test_the_id_does_not_change_with_the_text_and_the_content_hash_does() -> None:
    (before,) = cut([msg("a", stamp(0), "one")])
    (after,) = cut([msg("a", stamp(0), "one, edited")])
    assert before.segment_id == after.segment_id
    assert content_hash(before.text, "model-a", 8) != content_hash(after.text, "model-a", 8)


def test_the_content_hash_follows_the_text_the_model_and_the_dimensions() -> None:
    base = content_hash("2024-06-03\nMe: one", "model-a", 8)
    assert base == content_hash("2024-06-03\nMe: one", "model-a", 8)
    assert len(base) == 64
    assert base != content_hash("2024-06-03\nMe: two", "model-a", 8)
    assert base != content_hash("2024-06-03\nMe: one", "model-b", 8)
    assert base != content_hash("2024-06-03\nMe: one", "model-a", 16)


def test_a_stored_segment_holds_the_facts_and_the_hash_and_no_text() -> None:
    (segment,) = cut(run(3))
    kept = stored(segment, "model-a", 8)
    assert kept.content_hash == content_hash(segment.text, "model-a", 8)
    assert (kept.segment_id, kept.conversation_id, kept.message_count) == (segment.segment_id, CHAT, 3)
    assert (kept.first_message_id, kept.last_message_id) == ("m0", "m2")
    assert (kept.first_timestamp, kept.last_timestamp) == (segment.first_timestamp, segment.last_timestamp)
    assert not hasattr(kept, "text")


def test_the_messages_given_are_not_changed() -> None:
    messages = run(20)
    before = copy.deepcopy(messages)
    cut(messages)
    assert messages == before


BAD_TIMES = ["2024-06-03", "03/06/2024 09:00:00", "", "2024-06-03T09:00:00Z", "2024-13-40 09:00:00", None, 1717405200]


@pytest.mark.parametrize("bad", BAD_TIMES)
def test_a_time_that_can_not_be_read_is_a_typed_error_with_no_text(bad: str) -> None:
    with pytest.raises(SourceError) as caught:
        cut([msg("a", stamp(0), "one"), msg("b", bad, "MARKED-TEXT")])
    assert "MARKED-TEXT" not in str(caught.value)
    assert CHAT not in str(caught.value)


def test_cutting_reads_the_messages_as_a_stream() -> None:
    def stream():
        yield from run(40)

    assert sum(s.message_count for s in cut_segments(CHAT, stream())) == 40
