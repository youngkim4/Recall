from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from typing import Any

import pytest

from insights import engine, prompt, reading
from insights.thread import Note
from report_engine.ports import ChatMessage, LLMResult, LLMUsage
from tests.insights.kit import NOW, at, every, note, thread


class FakeLLM:
    """Answers each call with `reply(user_text)`; it counts the calls and can fail."""

    def __init__(self, reply: Callable[[str], Mapping[str, Any] | None], fail_on: str = "") -> None:
        self.reply = reply
        self.fail_on = fail_on
        self.seen: list[Sequence[ChatMessage]] = []

    def generate(self, messages, *, schema=None, max_output_tokens, reasoning=None) -> LLMResult:
        self.seen.append(messages)
        user = messages[-1]["content"]
        if self.fail_on and self.fail_on in user:
            raise RuntimeError("the service is down")
        assert schema == prompt.SCHEMA
        return LLMResult("{}", self.reply(user), LLMUsage(100, 0, 20), "completed")


def _recent() -> list[Note]:
    return [
        note("2025-11-02 18:00", "them", "we should get dinner when you're back", "a"),
        note("2025-11-02 18:05", "me", "yes!! i'll send you that article too", "b"),
        note("2025-11-20 09:00", "them", "my dad is in the hospital", "c"),
        note("2025-11-20 09:30", "me", "oh no, thinking of you", "d"),
    ]


def _items(*items: dict[str, Any]) -> Callable[[str], Mapping[str, Any]]:
    return lambda _user: {"items": list(items)}


class TestChecks:
    def test_refs_become_message_ids(self):
        llm = FakeLLM(_items({"kind": "plan", "summary": "Dinner with Lee never happened.", "refs": ["m1", "m2"], "done": False}))
        got = reading.read(llm, thread("lee", _recent()), NOW)
        assert [(item.kind, item.message_ids) for item in got.insights] == [("plan", ("a", "b"))]
        assert got.insights[0].when == at("2025-11-02 18:00")

    def test_an_unknown_ref_is_dropped_and_an_item_with_none_left_too(self):
        llm = FakeLLM(_items(
            {"kind": "news", "summary": "Lee's dad is in the hospital.", "refs": ["m3", "m99"], "done": False},
            {"kind": "plan", "summary": "Made up.", "refs": ["m42"], "done": False},
        ))
        got = reading.read(llm, thread("lee", _recent()), NOW)
        assert [item.message_ids for item in got.insights] == [("c",)]
        assert got.dropped == 1

    def test_a_promise_must_cite_my_own_message(self):
        llm = FakeLLM(_items(
            {"kind": "promise", "summary": "Lee's promise.", "refs": ["m1"], "done": False},
            {"kind": "promise", "summary": "Send Lee the article.", "refs": ["m2"], "done": False},
        ))
        got = reading.read(llm, thread("lee", _recent()), NOW)
        assert [item.message_ids for item in got.insights] == [("b",)]

    def test_done_items_and_repeats_are_dropped(self):
        same = {"kind": "news", "summary": "Lee's dad is sick.", "refs": ["m3"], "done": False}
        llm = FakeLLM(_items(same, dict(same), {"kind": "plan", "summary": "Dinner.", "refs": ["m1"], "done": True}))
        got = reading.read(llm, thread("lee", _recent()), NOW)
        assert len(got.insights) == 1
        assert got.dropped == 2

    @pytest.mark.parametrize("reply", [None, {}, {"items": "no"}, {"items": [1, "x", None]}])
    def test_a_reply_in_the_wrong_shape_gives_nothing(self, reply):
        got = reading.read(FakeLLM(lambda _user: reply), thread("lee", _recent()), NOW)
        assert got.insights == ()

    @pytest.mark.parametrize("refs", ["m1", None, [{"x": 1}], [["m1"]]])
    def test_refs_in_the_wrong_shape_drop_the_item(self, refs):
        llm = FakeLLM(_items({"kind": "plan", "summary": "x", "refs": refs, "done": False}))
        assert reading.read(llm, thread("lee", _recent()), NOW).insights == ()

    def test_an_unknown_kind_is_dropped(self):
        llm = FakeLLM(_items({"kind": "drift", "summary": "x", "refs": ["m1"], "done": False}))
        assert reading.read(llm, thread("lee", _recent()), NOW).insights == ()


class TestWindow:
    def test_only_the_last_ninety_days_and_nothing_after_now(self):
        notes = [note("2025-08-01", "me", "old"), *_recent(), note("2025-12-25", "them", "future")]
        assert [item.id for item in reading.window(thread("lee", notes), NOW)] == ["a", "b", "c", "d"]

    def test_a_long_window_keeps_the_newest(self):
        notes = [Note(f"n{place}", at("2025-12-01") + timedelta(minutes=place), place % 2 == 0, "x" * 1000) for place in range(80)]
        kept = reading.window(thread("lee", notes), NOW)
        assert kept[-1].id == "n79"
        assert sum(len(item.text) for item in kept) <= reading.MAX_CHARS

    def test_the_transcript_names_both_sides_and_refs(self):
        text = prompt.user_text("Lee", _recent(), NOW)
        assert "m1 18:00 Lee: we should get dinner" in text
        assert "m2 18:05 Me: yes!!" in text
        assert "--- 2025-11-20 Thu ---" in text
        assert "Today is 2025-12-20 Saturday" in text

    def test_a_one_sided_or_quiet_thread_is_not_read(self):
        assert reading.worth_reading(thread("lee", _recent()), NOW)
        assert not reading.worth_reading(thread("lee", [note("2025-12-01", "them", "code 1234")]), NOW)
        assert not reading.worth_reading(thread("lee", [note("2025-06-01", "them", "hi"), note("2025-06-01", "me", "hi")]), NOW)


class TestEngine:
    def test_counted_insights_need_no_model(self):
        made = thread("sam", every("2024-06-01", "2025-07-20", 7))
        report = engine.run([made], NOW)
        assert [item.kind for item in report.insights] == ["drift"]
        assert report.calls == 0
        assert [pick.thread for pick in report.picks] == ["sam"]

    def test_one_failed_call_leaves_the_others(self):
        lee = thread("lee", _recent())
        kim = thread("kim", _recent(), name="Kim")
        llm = FakeLLM(_items({"kind": "news", "summary": "Dad in hospital.", "refs": ["m3"], "done": False}), fail_on="The person is Kim.")
        report = engine.run([lee, kim], NOW, llm)
        assert report.calls == 2
        assert report.failed == 1
        assert [item.thread for item in report.insights if item.kind == "news"] == ["lee"]
        assert len(report.usage) == 1


class TestDigest:
    def test_heavier_and_fresher_reasons_win(self):
        from insights.digest import picks
        from insights.thread import Insight

        def made(key: str, kind: str, when: str, **figures: float) -> Insight:
            return Insight(kind, key, key.title(), "s", at(when), ("x",), tuple(figures.items()))

        found = [
            made("ann", "news", "2025-12-15"),
            made("bob", "plan", "2025-12-10"),
            made("cat", "drift", "2025-06-01", active_weeks=52.0),
            made("dan", "balance", "2025-12-01"),
            made("eve", "news", "2025-09-01"),
            made("fay", "birthday", "2024-12-22", days_until=2.0),
        ]
        ranked = picks(found, NOW)
        assert [pick.thread for pick in ranked] == ["fay", "ann", "cat"]
        assert all(pick.thread != "dan" for pick in picks(found, NOW, top=10))

        twice = [made("gus", "news", "2025-12-14"), made("gus", "news", "2025-12-15")]
        assert picks(twice, NOW)[0].score == 3.0
        assert len(picks(twice, NOW)[0].reasons) == 2
