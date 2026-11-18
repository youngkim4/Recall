"""conversation_digest: the story of one conversation, with the messages that back its moments."""

import logging

import pytest

from chat_engine.fake_digests import FakeDigests, moment, period
from chat_engine.render import estimate_tokens
from chat_engine.scope import Scope
from chat_engine.tool_digest import DIGESTS_PER_QUESTION
from tests.chat_engine.memory_bench import RAW_HANDLES, GrowingLabeler, MemoryBench, shown_refs
from tests.chat_engine.test_tools import Bench
from tests.chat_engine.tiny import (
    INES_PHONE,
    JONAS_PHONE,
    MAYA_EMAIL,
    MAYA_PHONE,
    UNSAVED_EMAIL,
    UNSAVED_PHONE,
)


def maya_story() -> list:
    return [
        period(
            "2021", "2021-03-04", "2021-03-04", 4, "Maya signs a lease and plans the move.",
            [
                moment(
                    "2021-03-04", "The lease is signed", "Maya tells you the news.", "the lease got signed btw",
                    ["a2", "a3"],
                ),
                moment("2021-03-04", "Moving day is set", "They move in April.", "", ["a4"]),
            ],
        ),
        period(
            "2023", "2023-07-11", "2023-07-11", 3, "The landlord raises the rent.",
            [moment("2023-07-11", "Rent goes up", "Fifty more a month.", "fifty more a month", ["a7"])],
        ),
        period("2025", "2025-12-24", "2025-12-24", 2, "A Christmas greeting and a plan for the move."),
    ]


def bench_for(story: list | None = None, **options: object) -> MemoryBench:
    fake_options = {key: options.pop(key) for key in list(options) if key in FAKE_OPTIONS}
    digests = FakeDigests({MAYA_PHONE: maya_story() if story is None else story}, **fake_options)
    return MemoryBench(digests=digests, **options)


FAKE_OPTIONS = {"usd", "cached", "failed", "error", "estimate_error", "during", "honour_cancel"}


def test_the_tool_is_there_only_with_the_port() -> None:
    outcome = Bench().run("conversation_digest", conversation="c1")
    assert not outcome.ok and 'There is no tool named "conversation_digest".' in outcome.text


def test_a_digest_gives_the_story_period_by_period_with_the_messages_of_its_moments() -> None:
    bench = bench_for()
    alias = bench.alias(MAYA_PHONE)
    prepared = bench.prepare("conversation_digest", conversation=alias)
    assert prepared.own_budget == 24_000 and not prepared.repeated
    outcome = prepared.run(prepared.own_budget)
    assert outcome.ok
    assert outcome.text.splitlines() == [
        "Digest of c1 · Maya Okafor · direct: 3 periods, 9 messages. "
        "A summary is not a quote: quote and cite only the message lines.",
        "",
        "== 2021 · 2021-03-04 to 2021-03-04 · 4 messages ==",
        "Maya signs a lease and plans the move.",
        '- 2021-03-04 · The lease is signed: Maya tells you the news. Quote: "the lease got signed btw"',
        "2021-03-04",
        "m1 18:22 Maya Okafor: yes!! the lease got signed btw",
        "m2 18:23 You: omg congrats",
        "- 2021-03-04 · Moving day is set: They move in April.",
        "2021-03-04",
        "m3 18:24 Maya Okafor: we move in on the first of April",
        "",
        "== 2023 · 2023-07-11 to 2023-07-11 · 3 messages ==",
        "The landlord raises the rent.",
        '- 2023-07-11 · Rent goes up: Fifty more a month. Quote: "fifty more a month"',
        "2023-07-11",
        "m4 09:05 Maya Okafor: fifty more a month",
        "",
        "== 2025 · 2025-12-24 to 2025-12-24 · 2 messages ==",
        "A Christmas greeting and a plan for the move.",
    ]
    assert [bench.registry.get(f"m{n}")["id"] for n in range(1, 5)] == ["a2", "a3", "a4", "a7"]
    assert outcome.summary == "Read the story of your conversation with Maya Okafor: 3 periods"
    assert bench.box.context.memory.digests.calls == [
        ("estimate", {"conversation_id": MAYA_PHONE, "since": None, "until": None}),
        ("digest", {"conversation_id": MAYA_PHONE, "since": None, "until": None}),
    ]


def test_the_status_line_says_whether_periods_must_be_built() -> None:
    new, cached = bench_for(usd=0.3), bench_for(usd=0.3, cached=True)
    assert new.prepare("conversation_digest", conversation=new.alias(MAYA_PHONE)).status == (
        "Reading your whole conversation with Maya Okafor"
    )
    assert cached.prepare("conversation_digest", conversation=cached.alias(MAYA_PHONE)).status == (
        "Recalling your conversation with Maya Okafor"
    )


def test_a_message_the_archive_does_not_hold_or_the_scope_does_not_allow_is_dropped_without_a_word() -> None:
    story = [
        period(
            "2021", "2021-03-04", "2021-03-04", 4, "A lease.",
            [moment("2021-03-04", "Signed", "It is signed.", "", ["gone", "j1", "a2", "b1"])],
        )
    ]
    bench = bench_for(story, scope=Scope(conversations=(MAYA_PHONE, MAYA_EMAIL)))
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert [m["id"] for m in bench.registry.messages()] == ["a2"]  # b1 is in the scope, in another conversation
    assert "sourdough" not in outcome.text and "lease scan" not in outcome.text and "gone" not in outcome.text
    assert "m1 18:22 Maya Okafor: yes!! the lease got signed btw" in outcome.text
    assert not [a for a in bench.read("message") if a["message_id"] not in {"gone", "j1", "a2", "b1"}]


def test_a_quote_is_shown_only_with_a_message_of_its_moment() -> None:
    story = [
        period(
            "2021", "2021-03-04", "2021-03-04", 4, "A lease.",
            [moment("2021-03-04", "Signed", "It is signed.", "nobody can check this", ["gone"])],
        )
    ]
    bench = bench_for(story)
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert "- 2021-03-04 · Signed: It is signed." in outcome.text
    assert "nobody can check this" not in outcome.text and len(bench.registry) == 0


def test_a_digest_above_the_cost_limit_is_not_built() -> None:
    bench = bench_for(usd=2.5)
    prepared = bench.prepare("conversation_digest", conversation=bench.alias(MAYA_PHONE), since="2021-01-01")
    outcome = prepared.run(prepared.own_budget)
    assert prepared.status == "" and not outcome.ok
    assert outcome.text == (
        "Reading c1 · Maya Okafor · direct, from 2021-01-01 whole would cost about $2.50: 3 periods are not "
        "built yet. That is above the limit of $2.00, so nothing was built. Narrow the days with since and "
        "until, or use the other tools."
    )
    assert "\n" not in outcome.text
    assert outcome.summary == "Did not read the conversation with Maya Okafor: it would cost about $2.50"
    assert [name for name, _ in bench.box.context.memory.digests.calls] == ["estimate"]
    assert bench.read("message") == []


@pytest.mark.parametrize(
    ("usd", "limit", "cached", "built"),
    [(2.0, 2.0, False, True), (2.01, 2.0, False, False), (0.01, 0.0, False, False), (9.0, 0.0, True, True)],
)
def test_the_cost_guard_compares_the_cost_of_what_is_not_cached(
    usd: float, limit: float, cached: bool, built: bool
) -> None:
    bench = bench_for(usd=usd, cached=cached, digest_usd_limit=limit)
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert outcome.ok is built
    assert ("digest" in [name for name, _ in bench.box.context.memory.digests.calls]) is built


def test_the_scope_of_the_user_limits_the_conversation_and_the_days() -> None:
    bench = bench_for(scope=Scope(conversations=(JONAS_PHONE,)))
    outside = bench.run("conversation_digest", conversation="Maya Okafor")
    assert not outside.ok and bench.box.context.memory.digests.calls == []
    dated = bench_for(scope=Scope(since="2023-01-01", until="2023-12-31"))
    outcome = dated.run("conversation_digest", conversation=dated.alias(MAYA_PHONE), since="2020-01-01")
    assert dated.box.context.memory.digests.calls == [
        ("estimate", {"conversation_id": MAYA_PHONE, "since": "2023-01-01", "until": "2023-12-31"}),
        ("digest", {"conversation_id": MAYA_PHONE, "since": "2023-01-01", "until": "2023-12-31"}),
    ]
    assert "The landlord raises the rent." in outcome.text
    assert "Maya signs a lease" not in outcome.text and "Christmas" not in outcome.text
    assert "[2 periods reach outside the days of the scope and are not shown.]" in outcome.text
    assert [m["id"] for m in dated.registry.messages()] == ["a7"]


def test_a_moment_outside_the_days_of_the_scope_is_dropped() -> None:
    story = [
        period(
            "2023", "2023-07-11", "2023-07-11", 3, "The rent goes up.",
            [
                moment("2021-03-04", "An old lease", "From another year.", "the lease got signed btw", ["a2"]),
                moment("2023-07-11", "Rent goes up", "Fifty more.", "", ["a7", "a2"]),
            ],
        )
    ]
    bench = bench_for(story, scope=Scope(since="2023-01-01"))
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert "An old lease" not in outcome.text and "lease got signed" not in outcome.text
    assert [m["id"] for m in bench.registry.messages()] == ["a7"]


def test_a_failure_of_the_port_is_one_plain_line(caplog: pytest.LogCaptureFixture) -> None:
    bench = bench_for(error=RuntimeError("Maya Okafor said secret things"))
    with caplog.at_level(logging.WARNING, logger="chat_engine"):
        outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert not outcome.ok
    assert outcome.text == (
        "The digest of c1 · Maya Okafor · direct could not be built now. Go on with the other tools."
    )
    assert outcome.summary == "Could not read the story of your conversation with Maya Okafor"
    assert len(bench.registry) == 0 and bench.read("message") == []
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "RuntimeError" in logged and "secret" not in logged and "Maya" not in logged


def test_a_failure_of_the_estimate_builds_nothing() -> None:
    bench = bench_for(estimate_error=RuntimeError("no price list"))
    prepared = bench.prepare("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    outcome = prepared.run(prepared.own_budget)
    assert prepared.status == "" and not outcome.ok and "could not be built now" in outcome.text
    assert [name for name, _ in bench.box.context.memory.digests.calls] == ["estimate"]


class _Odd:
    """A source that gives what the script says, whatever its form."""

    def __init__(self, cost: object, result: object) -> None:
        self._cost, self._result = cost, result
        self.digests = 0

    def estimate(self, conversation_id: str, **_days: object) -> object:
        return self._cost

    def digest(self, conversation_id: str, **_rest: object) -> object:
        self.digests += 1
        return self._result


@pytest.mark.parametrize(
    ("cost", "result"),
    [
        ("free", {"periods": [], "periods_failed": 0}),
        ({"periods": 1, "periods_cached": 0, "usd": "a lot"}, {"periods": [], "periods_failed": 0}),
        ({"periods": 1, "periods_cached": 0, "usd": float("nan")}, {"periods": [], "periods_failed": 0}),
        ({"periods": 1, "periods_cached": 0, "usd": -1.0}, {"periods": [], "periods_failed": 0}),
        ({"periods": 1, "periods_cached": 0, "usd": 0.1}, "a story"),
        ({"periods": 1, "periods_cached": 0, "usd": 0.1}, {"periods": "many", "periods_failed": 0}),
    ],
)
def test_an_answer_of_the_port_in_the_wrong_form_is_a_failure(cost: object, result: object) -> None:
    bench = MemoryBench(digests=_Odd(cost, result))
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert not outcome.ok and "could not be built now" in outcome.text


def test_a_period_in_the_wrong_form_counts_as_failed_and_the_rest_is_shown() -> None:
    good = maya_story()[1]
    rows = [
        "not a period",
        {**good, "first_day": "July"},
        {**good, "label": "ok", "moments": ["not a moment", {"title": 3}, good["moments"][0]], "messages": "x"},
    ]
    source = _Odd({"periods": 3, "periods_cached": 3, "usd": 0}, {"periods": rows, "periods_failed": 1})
    bench = MemoryBench(digests=source)
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert outcome.ok and "1 period, 0 messages" in outcome.text
    assert "[3 periods could not be built and are missing.]" in outcome.text
    assert "== ok · 2023-07-11 to 2023-07-11 · 0 messages ==" in outcome.text
    assert outcome.text.count("- 2023-07-11") == 1 and len(bench.registry) == 1


def test_a_conversation_with_nothing_to_digest_says_so_and_builds_nothing() -> None:
    bench = bench_for()
    outcome = bench.run("conversation_digest", conversation=bench.alias(JONAS_PHONE))
    assert outcome.ok and outcome.text == "Digest of c1 · Jonas Weber · direct: nothing to show."
    assert [name for name, _ in bench.box.context.memory.digests.calls] == ["estimate"]


def test_bad_arguments_are_refused_before_the_port_is_asked() -> None:
    bench = bench_for()
    alias = bench.alias(MAYA_PHONE)
    assert "conversation needs an alias" in bench.run("conversation_digest").text
    assert not bench.run("conversation_digest", conversation="c99").ok
    assert not bench.run("conversation_digest", conversation=alias, since="last year").ok
    assert not bench.run("conversation_digest", conversation=alias, since="2024-01-01", until="2023-01-01").ok
    assert bench.box.context.memory.digests.calls == []


def test_the_same_digest_is_asked_one_time_in_a_question() -> None:
    bench = bench_for()
    alias = bench.alias(MAYA_PHONE)
    first = bench.prepare("conversation_digest", conversation=alias)
    first.run(first.own_budget)
    again = bench.prepare("conversation_digest", conversation=f"[{alias.upper()}]", since=None, until="")
    assert again.repeated and again.status == ""
    other_days = bench.prepare("conversation_digest", conversation=alias, since="2023-01-01")
    assert not other_days.repeated
    assert [name for name, _ in bench.box.context.memory.digests.calls] == ["estimate", "digest", "estimate"]


def test_one_question_reads_at_most_three_conversations_whole() -> None:
    people = [MAYA_PHONE, MAYA_EMAIL, JONAS_PHONE, INES_PHONE]
    digests = FakeDigests({person: maya_story()[2:] for person in people})
    bench = MemoryBench(digests=digests)
    outcomes = [bench.run("conversation_digest", conversation=bench.alias(person)) for person in people]
    assert DIGESTS_PER_QUESTION == 3
    assert [outcome.ok for outcome in outcomes] == [True, True, True, False]
    assert outcomes[3].text == (
        "This question already read 3 conversations whole. Answer from them, or read around a moment."
    )
    assert [name for name, _ in digests.calls].count("digest") == 3


@pytest.mark.parametrize("honour_cancel", [True, False])
def test_a_cancel_during_the_digest_stops_the_tool_before_it_reads_the_archive(honour_cancel: bool) -> None:
    bench = bench_for(honour_cancel=honour_cancel)
    bench.box.context.memory.digests._during = bench.cancel.cancel
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert not outcome.ok and outcome.text == ""
    assert bench.read("message") == [] and len(bench.registry) == 0


def test_a_cancel_before_the_digest_builds_nothing() -> None:
    bench = bench_for()
    prepared = bench.prepare("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    bench.cancel.cancel()
    assert prepared.run(prepared.own_budget).text == ""
    assert [name for name, _ in bench.box.context.memory.digests.calls] == ["estimate"]


def test_no_raw_handle_of_a_summary_or_a_moment_reaches_the_result_or_the_status() -> None:
    story = [
        period(
            "Dana, +1 (720) 555-0199", "2025-01-15", "2025-01-16", 3,
            "Dana from the gym writes from 720-555-0199 and from rk.unsaved@example.org.",
            [
                moment(
                    "2025-01-16", "A second line: (720) 555-0199", "Mail goes to rk.unsaved@example.org.",
                    "my other line is (720) 555-0199 too", ["u3"],
                )
            ],
        )
    ]
    digests = FakeDigests({UNSAVED_PHONE: story}, usd=0.2)
    bench = MemoryBench(digests=digests)
    prepared = bench.prepare("conversation_digest", conversation="0199")
    assert not prepared.status  # "0199" is no alias and no exact name: the call is refused
    alias = bench.alias(UNSAVED_PHONE)
    prepared = bench.prepare("conversation_digest", conversation=alias)
    outcome = prepared.run(prepared.own_budget)
    seen = "\n".join([prepared.status, outcome.text, outcome.summary])
    assert prepared.status == "Reading your whole conversation with a number ending in 0199"
    assert "Dana from the gym writes from a number ending in 0199 and from an unsaved email contact." in seen
    assert "A second line: a number ending in 0199: Mail goes to an unsaved email contact." in seen
    assert 'Quote: "my other line is a number ending in 0199 too"' in seen
    for raw in [*RAW_HANDLES, UNSAVED_PHONE, UNSAVED_EMAIL]:
        assert raw not in seen, raw


def _long_story(periods: int, moments: int) -> list:
    rows = []
    for index in range(periods):
        day = f"20{10 + index:02d}-01-01"
        notes = [
            moment(day, f"Moment {index}.{n}", "Something happened that day. " * 3, "", ["a2"] if n == 0 else ["a4"])
            for n in range(moments)
        ]
        rows.append(period(f"Year {index}", day, day, 100, f"Summary {index}: " + "they talk a lot. " * 40, notes))
    return rows


def test_a_digest_that_does_not_fit_drops_weak_moments_and_keeps_every_period() -> None:
    story = _long_story(periods=6, moments=4)
    bench = bench_for(story, digest_tokens=2400)
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    text = outcome.text
    assert estimate_tokens(text) <= 2400
    assert all(f"== Year {index} ·" in text for index in range(6))
    assert all(f"Summary {index}: " + "they talk a lot. " * 39 in text for index in range(6))
    kept = [sum(1 for n in range(4) if f"Moment {index}.{n}:" in text) for index in range(6)]
    assert 0 < sum(kept) < 24 and max(kept) - min(kept) <= 1
    assert all(f"Moment {index}.0:" in text for index in range(6) if kept[index])  # the strongest stays
    dropped = 24 - sum(kept)
    assert text.splitlines()[-1] == (
        f"[Cut to fit: {dropped} moments are not shown. Narrow the days with since and until to read more.]"
    )
    assert shown_refs(text) == {f"m{n}" for n in range(1, len(bench.registry) + 1)}


def test_summaries_that_do_not_fit_are_cut_by_the_same_share() -> None:
    story = _long_story(periods=6, moments=2)
    bench = bench_for(story, digest_tokens=700)
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    text = outcome.text
    assert estimate_tokens(text) <= 700
    assert all(f"== Year {index} ·" in text for index in range(6))
    lengths = [len(line) for line in text.splitlines() if line.startswith("Summary ")]
    assert len(lengths) == 6 and max(lengths) - min(lengths) <= 6 and all(60 < size < 400 for size in lengths)
    assert text.count(" [cut]") == 6 and "Moment" not in text and len(bench.registry) == 0
    assert "[Cut to fit: each summary keeps about " in text.splitlines()[-1]
    assert "% of its text, and 12 moments are not shown." in text.splitlines()[-1]


@pytest.mark.parametrize("budget", [40, 120, 300, 900, 5000])
def test_a_digest_never_counts_more_tokens_than_its_budget(budget: int) -> None:
    bench = bench_for(_long_story(periods=8, moments=3), digest_tokens=budget)
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert estimate_tokens(outcome.text) <= budget
    assert shown_refs(outcome.text) == {f"m{n}" for n in range(1, len(bench.registry) + 1)}


def test_a_digest_that_does_not_fit_after_the_masking_leaves_no_message_registered() -> None:
    bench = bench_for(labeler=GrowingLabeler(), digest_tokens=600)
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert not outcome.ok and outcome.text.startswith("The result does not fit.")
    assert len(bench.registry) == 0
    assert [name for name, _ in bench.box.context.memory.digests.calls] == ["estimate", "digest"]
