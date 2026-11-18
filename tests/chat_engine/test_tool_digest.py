"""conversation_digest: the story of one conversation, with the messages that back its moments."""

import logging

import pytest

from chat_engine.fake_digests import FakeDigests, moment, period
from chat_engine.scope import Scope
from chat_engine.tool_digest import DIGESTS_PER_QUESTION
from tests.chat_engine.memory_bench import RAW_HANDLES, MemoryBench, bench_for, maya_story
from tests.chat_engine.test_tools import Bench
from tests.chat_engine.tiny import (
    INES_PHONE,
    JONAS_PHONE,
    MAYA_EMAIL,
    MAYA_PHONE,
    UNSAVED_EMAIL,
    UNSAVED_PHONE,
)


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


def test_a_quote_is_shown_only_when_a_shown_message_of_its_moment_holds_it() -> None:
    story = [
        period(
            "2021", "2021-03-04", "2021-03-04", 4, "A lease.",
            [
                moment("2021-03-04", "Signed", "It is signed.", "nobody can check this", ["gone"]),
                moment("2021-03-04", "Congrats", "You cheer.", "the lease got signed btw", ["a3"]),
                moment("2021-03-04", "The move", "A date.", "We move in on the  FIRST of April", ["a4"]),
            ],
        )
    ]
    bench = bench_for(story)
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    lines = [line for line in outcome.text.splitlines() if line.startswith("- ")]
    assert lines == [
        "- 2021-03-04 · Signed: It is signed.",  # no message of the moment is shown
        "- 2021-03-04 · Congrats: You cheer.",  # the message of the moment does not hold the quote
        '- 2021-03-04 · The move: A date. Quote: "We move in on the FIRST of April"',  # held: case and spaces
    ]
    assert [m["id"] for m in bench.registry.messages()] == ["a3", "a4"]


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


def test_the_quote_rule_of_the_digest_is_the_rule_of_the_answer_check() -> None:
    from chat_engine.verify import is_quote_of, unverified_quotes

    sources = ["yes!! the lease got signed btw", "we move in on the first of April"]
    for quote in ("the lease got signed btw", "We move in on the FIRST of April", "yes!! ... signed btw", "short"):
        assert is_quote_of(quote, sources) and unverified_quotes(f"“{quote}”", sources) == []
    for quote in ("the lease never got signed", "the lease got signed ... first of April"):
        assert not is_quote_of(quote, sources) and unverified_quotes(f"“{quote}”", sources) == [quote]


def test_a_moment_with_no_day_is_shown_only_when_the_scope_has_no_days() -> None:
    row = {
        "label": "2023", "first_day": "2023-07-11", "last_day": "2023-07-11", "messages": 3, "summary": "Rent.",
        "moments": [{"date": "in July", "title": "Rent goes up", "detail": "", "quote": "", "message_ids": ["a7"]}],
    }
    nameless = {key: value for key, value in row.items() if key != "label"}
    cost = {"periods": 2, "periods_cached": 2, "usd": 0}
    result = {"periods": [row, nameless], "periods_failed": 0}
    free = MemoryBench(digests=_Odd(cost, result))
    text = free.run("conversation_digest", conversation=free.alias(MAYA_PHONE)).text
    assert "- Rent goes up\n2023-07-11\nm1 09:05 Maya Okafor: fifty more a month" in text
    assert "[1 period could not be built and is missing.]" in text
    dated = MemoryBench(digests=_Odd(cost, result), scope=Scope(since="2023-01-01"))
    text = dated.run("conversation_digest", conversation=dated.alias(MAYA_PHONE)).text
    assert "Rent goes up" not in text and "Rent." in text and len(dated.registry) == 0
