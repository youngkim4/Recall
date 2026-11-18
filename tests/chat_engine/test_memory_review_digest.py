"""Findings of the review of the digest tool: scope of a moment, cancel, masking, budget, repeats."""

from typing import Any

import pytest

from chat_engine.digest_fit import MIN_DIGEST_TOKENS
from chat_engine.engine import EngineConfig
from chat_engine.fake_archive import FakeArchive
from chat_engine.fake_digests import moment, period
from chat_engine.handle_mask import mask_every_handle
from chat_engine.render import estimate_tokens
from chat_engine.scope import Scope
from tests.chat_engine.memory_bench import MemoryBench, bench_for, shown_refs
from tests.chat_engine.tiny import (
    CONVERSATIONS,
    INES_PHONE,
    JONAS_PHONE,
    MAYA_EMAIL,
    MAYA_PHONE,
    tiny_labeler,
    tiny_messages,
)

ONE_DAY = Scope(conversations=(MAYA_PHONE,), since="2021-03-04", until="2021-03-04")


# --- 1: a moment about messages outside the scope ---------------------------------


def test_a_moment_with_a_message_the_scope_refuses_is_dropped_and_counted() -> None:
    story = [
        period(
            "2021", "2021-03-04", "2021-03-04", 4, "A lease.",
            [
                moment(
                    "2021-03-04", "Rent goes up", "Fifty more a month (said on 2023-07-11)", "fifty more a month",
                    ["a7"],
                ),
                moment("2021-03-04", "Signed", "It is signed, and later it costs more.", "", ["a2", "a7"]),
                moment("2021-03-04", "Congrats", "You cheer.", "", ["a3"]),
                moment("2021-03-04", "Unknown", "The archive holds none of its messages.", "", ["gone"]),
            ],
        )
    ]
    bench = bench_for(story, scope=ONE_DAY)
    text = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE)).text
    for outside in ("Rent goes up", "Fifty more", "2023-07-11", "Signed", "costs more"):
        assert outside not in text, outside
    assert "- 2021-03-04 · Congrats: You cheer." in text
    assert "- 2021-03-04 · Unknown: The archive holds none of its messages." in text
    assert "[Outside the scope and not shown: 2 moments.]" in text
    assert [m["id"] for m in bench.registry.messages()] == ["a3"]


def test_a_moment_with_a_message_of_a_conversation_outside_the_scope_is_dropped() -> None:
    story = [
        period(
            "2021", "2021-03-04", "2021-03-04", 4, "A lease.",
            [
                moment("2021-03-04", "Bread", "Jonas bakes.", "", ["a2", "j1"]),
                moment("2021-03-04", "Scan", "The scan comes by mail.", "", ["a2", "b1"]),
            ],
        )
    ]
    both = bench_for(story, scope=Scope(conversations=(MAYA_PHONE, MAYA_EMAIL)))
    text = both.run("conversation_digest", conversation=both.alias(MAYA_PHONE)).text
    assert "Bread" not in text and "[Outside the scope and not shown: 1 moment.]" in text
    assert "- 2021-03-04 · Scan: The scan comes by mail." in text  # b1 is inside the scope: only its line goes
    assert [m["id"] for m in both.registry.messages()] == ["a2"]


def test_periods_and_moments_outside_the_scope_share_one_note() -> None:
    story = [
        period("2021", "2021-03-04", "2021-03-04", 4, "A lease.", [moment("2023-07-11", "Later", "Rent.", "", [])]),
        period("2023", "2023-07-11", "2023-07-11", 3, "Rent goes up."),
    ]
    bench = bench_for(story, scope=ONE_DAY)
    text = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE)).text
    assert "[Outside the scope and not shown: 1 period, 1 moment.]" in text and "Rent" not in text


# --- 2: nothing is read from the archive after a cancel -----------------------------


class _CancellingArchive(FakeArchive):
    """An archive that sees a cancel arrive while it gives one message."""

    cancel: Any = None

    def message(self, message_id: str) -> Any:
        found = super().message(message_id)
        self.cancel()
        return found


def test_a_cancel_while_the_messages_of_the_moments_are_read_stops_the_reading() -> None:
    archive = _CancellingArchive(tiny_messages(), CONVERSATIONS)
    bench = bench_for(archive=archive)
    archive.cancel = bench.cancel.cancel
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert not outcome.ok and outcome.text == ""
    assert len(bench.read("message")) == 1 and len(bench.registry) == 0


@pytest.mark.parametrize("story", [None, [period("2021", "2021-03-04", "2021-03-04", 4, "A lease.")]])
def test_a_cancel_before_a_digest_is_written_again_reads_nothing(story: Any) -> None:
    bench = bench_for(story)
    prepared = bench.prepare("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert prepared.run(prepared.own_budget).ok
    bench.cancel.cancel()
    before = len(bench.archive.calls)
    again = prepared.run(prepared.own_budget)
    assert again.text == "" and not again.ok and len(bench.archive.calls) == before


# --- 3: every handle of a digest text is masked -----------------------------------


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        ("call +14155550123 now", "call Maya Okafor now"),
        ("call (415) 555-0123 now", "call Maya Okafor now"),
        ("call 415.555.0123 now", "call Maya Okafor now"),
        ("mail maya.okafor@example.org now", "mail Maya Okafor now"),
        ("a stranger at +1 303 555 0100 wrote", "a stranger at a phone number wrote"),
        ("a stranger at x@y.com wrote", "a stranger at an email address wrote"),
        ("Dana is +1 (720) 555-0199", "Dana is a number ending in 0199"),
        ("write rk.unsaved@example.org", "write an unsaved email contact"),
        ("close in 2016-2019", "close in 2016-2019"),
        ("moved on 2016-06-01", "moved on 2016-06-01"),
        ("from 2016 to 2019", "from 2016 to 2019"),
        ("sent 2021-03-04 18:22 and 3/14/2024", "sent 2021-03-04 18:22 and 3/14/2024"),
        ("they talk 9:30-10:15 most days", "they talk 9:30-10:15 most days"),
        ("at 18:22 4155550123 called", "at 18:22 Maya Okafor called"),
        ("two: 303-555-0100 / 415-555-0123", "two: a phone number / Maya Okafor"),
        ("ext: 415-555-0123, 22", "ext: a phone number"),
        ("on 2021-03-04, 4155550123 called", "on 2021-03-04, Maya Okafor called"),
        ("", ""),
    ],
)
def test_every_phone_and_address_in_a_text_is_masked_and_a_date_is_not(text: str, masked: str) -> None:
    labeler = tiny_labeler()
    labeler.sender("+17205550199")
    labeler.sender("rk.unsaved@example.org")
    assert mask_every_handle(text, labeler) == masked


def test_a_labeler_with_no_list_of_saved_names_is_asked_for_the_sender() -> None:
    class Plain:
        def conversation(self, conversation: Any) -> str:
            return "a chat"

        def sender(self, handle: str) -> str:
            return "Maya Okafor" if "0123" in handle else "a number ending in 0100"

        def redact(self, text: str) -> str:
            return text

    assert mask_every_handle("415-555-0123 or 303-555-0100", Plain()) == "Maya Okafor or a phone number"


HANDLES = (
    "Maya (+14155550123, 415-555-0123, (415) 555-0123, maya.okafor@example.org) and a stranger "
    "(+1 303 555 0100, x@y.com) in 2016-2019, on 2016-06-01, from 2016 to 2019"
)
CLEAN = (
    "Maya (Maya Okafor, Maya Okafor, Maya Okafor, Maya Okafor) and a stranger "
    "(a phone number, an email address) in 2016-2019, on 2016-06-01, from 2016 to 2019"
)


def test_a_digest_masks_the_handles_of_saved_people_and_of_strangers() -> None:
    story = [
        period(
            f"Label {HANDLES}"[:40], "2021-03-04", "2021-03-04", 3, f"Summary: {HANDLES}",
            [moment("2021-03-04", f"Title {HANDLES}", f"Detail: {HANDLES}", "", ["a2"])],
        )
    ]
    bench = bench_for(story)
    text = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE)).text
    assert f"Summary: {CLEAN}" in text
    assert f"- 2021-03-04 · Title {CLEAN}: Detail: {CLEAN}" in text
    assert "== Label Maya (Maya Okafor, Maya Okafor, · 2021-03-04" in text
    for raw in ("4155550123", "415-555", "(415)", "maya.okafor@", "303 555", "x@y.com"):
        assert raw not in text, raw


def test_a_quote_that_the_masking_of_a_digest_changes_is_not_shown() -> None:
    rows = [*tiny_messages()]
    rows[1] = {**rows[1], "text": "my number is 415-555-0123, the lease got signed"}
    story = [
        period(
            "2021", "2021-03-04", "2021-03-04", 4, "A lease.",
            [
                moment("2021-03-04", "Number", "She gives it.", "my number is 415-555-0123", ["a2"]),
                moment("2021-03-04", "Signed", "It is signed.", "the lease got signed", ["a2"]),
            ],
        )
    ]
    bench = bench_for(story, archive=FakeArchive(rows, CONVERSATIONS))
    text = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE)).text
    assert "- 2021-03-04 · Number: She gives it.\n" in text
    assert '- 2021-03-04 · Signed: It is signed. Quote: "the lease got signed"' in text


# --- 4: a small budget and very many periods ----------------------------------------


def test_the_engine_refuses_a_digest_budget_below_the_floor() -> None:
    assert MIN_DIGEST_TOKENS == 600
    assert EngineConfig(model="m", max_output_tokens=100, digest_tokens=MIN_DIGEST_TOKENS).digest_tokens == 600
    with pytest.raises(ValueError, match="digest_tokens must be at least 600"):
        EngineConfig(model="m", max_output_tokens=100, digest_tokens=MIN_DIGEST_TOKENS - 1)


def _many(periods: int) -> list:
    notes = [moment("2021-03-04", f"T{n}", "d" * 50, "", ["a2", "a3"]) for n in range(5)]
    return [period(f"Y{n}", "2021-03-04", "2021-03-04", 3, "summary " * 20, notes) for n in range(periods)]


@pytest.mark.parametrize(("periods", "budget"), [(20, 600), (200, 1500), (1000, 24_000), (20, 100), (20, 10)])
def test_periods_that_do_not_fit_are_counted_in_the_header_and_in_one_line(periods: int, budget: int) -> None:
    bench = bench_for(_many(periods), digest_tokens=budget)
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    text = outcome.text
    shown = sum(1 for line in text.splitlines() if line.startswith("== "))
    assert estimate_tokens(text) <= budget and 0 <= shown < periods
    if budget >= 100:
        assert f": {shown} of {periods:,} periods shown, {shown * 3:,} messages in them." in text.splitlines()[0]
        assert f"[{periods - shown:,} periods not shown. Narrow the days with since and until.]" in text
        assert outcome.summary.endswith(f": {shown} of {periods:,} periods")
    if budget >= MIN_DIGEST_TOKENS:
        assert shown >= 4
        assert text.count("\nsummary summary") == shown  # every period that is shown has the start of its summary
        assert [f"== Y{n} ·" in text for n in range(shown)] == [True] * shown  # the first periods, in order
    assert shown_refs(text) == {f"m{n}" for n in range(1, len(bench.registry) + 1)}


def test_a_digest_that_fits_keeps_its_plain_header() -> None:
    bench = bench_for(_many(3))
    text = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE)).text
    assert ": 3 periods, 9 messages. A summary is not a quote" in text.splitlines()[0]
    assert "not shown" not in text


def test_a_long_label_is_cut() -> None:
    story = [period("Spring " * 40, "2021-03-04", "2021-03-04", 3, "A lease.")]
    bench = bench_for(story)
    text = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE)).text
    assert f"== {('Spring ' * 40)[:60].rstrip()}... · 2021-03-04 to 2021-03-04 · 3 messages ==" in text


# --- 7: the repeat of a digest that showed nothing, and the cost line ---------------


def test_a_repeat_after_a_failure_says_what_happened_and_asks_no_source() -> None:
    for options in ({"error": RuntimeError("down")}, {"estimate_error": RuntimeError("down")}):
        bench = bench_for(**options)
        alias = bench.alias(MAYA_PHONE)
        bench.run("conversation_digest", conversation=alias)
        calls = len(bench.box.context.memory.digests.calls)
        again = bench.prepare("conversation_digest", conversation=alias, since=None)
        outcome = again.run(again.own_budget)
        assert not again.repeated and not outcome.ok
        assert outcome.text == (
            "The digest of c1 · Maya Okafor · direct could not be built earlier in this question. "
            "It is not tried again. Go on with the other tools."
        )
        assert "moment" not in outcome.text and len(bench.box.context.memory.digests.calls) == calls


def test_a_repeat_after_a_cost_refusal_asks_for_the_cost_again() -> None:
    bench = bench_for(usd=2.5)
    alias = bench.alias(MAYA_PHONE)
    first = bench.run("conversation_digest", conversation=alias)
    bench.box.context.memory.digests._usd = 0.5  # a narrower digest was built in between: now it costs less
    again = bench.prepare("conversation_digest", conversation=alias)
    assert "$2.50" in first.text and not again.repeated
    assert again.status == "Reading your whole conversation with Maya Okafor"
    assert again.run(again.own_budget).ok


def test_a_repeat_of_a_digest_that_was_shown_is_a_repeat_and_names_no_moment() -> None:
    bench = bench_for()
    alias = bench.alias(MAYA_PHONE)
    bench.run("conversation_digest", conversation=alias)
    again = bench.prepare("conversation_digest", conversation=alias)
    assert again.repeated and again.answers_repeats
    assert again.run(again.own_budget).text == "This digest is above already. Use what it shows, or answer."
    empty = bench.prepare("conversation_digest", conversation=bench.alias(JONAS_PHONE))
    assert empty.run(empty.own_budget).ok
    assert bench.prepare("conversation_digest", conversation=bench.alias(JONAS_PHONE)).repeated


class _Costs:
    def __init__(self, periods: int, cached: int) -> None:
        self._cost = {"periods": periods, "periods_cached": cached, "usd": 2.5}

    def estimate(self, conversation_id: str, **_days: object) -> dict:
        return dict(self._cost)

    def digest(self, conversation_id: str, **_rest: object) -> dict:
        raise AssertionError("a digest above the limit is not built")


@pytest.mark.parametrize(
    ("periods", "cached", "words"),
    [
        (5, 2, " would cost about $2.50: 3 periods are not built yet. That is above"),
        (5, 4, " would cost about $2.50: 1 period is not built yet. That is above"),
        (5, 5, " would cost about $2.50. That is above"),
        (2, 5, " would cost about $2.50. That is above"),
        (0, 0, " would cost about $2.50. That is above"),
    ],
)
def test_the_cost_line_never_contradicts_the_estimate(periods: int, cached: int, words: str) -> None:
    bench = MemoryBench(digests=_Costs(periods, cached))
    outcome = bench.run("conversation_digest", conversation=bench.alias(INES_PHONE))
    assert not outcome.ok and words in outcome.text
    assert outcome.text.endswith("Narrow the days with since and until, or use the other tools.")
