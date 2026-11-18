"""The budget of a conversation digest: what is cut, what stays registered, what the masking adds."""

import pytest

from chat_engine.fake_digests import FakeDigests, moment, period
from chat_engine.render import Limits, estimate_tokens
from tests.chat_engine.memory_bench import GrowingLabeler, MemoryBench, bench_for, shown_refs
from tests.chat_engine.tiny import MAYA_PHONE, UNSAVED_PHONE


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


def test_a_tokenizer_that_counts_a_text_differently_later_leaves_no_message_without_its_line() -> None:
    seen: list[str] = []

    def count(text: str) -> int:
        if text.startswith("- "):
            seen.append(text)
        late = text.startswith("- ") and len(seen) > 3  # the three moments were counted for the plan
        return estimate_tokens(text) * (500 if late else 1)

    bench = bench_for(limits=Limits(count_tokens=count), digest_tokens=2000)
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert "The landlord raises the rent." in outcome.text and "Rent goes up" not in outcome.text
    assert outcome.text.splitlines()[-1].startswith("[Cut to fit: 3 moments are not shown.")
    assert len(bench.registry) == 0 and shown_refs(outcome.text) == set()


def test_a_summary_is_masked_before_it_is_measured_so_the_masks_do_not_break_the_budget() -> None:
    story = [period("2025", "2025-01-15", "2025-01-16", 3, "call 720-555-0199 " * 100)]
    bench = MemoryBench(digests=FakeDigests({UNSAVED_PHONE: story}), digest_tokens=800)
    outcome = bench.run("conversation_digest", conversation=bench.alias(UNSAVED_PHONE))
    assert estimate_tokens("call 720-555-0199 " * 100) < 800 < estimate_tokens("call a number ending in 0199 " * 100)
    assert outcome.ok and estimate_tokens(outcome.text) <= 800
    assert outcome.text.count("call a number ending in 0199") > 50 and " [cut]" in outcome.text
    assert "720" not in outcome.text
