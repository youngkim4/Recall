"""A section has a cap on its length.

The story renderer (commit ab20b5e) cuts a story at 200,000 characters, and
it reads a paragraph more slowly for each guarded mark in it. A story of four
capped sections stays far below that length and is read in good time.
"""

import time
from typing import Any

import pytest

from report_engine.fake_llm import FakeCall, FakeLLM, default_reply
from report_engine.markdown import render_markdown
from report_engine.models import SECTION_KEYS
from report_engine.reduce_step import MAX_SECTION_CHARS

from .story_reader import STORY_LIMIT, headings_of, story_html_steps, story_of, tame
from .support import SMALL, run

TITLE_ROOM = 100
WORST = {
    "one bracket": "[" * 1_000_000,
    "bracket pairs": "[]" * 500_000,
    "bracketed words": "[a]" * 400_000,
    "guarded brackets": "\\[" * 500_000,
    "heading lines": "# a\n" * 250_000,
    "heading lines of brackets": "# [\n" * 250_000,
    "hash pairs": "## " * 400_000,
    "quoted links": "> [x](a)\n" * 100_000,
    "quote marks": "'a" * 500_000,
    "emphasis": "*a* " * 250_000,
}


def with_sections(**texts: str) -> FakeLLM:
    def responder(call: FakeCall) -> Any:
        return {**default_reply(call), **texts} if call.kind == "reduce" else None

    return FakeLLM(responder)


def cap_warnings(report: Any) -> list[str]:
    return [warning for warning in report.warnings if "section" in warning]


def exactly(length: int) -> str:
    words = "a" + " word" * ((length - 1) // 5)
    return words + "x" * (length - len(words))


def test_a_section_at_the_cap_is_kept_whole() -> None:
    text = exactly(MAX_SECTION_CHARS)
    report = run(SMALL, llm=with_sections(arc=text))
    assert len(text) == MAX_SECTION_CHARS
    assert report.sections[0].markdown == text
    assert cap_warnings(report) == []


def test_a_section_over_the_cap_keeps_its_start_and_the_report_says_so() -> None:
    text = exactly(MAX_SECTION_CHARS + 1)
    report = run(SMALL, llm=with_sections(arc=text))
    arc = report.sections[0].markdown
    assert arc.startswith("a word word")
    assert arc.endswith("word \\[...\\]")
    assert len(arc.replace("\\", "")) <= MAX_SECTION_CHARS
    assert report.sections[1].markdown == "Fake dynamics text."
    assert report.status == "complete"
    assert cap_warnings(report) == [
        f"1 section was longer than {MAX_SECTION_CHARS:,} characters. "
        "The report has only its start."
    ]


def test_the_report_counts_the_sections_that_were_cut() -> None:
    long = "word " * MAX_SECTION_CHARS
    report = run(SMALL, llm=with_sections(arc=long, dynamics="short", big_picture=long))
    assert [len(section.markdown) > 1_000 for section in report.sections] == [
        True,
        False,
        False,
        True,
    ]
    assert cap_warnings(report) == [
        f"2 sections were longer than {MAX_SECTION_CHARS:,} characters. "
        "The report has only the start of each."
    ]
    assert "- Warning: 2 sections were longer than" in render_markdown(report)


def test_the_cut_of_a_section_does_not_fall_inside_a_word() -> None:
    text = "start " * 1_990 + "x" * 500 + " tail" * 2_000
    report = run(SMALL, llm=with_sections(arc=text))
    arc = report.sections[0].markdown
    assert arc.endswith("start \\[...\\]")
    assert "x" not in arc and "tail" not in arc


def test_the_cut_mark_of_a_section_reads_as_it_is_written() -> None:
    report = run(SMALL, llm=with_sections(arc="word " * MAX_SECTION_CHARS))
    page = story_html_steps(story_of(render_markdown(report)))
    assert "word [...]</p>" in page
    assert "a" not in headings_of(page).tags


def test_white_space_around_a_section_does_not_count() -> None:
    text = "\n\n  " + exactly(MAX_SECTION_CHARS) + "  \n\n"
    report = run(SMALL, llm=with_sections(arc=text))
    assert report.sections[0].markdown == text.strip()
    assert cap_warnings(report) == []


@pytest.mark.parametrize("name", sorted(WORST))
def test_the_story_stays_far_below_the_length_at_which_the_renderer_cuts_it(name: str) -> None:
    """Past its limit the renderer drops the end of a story. A cut there can
    leave a line of `=` under a line of text, which is a heading."""
    report = run(SMALL, llm=with_sections(**dict.fromkeys(SECTION_KEYS, WORST[name])))
    story = story_of(render_markdown(report))
    assert len(story) <= len(SECTION_KEYS) * 2 * MAX_SECTION_CHARS + TITLE_ROOM
    assert len(story) < STORY_LIMIT // 2
    assert tame(story) == story


def test_the_renderer_reads_the_worst_story_in_good_time() -> None:
    """Each backslash that guards a mark costs the renderer one more pass over
    its paragraph. With no cap this story takes the renderer about half a
    minute; with the cap it takes about a second."""
    report = run(SMALL, llm=with_sections(**dict.fromkeys(SECTION_KEYS, WORST["bracket pairs"])))
    story = story_of(render_markdown(report))
    started = time.perf_counter()
    page = story_html_steps(story)
    assert time.perf_counter() - started < 15.0
    assert "a" not in headings_of(page).tags
