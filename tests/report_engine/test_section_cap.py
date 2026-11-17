"""A section has a cap on its length, and the story of a report fits the server.

The server (commit 3563549) drops what is past `STORY_LIMIT` characters of a
story. A cut there can leave a line of `=` under a line of text, which is a
heading. The cap counts a section as the report holds it, after the cleaning,
so the story of four sections is always far shorter than that limit.
"""

from typing import Any

import pytest

from report_engine.errors import ReduceError
from report_engine.fake_llm import FakeCall, FakeLLM, default_reply
from report_engine.markdown import (
    LONGEST_WORD_CHARS,
    MAX_TITLE_CHARS,
    SECTION_CUT_MARK,
    _cut_at_word,
    clean_section,
    fit_section,
    render_markdown,
)
from report_engine.models import SECTION_KEYS, SECTION_TITLES, Section
from report_engine.reduce_step import MAX_SECTION_CHARS
from report_engine.transcript import ELISION_MARK

from .story_reader import STORY_LIMIT, headings_of, story_html_steps, story_of, tame
from .support import SMALL, run
from .test_models import report as make_report

TITLES = [SECTION_TITLES[key] for key in SECTION_KEYS]
# a section in a story: `### `, a title that the guards can make twice as
# long, a line break, the text, an empty line
SECTION_ROOM = MAX_SECTION_CHARS + 2 * MAX_TITLE_CHARS + 8
STORY_BOUND = len(SECTION_KEYS) * SECTION_ROOM
INJECT = "Injected heading\n====="


def repeated(unit: str) -> str:
    """One piece again and again, to six times the cap."""
    return unit * (6 * MAX_SECTION_CHARS // len(unit))


WORST = {
    "one bracket": repeated("["),
    "bracket pairs": repeated("[]"),
    "bracketed words": repeated("[a]"),
    "guarded brackets": repeated("\\["),
    "heading lines": repeated("# a\n"),
    "heading lines of brackets": repeated("# [\n"),
    "hash pairs": repeated("## "),
    "hash lines": repeated("#a\n"),
    "quoted links": repeated("> [x](a)\n"),
    "quote marks": repeated("'a"),
    "emphasis": repeated("*a* "),
    "tags": repeated("<b>a</b> "),
    "one long word": "a" * (2 * MAX_SECTION_CHARS) + " and the end",
    "backslashes only": repeated("\\" * 15 + " "),
}
# text that the cleaning makes longer, in paragraphs that the server reads fast
GROWTH = {
    "grown brackets": "\n\n".join(["[" * 11] * 2_000),
    "grown hash lines": "\n\n".join(["\n".join(["#a"] * 24)] * 600),
    "grown bold lines": "\n\n".join(["\n".join(["# a"] * 17)] * 600),
}
HOSTILE = {**WORST, **GROWTH}


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
    assert arc.endswith(f"word {SECTION_CUT_MARK}")
    assert len(arc) <= MAX_SECTION_CHARS
    assert report.sections[1].markdown == "Fake dynamics text."
    assert report.status == "complete"
    limit = f"{MAX_SECTION_CHARS:,} characters"
    assert cap_warnings(report) == [f"1 section was cut to {limit}. The report has only its start."]


def test_the_report_counts_the_sections_that_were_cut() -> None:
    long = "word " * MAX_SECTION_CHARS
    report = run(SMALL, llm=with_sections(arc=long, dynamics="short", big_picture=long))
    assert [len(section.markdown) > 1_000 for section in report.sections] == [
        True,
        False,
        False,
        True,
    ]
    limit = f"{MAX_SECTION_CHARS:,} characters"
    assert cap_warnings(report) == [
        f"2 sections were cut to {limit} each. The report has only the start of each."
    ]
    assert "- Warning: 2 sections were cut to" in render_markdown(report)


def test_the_cap_counts_the_text_as_the_report_holds_it() -> None:
    """The cleaning puts a backslash before each bracket. A text that is under
    the cap when the model writes it can be over it in the report."""
    text = "[]" * (MAX_SECTION_CHARS // 2 - 100)
    report = run(SMALL, llm=with_sections(arc=text))
    arc = report.sections[0].markdown
    assert len(text) < MAX_SECTION_CHARS < len(clean_section(text))
    assert len(arc) <= MAX_SECTION_CHARS
    assert arc.endswith(SECTION_CUT_MARK)
    # the warning is true for this text too: it does not say how long the reply was
    assert cap_warnings(report) == [
        f"1 section was cut to {MAX_SECTION_CHARS:,} characters. The report has only its start."
    ]


def test_a_long_text_that_is_short_after_the_cleaning_is_kept_whole() -> None:
    text = "<b></b>" * MAX_SECTION_CHARS + "The words that stay."
    report = run(SMALL, llm=with_sections(arc=text))
    assert report.sections[0].markdown == "The words that stay."
    assert cap_warnings(report) == []


def test_a_section_with_no_text_after_the_cleaning_is_an_error_at_any_length() -> None:
    """A long reply of markup only is as empty as a short one."""
    empties = [
        "<b></b>" * 10,
        "<b></b>" * (3 * MAX_SECTION_CHARS),
        "\x00" * (3 * MAX_SECTION_CHARS),
    ]
    for text in empties:
        with pytest.raises(ReduceError, match="section 'arc' is empty"):
            run(SMALL, llm=with_sections(arc=text))


def test_a_word_longer_than_any_real_word_is_cut_inside() -> None:
    kept, cut = fit_section(WORST["one long word"])
    room = MAX_SECTION_CHARS - len(SECTION_CUT_MARK) - 1
    assert cut
    assert kept == f"{'a' * room} {SECTION_CUT_MARK}"


def test_a_cut_goes_back_to_the_start_of_a_word_of_usual_length_only() -> None:
    """The cut falls 63 characters into a word: the word goes. One character
    more, and the word is longer than any real one: it is cut inside."""
    text = "aa " + "w" * 100
    near = len("aa ") + LONGEST_WORD_CHARS
    assert _cut_at_word(text, near - 1) == "aa"
    assert _cut_at_word(text, near) == "aa " + "w" * LONGEST_WORD_CHARS


def test_a_cut_takes_backslashes_and_white_space_off_the_end() -> None:
    assert _cut_at_word("word \\ \\ tail", 8) == "word"
    assert _cut_at_word("word\n\\\n\\\ttail", 8) == "word"
    assert _cut_at_word("word  \t tail", 8) == "word"


def test_the_cut_of_a_section_does_not_fall_inside_a_word() -> None:
    lead = "start " * ((MAX_SECTION_CHARS - 60) // 6)
    report = run(SMALL, llm=with_sections(arc=lead + "x" * 500 + " tail" * 2_000))
    arc = report.sections[0].markdown
    assert arc.endswith(f"start {SECTION_CUT_MARK}")
    assert "x" not in arc and "tail" not in arc


def test_the_cut_mark_of_a_section_reads_as_it_is_written() -> None:
    report = run(SMALL, llm=with_sections(arc="word " * MAX_SECTION_CHARS))
    page = story_html_steps(story_of(render_markdown(report)))
    assert f"word {ELISION_MARK}</p>" in page
    assert "a" not in headings_of(page).tags
    assert clean_section(ELISION_MARK) == SECTION_CUT_MARK


def test_white_space_around_a_section_does_not_count() -> None:
    text = "\n\n  " + exactly(MAX_SECTION_CHARS) + "  \n\n"
    report = run(SMALL, llm=with_sections(arc=text))
    assert report.sections[0].markdown == text.strip()
    assert cap_warnings(report) == []


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_a_cut_section_is_clean_text_within_the_cap(name: str) -> None:
    text = HOSTILE[name]
    kept, cut = fit_section(text)
    assert len(kept) <= MAX_SECTION_CHARS
    assert not kept.endswith("\\")
    assert clean_section(kept) == kept
    assert fit_section(kept) == (kept, False)
    assert cut == (len(clean_section(text)) > MAX_SECTION_CHARS)
    if cut:
        assert kept.endswith(SECTION_CUT_MARK)


@pytest.mark.parametrize("backslashes", range(6))
def test_a_cut_leaves_no_backslash_without_its_mark(backslashes: int) -> None:
    """A cut can fall between a backslash and the bracket that it guards."""
    for lead in range(MAX_SECTION_CHARS - 40, MAX_SECTION_CHARS - 4, 3):
        text = "a" * lead + " " + "\\" * backslashes + "[b] " + "word " * 40
        kept, cut = fit_section(text)
        assert cut and len(kept) <= MAX_SECTION_CHARS
        assert kept.endswith(f" {SECTION_CUT_MARK}")
        assert not kept[: -len(SECTION_CUT_MARK)].rstrip().endswith("\\")
        assert clean_section(kept) == kept
        assert "a" not in headings_of(story_html_steps(kept)).tags


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_the_story_fits_the_server_with_room(name: str) -> None:
    """Past its limit the server drops the end of a story. So the story of a
    report is never longer than three quarters of that limit."""
    text = HOSTILE[name]
    report = run(SMALL, llm=with_sections(**dict.fromkeys(SECTION_KEYS, text)))
    story = story_of(render_markdown(report))
    assert all(len(section.markdown) <= MAX_SECTION_CHARS for section in report.sections)
    assert len(story) <= STORY_BOUND
    assert tame(story) == story


def test_the_bound_of_a_story_leaves_a_quarter_of_the_server_limit_free() -> None:
    """`STORY_LIMIT` is the number of the real server where its module can be
    imported. When that number falls, this test fails and the cap must follow."""
    assert STORY_BOUND <= STORY_LIMIT * 3 // 4


@pytest.mark.parametrize("form", sorted(GROWTH))
def test_no_heading_can_be_put_where_the_server_cuts_a_story(form: str) -> None:
    """Found by the review of 01f5d17. Three sections grew in the cleaning
    until the story was longer than the server's limit. The fourth one put a
    line of `=` under a line of text where the server cut, and the page
    showed that text as a heading."""
    last = SECTION_KEYS[-1]
    for filler in (10, 2_000, MAX_SECTION_CHARS - 200):
        tail = "a" * filler + "\n\n" + INJECT + "x" * (MAX_SECTION_CHARS - filler - 30)
        texts = {**dict.fromkeys(SECTION_KEYS[:3], GROWTH[form]), last: tail}
        report = run(SMALL, llm=with_sections(**texts))
        story = story_of(render_markdown(report))
        assert len(story) <= STORY_BOUND
        assert story[:STORY_LIMIT] == story
        found = headings_of(story_html_steps(story)).found
        assert found == [(3, 0, title) for title in TITLES]


def test_a_long_section_of_backslashes_keeps_its_start() -> None:
    """A cut takes the backslashes off the end of the text. When that leaves
    nothing, the text stays as it is: the report has the start of the section,
    as its warning says."""
    for unit in ("\\ ", "\\\t", "\\\n"):
        kept, cut = fit_section(unit * 5_000)
        assert cut
        assert MAX_SECTION_CHARS // 2 < len(kept) <= MAX_SECTION_CHARS
        assert kept.startswith("\\") and kept.endswith(f" {SECTION_CUT_MARK}")
        assert clean_section(kept) == kept
        assert fit_section(kept) == (kept, False)


@pytest.mark.parametrize(
    "title",
    [
        "T" * 100_000,
        "word " * 20_000,
        "[x](https://attacker.example) " * 5_000,
        "\\" * 50_000,
        "# " * 50_000,
        "title\n\n# Injected\n=====\n" * 5_000,
    ],
)
def test_a_long_title_made_by_hand_is_cut_when_it_is_written(title: str) -> None:
    """`Section.title` has no limit. A long one pushed the story past the limit
    of the server, where a cut after the `-` of a list line made a heading."""
    body = "ab word\n- item\n\n" * 470
    sections = tuple(Section(key=key, title=title, markdown=body) for key in SECTION_KEYS)
    story = story_of(render_markdown(make_report(sections=sections)))
    assert len(story) <= STORY_BOUND
    assert tame(story) == story
    found = headings_of(story_html_steps(story))
    assert [level for level, _depth, _words in found.found] == [3] * len(SECTION_KEYS)
    assert "a" not in found.tags


def test_a_title_of_the_engine_is_written_whole() -> None:
    story = story_of(render_markdown(make_report()))
    assert [line for line in story.splitlines() if line.startswith("### ")] == [
        f"### {title}" for title in TITLES
    ]
    assert all(len(title) <= MAX_TITLE_CHARS for title in TITLES)


def test_a_report_made_by_hand_is_cut_when_it_is_written() -> None:
    """`render_markdown` takes any report. Its story fits the server too."""
    sections = tuple(
        Section(key=key, title=SECTION_TITLES[key], markdown="[word] " * 20_000)
        for key in SECTION_KEYS
    )
    story = story_of(render_markdown(make_report(sections=sections)))
    assert len(story) <= STORY_BOUND
    assert story.count(SECTION_CUT_MARK) == len(SECTION_KEYS)
    assert tame(story) == story
