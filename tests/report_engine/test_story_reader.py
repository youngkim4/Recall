"""The copy of the server's story steps, and what the server shows for a story.

The first part runs only where `safe_html` can be imported. It compares each
copied step with the real one, through the public names of that module.

The second part states which stories the server renders, and which ones it
shows as plain paragraphs because they ask for too much work.
"""

from typing import Any

import pytest

from report_engine.markdown import MAX_SECTION_CHARS, render_markdown

from . import story_reader
from .story_reader import (
    headings_of,
    plain,
    prepared,
    story_of,
    story_page,
    story_work,
    work_share,
)
from .test_contract_markdown import HOSTILE_BODIES, HOSTILE_LINKS, TITLES, with_bodies
from .test_section_cap import HOSTILE

PROSE = [
    "You wrote first on most days, and Mina answered within the hour.",
    'In March the tone changed: the messages got shorter, and "see you soon" became rare.',
    "It wasn't a fight; it was a slow drift that neither of you named.",
    "By June you were planning the trip again, with lists, dates and a lot of jokes!",
]
# a quote mark or an apostrophe in almost every clause
DIALOGUE = [
    '"I\'m here," you wrote; "so am I," Mina wrote back.',
    "\"Don't go,\" she said, and you didn't, and that's what changed it.",
]
WORD_LIMIT_CHARS = 3_000
RAW_STORIES = [
    '# A heading\n\nsee [the plan](https://example.com/x "a title") now\n\n> a quoted line',
    "one\n\n\n\n\n\ntwo " + "!" * 40 + " three " + "\\" * 40 + "[four](https://example.com)",
    "- first\n- second\n- third\n\n1. one\n2. two\n\n| a | b |\n|---|---|\n| 1 | 2 |",
    "word " * 20_000,
    "Title\n=====\n\nIt's \"quoted\" & <b>bold</b>, isn't it?\n\n   \n\nend",
    "[]" * 40_000,
]


def written(body: str) -> str:
    """The story of a report whose four sections hold `body`."""
    return story_of(render_markdown(with_bodies(body)))


def prose(chars: int, sentences: list[str], per_paragraph: int = 0) -> str:
    """Invented prose of `chars` characters, in paragraphs or as one paragraph."""
    count = chars // min(len(sentence) for sentence in sentences) + 1
    parts = [sentences[index % len(sentences)] for index in range(count)]
    if not per_paragraph:
        return " ".join(parts)[:chars].rstrip()
    groups = [" ".join(parts[at : at + per_paragraph]) for at in range(0, count, per_paragraph)]
    return "\n\n".join(groups)[:chars].rstrip()


SAMPLES = [*RAW_STORIES, *(written(body) for body in HOSTILE_BODIES[:8] + HOSTILE_LINKS[:8])]


def real() -> Any:
    return pytest.importorskip("safe_html")


def test_the_copied_limits_are_the_limits_of_the_server() -> None:
    """A change of a limit in the server fails here. Then read `story_html`
    again, bring `story_reader.py` to it, and check the cap of a section."""
    safe_html = real()
    assert safe_html.STORY_LIMIT == story_reader.COPIED_STORY_LIMIT
    assert safe_html.STORY_WORK_LIMIT == story_reader.COPIED_STORY_WORK_LIMIT
    assert list(safe_html.STORY_EXTENSIONS) == story_reader.STORY_EXTENSIONS


@pytest.mark.parametrize("text", SAMPLES)
def test_each_copied_step_gives_what_the_server_gives(text: str) -> None:
    safe_html = real()
    assert prepared(text) == safe_html.prepared(text)
    assert story_work(prepared(text)) == safe_html.story_work(safe_html.prepared(text))


@pytest.mark.parametrize("text", SAMPLES)
def test_the_copied_page_agrees_with_the_page_of_the_server(text: str) -> None:
    safe_html = real()
    page = safe_html.story_html(text)
    if work_share(text) > 1:
        assert page == plain(text) == story_page(text)
        return
    ours, theirs = headings_of(story_page(text)), headings_of(page)
    assert ours.found == theirs.found
    assert ours.tags.count("a") == theirs.tags.count("a")


def test_a_report_within_the_word_limits_of_the_prompt_is_rendered() -> None:
    """The prompt asks for 450 words at most in a section. Such a story asks
    for less than half of the work limit, also when each section is one
    paragraph with a quote mark in almost every clause."""
    for sentences in (PROSE, DIALOGUE):
        story = written(prose(WORD_LIMIT_CHARS, sentences))
        assert work_share(story) < 0.5
        assert headings_of(story_page(story)).found == [(3, 0, title) for title in TITLES]


def test_sections_in_paragraphs_are_rendered_up_to_the_cap() -> None:
    for sentences in (PROSE, DIALOGUE):
        story = written(prose(MAX_SECTION_CHARS, sentences, per_paragraph=4))
        assert work_share(story) < 0.25
        assert headings_of(story_page(story)).found == [(3, 0, title) for title in TITLES]


def test_one_long_paragraph_of_plain_prose_in_each_section_is_rendered() -> None:
    story = written(prose(MAX_SECTION_CHARS, PROSE))
    assert work_share(story) < 0.5
    assert len(headings_of(story_page(story)).found) == len(TITLES)


def test_one_long_paragraph_full_of_quote_marks_is_shown_as_plain_paragraphs() -> None:
    """The server reads a paragraph again for each mark in it. Past its work
    limit it does not render: the page has no chapter, the titles of the
    sections are lines of text, and the marks show as they are written."""
    story = written(prose(MAX_SECTION_CHARS, DIALOGUE))
    page = story_page(story)
    assert work_share(story) > 1
    assert page == plain(story)
    assert set(headings_of(page).tags) == {"p"}
    assert "<p>### The Arc\n" in page


def test_the_plain_page_shows_a_guard_as_it_is_written() -> None:
    story = written("[]" * (MAX_SECTION_CHARS // 4))
    page = story_page(story)
    assert page == plain(story)
    assert "\\[\\]" in page


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_hostile_text_is_rendered_or_plain_and_never_active(name: str) -> None:
    story = written(HOSTILE[name])
    found = headings_of(story_page(story))
    assert "a" not in found.tags and "img" not in found.tags
    assert found.found in ([], [(3, 0, title) for title in TITLES])
