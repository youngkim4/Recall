"""The report file against the code that reads it in the redesign (commit ab20b5e).

The readers are copies of `report_contact`, `markdown_section` and
`report_scope` in `ui_server.py`, and of the steps of `story_html` in
`safe_html.py`, as CONTRACTS.md at that commit names them. They are in
`story_reader.py`. When `safe_html` can be imported, the real `story_html`
is used too.
"""

import time
from typing import Any

import markdown as markdown_lib
import pytest

from report_engine.errors import StoreError
from report_engine.markdown import clean_section, render_markdown
from report_engine.models import SECTION_KEYS, SECTION_TITLES, Scope, Section

from .story_reader import (
    STORY_EXTENSIONS,
    headings_of,
    markdown_section,
    report_contact,
    stated_scope,
    story_html_steps,
    story_of,
)
from .test_models import report as make_report

TITLES = [SECTION_TITLES[key] for key in SECTION_KEYS]
BACKSLASH = "\\"


def guarded_link(slashes: int) -> str:
    """A link with a run of backslashes before each of its brackets."""
    run = BACKSLASH * slashes
    return f"see {run}[the plan{run}](https://attacker.example/x) now"


def with_bodies(body: str, **fields: Any) -> Any:
    sections = tuple(
        Section(key=key, title=SECTION_TITLES[key], markdown=clean_section(body))
        for key in SECTION_KEYS
    )
    return make_report(sections=sections, **fields)


def test_the_reader_of_the_heading_gives_no_name_for_an_empty_file() -> None:
    assert report_contact("") == ""


@pytest.mark.parametrize(
    "conversation",
    [
        "+12025550143",
        "theo@example.com",
        "chat000000000000000042",
        "odd#name[1]@example.com",
        "back\\slash@example.com",
    ],
)
def test_the_heading_gives_the_conversation_id_back_as_it_is(conversation: str) -> None:
    text = render_markdown(make_report(conversation=conversation))
    assert report_contact(text) == conversation
    assert text.splitlines()[0] == f"# Conversation Analysis - {conversation}"


def test_a_conversation_id_stays_on_the_first_line() -> None:
    text = render_markdown(make_report(conversation="first\n# second\nthird"))
    assert text.splitlines()[0] == "# Conversation Analysis - first # second third"
    assert text.splitlines()[1] == ""


@pytest.mark.parametrize(
    "scope, stated",
    [
        (Scope(since="2024-06-01", until="2024-06-30"), "2024-06-01 to 2024-06-30"),
        (Scope(since="2024-06-01"), "from 2024-06-01"),
        (Scope(until="2024-06-30"), "through 2024-06-30"),
        (Scope(), "all dates"),
    ],
)
def test_the_reader_of_the_scope_finds_the_line_that_was_written(scope: Scope, stated: str) -> None:
    text = render_markdown(
        make_report(scope=scope, warnings=("- Scope: from 1999-01-01", "x\n- Scope: all dates"))
    )
    assert stated_scope(text) == stated


def test_the_story_is_everything_after_the_summary_marker_and_nothing_else() -> None:
    text = render_markdown(with_bodies("One line.\n\nplain ## Summary and ## Overview inside"))
    story = story_of(text)
    assert story.startswith(f"### {TITLES[0]}\nOne line.")
    assert [line[4:] for line in story.splitlines() if line.startswith("### ")] == TITLES
    assert markdown_section(text, "Overview").startswith("- Scope: ")


HOSTILE_BODIES = [
    "#Injected",
    "#Injected\n#Twice",
    "####### seven",
    "# A heading\n## Another",
    "Title\n=====",
    "Title\n-----\nmore\nSecond\n---",
    "Title\n-",
    "Title\n= = =\nTitle\n=-=",
    "text\n#tag at the start of a line",
    " #one space\n   #three spaces",
    "- #in a list\n- # also in a list\n1. #numbered",
    "> #in a quote\n> # also in a quote\n> Title\n> ===",
    "```\n# in a fence\n```",
    "\\#escaped already",
    "&#35; entity\n&num; entity",
    "<h1>raw</h1>\n<h2 onclick=x>raw</h2>",
    "1234567890. # a list number of ten digits",
    "> 1234567890. # the same in a quote",
    "123456789012345678901234567890. # thirty digits",
    "- item\n\n    # in the second block of a list item",
    "1. one\n\n\t# after a tab",
    # the renderer takes U+0002 and U+0003 out before it reads a line
    "\x02# behind a character that the renderer removes",
    "\x03#Injected",
    "\x02\x03\x02 # behind three of them",
    "Title\n\x02=====",
    "Title\n==\x03===",
    "> \x02# in a quote",
]

HOSTILE_LINKS = [
    "see [the plan](https://attacker.example/x) now",
    "see [a [b] c](https://attacker.example/x) now",
    "see [a\\]b](https://attacker.example/x) now",
    "see [a\nb](https://attacker.example/x) now",
    'see [a](https://attacker.example/x "a title") now',
    "see [a](<https://attacker.example/x>) now",
    "click [here]\n\n[here]: https://attacker.example/x",
    "click [here]\n\n[here]:\nhttps://attacker.example/x",
    "click [here]\n\n> [here]: https://attacker.example/x",
    "click [here]\n\n- [here]: https://attacker.example/x",
    "[h] go\n\n>[h]:\n>https://attacker.example",
    "click [][]\n\n[]: https://attacker.example/x",
    "click [here][ref]\n\n[ref]: https://attacker.example/x",
    "![a picture](https://attacker.example/p.png)",
    "![a [b] c](https://attacker.example/p.png)",
    "![pic][ref]\n\n[ref]: https://attacker.example/p.png",
    "<https://attacker.example/x>",
    "<a href='https://attacker.example/x'>go</a>",
    "\\[a](https://attacker.example/x) and \\\\[b](https://attacker.example/y)",
    # the renderer cuts a run of 17 backslashes to 16, which frees the bracket after it
    guarded_link(15),
    guarded_link(16),
    guarded_link(17),
    guarded_link(18),
    guarded_link(33),
    guarded_link(400),
    "!" + guarded_link(16),
    # the renderer takes U+0002 and U+0003 out, which joins what stands around them
    "see \\\x02[the plan\\\x03](https://attacker.example/x) now",
    "see \\\\\x02\\[the plan\\\\\x03\\](https://attacker.example/x) now",
]


@pytest.mark.parametrize("body", HOSTILE_BODIES)
def test_no_section_text_makes_a_heading_in_the_story(body: str) -> None:
    """The page cuts the story into chapters at its headings (readStory). A
    heading that the model or a message could add would move the chapters."""
    text = render_markdown(with_bodies(f"Before.\n\n{body}\n\nAfter."))
    story = story_of(text)
    found = headings_of(story_html_steps(story)).found
    assert found == [(3, 0, title) for title in TITLES]


@pytest.mark.parametrize("body", HOSTILE_BODIES)
def test_stored_section_text_makes_no_heading_when_it_is_rendered_alone(body: str) -> None:
    """A page that reads the JSON report renders `Section.markdown` itself."""
    cleaned = clean_section(f"Before.\n\n{body}\n\nAfter.")
    assert headings_of(story_html_steps(cleaned)).found == []
    assert clean_section(cleaned) == cleaned


@pytest.mark.parametrize("body", HOSTILE_LINKS)
def test_no_section_text_makes_a_link_or_an_image(body: str) -> None:
    """The story is written from the messages of other people. A link in it
    would be a link that one of them chose."""
    cleaned = clean_section(f"Before.\n\n{body}\n\nAfter.")
    tags = headings_of(story_html_steps(cleaned)).tags
    assert "a" not in tags and "img" not in tags
    raw = headings_of(markdown_lib.markdown(cleaned, extensions=STORY_EXTENSIONS)).tags
    assert "a" not in raw and "img" not in raw
    assert clean_section(cleaned) == cleaned
    assert cleaned.startswith("Before.") and cleaned.endswith("After.")


@pytest.mark.parametrize("body", HOSTILE_LINKS)
def test_the_real_story_renderer_shows_no_link(body: str) -> None:
    safe_html = pytest.importorskip("safe_html")
    text = render_markdown(with_bodies(f"Before.\n\n{body}\n\nAfter."))
    story = story_of(text)
    tags = headings_of(safe_html.story_html(story)).tags
    assert "a" not in tags and "img" not in tags


def test_brackets_read_as_they_were_written() -> None:
    cleaned = clean_section("You said [sic] twice, and a[0] once.")
    assert "You said [sic] twice, and a[0] once." in story_html_steps(cleaned)


@pytest.mark.parametrize(
    "line",
    [
        "# a" + " " * 20_000 + "b",
        "# " + "a " * 10_000 + "#" * 10_000,
        "[" * 50_000 + "](x)",
        "[" * 100_000,
        "![" * 30_000 + "](x)" * 30_000,
        "<a" * 50_000,
        "1" * 100_000 + " # x",
        "> " * 30_000 + "# x",
        "-" * 100_000 + " x",
        " " * 100_000 + "=",
    ],
)
def test_no_line_makes_the_cleaning_slow(line: str) -> None:
    """A model can be steered to write such a line. The cleaning must not stop a run."""
    started = time.perf_counter()
    cleaned = clean_section(line)
    assert time.perf_counter() - started < 2.0
    assert clean_section(cleaned) == cleaned


def test_a_conversation_id_that_holds_a_section_marker_is_refused() -> None:
    """The server finds a section with text.find. The id must stay as it is in
    the first line, so an id that holds a marker can not be written."""
    for conversation in ("chat ## Summary", "x ## Overview y"):
        with pytest.raises(StoreError, match="conversation id"):
            render_markdown(make_report(conversation=conversation))
    assert render_markdown(make_report(conversation="a#b##c")).startswith(
        "# Conversation Analysis - a#b##c\n"
    )


def test_lists_quotes_tables_and_emphasis_survive_the_story_renderer() -> None:
    body = (
        "It **began** in *June*.\n\n"
        "- first\n- second\n\n"
        "1. one\n2. two\n\n"
        "> a quoted line\n\n"
        "| Month | Messages |\n|-------|----------|\n| June | 120 |"
    )
    text = render_markdown(with_bodies(body))
    story = story_of(text)
    tags = headings_of(story_html_steps(story)).tags
    for tag in ("strong", "em", "ul", "ol", "blockquote", "table"):
        assert tags.count(tag) == len(SECTION_KEYS), tag


def test_a_hash_that_is_not_at_the_start_of_a_line_reads_as_it_was_written() -> None:
    cleaned = clean_section("You were #1 in June.\n#1 again in July.")
    html = story_html_steps(cleaned)
    assert "You were #1 in June." in html
    assert "#1 again in July." in html
    assert "\\" not in html


@pytest.mark.parametrize("body", HOSTILE_BODIES)
def test_the_real_story_renderer_agrees(body: str) -> None:
    safe_html = pytest.importorskip("safe_html")
    text = render_markdown(with_bodies(f"Before.\n\n{body}\n\nAfter."))
    story = story_of(text)
    found = headings_of(safe_html.story_html(story)).found
    assert found == [(3, 0, title) for title in TITLES]
