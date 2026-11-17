"""The text of a report against the cuts that the story renderer makes.

`story_html` (commit ab20b5e) cuts a story before it reads it: a run of one
mark to 16, a run of empty lines to two, the whole to 200,000 characters. The
renderer under it then takes U+0002 and U+0003 out. Each of these can take a
backslash from the bracket that it guards, or bring a `#` to the start of a
line. The engine writes text that none of them changes.
"""

import unicodedata

import pytest

from report_engine.markdown import clean_section, render_markdown
from report_engine.models import SECTION_KEYS, SECTION_TITLES, Moment, Section

from .story_reader import headings_of, markdown_section, story_html_steps, story_of, tame
from .test_contract_markdown import BACKSLASH, HOSTILE_BODIES, HOSTILE_LINKS, guarded_link
from .test_models import report as make_report

TITLES = [SECTION_TITLES[key] for key in SECTION_KEYS]
STX, ETX = "\x02", "\x03"
LAUGH = "\U0001f602"
ACCENT = "́"
MARKS = [BACKSLASH, *"[]#!*-=><()`~|+.:\"'&/", LAUGH, ACCENT]
RUN_LENGTHS = [15, 16, 17, 18, 40, 1_000]
CONTROL_CODES = [code for code in range(0xA0) if unicodedata.category(chr(code)) == "Cc"]


def story_with(body: str) -> str:
    """The story of a report whose first section holds `body`."""
    sections = tuple(
        Section(
            key=key,
            title=SECTION_TITLES[key],
            markdown=clean_section(body if key == SECTION_KEYS[0] else "Plain words."),
        )
        for key in SECTION_KEYS
    )
    return story_of(render_markdown(make_report(sections=sections)))


def run_bodies(mark: str, length: int) -> list[str]:
    run = mark * length
    return [
        f"Before {run} after.",
        f"{run} at the start of a line",
        f"at the end of a line {run}",
        f"# {run}",
        f"# {run} words {run}",
        f"see {run}[the plan{run}](https://attacker.example/x) now",
        f"{run}<b>{run}</b>{run}",
        f"- {run}\n> {run}\n1. {run}",
    ]


@pytest.mark.parametrize("mark", MARKS)
def test_the_renderer_cuts_nothing_out_of_a_story(mark: str) -> None:
    """A cut inside a run of backslashes takes the guard off the bracket after
    it. So no run of one mark in a story is long enough to be cut."""
    for length in RUN_LENGTHS:
        for body in run_bodies(mark, length):
            case = (length, body[:48])
            cleaned = clean_section(f"Before.\n\n{body}\n\nAfter.")
            assert tame(cleaned) == cleaned, case
            assert clean_section(cleaned) == cleaned, case
            story = story_with(f"Before.\n\n{body}\n\nAfter.")
            assert tame(story) == story, case
            page = headings_of(story_html_steps(story))
            assert "a" not in page.tags and "img" not in page.tags, case
            assert page.found == [(3, 0, title) for title in TITLES], case


@pytest.mark.parametrize("slashes", range(41))
def test_a_run_of_backslashes_of_any_length_frees_no_bracket(slashes: int) -> None:
    run = BACKSLASH * slashes
    bodies = [
        guarded_link(slashes),
        f"see {run}[the plan](https://attacker.example/x) now",
        f"see [the plan{run}](https://attacker.example/x) now",
        f"see !{run}[a picture{run}](https://attacker.example/p.png) now",
        f"{run}[ref{run}] and more\n\n{run}[ref{run}]: https://attacker.example/x",
    ]
    for body in bodies:
        cleaned = clean_section(body)
        tags = headings_of(story_html_steps(cleaned)).tags
        assert "a" not in tags and "img" not in tags, body
        assert tame(cleaned) == cleaned, body
        assert clean_section(cleaned) == cleaned, body


@pytest.mark.parametrize("body", HOSTILE_BODIES + HOSTILE_LINKS)
def test_the_renderer_changes_no_hostile_story_before_it_reads_it(body: str) -> None:
    story = story_with(f"Before.\n\n{body}\n\nAfter.")
    assert tame(story) == story
    assert STX not in story and ETX not in story


@pytest.mark.parametrize("empty", range(9))
def test_the_renderer_cuts_no_run_of_empty_lines(empty: int) -> None:
    gap = "\n" * (empty + 1)
    bodies = [
        f"one{gap}two",
        f"one{gap}====={gap}-----{gap}two",
        f"one{gap}[a]: https://attacker.example/x{gap}two",
        f"one{gap}  \t {gap}two",
    ]
    for body in bodies:
        cleaned = clean_section(body)
        assert tame(cleaned) == cleaned, body
        assert clean_section(cleaned) == cleaned, body
        assert cleaned.startswith("one") and cleaned.endswith("two"), body


def test_a_paragraph_break_is_kept() -> None:
    written = "one\n\ntwo\n\n\nthree\n\n\n\n\n\nfour"
    assert clean_section(written) == "one\n\ntwo\n\n\nthree\n\n\nfour"


@pytest.mark.parametrize("code", CONTROL_CODES)
def test_no_control_character_is_left_in_section_text(code: int) -> None:
    """The renderer takes U+0002 and U+0003 out of a text before it reads it.
    What stood behind one is then at the start of a line, or next to a
    backslash. A tab and a line break are the only control characters kept."""
    char = chr(code)
    body = (
        f"one{char}two\n"
        f"{char}# Injected\n"
        f"Title\n{char}=====\n"
        f"see {BACKSLASH}{char}[the plan{BACKSLASH}{char}](https://attacker.example/x) now"
    )
    cleaned = clean_section(body)
    assert {c for c in cleaned if unicodedata.category(c) == "Cc"} <= {"\t", "\n"}
    page = headings_of(story_html_steps(cleaned))
    assert page.found == []
    assert "a" not in page.tags
    assert clean_section(cleaned) == cleaned
    assert cleaned.startswith("one") and cleaned.endswith("now")


def test_text_around_a_removed_character_is_kept() -> None:
    assert clean_section(f"a{STX}b{ETX}c \x00d\x7fe") == "abc de"


def hostile_inline(slashes: int) -> list[str]:
    run = BACKSLASH * slashes
    return [
        guarded_link(slashes),
        f"{run}# not a heading",
        f"{STX}{run}[a{ETX}{run}](https://attacker.example/x)",
        f"{'!' * 40} {run} {'#' * 40} {'[' * 40}",
        f"{run}{'"' * 15}",
        f"{'"' * 15} words {'"' * 15}",
    ]


@pytest.mark.parametrize("slashes", [0, 1, 7, 8, 9, 15, 16, 17, 33, 400])
def test_moment_text_and_warnings_keep_their_guards(slashes: int) -> None:
    """The server reads the moments from another file today. The text of the
    Key Events and of a warning is safe all the same when a page renders it."""
    for text in hostile_inline(slashes):
        moment = Moment(
            date="2024-06-03",
            title=text,
            detail=text,
            category="milestone",
            score=0.7,
            quote=text,
            message_ids=("11",),
            quote_verified=True,
            quote_message_id="11",
            quote_speaker="Mina",
        )
        written = render_markdown(make_report(moments=(moment,), warnings=(text,), model=text))
        levels = {"Overview": [], "Key Events": [3]}
        for heading, expected in levels.items():
            section = markdown_section(written, heading)
            assert tame(section) == section, (heading, text[:48])
            assert STX not in section and ETX not in section
            page = headings_of(story_html_steps(section))
            assert "a" not in page.tags and "img" not in page.tags, (heading, text[:48])
            assert [level for level, _depth, _words in page.found] == expected


def test_a_short_run_reads_as_it_was_written() -> None:
    """Fifteen of one mark is more than writing uses. Nothing shorter changes."""
    line = f"So{'!' * 15} and {LAUGH * 15} and {'.' * 15} and {BACKSLASH * 15}x"
    assert clean_section(line) == line
    assert clean_section(f"wait{'.' * 300} what{'?' * 300}") == f"wait{'.' * 15} what{'?' * 15}"


def test_a_table_keeps_its_shape_when_its_rule_is_long() -> None:
    body = f"| Month | Messages |\n|{'-' * 40}|{'-' * 40}|\n| June | 120 |"
    tags = headings_of(story_html_steps(clean_section(body))).tags
    assert tags.count("table") == 1 and tags.count("td") == 2
