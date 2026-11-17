"""The story seam on the joined branch, with the real `safe_html`.

The report engine writes the story of a report. `story_html` renders it, and
the Reports page cuts the page into chapters at its headings. The engine
promises three things, and each one has tests here:

1. An ordinary report gives exactly four `h3` headings, no other heading and
   no link, and it is rendered: it is not the plain fall-back.
2. The story of the worst texts of the engine's own tests is not longer than
   `safe_html.STORY_LIMIT`, so the server's cut does not apply.
3. Hostile text at the engine's cuts gives no heading but the four, no link
   and no image.

`story_html` shows a story as plain paragraphs when its estimate of the work is
past `STORY_WORK_LIMIT`, or when the render uses more than `STORY_SECONDS` of
processor time. Plain paragraphs have no heading at all and nothing active.
For the heavy hostile cases both outcomes keep the promise. The fragments at a
cut and the hand-made titles render in milliseconds, so there the test asks
for the four headings.

`story_cases.py` holds the text and says how the sample was chosen.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import safe_html
from report_engine.markdown import MAX_SECTION_CHARS
from report_engine.models import SECTION_KEYS, SECTION_TITLES
from tests.integration import story_cases as cases
from tests.report_engine.story_reader import headings_of
from tests.report_engine.test_section_cap import WORST

LIMIT = safe_html.STORY_LIMIT
FOUR_TITLES = tuple((3, 0, SECTION_TITLES[key]) for key in SECTION_KEYS)
FOUR_PLACES = ((3, 0),) * len(SECTION_KEYS)
ORDINARY = {
    "the words of the fake model": {},
    "prose of 3,000 characters a section": {
        key: cases.prose(3_000, seed) for seed, key in enumerate(SECTION_KEYS)
    },
    "prose at the cap of a section": {
        key: cases.prose(MAX_SECTION_CHARS, 9 + seed) for seed, key in enumerate(SECTION_KEYS)
    },
}


@dataclass(frozen=True)
class Page:
    """What the server shows for a story."""

    html: str
    plain: bool
    headings: tuple[tuple[int, int, str], ...]
    tags: frozenset[str]


def page_of(story: str) -> Page:
    html = safe_html.story_html(story)
    found = headings_of(html)
    return Page(
        html=html,
        plain=html == safe_html._plain(story),
        headings=tuple(found.found),
        tags=frozenset(found.tags),
    )


def work_share(story: str) -> float:
    """The server's estimate of the work of a story, as a part of its limit."""
    return safe_html.story_work(safe_html.prepared(story)) / safe_html.STORY_WORK_LIMIT


def faults(page: Page, *, by_hand: bool = False, plain_is_fine: bool = False) -> list[str]:
    """What breaks the promise: a heading that is not one of the four, a link, an image.

    `by_hand`: the titles are hostile text, so only the level and the place of each heading
    are known. `plain_is_fine`: plain paragraphs with no heading keep the promise too."""
    found = [f"a tag {tag}" for tag in ("a", "img") if tag in page.tags]
    if page.plain and plain_is_fine:
        return found + (["a heading in plain paragraphs"] if page.headings else [])
    if page.plain:
        return [*found, "the plain fall-back"]
    if by_hand and tuple(heading[:2] for heading in page.headings) != FOUR_PLACES:
        return [*found, f"headings {page.headings!r}"]
    if not by_hand and page.headings != FOUR_TITLES:
        return [*found, f"headings {page.headings!r}"]
    return found


@pytest.mark.parametrize("name", sorted(ORDINARY))
def test_an_ordinary_report_is_four_chapters_and_nothing_active(name: str) -> None:
    story, report = cases.story_with(**ORDINARY[name])
    page = page_of(story)
    assert report.warnings == ()
    assert not page.plain
    assert page.headings == FOUR_TITLES
    assert not page.tags & {"a", "img", "h1", "h2", "h4", "h5", "h6"}
    assert len(story) <= LIMIT and work_share(story) < 1
    if ORDINARY[name]:
        # the story was rendered: its bold words are bold
        assert "strong" in page.tags


@pytest.mark.parametrize("name", sorted(WORST))
def test_the_story_of_a_worst_text_fits_the_server(name: str) -> None:
    story, report = cases.story_of_one_text(WORST[name])
    assert all(len(section.markdown) <= MAX_SECTION_CHARS for section in report.sections)
    assert len(story) <= LIMIT
    # the server cuts nothing from it: not its end, not a run of one mark, not empty lines
    assert safe_html._tamed(story) == story
    assert faults(page_of(story), plain_is_fine=True) == []


@pytest.mark.parametrize("form", sorted(cases.GROWTH))
def test_a_crafted_story_reaches_no_cut_of_the_server_and_makes_no_heading(form: str) -> None:
    """Three sections grow in the cleaning, and the fourth one holds a line of `=` under a
    line of text. Before the cap of a section counted the cleaned text, such a story was
    longer than the server's limit, and the server's cut made that line a heading."""
    # the lead that would put the line at the server's cut is longer than a section can be
    assert cases.lead_to_reach(LIMIT, form) > MAX_SECTION_CHARS
    for lead in cases.CRAFTED_FILLERS:
        story = cases.crafted_story(form, lead)
        page = page_of(story)
        assert cases.INJECT in story and len(story) <= LIMIT
        assert faults(page, plain_is_fine=True) == [], lead
        assert not [words for _level, _depth, words in page.headings if "Injected" in words]


@pytest.mark.parametrize(
    ("first", "name"), list(enumerate(cases.ALL_CUT_FRAGMENTS)), ids=list(cases.ALL_CUT_FRAGMENTS)
)
def test_a_fragment_at_the_cut_of_a_section_is_not_active(first: int, name: str) -> None:
    fragment = cases.ALL_CUT_FRAGMENTS[name]
    places = cases.sampled_places(fragment, first)
    found: list[tuple[int, str]] = []
    for back in places:
        story, report = cases.story_with(arc=cases.cut_section(fragment, back))
        # every one of these sections is over the cap, so the engine cut it
        assert [warning for warning in report.warnings if "section was cut" in warning], back
        found += [(back, fault) for fault in faults(page_of(story))]
    # the sample goes from the cap to a place where the whole fragment is kept
    assert places[0] < cases.STEP and places[-1] == cases.span_of(fragment) - 1
    assert found == []


@pytest.mark.parametrize("name", sorted(cases.TITLE_FRAGMENTS))
def test_a_title_made_by_hand_is_not_active(name: str) -> None:
    found: list[tuple[int, str]] = []
    for lead in cases.TITLE_LEADS:
        title = cases.title_with(cases.TITLE_FRAGMENTS[name], lead)
        story = cases.story_with_titles([title] * len(SECTION_KEYS))
        found += [(lead, fault) for fault in faults(page_of(story), by_hand=True)]
    assert found == []


@pytest.mark.parametrize("name", sorted(cases.LONG_TITLES))
def test_a_long_title_made_by_hand_is_cut_before_the_server_reads_it(name: str) -> None:
    titles = [cases.LONG_TITLES[name]] * len(SECTION_KEYS)
    story = cases.story_with_titles(titles, cases.FULL_SECTION)
    assert len(story) <= LIMIT
    assert faults(page_of(story), by_hand=True, plain_is_fine=True) == []


@pytest.mark.parametrize("name", sorted(cases.BACKSLASH_SECTIONS))
def test_a_section_of_backslashes_is_not_active(name: str) -> None:
    story, _report = cases.story_of_one_text(cases.BACKSLASH_SECTIONS[name])
    assert len(story) <= LIMIT
    assert faults(page_of(story), plain_is_fine=True) == []


def test_dialogue_in_one_long_paragraph_is_shown_as_plain_paragraphs() -> None:
    """The one way that ordinary words reach the plain fall-back: each section is one long
    paragraph with many quote marks and apostrophes. The estimate of the work decides it,
    not a clock, so the answer is the same on every machine. The page then has no chapter:
    the four titles are lines of text."""
    rendered = {
        "3,000 characters in one paragraph": cases.dialogue(3_000),
        "6,000 characters in one paragraph": cases.dialogue(6_000),
        "the cap in paragraphs of 800": cases.dialogue(MAX_SECTION_CHARS, 800),
    }
    for label, text in rendered.items():
        story, _report = cases.story_of_one_text(text)
        assert work_share(story) < 1, label
        assert faults(page_of(story)) == [], label

    story, report = cases.story_of_one_text(cases.dialogue(MAX_SECTION_CHARS))
    page = page_of(story)
    assert report.warnings == ()
    assert work_share(story) > 1
    assert page.plain and page.headings == () and page.tags == {"p"}
    assert all(f"<p>### {SECTION_TITLES[key]}\n" in page.html for key in SECTION_KEYS)
