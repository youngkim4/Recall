"""Hostile text for the story seam, and the sample of it that the tests render.

The server renders the story of a report with `story_html` in `safe_html.py`,
and the Reports page cuts the result into chapters at its headings. The report
engine promises that the only headings are its four section titles and that no
link gets through.

The review session probed this seam with four scripts (`story_probe.py`,
`cut_probe.py`, `cut_probe_slashes.py`, `title_probe.py`). With the engine at
`22fea4f` and the `safe_html` of the base they rendered 12,239 reports, with
hostile text at every place of the engine's cuts. No report had a heading but
the four, a link or an image.

This module holds the text of those probes. All of it is invented. The tests
render a sample, so that the folder runs in seconds:

- A cut fragment stands before the cap of a section. The full probe moved it
  through every place of `span_of(fragment)`. The sample takes each third
  place where the engine's cut can meet the fragment (from the cap to
  `MEETING_ROOM` places past the end of the fragment), and two places where
  the fragment is whole. The first place differs from fragment to fragment, so
  the fragments together cover every remainder of three.
- A title fragment stands behind every lead of the full probe. A report with
  hand-made titles needs no run, so nothing is left out there.
- The crafted forms, the long titles and the sections of backslashes are the
  cases of the probes, all of them.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from functools import cache

from report_engine import Report, render_markdown
from report_engine.markdown import MAX_SECTION_CHARS, MAX_TITLE_CHARS, fit_section
from report_engine.models import SECTION_KEYS
from tests.report_engine.story_reader import story_of
from tests.report_engine.support import SMALL, run
from tests.report_engine.test_section_cap import with_sections

ADDRESS = "http://example.com/x"
INJECT = "Injected heading\n====="
TAIL = " tail" * 400
STEP = 3
# the cut mark takes 8 characters of the cap, and the guards make a fragment longer
MEETING_ROOM = 24

CUT_FRAGMENTS = {
    "underline, then a space": "\n\nInjected heading\n===== tail",
    "underline, then a word": "\n\nInjected heading\n=====tail",
    "dashes, then a tab": "\n\nInjected heading\n-----\ttail",
    "underline, then a no-break space": "\n\nInjected heading\n=====\u00a0tail",
    "underline, then a guarded bracket": "\n\nInjected heading\n===== \\[x\\] tail",
    "underline with spaces after it": "\n\nInjected heading\n=====   \n\ntail",
    "long underline": "\n\nInjected heading\n==============================x tail",
    "underline in a quote": "\n\n> Injected heading\n> =====x tail",
    "underline in a list": "\n\n- Injected heading\n  =====x tail",
    "underline under bold": "\n\n**Injected heading**\n=====x tail",
    "underline after a line separator": "\n\nInjected heading\u2028=====x tail",
    "underline after a vertical tab": "\n\nInjected heading\x0b=====x tail",
    "underline after a carriage return": "\n\nInjected heading\r\n=====x tail",
    "hash heading": "\n\n# Injected heading\n\ntail",
    "hash heading, no space": "\n\n#Injected heading\n\ntail",
    "closed hash heading": "\n\n## Injected heading ##\n\ntail",
    "hash after two hashes": "\n\n##\\# Injected heading\n\ntail",
    "plain link": f" [Injected]({ADDRESS}) tail",
    "reference link": f" [Injected][1]\n\n[1]: {ADDRESS}\n\ntail",
    "image": f" ![Injected]({ADDRESS}) tail",
    "tag link": f' <a href="{ADDRESS}">Injected</a> tail',
    "angle link": f" <{ADDRESS}> tail",
    "link behind one backslash": f" \\[Injected]({ADDRESS}) tail",
    "link behind two backslashes": f" \\\\[Injected]({ADDRESS}) tail",
    "link behind three backslashes": f" \\\\\\[Injected]({ADDRESS}) tail",
    "link with a guarded address": f" [Injected]\\({ADDRESS}) tail",
    "nested link": f" [[Injected]({ADDRESS})]({ADDRESS}) tail",
    "link over two lines": f" [Injected\nheading]({ADDRESS}) tail",
}
SLASH_FRAGMENTS = {
    **{
        f"link behind {count} backslashes": " " + "\\" * count + f"[Injected]({ADDRESS}) tail"
        for count in (4, 7, 8, 14, 15, 16, 17, 30, 31)
    },
    "link behind runs of fifteen": " " + ("\\" * 15 + " ") * 3 + f"[Injected]({ADDRESS}) tail",
    "underline behind backslashes": "\n\nInjected heading\n" + "\\" * 15 + "\n=====x tail",
    "link in a line of backslash pairs": (
        " " + "\\\\ " * 20 + f"[Injected]({ADDRESS}) " + "\\\\ " * 20 + "tail"
    ),
}
ALL_CUT_FRAGMENTS = {**CUT_FRAGMENTS, **SLASH_FRAGMENTS}

TITLE_FRAGMENTS = {
    "plain link": f"[Injected]({ADDRESS})",
    "link behind one backslash": f"\\[Injected]({ADDRESS})",
    "link behind two backslashes": f"\\\\[Injected]({ADDRESS})",
    "link behind seven backslashes": "\\" * 7 + f"[Injected]({ADDRESS})",
    "link behind sixteen backslashes": "\\" * 16 + f"[Injected]({ADDRESS})",
    "image": f"![Injected]({ADDRESS})",
    "tag link": f'<a href="{ADDRESS}">Injected</a>',
    "line break and underline": "Injected\n=====",
    "line break and hash": "Injected\n# Second heading",
    "closing hashes": "Injected ## Second ##",
    "line separator": "Injected\u2028# Second heading",
    "control character": "Injected\x02# Second heading",
}
TITLE_LEADS = range(MAX_TITLE_CHARS + 6)
LONG_TITLES = {
    "letters": "A" * 100_000,
    "words": "word " * 20_000,
    "brackets": "[" * 100_000,
    "bracket pairs": "[]" * 50_000,
    "backslashes": "\\" * 100_000,
    "hashes": "#" * 100_000,
    "underline marks": "=" * 100_000,
}
FULL_SECTION = "word " * (MAX_SECTION_CHARS // 5 + 500)
BACKSLASH_SECTIONS = {
    "backslashes": "\\" * 30_000,
    "backslashes and spaces": "\\ " * 15_000,
    "backslash lines": "\\\n" * 15_000,
    "runs of fifteen": ("\\" * 15 + " ") * 2_000,
    "runs, then a link": ("\\" * 15 + " ") * 600 + f"[Injected]({ADDRESS})",
    "backslashes, then an underline": "Injected heading\n" + "\\" * 9_000 + "\n=====x",
    "words, then backslashes at the cut": "ab " * 2_650 + "\\" * 200 + f"[Injected]({ADDRESS})",
}
# one paragraph of raw text, and how many of them a section holds: text that the cleaning makes longer
GROWTH = {
    "guarded brackets": ("[" * 11, 466),
    "guarded hash lines": ("\n".join(["#a"] * 24), 164),
    "bold lines": ("\n".join(["# a"] * 17), 173),
}
CRAFTED_FILLERS = (10, 2_000, MAX_SECTION_CHARS - len(INJECT) - 10)
PROSE_WORDS = "river window paper quiet morning coffee station yellow answer follow garden winter".split()
CLAUSE = "she said the plan's fine and it didn't matter"


def story_with(**texts: str) -> tuple[str, Report]:
    """The story of a report whose sections the fake model wrote, and the report."""
    report = run(SMALL, llm=with_sections(**texts))
    return story_of(render_markdown(report)), report


def story_of_one_text(text: str) -> tuple[str, Report]:
    """The story of a report with the same text in all four sections."""
    return story_with(**dict.fromkeys(SECTION_KEYS, text))


def filler(chars: int) -> str:
    return "ab " * (chars // 3) + "x" * (chars % 3)


def span_of(fragment: str) -> int:
    """How many places before the cap the full probe gave a fragment."""
    grown = len(fit_section(fragment + TAIL)[0])
    return max(len(fragment), min(grown, 200)) + 80


def sampled_places(fragment: str, first: int) -> list[int]:
    """The places before the cap where the sample puts a fragment. See the module text."""
    span = span_of(fragment)
    meeting = range(first % STEP, len(fragment) + MEETING_ROOM, STEP)
    return sorted({*meeting, span * 2 // 3, span - 1})


def cut_section(fragment: str, back: int) -> str:
    """A section that is over the cap, with the fragment `back` places before the cap."""
    return filler(MAX_SECTION_CHARS - back) + fragment + TAIL


def title_with(fragment: str, lead: int) -> str:
    return ("ab " * 40)[:lead] + fragment + " tail" * 30


@cache
def _plain_report() -> Report:
    return run(SMALL, llm=with_sections())


def story_with_titles(titles: Sequence[str], text: str | None = None) -> str:
    """The story of a report made by hand: its section titles, and its text when one is given."""
    report = _plain_report()
    changed = tuple(
        section.model_copy(update={"title": title, **({"markdown": text} if text is not None else {})})
        for section, title in zip(report.sections, titles, strict=True)
    )
    return story_of(render_markdown(report.model_copy(update={"sections": changed})))


def grown(form: str) -> str:
    """A section of the form that fills the cap after the cleaning."""
    paragraph, count = GROWTH[form]
    body = "\n\n".join([paragraph] * count)
    room = MAX_SECTION_CHARS - len(body) - 2
    return body + ("\n\n" + "a" * room if room > 0 else "")


def crafted_story(form: str, lead: int) -> str:
    """Three sections that grow in the cleaning, and a fourth one with a line of `=` under
    a line of text, `lead` characters into it."""
    tail = MAX_SECTION_CHARS - lead - len(INJECT) - 4
    last = "a" * lead + "\n\n" + INJECT + "x" * max(tail, 1)
    texts = {**dict.fromkeys(SECTION_KEYS[:3], grown(form)), SECTION_KEYS[3]: last}
    return story_with(**texts)[0]


def lead_to_reach(limit: int, form: str) -> int:
    """The lead that would put the line of `=` at the place where the server cuts a story."""
    story = crafted_story(form, 10)
    return 10 + limit - (story.index(INJECT) + len(INJECT))


def prose(chars: int, seed: int) -> str:
    """Ordinary writing in paragraphs, with some bold words and some quoted words."""
    rng = random.Random(seed)
    parts: list[str] = []
    size = 0
    while size < chars:
        sentences = []
        for _ in range(rng.randint(3, 6)):
            words = [rng.choice(PROSE_WORDS) for _ in range(rng.randint(8, 18))]
            if rng.random() < 0.3:
                words[2] = f"**{words[2]}**"
            if rng.random() < 0.3:
                words[5] = f'"{words[5]}"'
            sentences.append(" ".join(words).capitalize() + rng.choice([".", ".", "?", "!"]))
        parts.append(" ".join(sentences))
        size += len(parts[-1]) + 2
    return "\n\n".join(parts)[:chars].rstrip()


def _clauses(chars: int) -> list[str]:
    """Clauses with apostrophes, to `chars` characters. Each fourth one is in quote marks."""
    found: list[str] = []
    size = 0
    while size < chars:
        found.append(f'"{CLAUSE}"' if len(found) % 4 == 0 else CLAUSE)
        size += len(found[-1]) + 2
    return found


def dialogue(chars: int, paragraph_chars: int = 0) -> str:
    """Everyday dialogue. `paragraph_chars` of 0 gives one paragraph."""
    paragraphs: list[list[str]] = [[]]
    held = 0
    for clause in _clauses(chars):
        paragraphs[-1].append(clause)
        held += len(clause) + 2
        if paragraph_chars and held >= paragraph_chars:
            paragraphs.append([])
            held = 0
    joined = "\n\n".join(", ".join(part) + "." for part in paragraphs if part)
    return joined[:chars].rstrip(' ,"')
