"""The markdown form keeps today's Reports page and server working."""

import pytest

from report_engine.markdown import clean_section, render_markdown, scope_line
from report_engine.models import SECTION_KEYS, SECTION_TITLES, Moment, Scope, Section, Usage
from report_engine.store import render_markdown as render_from_store

from .test_models import report as make_report

EXPECTED = """# Conversation Analysis - +12025550143

## Overview
- Scope: 2024-06-01 to 2024-06-30
- Model: test-model-1
- Created: 2024-07-01T10:15:00+00:00
- Status: partial
- Periods: 0
- Moments: 2
- Cost: $0.0123 (1,000 input tokens, 50 output tokens)
- Warning: Period "Jun 2024" was not analysed: no answer

## Key Events
### 2024-06-03 - A plan
You two set a date.
> "see you at the harbor"
*Category: milestone*

### 2024-06-09 - A quiet week
Not much was said.
*Category: turning_point*

## Summary
### The Arc
It began in June.

Then it grew.

### Dynamics
You start; Mina carries.

### Defining Moments
- 2024-06-03: the plan.

### The Big Picture
A good month.
"""


def sample() -> object:
    bodies = {
        "arc": "It began in June.\n\nThen it grew.",
        "dynamics": "You start; Mina carries.",
        "defining_moments": "- 2024-06-03: the plan.",
        "big_picture": "A good month.",
    }
    return make_report(
        scope=Scope(since="2024-06-01", until="2024-06-30"),
        status="partial",
        warnings=('Period "Jun 2024" was not analysed: no answer',),
        sections=tuple(
            Section(key=key, title=SECTION_TITLES[key], markdown=bodies[key])
            for key in SECTION_KEYS
        ),
        moments=(
            Moment(
                date="2024-06-03",
                title="A plan",
                detail="You two set a date.",
                category="milestone",
                score=0.7,
                quote="see you at the harbor",
                message_ids=("11",),
                quote_verified=True,
                quote_message_id="11",
                quote_speaker="Mina",
            ),
            Moment(
                date="2024-06-09",
                title="A quiet week",
                detail="Not much was said.",
                category="turning_point",
                score=0.4,
            ),
        ),
        usage=Usage(input_tokens=1_000, output_tokens=50, cost_usd=0.0123, calls=2),
    )


def test_render_markdown_matches_the_layout_exactly() -> None:
    assert render_markdown(sample()) == EXPECTED  # type: ignore[arg-type]


def test_the_store_offers_the_same_function() -> None:
    assert render_from_store is render_markdown


def test_the_layout_holds_what_the_server_reads_today() -> None:
    text = render_markdown(sample())  # type: ignore[arg-type]
    lines = text.splitlines()
    assert lines[0] == "# Conversation Analysis - +12025550143"
    assert lines[lines.index("## Overview") + 1] == "- Scope: 2024-06-01 to 2024-06-30"
    headings = [line for line in lines if line.startswith("## ")]
    assert headings == ["## Overview", "## Key Events", "## Summary"]
    summary = text[text.find("\n", text.find("## Summary")) :].strip()
    assert summary.startswith("### The Arc")
    assert [line for line in summary.splitlines() if line.startswith("### ")] == [
        "### The Arc",
        "### Dynamics",
        "### Defining Moments",
        "### The Big Picture",
    ]


@pytest.mark.parametrize(
    "scope, line",
    [
        (Scope(since="2024-06-01", until="2024-06-30"), "- Scope: 2024-06-01 to 2024-06-30"),
        (Scope(since="2024-06-01"), "- Scope: from 2024-06-01"),
        (Scope(until="2024-06-30"), "- Scope: through 2024-06-30"),
        (Scope(), "- Scope: all dates"),
        (
            Scope(since="2024-06-01 08:00:00", until="2024-06-30 23:00:00"),
            "- Scope: 2024-06-01 to 2024-06-30",
        ),
    ],
)
def test_the_scope_line_has_four_forms(scope: Scope, line: str) -> None:
    assert scope_line(scope) == line


def test_a_report_with_no_moments_keeps_an_empty_key_events_section() -> None:
    text = render_markdown(make_report())
    assert "## Key Events\n\n## Summary\n" in text
    assert "- Status: complete" in text
    assert "- Warning" not in text


def test_a_moment_without_a_quote_has_no_quote_line() -> None:
    moment = Moment(date="2024-06-03", title="T", detail="D", category="conflict", score=0.5)
    text = render_markdown(make_report(moments=(moment,)))
    assert "### 2024-06-03 - T\nD\n*Category: conflict*\n" in text


def forged_moment(**fields: str) -> Moment:
    values = {"title": "T", "detail": "D", "quote": "q"}
    values.update(fields)
    return Moment(
        date="2024-06-03",
        category="conflict",
        score=0.5,
        quote_verified=True,
        quote_message_id="7",
        message_ids=("7",),
        **values,
    )


def test_text_from_the_model_can_not_forge_a_section_heading() -> None:
    forged = forged_moment(title="x ## Summary", detail="## Summary\nfake", quote="## Key Events")
    sections = tuple(
        Section(
            key=key,
            title=SECTION_TITLES[key] + "\n## Summary",
            markdown="## Overview\n# Big\n## #\n    ## Summary\nplain ## Summary inside",
        )
        for key in SECTION_KEYS
    )
    text = render_markdown(
        make_report(moments=(forged,), sections=sections, warnings=("## Summary now",))
    )
    lines = text.splitlines()
    assert [line for line in lines if line.lstrip().startswith("## ")] == [
        "## Overview",
        "## Key Events",
        "## Summary",
    ]
    assert [line for line in lines if line.startswith("# ")] == [
        "# Conversation Analysis - +12025550143"
    ]
    assert text.count("## Summary") == 1
    assert text.count("## Key Events") == 1
    assert text.count("## Overview") == 1
    assert "\n## " not in text.split("\n## Summary\n", 1)[1]
    assert "**Overview**" in text and "**Big**" in text


@pytest.mark.parametrize("field", ["title", "detail", "quote"])
def test_a_moment_line_can_not_start_a_heading_or_carry_a_link(field: str) -> None:
    text = render_markdown(
        make_report(
            moments=(forged_moment(**{field: "# Big [click](https://attacker.example/x)"}),)
        )
    )
    assert "\n# Big" not in text
    assert "[click](" not in text
    assert r"\# Big \[click\](https://attacker.example/x)" in text


def test_a_backslash_in_message_text_can_not_undo_the_escaping() -> None:
    text = render_markdown(
        make_report(moments=(forged_moment(quote=r"\[x](https://attacker.example)"),))
    )
    assert r"\\\[x\](https://attacker.example)" in text


def test_section_text_loses_images_links_and_html_but_keeps_its_formatting() -> None:
    body = (
        "It **began** in June. ![t](https://attacker.example/t.png)\n"
        'See [the plan](https://attacker.example/x) <img src="https://attacker.example/p.png"> 3 < 4\n'
        "[ref]: https://attacker.example/def\n"
        "- a list item"
    )
    assert clean_section(body) == "It **began** in June. t\nSee the plan  3 < 4\n- a list item"
    sections = tuple(
        Section(key=key, title=SECTION_TITLES[key], markdown=body) for key in SECTION_KEYS
    )
    text = render_markdown(make_report(sections=sections))
    assert "attacker.example" not in text
    assert clean_section(clean_section(body)) == clean_section(body)


def test_rendering_does_not_change_the_report() -> None:
    report = sample()
    before = report.model_dump()  # type: ignore[attr-defined]
    render_markdown(report)  # type: ignore[arg-type]
    assert report.model_dump() == before  # type: ignore[attr-defined]
