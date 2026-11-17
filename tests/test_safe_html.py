"""Tests for the allowlist that rendered report text passes through."""
import time

import pytest

from safe_html import STORY_LIMIT, STORY_WORK_LIMIT, prepared, sanitize, story_html, story_work
from tests.html_shape import assert_only_allowed, shape


def test_ordinary_writing_passes_through_unchanged():
    html = (
        "<h2>Timeline</h2>\n<p>They <em>made</em> <strong>plans</strong>.</p>\n"
        "<blockquote>\n<p>see you at 8</p>\n</blockquote>\n"
        "<ul>\n<li>one</li>\n<li>two</li>\n</ul>\n<hr>\n"
        '<p><a href="https://example.com/menu" rel="noopener noreferrer">the menu</a><br>next line</p>'
    )

    assert sanitize(html) == html


def test_nothing_gives_nothing():
    assert sanitize("") == ""
    assert sanitize(None) == ""


@pytest.mark.parametrize("html", [
    '<p onmouseover="alert(1)">hi</p>',
    '<p style="background:url(javascript:alert(1))">hi</p>',
    '<p class="c" id="i" title="t" data-x="1" tabindex="0" autofocus>hi</p>',
    '<P ONCLICK="alert(1)">hi</P>',
    "<p onclick=alert(1)>hi</p>",
    "<p\nonclick='alert(1)'>hi</p>",
])
def test_a_tag_keeps_no_attribute_it_was_not_given(html):
    assert sanitize(html) == "<p>hi</p>"


def test_code_that_would_run_is_dropped_with_what_is_in_it():
    html = "<p>a</p><script>alert(1)</script><style>p { color: red }</style><p>b</p>"

    assert sanitize(html) == "<p>a</p><p>b</p>"


def test_a_script_tag_in_capitals_is_dropped_too():
    assert sanitize("<p>a</p><SCRIPT>alert(1)</SCRIPT>") == "<p>a</p>"


def test_a_script_left_open_takes_the_rest_with_it():
    assert sanitize("<p>a</p><script>alert(1)") == "<p>a</p>"


def test_a_tag_that_is_not_allowed_goes_and_its_words_stay():
    assert sanitize("<div><span>hi</span> <b>there</b></div>") == "hi there"


@pytest.mark.parametrize("html", [
    '<svg onload="alert(1)"><circle r="1"/></svg>',
    '<iframe src="javascript:alert(1)"></iframe>',
    '<object data="x"></object><embed src="x">',
    '<form action="https://example.com"><input value="y"><button>go</button></form>',
    '<math><mtext></mtext></math>',
    '<link rel="stylesheet" href="https://example.com/x.css">',
    '<meta http-equiv="refresh" content="0;url=https://example.com">',
    '<base href="https://example.com/">',
    "<scr<script>ipt>alert(1)</scr</script>ipt>",
    "<<script>script>alert(1)<</script>/script>",
])
def test_no_tag_outside_the_list_survives(html):
    cleaned = sanitize(html)

    assert_only_allowed(cleaned)
    assert "<script" not in cleaned.lower()


@pytest.mark.parametrize("html", [
    "<textarea><script>alert(1)</script></textarea>",
    "<title><img src=x onerror=alert(1)></title>",
    "<noscript><p title='</noscript><img src=x onerror=alert(1)>'></noscript>",
    "<xmp><script>alert(1)</script></xmp>",
    "<plaintext><script>alert(1)</script>",
    "<script/>alert(1)</script><p>a</p>",
    "<style/><img src=x onerror=alert(1)>",
    "<p>a\x00<scr\x00ipt>alert(1)</scr\x00ipt></p>",
    "<svg><style><img src=x onerror=alert(1)></style></svg>",
    "<math><mi xlink:href='javascript:alert(1)'>x</mi></math>",
    "<a href='https://example.com' href='javascript:alert(1)' onclick='alert(1)'>x</a>",
    "<a href='javascript:alert(1)' href='https://example.com'>x</a>",
    "<p><a href='https://example.com'><a href='javascript:alert(1)'>x</a></a></p>",
    "&lt;img src=x onerror=alert(1)&gt;",
    "<p>&#60;script&#62;alert(1)&#60;/script&#62;</p>",
    "<img alt='\"><script>alert(1)</script>'>",
    "<ol start='1\" onmouseover=\"alert(1)'><li>x</li></ol>",
])
def test_what_comes_out_can_only_be_read(html):
    cleaned = sanitize(html)

    assert_only_allowed(cleaned)
    assert "<script" not in cleaned.lower()
    assert "<img" not in cleaned.lower()
    assert sanitize(cleaned) == cleaned


def test_the_first_of_two_addresses_is_the_one_judged():
    # a browser keeps the first attribute of a name, so that one decides
    assert sanitize("<a href='https://example.com' href='javascript:alert(1)'>x</a>") == '<a href="https://example.com" rel="noopener noreferrer">x</a>'
    assert sanitize("<a href='javascript:alert(1)' href='https://example.com'>x</a>") == "x"


def test_html_the_reader_cannot_parse_is_shown_as_text(monkeypatch):
    import safe_html

    def refuse(self, data):
        raise AssertionError("unexpected char in declaration")

    monkeypatch.setattr(safe_html._Rebuilder, "feed", refuse)

    assert sanitize('<p onclick="alert(1)">hi</p>') == "&lt;p onclick=&quot;alert(1)&quot;&gt;hi&lt;/p&gt;"


def test_a_picture_becomes_its_caption():
    html = '<p>Before <img src="https://example.com/p.png" alt="a &lt;b&gt; photo" onerror="alert(1)"> after</p>'

    assert sanitize(html) == "<p>Before a &lt;b&gt; photo after</p>"


def test_a_picture_with_no_caption_leaves_nothing():
    assert sanitize('<p>Before <img src="https://example.com/p.png"> after</p>') == "<p>Before  after</p>"


@pytest.mark.parametrize("href", [
    "https://example.com/menu",
    "http://example.com/",
    "mailto:hi@example.com",
])
def test_a_link_to_an_ordinary_address_stays(href):
    assert sanitize(f'<a href="{href}">x</a>') == f'<a href="{href}" rel="noopener noreferrer">x</a>'


def test_a_link_address_is_trimmed_before_it_is_judged():
    assert sanitize('<a href="  HTTPS://Example.com/a b ">x</a>') == '<a href="HTTPS://Example.com/a b" rel="noopener noreferrer">x</a>'


@pytest.mark.parametrize("href", [
    "javascript:alert(1)",
    "JaVaScRiPt:alert(1)",
    " javascript:alert(1)",
    "java\tscript:alert(1)",
    "java\nscript:alert(1)",
    "java&#10;script:alert(1)",
    "&#1;javascript:alert(1)",
    "&#106;avascript:alert(1)",
    "vbscript:msgbox(1)",
    "data:text/html;base64,PHNjcmlwdD4=",
    "//example.com/path",
    "/api/report?path=x",
    "note",
    "#fn:1",
    "",
])
def test_a_link_to_anything_else_loses_the_link_and_keeps_the_words(href):
    assert sanitize(f'<p><a href="{href}">x</a> y</p>') == "<p>x y</p>"


def test_a_link_with_no_address_is_only_its_words():
    assert sanitize("<a>x</a>") == "x"
    assert sanitize('<a name="top" onclick="alert(1)">x</a>') == "x"


def test_a_link_address_cannot_close_its_own_quotes():
    html = sanitize("<a href='https://example.com/\"onmouseover=\"alert(1)'>x</a>")

    assert html == '<a href="https://example.com/&quot;onmouseover=&quot;alert(1)" rel="noopener noreferrer">x</a>'
    assert_only_allowed(html)


def test_a_link_keeps_an_address_with_an_ampersand():
    html = '<a href="https://example.com/a?b=1&amp;c=2" rel="noopener noreferrer">x</a>'

    assert sanitize(html) == html


def test_text_is_escaped_again_on_the_way_out():
    html = "<p>&lt;script&gt;alert(1)&lt;/script&gt; &amp; &quot;more&quot;</p>"

    cleaned = sanitize(html)

    assert "<script" not in cleaned
    assert cleaned == "<p>&lt;script&gt;alert(1)&lt;/script&gt; &amp; &quot;more&quot;</p>"


def test_a_numbered_list_keeps_where_it_starts():
    assert sanitize('<ol start="3"><li>three</li></ol>') == '<ol start="3"><li>three</li></ol>'


@pytest.mark.parametrize("start", ["3 onclick=alert(1)", "-1", "three", "", "1e3", "１２"])
def test_a_list_start_that_is_not_a_plain_number_is_dropped(start):
    assert sanitize(f'<ol start="{start}"><li>x</li></ol>') == "<ol><li>x</li></ol>"


def test_a_table_keeps_its_cells_and_loses_its_styles():
    html = (
        '<table><thead><tr><th style="text-align: left;">a</th></tr></thead>'
        '<tbody><tr><td style="text-align: right;" onclick="alert(1)">1</td></tr></tbody></table>'
    )

    assert sanitize(html) == "<table><thead><tr><th>a</th></tr></thead><tbody><tr><td>1</td></tr></tbody></table>"


def test_a_tag_left_open_is_closed():
    assert sanitize("<p><em>hi") == "<p><em>hi</em></p>"


def test_a_closing_tag_with_nothing_to_close_is_dropped():
    assert sanitize("</div></p>hi</em>") == "hi"


def test_tags_closed_out_of_order_come_out_in_order():
    assert sanitize("<p><em>a</p>b</em>") == "<p><em>a</em></p>b"


def test_a_line_break_and_a_rule_are_written_once():
    assert sanitize("a<br>b<br/>c<br></br><hr />") == "a<br>b<br>c<br><hr>"


def test_comments_and_declarations_are_dropped():
    html = "<!-- note --><!DOCTYPE html><?php echo 1 ?><![CDATA[x]]><p>a</p>"

    assert sanitize(html) == "<p>a</p>"


def test_code_is_left_as_written_unless_asked():
    html = "<p><code>a &amp;lt; b</code></p>"

    assert sanitize(html) == html


def test_code_escaped_twice_shows_what_was_typed():
    # a caller that escapes before rendering gets code escaped a second time
    # by the renderer; one layer comes off so the reader sees the characters
    html = "<p><code>don&amp;#x27;t &amp;quot;wait&amp;quot; &amp;lt;3</code></p>"

    assert sanitize(html, code_escaped_twice=True) == "<p><code>don&#x27;t &quot;wait&quot; &lt;3</code></p>"


def test_a_code_block_escaped_twice_shows_what_was_typed():
    html = "<pre><code>if a &amp;lt; b &amp;amp;&amp;amp; c:\n    go()\n</code></pre>"

    assert sanitize(html, code_escaped_twice=True) == "<pre><code>if a &lt; b &amp;&amp; c:\n    go()\n</code></pre>"


def test_only_code_loses_a_layer():
    html = "<p>a &amp;lt; b <code>c &amp;lt; d</code> e &amp;lt; f</p>"

    assert sanitize(html, code_escaped_twice=True) == "<p>a &amp;lt; b <code>c &lt; d</code> e &amp;lt; f</p>"


def test_markup_typed_inside_code_stays_text():
    html = "<code>&amp;lt;script&amp;gt;alert(1)&amp;lt;/script&amp;gt;</code>"

    cleaned = sanitize(html, code_escaped_twice=True)

    assert cleaned == "<code>&lt;script&gt;alert(1)&lt;/script&gt;</code>"
    assert shape(cleaned).tags == ["code"]


@pytest.mark.parametrize("html", [
    '<p onmouseover="alert(1)">hi</p><script>x</script>',
    '<a href="https://example.com/a?b=1&amp;c=2">x</a> <img alt="cap">',
    "<p><em>a</p>b</em><ol start='2'><li>x",
    "<code>&amp;lt;b&amp;gt;</code> &lt;b&gt; <b>bold</b>",
    "<table><tr><td style='x'>1</td></tr></table>",
])
def test_cleaning_twice_changes_nothing_more(html):
    once = sanitize(html)

    assert sanitize(once) == once
    assert_only_allowed(once)


def test_a_link_says_nothing_to_where_it_goes():
    # no referrer and no handle on the page that opened it
    assert story_html("[a link](https://a.example/x)") == (
        '<p><a href="https://a.example/x" rel="noopener noreferrer">a link</a></p>'
    )


@pytest.mark.parametrize(
    "hostile",
    [
        "`" * 5000,
        "[" * 5000,
        "&" * 40000,
        "\n" * 40000 + "> quoted",
        "*" * 5000 + "x",
    ],
)
def test_a_story_made_to_be_slow_is_still_quick(hostile):
    started = time.monotonic()
    rendered = story_html(hostile)
    assert time.monotonic() - started < 2.0
    assert_only_allowed(rendered)


@pytest.mark.parametrize("hostile", ["- " * 6000 + "x", "> " * 6000 + "x", ("  " * 400 + "- x\n") * 3])
def test_a_story_nested_too_deep_is_shown_as_plain_text(hostile):
    rendered = story_html(hostile)
    assert_only_allowed(rendered)
    assert "x" in rendered


def test_a_story_longer_than_any_report_is_cut():
    rendered = story_html("word " * 200_000)
    assert len(rendered) < 300_000
    assert_only_allowed(rendered)


# ---- a story cannot make the page wait

# What the renderer is slow to read. Each of these, repeated, took from
# seconds to minutes at a length a story was allowed to have; "__\\" took
# twenty seconds at 2,400 characters.
SLOW_UNITS = [
    "[a](", "![a](", "> ", "**a *b ", "*a ", '](a "', "![", "[a][", "`a ", "_a ", "#\n",
    "__\\", "![\\", "[\\*", '["', "a\n\n", "it's\n\n", "- a\n", "- __\\\n",
]


def repeated(unit, length):
    return (unit * (length // len(unit) + 1))[:length]


def seconds_to_render(text):
    started = time.perf_counter()
    rendered = story_html(text)
    return time.perf_counter() - started, rendered


@pytest.mark.parametrize("unit", SLOW_UNITS)
@pytest.mark.parametrize("length", [STORY_LIMIT, 200_000])
def test_a_slow_shape_as_long_as_a_story_can_be_is_still_quick(unit, length):
    took, rendered = seconds_to_render(repeated(unit, length))

    assert took < 3.0
    assert_only_allowed(rendered)


def most_that_is_rendered(unit):
    """The longest paragraph of this unit that is still read as markdown."""
    low, high = 0, STORY_LIMIT
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (middle, high) if story_work(prepared(repeated(unit, middle))) <= STORY_WORK_LIMIT else (low, middle)
    return repeated(unit, low)


@pytest.mark.parametrize("unit", SLOW_UNITS)
def test_the_most_of_a_slow_shape_that_the_renderer_is_given_is_still_quick(unit):
    hostile = most_that_is_rendered(unit)
    assert len(hostile) > 100

    # the renderer reads it, or is stopped while it does
    took, rendered = seconds_to_render(hostile)

    assert took < 3.0
    assert_only_allowed(rendered)


def test_a_story_that_asks_for_too_much_work_is_shown_as_plain_paragraphs():
    hostile = repeated("[a](", 20_000)
    assert story_work(prepared(hostile)) > STORY_WORK_LIMIT

    rendered = story_html(hostile + "\n\n**not read**")

    assert "<strong>" not in rendered
    assert "**not read**" in rendered
    assert_only_allowed(rendered)
    assert set(shape(rendered).tags) == {"p"}


def test_a_story_that_takes_the_renderer_too_long_is_shown_as_plain_paragraphs():
    # short enough to be given to the renderer, which then starts it again and again
    hostile = repeated("__\\", 2_000)
    assert story_work(prepared(hostile)) <= STORY_WORK_LIMIT

    took, rendered = seconds_to_render(hostile + "\n\n**not read**")

    assert took < 3.0
    assert "<strong>" not in rendered
    assert "**not read**" in rendered
    assert set(shape(rendered).tags) == {"p"}


def test_a_list_is_counted_an_item_at_a_time():
    items = "\n".join(f"- **Label {number}:** it's a line with \"words\" in it, and more." for number in range(400))

    rendered = story_html(items)

    # as one paragraph of this length it would be far past the limit
    assert story_work(prepared(items)) * 4 < STORY_WORK_LIMIT
    assert rendered.count("<li>") == rendered.count("<strong>") == 400


def an_ordinary_story(length):
    """Writing the way a report is written: headings, paragraphs with marks
    and quotes in them, lists, a quoted line, a table."""
    paragraph = (
        'On 2024-06-10 you wrote "see you at 8", and it\'s clear they didn\'t mind. '
        "**The plans** they made that summer were mostly about dinner. "
        "*You wrote first* most mornings, and they answered late but warmly. "
    ) * 4
    parts = [
        "### Timeline",
        paragraph,
        "\n".join(f"- **Phase {number}:** it's a line with \"words\" in it, and more." for number in range(1, 9)),
        "> see you at 8",
        "| Year | Messages |\n|---|---|\n| 2023 | 410 |\n| 2024 | 1,932 |",
        "### Takeaways",
        paragraph,
    ]
    story = "\n\n".join(parts)
    return "\n\n".join([story] * (length // len(story)))


def test_an_ordinary_story_as_long_as_a_story_can_be_is_rendered():
    story = an_ordinary_story(STORY_LIMIT)
    assert STORY_LIMIT * 0.9 < len(story) <= STORY_LIMIT

    took, rendered = seconds_to_render(story)

    assert took < 3.0
    assert rendered.count("<h3>") == story.count("### ")
    assert rendered.count("<strong>") == story.count("**") // 2
    assert rendered.count("<table>") == story.count("| Year |")
    # writing this full of marks, at this length, is still well inside the limit
    assert story_work(prepared(story)) * 3 < STORY_WORK_LIMIT


def test_a_link_title_far_longer_than_a_title_is_not_looked_for():
    # the search for the end of a title is bounded, so it cannot be made slow
    took, rendered = seconds_to_render(repeated('](a "', STORY_LIMIT))

    assert took < 3.0
    assert_only_allowed(rendered)
