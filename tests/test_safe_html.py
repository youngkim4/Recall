"""Tests for the allowlist that rendered report text passes through."""
import time

import pytest

from safe_html import sanitize, story_html
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
