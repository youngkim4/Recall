"""Reads back the tags and attributes a piece of HTML carries, for tests."""
from html.parser import HTMLParser

from safe_html import ALLOWED_TAGS

ALLOWED_ATTRIBUTES = {("a", "href"), ("a", "rel"), ("ol", "start")}
ORDINARY_ADDRESSES = ("http://", "https://", "mailto:")


class Shape(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.attributes = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.extend((tag, name, value) for name, value in attrs)


def shape(html):
    reader = Shape()
    reader.feed(html)
    reader.close()
    return reader


def assert_only_allowed(html):
    found = shape(html)
    assert set(found.tags) <= ALLOWED_TAGS, found.tags
    for tag, name, value in found.attributes:
        assert (tag, name) in ALLOWED_ATTRIBUTES, (tag, name, value)
        if name == "href":
            assert value.lower().startswith(ORDINARY_ADDRESSES), value
        if name == "start":
            assert value.isascii() and value.isdigit(), value
        if name == "rel":
            assert value == "noopener noreferrer", value
