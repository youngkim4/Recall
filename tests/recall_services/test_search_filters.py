"""search_filters.py: the filters of /api/search and /api/stats, read from a query string."""

from __future__ import annotations

import pytest

from recall_services.search_filters import NO_FILTERS, Filters, read_filters, read_order, read_prefix


def test_nothing_given_is_no_filter() -> None:
    found = read_filters({})
    assert found == NO_FILTERS
    assert found.is_empty
    assert found.payload() == {"contacts": [], "since": "", "until": "", "from": "", "kind": "", "attachments": ""}
    assert found.archive_args() == {
        "conversations": None,
        "since": None,
        "until": None,
        "from_me": None,
        "kind": None,
        "attachments": None,
    }


def test_every_filter() -> None:
    found = read_filters(
        {
            "contacts": " +15550170001, ines@example.org ,,+15550170001",
            "contact": "chat900100",
            "since": "2024-03-01",
            "until": "2024-03-31",
            "from": "them",
            "kind": "direct",
            "attachments": "image",
        }
    )
    # the one `contact` of today comes first, then the list without repeats
    assert found.contacts == ("chat900100", "+15550170001", "ines@example.org")
    assert not found.is_empty
    assert found.payload() == {
        "contacts": ["chat900100", "+15550170001", "ines@example.org"],
        "since": "2024-03-01",
        "until": "2024-03-31",
        "from": "them",
        "kind": "direct",
        "attachments": "image",
    }
    assert found.archive_args() == {
        "conversations": ["chat900100", "+15550170001", "ines@example.org"],
        "since": "2024-03-01",
        "until": "2024-03-31",
        "from_me": False,
        "kind": "direct",
        "attachments": "image",
    }


def test_from_me() -> None:
    assert read_filters({"from": "me"}).archive_args()["from_me"] is True


@pytest.mark.parametrize("choice", ["any", "none", "image", "video", "audio", "other"])
def test_each_attachment_choice(choice: str) -> None:
    assert read_filters({"attachments": choice}).attachments == choice


def test_with_contact_adds_one_conversation() -> None:
    assert NO_FILTERS.with_contact("") is NO_FILTERS
    assert Filters(contacts=("a",)).with_contact("b").contacts == ("b", "a")
    assert Filters(contacts=("a",)).with_contact("a").contacts == ("a",)


@pytest.mark.parametrize(
    ("params", "sentence"),
    [
        ({"since": "03/01/2024"}, "since must be a day written as YYYY-MM-DD."),
        ({"until": "2024-02-30"}, "until must be a day written as YYYY-MM-DD."),
        ({"from": "you"}, "from must be me, them or empty."),
        ({"kind": "person"}, "kind must be direct, group or empty."),
        ({"attachments": "sticker"}, "attachments must be any, none, image, video, audio, other or empty."),
    ],
)
def test_a_value_that_is_not_allowed_is_refused_in_one_sentence(params: dict[str, str], sentence: str) -> None:
    with pytest.raises(ValueError) as refused:
        read_filters(params)
    assert str(refused.value) == sentence


def test_order() -> None:
    assert read_order({}) == "newest"
    assert [read_order({"order": order}) for order in ("newest", "oldest", "relevance")] == [
        "newest",
        "oldest",
        "relevance",
    ]
    with pytest.raises(ValueError, match="^order must be newest, oldest or relevance.$"):
        read_order({"order": "best"})


def test_prefix() -> None:
    # no `prefix` at all is today's search as the reader types: the app drops an empty value too
    assert read_prefix({}) is True
    assert read_prefix({"prefix": ""}) is True
    assert read_prefix({"prefix": "true"}) is True
    assert read_prefix({"prefix": "FALSE"}) is False
    # a route can choose its own default for a request with no `prefix`
    assert read_prefix({}, default=False) is False
    assert read_prefix({"prefix": ""}, default=False) is False
    assert read_prefix({"prefix": "true"}, default=False) is True
    with pytest.raises(ValueError, match="^prefix must be true or false.$"):
        read_prefix({"prefix": "1"})
