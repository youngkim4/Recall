"""search.py: the ranges of a result, in UTF-16 units of the text that the result shows."""

from __future__ import annotations

from archive import Archive
from recall_services import search
from recall_services.messages import MESSAGE_TEXT_LIMIT
from recall_services.names import Names
from tests.recall_services import world

PLACEHOLDER = "￼"


def ranges_of(found: dict[str, object]) -> dict[str, list[list[int]]]:
    return {str(result["text"]): result["ranges"] for result in found["results"]}  # type: ignore[union-attr]


def test_each_result_has_the_ranges_of_its_words(archive: Archive, saved: Names) -> None:
    found = search.payload(archive, saved, "flat")
    assert ranges_of(found) == {"The wheelbarrow has a flat tire": [[22, 26]]}


def test_a_word_inside_another_word_has_no_range(archive: Archive, saved: Names) -> None:
    # "tire" is a whole word: "Tire" is one too, case aside; nothing inside a longer word
    found = search.payload(archive, saved, "tire")
    assert ranges_of(found) == {"Tire fixed": [[0, 4]], "The wheelbarrow has a flat tire": [[27, 31]]}


def test_a_prefix_marks_the_whole_word(archive: Archive, saved: Names) -> None:
    found = search.payload(archive, saved, "wheelb", as_you_type=True)
    assert ranges_of(found) == {
        "The wheelbarrow has a flat tire": [[4, 15]],
        "Who has the wheelbarrow?": [[12, 23]],
    }


def test_the_ranges_follow_the_trimmed_text(archive: Archive, saved: Names) -> None:
    found = search.payload(archive, saved, "lemons")
    assert ranges_of(found) == {"Picked the first lemons today": [[17, 23]]}


def test_a_word_past_the_cut_has_no_range(archive: Archive, saved: Names) -> None:
    (result,) = search.payload(archive, saved, world.LONG_TEXT_WORD)["results"]
    assert result["ranges"] == []
    pears = search.payload(archive, saved, "pear")["results"][0]
    assert pears["ranges"] and all(end <= len(pears["text"]) - 1 for _start, end in pears["ranges"])


def test_with_no_words_there_are_no_ranges(archive: Archive, saved: Names) -> None:
    found = search.payload(archive, saved, "", contact=world.GARDEN)
    assert found["results"] and all(result["ranges"] == [] for result in found["results"])


def test_utf16_units_after_an_emoji() -> None:
    # the emoji is two UTF-16 units: the word after it starts at 3, not 2
    assert search.shown_ranges("🙂 hi", "🙂 hi", [[2, 4]]) == [[3, 5]]


def test_the_placeholder_and_outer_space_are_not_shown() -> None:
    raw = f"  {PLACEHOLDER} ok {PLACEHOLDER}go  "
    assert search.shown_ranges(raw, "ok go", [[4, 6], [8, 10]]) == [[0, 2], [3, 5]]


def test_a_range_across_the_cut_is_cut() -> None:
    raw = "a" * (MESSAGE_TEXT_LIMIT - 2) + " wordy end"
    shown = "a" * (MESSAGE_TEXT_LIMIT - 2) + "…"
    assert search.shown_ranges(raw, shown, [[0, 3], [MESSAGE_TEXT_LIMIT - 1, MESSAGE_TEXT_LIMIT + 4]]) == [[0, 3]]
    long_word = "b" * (MESSAGE_TEXT_LIMIT + 10)
    cut = long_word[:MESSAGE_TEXT_LIMIT] + "…"
    assert search.shown_ranges(long_word, cut, [[MESSAGE_TEXT_LIMIT - 2, MESSAGE_TEXT_LIMIT + 5]]) == [
        [MESSAGE_TEXT_LIMIT - 2, MESSAGE_TEXT_LIMIT]
    ]


def test_a_range_on_hidden_characters_only_is_dropped() -> None:
    assert search.shown_ranges(f"{PLACEHOLDER} ok", "ok", [[0, 1]]) == []
