"""A build: what it embeds, what it reuses, what it removes, and what it leaves when it stops."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from meaning_index.batching import BatchLimits
from meaning_index.build import build_index
from meaning_index.errors import (
    BuildCancelled,
    EmbeddingKeyError,
    EmbeddingRateLimited,
    IndexBusyError,
    SourceError,
)
from meaning_index.fake_embedder import FakeEmbedder
from meaning_index.shapes import BuildProgress, BuildResult
from meaning_index.store import IndexStore
from tests.meaning_index.support import ListSource, ScriptedEmbedder, cut_texts, msg, notes, run, stamp, words

FILES = ("segments.sqlite3", "vectors.npy", "meta.json")


def small() -> dict:
    """Three conversations: 2, 3 and 1 segments."""
    return {
        "chat-a": run(3, prefix="a") + run(3, prefix="a-late", start=600),
        "chat-b": run(40, prefix="b"),
        "chat-c": [msg("c0", stamp(0), "the ferry leaves at nine")],
    }


def build(source: ListSource, folder: Path, embedder: object = None, **named: object) -> BuildResult:
    named.setdefault("count_tokens", words)
    named.setdefault("sleep", lambda seconds: None)
    return build_index(source, embedder or FakeEmbedder(), IndexStore(folder), **named)  # type: ignore[arg-type]


def content(folder: Path) -> dict[str, bytes]:
    return {name: (folder / name).read_bytes() for name in FILES}


def ids(folder: Path) -> list[str]:
    found = IndexStore(folder).read()
    assert found is not None
    return [segment.segment_id for segment in found.segments]


def test_a_first_build_embeds_every_segment_in_the_order_of_the_source(folder: Path) -> None:
    source, embedder = ListSource(small()), FakeEmbedder()
    result = build(source, folder, embedder)
    texts = cut_texts(source)
    assert result == BuildResult(segments=6, embedded=6, reused=0, removed=0, tokens=sum(words(t) for t in texts))
    assert embedder.texts == texts
    status = IndexStore(folder).status()
    assert (status.state, status.segments, status.model, status.dimensions) == ("ready", 6, "fake-embedding-1", 64)
    assert status.covered_until == stamp(602)


def test_the_stored_segments_follow_the_source_and_the_vectors_have_unit_length(folder: Path) -> None:
    build(ListSource(small()), folder)
    found = IndexStore(folder).read()
    assert found is not None
    assert [s.conversation_id for s in found.segments] == ["chat-a", "chat-a", "chat-b", "chat-b", "chat-b", "chat-c"]
    assert [s.message_count for s in found.segments] == [3, 3, 16, 16, 8, 1]
    assert (found.segments[0].first_message_id, found.segments[0].last_message_id) == ("a0", "a2")
    lengths = np.linalg.norm(np.asarray(found.vectors, dtype=np.float64), axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-3)


def test_a_build_with_nothing_new_embeds_nothing(folder: Path) -> None:
    build(ListSource(small()), folder)
    before = np.array(IndexStore(folder).read().vectors)  # type: ignore[union-attr]
    embedder = FakeEmbedder()
    result = build(ListSource(small()), folder, embedder)
    assert result == BuildResult(segments=6, embedded=0, reused=6, removed=0, tokens=0)
    assert embedder.calls == []
    assert np.array_equal(np.asarray(IndexStore(folder).read().vectors), before)  # type: ignore[union-attr]


def test_only_a_new_segment_is_embedded(folder: Path) -> None:
    data = small()
    build(ListSource(data), folder)
    data["chat-c"] = [*data["chat-c"], msg("c1", stamp(900), "a new line about the canoe")]
    embedder = FakeEmbedder()
    result = build(ListSource(data), folder, embedder)
    assert result == BuildResult(segments=7, embedded=1, reused=6, removed=0, tokens=7)
    assert embedder.texts == ["2024-06-04\nThem: a new line about the canoe"]


def test_a_segment_that_grows_is_embedded_again_and_its_old_form_is_removed(folder: Path) -> None:
    data = small()
    build(ListSource(data), folder)
    before = ids(folder)
    data["chat-c"] = [*data["chat-c"], msg("c1", stamp(5), "see you there")]
    embedder = FakeEmbedder()
    result = build(ListSource(data), folder, embedder)
    assert (result.segments, result.embedded, result.reused, result.removed) == (6, 1, 5, 1)
    assert len(embedder.texts) == 1
    assert ids(folder)[:5] == before[:5]
    assert ids(folder)[5] != before[5]


def test_an_edited_text_keeps_the_segment_and_gets_a_new_vector(folder: Path) -> None:
    data = small()
    build(ListSource(data), folder)
    before = ids(folder)
    data["chat-c"] = [msg("c0", stamp(0), "the ferry leaves at ten")]
    result = build(ListSource(data), folder)
    assert (result.embedded, result.reused, result.removed) == (1, 5, 0)
    assert ids(folder) == before


def test_segments_that_the_source_no_longer_holds_are_removed(folder: Path) -> None:
    data = small()
    build(ListSource(data), folder)
    kept_rows = np.array(IndexStore(folder).read().vectors)[[0, 1, 5]]  # type: ignore[union-attr]
    del data["chat-b"]
    embedder = FakeEmbedder()
    result = build(ListSource(data), folder, embedder)
    assert result == BuildResult(segments=3, embedded=0, reused=3, removed=3, tokens=0)
    assert embedder.calls == []
    found = IndexStore(folder).read()
    assert found is not None
    assert [s.conversation_id for s in found.segments] == ["chat-a", "chat-a", "chat-c"]
    assert np.array_equal(np.asarray(found.vectors), kept_rows)


def test_a_reused_vector_goes_to_its_segment_when_the_order_of_the_source_changes(folder: Path) -> None:
    data = small()
    build(ListSource(data), folder)
    first = IndexStore(folder).read()
    assert first is not None
    by_id = {s.segment_id: np.array(first.vectors[n]) for n, s in enumerate(first.segments)}
    turned = dict(reversed(list(data.items())))
    result = build(ListSource(turned), folder)
    assert result.embedded == 0
    again = IndexStore(folder).read()
    assert again is not None
    assert [s.conversation_id for s in again.segments][0] == "chat-c"
    for row, segment in enumerate(again.segments):
        assert np.array_equal(np.asarray(again.vectors[row]), by_id[segment.segment_id])


@pytest.mark.parametrize("other", [FakeEmbedder(model="another-model"), FakeEmbedder(dimensions=32)])
def test_another_model_or_another_size_embeds_everything_again(folder: Path, other: FakeEmbedder) -> None:
    build(ListSource(small()), folder)
    result = build(ListSource(small()), folder, other)
    assert (result.segments, result.embedded, result.reused, result.removed) == (6, 6, 0, 0)
    status = IndexStore(folder).status()
    assert (status.model, status.dimensions) == (other.model, other.dimensions)
    assert IndexStore(folder).read().vectors.shape == (6, other.dimensions)  # type: ignore[union-attr]


def test_two_segments_with_one_text_are_one_text_of_a_request(folder: Path) -> None:
    same = [msg("x", stamp(0), "ok")]
    embedder = FakeEmbedder()
    result = build(ListSource({"chat-a": same, "chat-b": same}), folder, embedder)
    assert (result.segments, result.embedded, result.tokens) == (2, 2, 3)
    assert embedder.texts == ["2024-06-03\nThem: ok"]
    found = IndexStore(folder).read()
    assert found is not None
    assert np.array_equal(np.asarray(found.vectors[0]), np.asarray(found.vectors[1]))
    assert found.segments[0].segment_id != found.segments[1].segment_id


def test_a_conversation_that_the_source_names_two_times_is_read_one_time(folder: Path) -> None:
    class Twice(ListSource):
        def conversation_ids(self) -> Sequence[str]:
            return [*super().conversation_ids(), "chat-c"]

    source = Twice(small())
    assert build(source, folder).segments == 6
    assert source.reads == 3


def test_a_source_with_no_message_gives_an_index_with_no_segment(folder: Path) -> None:
    embedder = FakeEmbedder()
    result = build(ListSource({}), folder, embedder)
    assert result == BuildResult(segments=0, embedded=0, reused=0, removed=0, tokens=0)
    assert embedder.calls == []
    assert IndexStore(folder).status().state == "ready"


def test_requests_hold_96_texts_and_follow_the_token_counter(folder: Path) -> None:
    embedder = ScriptedEmbedder()
    build(ListSource({"chat-a": notes(250)}), folder, embedder)
    assert embedder.sizes == [96, 96, 58]
    heavy = ScriptedEmbedder()
    build(ListSource({"chat-a": notes(7)}), folder / "heavy", heavy, count_tokens=lambda text: 40_000)
    assert heavy.sizes == [2, 2, 2, 1]


def test_a_rate_limit_waits_through_the_sleep_that_the_caller_gives(folder: Path) -> None:
    slept: list[float] = []
    embedder = ScriptedEmbedder({1: EmbeddingRateLimited("wait", retry_after_s=12.0)})
    result = build(ListSource(small()), folder, embedder, sleep=slept.append)
    assert slept == [12.0]
    assert result.embedded == 6


def test_the_limits_of_a_request_can_be_given(folder: Path) -> None:
    embedder = ScriptedEmbedder()
    build(ListSource({"chat-a": notes(10)}), folder, embedder, limits=BatchLimits(max_texts=4))
    assert embedder.sizes == [4, 4, 2]


# --- a build that stops ------------------------------------------------------


def test_a_build_that_fails_leaves_the_old_index_as_it_was(folder: Path) -> None:
    data = small()
    build(ListSource(data), folder)
    before = content(folder)
    data["chat-d"] = notes(5, prefix="d")
    with pytest.raises(EmbeddingKeyError):
        build(ListSource(data), folder, ScriptedEmbedder({1: EmbeddingKeyError("no key")}, dimensions=64, model="fake-embedding-1"))
    assert content(folder) == before
    assert IndexStore(folder).status().segments == 6
    assert not list(folder.glob("*.new"))


def test_a_first_build_that_fails_leaves_no_index(folder: Path) -> None:
    with pytest.raises(EmbeddingKeyError):
        build(ListSource(small()), folder, ScriptedEmbedder({1: EmbeddingKeyError("no key")}))
    assert IndexStore(folder).status().state == "none"
    assert not any((folder / name).exists() for name in FILES)


def test_a_build_that_is_cancelled_leaves_the_old_index_as_it_was(folder: Path) -> None:
    data = {"chat-a": notes(100)}
    build(ListSource(data), folder)
    before = content(folder)
    data["chat-b"] = notes(250, prefix="b", word="memo")
    embedder = ScriptedEmbedder(dimensions=64, model="fake-embedding-1")
    with pytest.raises(BuildCancelled) as caught:
        build(ListSource(data), folder, embedder, cancel=lambda: len(embedder.sizes) >= 2)
    assert embedder.sizes == [96, 96]
    assert (caught.value.texts_embedded, caught.value.tokens) == (192, 192 * 4)
    assert content(folder) == before


def test_cancel_is_looked_at_before_the_first_request(folder: Path) -> None:
    embedder = FakeEmbedder()
    source = ListSource(small())
    with pytest.raises(BuildCancelled):
        build(source, folder, embedder, cancel=lambda: True)
    assert embedder.calls == []
    assert source.reads == 0
    assert IndexStore(folder).status().state == "none"


class Token:
    """The shape of report_engine's CancelToken."""

    def __init__(self) -> None:
        self.cancelled = False


@pytest.mark.parametrize("kind", ["event", "token"])
def test_cancel_can_be_an_event_or_a_token(folder: Path, kind: str) -> None:
    event, token = threading.Event(), Token()

    def stop(number: int, texts: Sequence[str]) -> None:
        event.set()
        token.cancelled = True

    embedder = ScriptedEmbedder(stop)
    with pytest.raises(BuildCancelled):
        build(ListSource({"chat-a": notes(200)}), folder, embedder, cancel=event if kind == "event" else token)
    assert embedder.sizes == [96]


def test_a_cancel_that_is_no_known_kind_is_refused(folder: Path) -> None:
    with pytest.raises(TypeError):
        build(ListSource(small()), folder, cancel=object())


def test_a_build_that_stopped_does_not_pay_two_times(folder: Path) -> None:
    source = ListSource({"chat-a": notes(250)})
    texts = cut_texts(source)
    first = ScriptedEmbedder({3: EmbeddingKeyError("no key")}, dimensions=64, model="fake-embedding-1")
    with pytest.raises(EmbeddingKeyError):
        build(source, folder, first)
    assert first.texts == texts[:192]
    assert (folder / "work.sqlite3").exists()

    second = FakeEmbedder()
    result = build(source, folder, second)
    assert second.texts == texts[192:]
    assert result == BuildResult(segments=250, embedded=58, reused=192, removed=0, tokens=58 * 4)
    assert not (folder / "work.sqlite3").exists()

    whole = FakeEmbedder()
    build(source, folder / "whole", whole)
    paid = IndexStore(folder).read()
    fresh = IndexStore(folder / "whole").read()
    assert paid is not None and fresh is not None
    assert np.array_equal(np.asarray(paid.vectors), np.asarray(fresh.vectors))


def test_a_cancelled_build_does_not_pay_two_times_either(folder: Path) -> None:
    source = ListSource({"chat-a": notes(250)})
    first = ScriptedEmbedder(dimensions=64, model="fake-embedding-1")
    with pytest.raises(BuildCancelled):
        build(source, folder, first, cancel=lambda: len(first.sizes) >= 1)
    second = FakeEmbedder()
    result = build(source, folder, second)
    assert len(second.texts) == 250 - 96
    assert (result.embedded, result.reused) == (154, 96)


def test_what_was_paid_for_another_model_is_not_given_to_this_one(folder: Path) -> None:
    source = ListSource({"chat-a": notes(120)})
    with pytest.raises(EmbeddingKeyError):
        build(source, folder, ScriptedEmbedder({2: EmbeddingKeyError("no key")}, dimensions=64, model="model-one"))
    other = FakeEmbedder(model="model-two")
    assert build(source, folder, other).embedded == 120
    assert len(other.texts) == 120


def test_the_index_is_swapped_only_at_the_end(folder: Path) -> None:
    seen: list[tuple[str, int]] = []

    def look(number: int, texts: Sequence[str]) -> None:
        status = IndexStore(folder).status()
        seen.append((status.state, status.segments))

    build(ListSource(small()), folder, ScriptedEmbedder(look))
    assert seen == [("none", 0)]
    data = small()
    data["chat-d"] = notes(3, prefix="d")
    build(ListSource(data), folder, ScriptedEmbedder(look))
    assert seen[-1] == ("ready", 6)
    assert IndexStore(folder).status().segments == 9


def test_a_second_build_of_the_folder_at_the_same_time_is_refused(folder: Path) -> None:
    caught: list[Exception] = []

    def second(number: int, texts: Sequence[str]) -> None:
        try:
            build(ListSource(small()), folder)
        except IndexBusyError as error:
            caught.append(error)

    build(ListSource(small()), folder, ScriptedEmbedder(second))
    assert len(caught) == 1


def test_a_source_that_fails_is_a_typed_error_and_the_index_is_as_it_was(folder: Path) -> None:
    class Broken(ListSource):
        def messages(self, conversation_id: str):  # type: ignore[no-untyped-def]
            if conversation_id == "chat-b":
                raise RuntimeError("could not read MARKED-TEXT-5512")
            return super().messages(conversation_id)

    build(ListSource(small()), folder)
    before = content(folder)
    with pytest.raises(SourceError) as caught:
        build(Broken(small()), folder)
    assert "MARKED" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert content(folder) == before


def test_a_source_that_can_not_list_its_conversations_is_a_typed_error(folder: Path) -> None:
    class Closed(ListSource):
        def conversation_ids(self) -> Sequence[str]:
            raise RuntimeError("closed")

    with pytest.raises(SourceError):
        build(Closed(small()), folder)


# --- progress ----------------------------------------------------------------


def test_progress_tells_the_three_phases_in_order(folder: Path) -> None:
    events: list[BuildProgress] = []
    build(ListSource({"chat-a": notes(120), "chat-b": notes(3, prefix="b", word="memo")}), folder, progress=events.append)
    phases = [event.phase for event in events]
    assert phases == sorted(phases, key=["read", "embed", "write"].index)
    by_phase = {phase: [(e.done, e.total) for e in events if e.phase == phase] for phase in ("read", "embed", "write")}
    assert by_phase["read"] == [(0, 2), (1, 2), (2, 2)]
    assert by_phase["embed"] == [(0, 123), (96, 123), (123, 123)]
    assert by_phase["write"][0] == (0, 123)
    assert by_phase["write"][-1] == (123, 123)
