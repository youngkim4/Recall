"""The two memory ports, the fakes for them, and `message` of the archive port."""

import pytest

from chat_engine.archive_adapter import ArchiveAdapter
from chat_engine.cancel import CancelToken
from chat_engine.fake_archive import FakeArchive, make_message
from chat_engine.fake_digests import DigestStopped, FakeDigests, moment, period
from chat_engine.fake_meaning import FakeMeaning, hit
from chat_engine.frame_archive import FrameArchive
from tests.chat_engine.frames import frame_of
from tests.chat_engine.tiny import JONAS_PHONE, MAYA_PHONE, tiny_archive, tiny_messages


def test_the_fake_archive_gives_one_message_by_its_id() -> None:
    archive = tiny_archive()
    found = archive.message("j3")
    assert found is not None and found["text"] == "its name is Clint Yeastwood"
    assert found["conversation"] == JONAS_PHONE
    assert archive.message("nope") is None
    assert archive.calls == [("message", {"message_id": "j3"}), ("message", {"message_id": "nope"})]
    found["text"] = "tampered"
    assert archive.message("j3")["text"] == "its name is Clint Yeastwood"


def test_a_system_row_is_not_a_message_of_the_port() -> None:
    rows = [
        make_message("s1", "chat9", "2024-01-01 10:00:00", "hello all", sender="+15550000001"),
        make_message("s2", "chat9", "2024-01-01 10:01:00", "Maya named the group hello", kind="system"),
    ]
    archive = FakeArchive(rows, [{"id": "chat9", "is_group": True}])
    assert archive.message("s2") is None
    assert ArchiveAdapter(_Plain(rows)).message("s2") is None
    assert ArchiveAdapter(_Plain(rows)).message("s1")["text"] == "hello all"


class _Plain:
    """The real archive gives every row by its id, also a system row."""

    def __init__(self, rows: list) -> None:
        self._rows = {row["id"]: row for row in rows}
        self.asked: list[str] = []

    def message(self, message_id: str) -> dict | None:
        self.asked.append(message_id)
        return self._rows.get(message_id)


def test_the_adapter_asks_the_archive_for_one_message() -> None:
    inner = _Plain(tiny_messages())
    adapter = ArchiveAdapter(inner)
    assert adapter.message("a2") == tiny_archive().message("a2")
    assert adapter.message("zz") is None
    assert inner.asked == ["a2", "zz"]


def test_the_frame_archive_gives_the_same_message_as_the_fake() -> None:
    frame = FrameArchive(frame_of(tiny_messages()))
    fake = tiny_archive()
    for message_id in ("a1", "a9", "g5", "p3"):
        ours, theirs = frame.message(message_id), fake.message(message_id)
        assert ours is not None and theirs is not None
        for key in ("id", "conversation", "timestamp", "is_from_me", "sender", "text", "kind"):
            assert ours[key] == theirs[key], (message_id, key)
    assert frame.message("nope") is None


def test_the_meaning_fake_returns_the_scripted_hits_and_records_the_call() -> None:
    hits = [hit(MAYA_PHONE, "a1", "a4", 0.9), hit(JONAS_PHONE, "j1", "j3", 0.5)]
    fake = FakeMeaning(hits)
    found = fake.search("the lease", conversations=[MAYA_PHONE], since="2021-01-01", until=None, limit=1)
    assert found == [{"conversation": MAYA_PHONE, "first_message_id": "a1", "last_message_id": "a4", "score": 0.9}]
    assert fake.calls == [
        {"query": "the lease", "conversations": [MAYA_PHONE], "since": "2021-01-01", "until": None, "limit": 1}
    ]
    found[0]["score"] = 0.0
    assert fake.search("again", conversations=None, since=None, until=None, limit=5)[0]["score"] == 0.9


def test_the_meaning_fake_can_answer_each_query_and_can_fail() -> None:
    fake = FakeMeaning(by_query={"bread": [hit(JONAS_PHONE, "j1", "j3")]})
    assert len(fake.search("bread", conversations=None, since=None, until=None, limit=5)) == 1
    assert fake.search("lease", conversations=None, since=None, until=None, limit=5) == []
    broken = FakeMeaning(error=RuntimeError("the index is gone"))
    with pytest.raises(RuntimeError):
        broken.search("bread", conversations=None, since=None, until=None, limit=5)
    assert len(broken.calls) == 1


def _story() -> list:
    return [
        period("2021", "2021-03-04", "2021-03-04", 4, "They sign a lease.", [moment("2021-03-04", "Lease", "Signed.")]),
        period("2023", "2023-07-11", "2023-07-11", 3, "The rent goes up."),
    ]


def test_the_digest_fake_returns_the_scripted_periods_and_records_the_calls() -> None:
    fake = FakeDigests({MAYA_PHONE: _story()}, usd=0.4)
    cost = fake.estimate(MAYA_PHONE, since=None, until="2023-12-31")
    assert cost == {"periods": 2, "periods_cached": 0, "usd": 0.4}
    token = CancelToken()
    result = fake.digest(MAYA_PHONE, since=None, until="2023-12-31", cancel=token)
    assert [p["label"] for p in result["periods"]] == ["2021", "2023"] and result["periods_failed"] == 0
    assert result["periods"][0]["moments"][0] == {
        "date": "2021-03-04", "title": "Lease", "detail": "Signed.", "quote": "", "message_ids": [],
    }
    assert fake.calls == [
        ("estimate", {"conversation_id": MAYA_PHONE, "since": None, "until": "2023-12-31"}),
        ("digest", {"conversation_id": MAYA_PHONE, "since": None, "until": "2023-12-31"}),
    ]
    assert fake.estimate(JONAS_PHONE, since=None, until=None) == {"periods": 0, "periods_cached": 0, "usd": 0.0}
    assert fake.digest(JONAS_PHONE, since=None, until=None, cancel=token) == {"periods": [], "periods_failed": 0}


def test_a_cached_digest_costs_nothing() -> None:
    fake = FakeDigests({MAYA_PHONE: _story()}, usd=3.0, cached=True)
    assert fake.estimate(MAYA_PHONE, since=None, until=None) == {"periods": 2, "periods_cached": 2, "usd": 0.0}


def test_the_digest_fake_can_fail_in_each_call() -> None:
    token = CancelToken()
    with pytest.raises(RuntimeError):
        FakeDigests({}, estimate_error=RuntimeError("no price")).estimate(MAYA_PHONE, since=None, until=None)
    broken = FakeDigests({MAYA_PHONE: _story()}, error=RuntimeError("the map step failed"), failed=1)
    assert broken.estimate(MAYA_PHONE, since=None, until=None)["periods"] == 2
    with pytest.raises(RuntimeError):
        broken.digest(MAYA_PHONE, since=None, until=None, cancel=token)
    partial = FakeDigests({MAYA_PHONE: _story()}, failed=1)
    assert partial.digest(MAYA_PHONE, since=None, until=None, cancel=token)["periods_failed"] == 1


def test_the_digest_fake_honours_a_cancel() -> None:
    token = CancelToken()
    fake = FakeDigests({MAYA_PHONE: _story()}, during=token.cancel)
    with pytest.raises(DigestStopped):
        fake.digest(MAYA_PHONE, since=None, until=None, cancel=token)
    deaf = FakeDigests({MAYA_PHONE: _story()}, during=token.cancel, honour_cancel=False)
    assert len(deaf.digest(MAYA_PHONE, since=None, until=None, cancel=token)["periods"]) == 2

