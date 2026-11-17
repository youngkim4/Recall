"""`FrameArchive`: the archive port over an export frame. Invented rows only."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from chat_engine.eval.synthetic import build_synthetic_archive
from chat_engine.fake_archive import FakeArchive, make_message
from chat_engine.frame_archive import FrameArchive
from tests.chat_engine.frames import frame_of
from tests.chat_engine.tiny import (
    CONVERSATIONS,
    GROUP,
    JONAS_PHONE,
    MAYA_EMAIL,
    MAYA_PHONE,
    UNSAVED_PHONE,
    tiny_messages,
)

SYNTHETIC = build_synthetic_archive()
QUERIES = [
    "lease", "LEASE", "lease signed", "the lease", '"the lease"', '"lease the"', "sourd*", "a*", "lis*  flights",
    "ótimo", "otimo", "apartamento cozinha", "zzzz", "", '"', '"lease', "lease  ", "  ", "marathon", "sourdough",
    '"sourdough starter"', "starter name", "ferry", "you", "the", "it's", "pinecone-42-lantern", "42",
]
WORLDS = {
    "tiny": (tiny_messages(), CONVERSATIONS),
    "synthetic": (list(SYNTHETIC.messages), list(SYNTHETIC.conversations)),
}


def _pair(name: str) -> tuple[FrameArchive, FakeArchive]:
    messages, conversations = WORLDS[name]
    return FrameArchive(frame_of(messages, conversations)), FakeArchive(messages, conversations)


def _ids(messages: list) -> list[str]:
    return [message["id"] for message in messages]


def _same_messages(left: list, right: list) -> None:
    assert _ids(left) == _ids(right)
    for one, other in zip(left, right, strict=True):
        for key in ("conversation", "timestamp", "is_from_me", "sender", "text", "kind", "attachments"):
            assert one[key] == other[key], (one["id"], key)


@pytest.fixture(scope="module", params=sorted(WORLDS))
def pair(request: pytest.FixtureRequest) -> tuple[FrameArchive, FakeArchive]:
    return _pair(request.param)


# --- the same answers as the fake archive -------------------------------------


def test_summary_and_conversations_are_the_same(pair: tuple[FrameArchive, FakeArchive]) -> None:
    frame, fake = pair
    assert frame.summary() == fake.summary()
    for order in ("count", "recent"):
        ours, theirs = frame.conversations(order=order), fake.conversations(order=order)
        assert [c["id"] for c in ours] == [c["id"] for c in theirs]
        for one, other in zip(ours, theirs, strict=True):
            for key in ("is_group", "name", "message_count", "sent_count", "received_count",
                        "first_timestamp", "last_timestamp"):
                assert one[key] == other[key], (one["id"], key)
            assert frame.conversation(one["id"]) == one
    assert frame.conversations(limit=2, offset=1) == frame.conversations()[1:3]
    assert frame.conversations(limit=2, offset=-4) == frame.conversations()[:2]
    assert frame.conversation("nobody") is None


def test_pages_and_context_are_the_same(pair: tuple[FrameArchive, FakeArchive]) -> None:
    frame, fake = pair
    for conversation in fake.conversations()[:4]:
        cid = conversation["id"]
        whole = fake.messages(cid, limit=100_000)["messages"]
        assert _ids(frame.messages(cid, limit=100_000)["messages"]) == _ids(whole)
        anchors = {whole[0]["id"], whole[len(whole) // 2]["id"], whole[-1]["id"]}
        for limit in (1, 2, 5, 50):
            ours, theirs = frame.messages(cid, limit=limit), fake.messages(cid, limit=limit)
            _same_messages(ours["messages"], theirs["messages"])
            for anchor in anchors:
                for cursor in ("around", "before", "after"):
                    ours = frame.messages(cid, limit=limit, **{cursor: anchor})
                    theirs = fake.messages(cid, limit=limit, **{cursor: anchor})
                    assert _ids(ours["messages"]) == _ids(theirs["messages"]), (cursor, anchor, limit)
                    assert (ours["has_older"], ours["has_newer"]) == (theirs["has_older"], theirs["has_newer"])
        for anchor in anchors:
            for radius in (0, 1, 6):
                _same_messages(frame.context(anchor, radius=radius), fake.context(anchor, radius=radius))


def test_search_is_the_same(pair: tuple[FrameArchive, FakeArchive]) -> None:
    frame, fake = pair
    top = [c["id"] for c in fake.conversations()[:2]]
    filters: list[dict[str, Any]] = [
        {},
        {"conversations": top},
        {"conversations": top[:1], "from_me": True},
        {"since": "2023-01-01", "until": "2024-12-31"},
        {"from_me": False, "since": "2021-03-04"},
        {"conversations": []},
        {"conversations": ["nobody"]},
    ]
    for query in QUERIES:
        for extra in filters:
            for order in ("newest", "oldest"):
                ours = frame.search(query, limit=7, order=order, **extra)
                theirs = fake.search(query, limit=7, order=order, **extra)
                assert ours["total"] == theirs["total"], (query, extra)
                assert _ids([hit["message"] for hit in ours["hits"]]) == _ids(
                    [hit["message"] for hit in theirs["hits"]]
                ), (query, extra, order)
                assert [hit["ranges"] for hit in ours["hits"]] == [hit["ranges"] for hit in theirs["hits"]]
            assert frame.count(text=query, **extra) == fake.count(text=query, **extra), (query, extra)
        paged = frame.search(query, limit=3, offset=2)
        assert _ids([h["message"] for h in paged["hits"]]) == _ids(
            [h["message"] for h in fake.search(query, limit=3, offset=2)["hits"]]
        )
    ranked = frame.search("lease", limit=50, order="relevance")
    assert ranked["total"] == fake.search("lease")["total"]
    assert sorted(_ids([h["message"] for h in ranked["hits"]])) == sorted(
        _ids([h["message"] for h in fake.search("lease", limit=50)["hits"]])
    )


def test_counts_are_the_same(pair: tuple[FrameArchive, FakeArchive]) -> None:
    frame, fake = pair
    top = [c["id"] for c in fake.conversations()[:3]]
    for extra in (
        {}, {"conversations": top}, {"conversations": top[0]}, {"from_me": True},
        {"since": "2024-01-01"}, {"until": "2023-06-30", "from_me": False}, {"text": "lease"},
        {"text": "the", "conversations": top[:1]}, {"conversations": []},
    ):
        assert frame.count(**extra) == fake.count(**extra), extra
        for group in ("day", "month", "year", "weekday_hour", "conversation", "sender"):
            assert frame.count(group_by=group, **extra) == fake.count(group_by=group, **extra), (group, extra)


def test_stats_are_the_same(pair: tuple[FrameArchive, FakeArchive]) -> None:
    frame, fake = pair
    for conversation in [*fake.conversations()[:5], {"id": "nobody"}]:
        periods = ({}, {"since": "2023-01-01"}, {"since": "2021-01-01", "until": "2024-12-31"}, {"until": "1999-01-01"})
        for days in periods:
            ours, theirs = frame.stats(conversation["id"], **days), fake.stats(conversation["id"], **days)
            assert ours.keys() == theirs.keys()
            for key, value in theirs.items():
                if isinstance(value, float):
                    assert ours[key] == pytest.approx(value), key
                else:
                    assert ours[key] == value, key


def test_the_scripted_questions_pass_on_the_frame() -> None:
    from chat_engine.engine import EngineConfig
    from chat_engine.eval.harness import evaluate
    from chat_engine.eval.oracle import SCRIPTS
    from chat_engine.eval.questions import build_questions
    from chat_engine.fake_llm import FakeLLM

    frame = frame_of(SYNTHETIC.messages, SYNTHETIC.conversations)

    class OnFrame:
        """The synthetic archive, read through the export frame."""

        def __getattr__(self, name: str) -> Any:
            return getattr(SYNTHETIC, name)

        def archive(self) -> FrameArchive:
            return FrameArchive(frame)

    config = EngineConfig(model="scripted", max_output_tokens=600)
    rows = evaluate(OnFrame(), build_questions(SYNTHETIC), lambda q: FakeLLM(SCRIPTS[q.id]()), config)  # type: ignore[arg-type]
    failed = {
        row.id: (row.error or [name for name, ok in row.checks.items() if not ok]) for row in rows if not row.passed
    }
    assert failed == {}


# --- what the frame holds -----------------------------------------------------


def _extra(message_id: str, text: str, **fields: str) -> dict[str, str]:
    row = {
        "message_id": message_id, "timestamp": "2024-03-01 10:00:00", "sender": JONAS_PHONE, "text": text,
        "chat_id": JONAS_PHONE, "is_from_me": "0",
    }
    return {**row, **fields}


def _with(*extra: dict[str, str]) -> FrameArchive:
    return FrameArchive(frame_of(tiny_messages(), CONVERSATIONS, extra))


def test_reaction_phrases_are_not_messages() -> None:
    plain = _with()
    reactions = [
        _extra("r1", "Loved “sourdough starter is alive”"),
        _extra("r2", 'Laughed at "its name is Clint Yeastwood"'),
        _extra("r3", "Emphasized “a text\nof two lines”", is_from_me="1", sender=""),
        _extra("r4", "Liked “sourdough again this weekend?”"),
        _extra("r5", "Disliked “sourdough”"),
        _extra("r6", "Questioned “sourdough”"),
    ]
    typed = [
        _extra("t1", 'Loved "The Sourdough Show" so much'),
        _extra("t2", "Liked it a lot, sourdough is great"),
        _extra("t3", "Loved"),
        _extra("t4", "loved “sourdough”"),
    ]
    archive = _with(*reactions, *typed)
    assert archive.dropped["reactions"] == 6
    assert archive.summary()["messages"] == plain.summary()["messages"] + 4
    found = [hit["message"]["id"] for hit in archive.search("sourdough", limit=50)["hits"]]
    assert sorted(found) == ["j1", "j4", "t1", "t2", "t4"]
    assert archive.count(text="sourdough") == 5
    ids = _ids(archive.messages(JONAS_PHONE, limit=100)["messages"])
    assert not {"r1", "r2", "r3", "r4", "r5", "r6"} & set(ids) and {"t1", "t2", "t3", "t4"} <= set(ids)
    assert archive.conversation(JONAS_PHONE)["message_count"] == plain.conversation(JONAS_PHONE)["message_count"] + 4
    assert archive.context("r1") == []


def test_attachment_rows_have_no_text_and_keep_their_types() -> None:
    archive = _with(
        _extra("x1", "￼", attachment_types="image/heic"),
        _extra("x2", " ￼￼ ", attachment_types="image/png|video/quicktime"),
        _extra("x3", "[Attachment] image/jpeg | ~/Library/Messages/secret-name.jpeg", attachment_types="image/jpeg"),
        _extra("x4", "[Attachment]"),
        _extra("x5", "look ￼ at this", attachment_types="image/gif"),
        _extra("x6", "   "),
    )
    rows = {m["id"]: m for m in archive.messages(JONAS_PHONE, limit=100)["messages"]}
    assert [rows[i]["kind"] for i in ("x1", "x2", "x3", "x4", "x5", "x6")] == [
        "attachment", "attachment", "attachment", "attachment", "text", "text",
    ]
    assert [rows[i]["text"] for i in ("x1", "x2", "x3", "x4")] == ["", "", "", ""]
    assert rows["x2"]["attachments"] == ["image/png", "video/quicktime"]
    assert rows["x4"]["attachments"] == [] and rows["x3"]["attachments"] == ["image/jpeg"]
    assert rows["x5"]["text"] == "look  at this" and rows["x5"]["attachments"] == ["image/gif"]
    assert rows["x6"]["text"] == "   "
    assert archive.search("secret", limit=5)["total"] == 0
    assert archive.search("attachment", limit=5)["total"] == 0


def test_rows_that_cannot_be_read_are_dropped_and_counted() -> None:
    archive = _with(
        _extra("j1", "a second row with the id of another"),
        _extra("b1", "no day", timestamp="yesterday"),
        _extra("b2", "no such day", timestamp="2024-02-31 10:00:00"),
        _extra("b3", "short day", timestamp="2024-3-1 10:00:00"),
        _extra("b4", "no conversation", chat_id=""),
        _extra("", "no id"),
        _extra("ok", "this row is fine"),
    )
    assert archive.dropped == {"bad_rows": 5, "duplicates": 1, "reactions": 0}
    assert archive.summary()["messages"] == len(tiny_messages()) + 1
    assert archive.search("fine")["hits"][0]["message"]["id"] == "ok"
    assert archive.search("another")["total"] == 0


def test_a_frame_without_a_needed_column_is_refused() -> None:
    frame = frame_of(tiny_messages(), CONVERSATIONS).drop(columns=["chat_id", "text"])
    with pytest.raises(ValueError, match="text, chat_id"):
        FrameArchive(frame)
    bare = frame_of(tiny_messages(), CONVERSATIONS).drop(
        columns=["chat_display_name", "service", "attachment_types", "attachment_files"]
    )
    assert FrameArchive(bare).summary()["messages"] == len(tiny_messages())


def test_numbers_in_the_frame_are_read_as_text() -> None:
    frame = frame_of(tiny_messages()[:3])
    frame["message_id"] = [101, 102, 103]
    frame["is_from_me"] = [1.0, 0.0, float("nan")]
    archive = FrameArchive(frame)
    rows = archive.messages(MAYA_PHONE)["messages"]
    assert _ids(rows) == ["101", "102", "103"]
    assert [row["is_from_me"] for row in rows] == [True, False, False]


def test_groups_get_their_name_and_their_members_from_the_rows() -> None:
    archive = _with()
    group = archive.conversation(GROUP)
    assert group["is_group"] and group["name"] == "Lisbon crew"
    assert group["members"] == [MAYA_PHONE, JONAS_PHONE, UNSAVED_PHONE]
    direct = archive.conversation(MAYA_EMAIL)
    assert not direct["is_group"] and direct["name"] == "" and direct["members"] == [MAYA_EMAIL]
    assert direct["person_id"] is None


def test_what_comes_back_is_a_copy() -> None:
    archive = _with()
    page = archive.messages(GROUP)
    page["messages"][0]["text"] = "changed"
    page["messages"][-1]["attachments"].append("x/y")
    archive.conversation(GROUP)["members"].append("someone")
    archive.conversations()[0]["name"] = "changed"
    again = archive.messages(GROUP)["messages"]
    assert again[0]["text"] == "flights to Lisbon are booked" and again[-1]["attachments"] == ["image/heic"]
    assert len(archive.conversation(GROUP)["members"]) == 3
    assert archive.conversations()[0]["name"] != "changed"


# --- words the index does not split -------------------------------------------


def test_words_without_spaces_and_signs_match_as_plain_text() -> None:
    archive = _with(
        _extra("k1", "明日は東京に行く"),
        _extra("k2", "pizza 🍕 tonight? :)"),
        _extra("k3", "PIZZA🍕"),
        _extra("k4", "서울에서 만나요"),
    )
    assert _ids([h["message"] for h in archive.search("東京")["hits"]]) == ["k1"]
    assert archive.search("東京")["hits"][0]["ranges"] == [[3, 5]]
    assert archive.search("京都")["total"] == 0
    assert archive.count(text="🍕") == 2
    assert archive.search("pizza 🍕")["total"] == 2
    assert archive.search("pizza🍕")["hits"][0]["message"]["id"] == "k3"
    assert archive.search("서울")["total"] == 1
    assert archive.search(":)")["total"] == 1
    assert archive.search("🍕", conversations=[MAYA_PHONE])["total"] == 0


def test_a_very_long_text_is_kept_whole_and_searched_at_its_start() -> None:
    long = "opening words here " + ("filler " * 60_000) + "closingword"
    archive = FrameArchive(frame_of([make_message("big", MAYA_PHONE, "2024-01-01 10:00:00", long)]), index_chars=5_000)
    assert archive.messages(MAYA_PHONE)["messages"][0]["text"] == long
    assert archive.search("opening")["total"] == 1
    assert archive.search("closingword")["total"] == 0
    assert archive.search('"opening words"')["hits"][0]["ranges"] == [[0, 13]]


# --- arguments ----------------------------------------------------------------


def test_bad_arguments_are_refused_as_the_contract_says() -> None:
    archive = _with()
    with pytest.raises(ValueError):
        archive.search("lease", since="last week")
    with pytest.raises(ValueError):
        archive.count(until="2024-13-01")
    with pytest.raises(ValueError):
        archive.count(group_by="decade")
    with pytest.raises(ValueError):
        archive.messages(MAYA_PHONE, before="a1", after="a2")
    with pytest.raises(TypeError):
        archive.search(None)  # type: ignore[arg-type]
    assert archive.messages(MAYA_PHONE, around="nobody") == {"messages": [], "has_older": False, "has_newer": False}
    assert archive.messages(MAYA_PHONE, around="j1")["messages"] == []
    assert archive.messages("nobody")["messages"] == []
    assert archive.context("nobody") == []
    assert archive.search("lease", limit=0)["hits"] == [] and archive.search("lease", limit=0)["total"] > 0


# --- the file -----------------------------------------------------------------


def test_from_csv_reads_text_as_it_is(tmp_path: Path) -> None:
    tricky = [
        _extra("0042", 'she said "no, never", then left'),
        _extra("0043", "line one\nline two, with a comma"),
        _extra("0044", "NA"),
        _extra("0045", "null"),
        _extra("0046", "None"),
        _extra("0047", "nan"),
    ]
    path = tmp_path / "export.csv"
    frame_of(tiny_messages(), CONVERSATIONS, tricky).to_csv(path, index=False)
    archive = FrameArchive.from_csv(path)
    rows = {m["id"]: m["text"] for m in archive.messages(JONAS_PHONE, limit=100)["messages"]}
    assert [rows[i] for i in ("0042", "0043", "0044", "0045", "0046", "0047")] == [
        'she said "no, never", then left', "line one\nline two, with a comma", "NA", "null", "None", "nan",
    ]
    assert archive.summary()["messages"] == len(tiny_messages()) + 6
    assert archive.search("lease")["total"] == FakeArchive(tiny_messages(), CONVERSATIONS).search("lease")["total"]


def test_the_index_is_built_once_and_only_when_a_search_needs_it() -> None:
    archive = _with()
    assert archive.index_built is False
    archive.count()
    archive.messages(MAYA_PHONE)
    assert archive.index_built is False
    archive.search("lease")
    assert archive.index_built is True
    assert archive.build_index() is archive.build_index()


def test_an_empty_frame_is_an_empty_archive() -> None:
    archive = FrameArchive(pd.DataFrame(columns=["message_id", "timestamp", "sender", "text", "chat_id", "is_from_me"]))
    assert archive.summary() == {"messages": 0, "conversations": 0, "first_timestamp": None, "last_timestamp": None}
    assert archive.conversations() == [] and archive.search("x") == {"total": 0, "hits": []}
    assert archive.count() == 0 and archive.count(group_by="month") == []
    assert archive.stats("nobody")["total_messages"] == 0
