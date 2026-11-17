"""Regression tests for the findings of the last review."""

import contextlib
import io
import json
import unicodedata
from typing import Any

import pytest

from chat_engine.archive_adapter import ArchiveAdapter
from chat_engine.cancel import CancelToken
from chat_engine.fake_archive import FakeArchive, make_message
from chat_engine.fake_llm import call, say
from chat_engine.frame_archive import FrameArchive
from chat_engine.labeler import MaskingLabeler
from chat_engine.local_answer import local_answer
from chat_engine.registry import ConversationAliases, EvidenceRegistry
from chat_engine.render import Limits, estimate_tokens
from chat_engine.scope import Scope
from chat_engine.session import ChatSession
from chat_engine.tools import ToolBox
from chat_engine.verify import unverified_quotes
from tests.chat_engine.frames import frame_of
from tests.chat_engine.helpers import all_text, last_user, make_engine, ref_of_line, tool_results
from tests.chat_engine.tiny import (
    CONVERSATIONS,
    MAYA_PHONE,
    SAVED_NAMES,
    UNSAVED_EMAIL,
    UNSAVED_PHONE,
    tiny_archive,
    tiny_labeler,
    tiny_messages,
)

ODD = "o'brien.unsaved@example.org"
ACCENT = "josé.unsaved@example.org"
KNOWN = [UNSAVED_PHONE, UNSAVED_EMAIL, "+447911123456", "+17203142024", ODD, ACCENT]


def _labeler() -> MaskingLabeler:
    return MaskingLabeler(SAVED_NAMES, known_handles=KNOWN)


# --- 1 and 3: every address of an unsaved person ------------------------------


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        ("메일은 rk.unsaved@example.org입니다", "메일은 an unsaved email contact입니다"),
        ("자료는 rk.unsaved@example.org로 보내줘", "자료는 an unsaved email contact로 보내줘"),
        ("メールはrk.unsaved@example.orgです", "メールはan unsaved email contactです"),
        ("邮箱是rk.unsaved@example.org吗", "邮箱是an unsaved email contact吗"),
        ("it is _rk.unsaved@example.org_ now", "it is _an unsaved email contact_ now"),
        ("rk.unsaved@example.org-thanks", "an unsaved email contact-thanks"),
        ("rk.unsaved@example.org2", "an unsaved email contact2"),
        ("write to o'brien.unsaved@example.org now", "write to an unsaved email contact now"),
        ("write to O'Brien.Unsaved@Example.org now", "write to an unsaved email contact now"),
        ("write to josé.unsaved@example.org now", "write to an unsaved email contact now"),
    ],
)
def test_an_unsaved_address_is_masked_next_to_any_character(text: str, masked: str) -> None:
    assert _labeler().redact(text) == masked


def test_an_unsaved_address_next_to_korean_never_reaches_the_model() -> None:
    texts = [
        "체육관 사람 메일은 rk.unsaved@example.org입니다",
        "체육관 자료는 rk.unsaved@example.org로 보내줘",
        "체육관 gym guy: room 4, 555-0199",
    ]
    rows = [
        *tiny_messages(),
        *(
            make_message(f"k{i}", MAYA_PHONE, f"2025-03-03 10:0{i}:00", text, sender=MAYA_PHONE)
            for i, text in enumerate(texts)
        ),
    ]
    engine, fake, _ = make_engine(
        [[call("search_messages", "s1", query="체육관")], say("ok")],
        archive=FakeArchive(rows, CONVERSATIONS), labeler=MaskingLabeler(SAVED_NAMES),
    )
    events: list[dict] = []
    result = engine.ask("체육관 사람 메일이 뭐였지? rk.unsaved@example.org였나", on_event=events.append)
    seen = "\n".join(all_text(request) for request in fake.requests)
    seen += "\n".join(event.get("text", "") for event in events if event["type"] == "status")
    seen += "\n".join(step["text"] for step in result["steps"])
    assert "rk.unsaved" not in seen and "555-0199" not in seen
    assert "room 4, a number ending in 0199" in seen
    assert "메일은 an unsaved email contact입니다" in seen


def test_an_address_that_is_not_a_usual_address_is_still_a_handle_of_the_archive() -> None:
    rows = [
        *tiny_messages(),
        make_message("o1", ODD, "2025-03-03 10:00:00", "hello from the kayak club", sender=ODD),
        make_message("o2", MAYA_PHONE, "2025-03-04 10:00:00", f"kayak: write to {ODD}", sender=MAYA_PHONE),
    ]
    described = [*CONVERSATIONS, {"id": ODD, "is_group": False, "name": "", "members": [ODD]}]
    engine, fake, _ = make_engine(
        [say("ok")], archive=FakeArchive(rows, described), labeler=MaskingLabeler(SAVED_NAMES)
    )
    engine.ask("who has the kayak?")
    text = all_text(fake.requests[0])
    assert "brien" not in text and "kayak: write to an unsaved email contact" in text


def test_a_second_handle_of_a_direct_thread_in_an_export_is_masked() -> None:
    dana = "+17205550199"
    rows = [
        make_message("d1", dana, "2025-01-15 14:00:00", "hey this is Dana, kayak friday?", sender=dana),
        make_message("d2", dana, "2025-01-15 14:05:00", "kayak: my mail is dana.unsaved@example.org",
                     sender="dana.unsaved@example.org"),
        make_message("d3", MAYA_PHONE, "2025-01-16 14:05:00", "kayak: ask dana.unsaved@example.org", sender=MAYA_PHONE),
    ]
    frame = FrameArchive(frame_of(rows))
    assert frame.conversation(dana)["members"] == [dana, "dana.unsaved@example.org"]
    for archive in (frame, FakeArchive(rows)):
        engine, fake, _ = make_engine([say("ok")], archive=archive, labeler=MaskingLabeler(SAVED_NAMES))
        engine.ask("who has the kayak?")
        assert "dana.unsaved" not in all_text(fake.requests[0])


def test_a_group_name_made_of_two_numbers_is_masked_number_by_number() -> None:
    group = {"id": "chat9", "is_group": True, "name": "+13035550144, +13035550155", "members": ["+13035550144"]}
    assert MaskingLabeler({}).conversation(group) == "a number ending in 0144, a number ending in 0155"
    saved = MaskingLabeler({"+13035550155": "Casey"})
    assert saved.conversation(group) == "a number ending in 0144, Casey"


# --- 4: the local form of a number, and marks between its groups --------------


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        ("room 4, 555-0199", "room 4, a number ending in 0199"),
        ("555-0199, 2pm", "a number ending in 0199, 2pm"),
        ("555-0199 / 555-0123", "a number ending in 0199 / 555-0123"),
        ("Dana 555-0199 (2nd line)", "Dana a number ending in 0199 (2nd line)"),
        ("call 720 -- 555 -- 0199 now", "call a number ending in 0199 now"),
        ("call +1 720 555\n0199 now", "call a number ending in 0199 now"),
        ("room 4, 720-555-0199", "room 4, a number ending in 0199"),
        ("gate 12: (720) 555-0199", "gate 12: a number ending in 0199"),
    ],
)
def test_a_number_is_masked_next_to_other_digits(text: str, masked: str) -> None:
    assert _labeler().redact(text) == masked


@pytest.mark.parametrize(
    "text",
    [
        "Other cell 303 555 0199",
        "Other cell (303) 555-0199",
        "we met on 3/14/2024 at noon",
        "the code 3-1-4-2-0-2-4 opens it",
        "Maya is 415-555-0123, room 4, 555-0123",
        "totals 2024-01: 1,234 and 2024-02: 987",
    ],
)
def test_digits_that_are_not_the_number_of_an_unsaved_person_stay(text: str) -> None:
    assert _labeler().redact(text) == text


# --- 2: the session after a question that did not finish ----------------------


def test_a_late_close_of_a_cancelled_question_does_not_touch_a_later_question() -> None:
    session, cancel = ChatSession(), CancelToken()
    engine, _, _ = make_engine([say("Nothing found.", chunk=4)])
    first = engine.stream("zzz qqq?", session=session, cancel=cancel)
    for event in first:
        if event["type"] == "delta":
            cancel.cancel()
        if event["type"] == "done":
            break
    assert event["payload"]["cancelled"] and len(session.registry) == 0

    def second(request: Any) -> list:
        return say(f'Jonas named it "Clint Yeastwood" [{ref_of_line(last_user(request), "Clint Yeastwood")}].')

    engine, _, _ = make_engine([second])
    engine.ask("what did Jonas name his sourdough starter?", session=session)
    ref = session.turns[0].cited[0]
    before = json.dumps(session.to_dict(), sort_keys=True)
    first.close()
    assert json.dumps(session.to_dict(), sort_keys=True) == before
    assert session.registry.get(ref)["text"] == "its name is Clint Yeastwood"


def test_a_question_that_was_left_open_is_rolled_back_when_the_next_one_starts() -> None:
    session = ChatSession()
    engine, _, _ = make_engine([say("The lease was signed.", chunk=4)])
    left_open = engine.stream("what did Maya say about the lease?", session=session)
    for event in left_open:
        if event["type"] == "delta":
            break
    assert len(session.registry) > 0
    fresh = ChatSession()
    for chat in (session, fresh):
        engine, _, _ = make_engine([say("It is Clint Yeastwood.")])
        engine.ask("what did Jonas name his sourdough starter?", session=chat)
    assert session.to_dict() == fresh.to_dict()
    left_open.close()
    assert session.to_dict() == fresh.to_dict()


# --- 5: an archive with no reader, behind the adapter -------------------------


@pytest.mark.parametrize(("order", "first_line"), [("oldest", "First 3 messages"), ("newest", "Latest 3 messages")])
def test_the_adapter_walks_pages_when_the_archive_has_no_reader(order: str, first_line: str) -> None:
    script = [[call("search_messages", "s1", query="", conversations=["Jonas Weber"], order=order, limit=3)], say("ok")]
    engine, fake, _ = make_engine(script, archive=ArchiveAdapter(tiny_archive()))
    events = list(engine.stream("zz?"))
    assert events[-1]["type"] == "done"
    assert tool_results(fake.requests[1])[0].startswith(first_line)
    assert ArchiveAdapter(tiny_archive()).iter_messages is None


def test_the_adapter_reads_in_order_when_the_archive_can() -> None:
    class InOrder(FakeArchive):
        def iter_messages(self, conversation_id: str, **keywords: Any) -> Any:
            self._record("iter_messages", conversation_id=conversation_id, **keywords)
            return iter(self._filtered([conversation_id], keywords.get("since"), keywords.get("until"), None))

    inner = InOrder(tiny_messages(), CONVERSATIONS)
    reader = ArchiveAdapter(inner).iter_messages
    assert [m["id"] for m in reader(MAYA_PHONE, since="2023-01-01", until="2023-12-31")] == ["a5", "a6", "a7"]
    assert inner.calls[-1] == (
        "iter_messages",
        {"conversation_id": MAYA_PHONE, "since": "2023-01-01", "until": "2023-12-31", "include_system": False},
    )


# --- 6: search terms that the archive takes as plain text ---------------------


def _plain_text_archives() -> tuple[FrameArchive, FakeArchive]:
    chat = "+15550000001"
    rows = [
        make_message("k1", chat, "2024-03-01 10:00:00", "내일 학교에서 만나자", sender=chat),
        make_message("k2", chat, "2024-03-01 10:01:00", "학교 도서관 앞", sender=chat),
        make_message("k3", chat, "2024-03-01 10:02:00", "the lease - signed today", sender=chat),
        make_message("k4", chat, "2024-03-01 10:03:00", "the lease got signed", sender=chat),
        make_message("k5", chat, "2024-03-01 10:04:00", "filler " * 3000 + "東京タワー at the end", sender=chat),
        make_message("k6", chat, "2024-03-01 10:05:00", "東京タワー at the start", sender=chat),
        make_message("k7", chat, "2024-03-01 10:06:00", unicodedata.normalize("NFD", "가방 がっこう"), sender=chat),
    ]
    return FrameArchive(frame_of(rows)), FakeArchive(rows)


def test_a_star_after_a_word_without_spaces_and_a_text_in_another_form_are_found() -> None:
    frame, _ = _plain_text_archives()
    assert sorted(h["message"]["id"] for h in frame.search("학교*")["hits"]) == ["k1", "k2"]
    assert frame.search("학*")["total"] == 2 and frame.search("東京*")["total"] == 1
    assert [h["message"]["id"] for h in frame.search("가방")["hits"]] == ["k7"]
    found = frame.search("東京タワー")
    assert [h["message"]["id"] for h in found["hits"]] == ["k6"] and found["hits"][0]["ranges"] == [[0, 5]]
    assert frame.count(text="東京タワー") == 1


def test_a_sign_alone_in_a_query_of_the_model_is_not_a_search_term() -> None:
    for archive in _plain_text_archives():
        searches = [
            call("search_messages", "s1", query="lease - signed"),
            call("search_messages", "s2", query="lease & signed"),
        ]
        script = [searches, say("ok")]
        engine, fake, _ = make_engine(script, archive=archive)
        engine.ask("zz?")
        for result in tool_results(fake.requests[1]):
            assert "2 messages match" in result, result.splitlines()[0]


# --- 7: a reference that the answer could not cite ----------------------------


def test_a_reference_that_could_not_be_cited_is_not_kept_for_the_next_turn() -> None:
    session = ChatSession()
    engine, _, archive = make_engine([say("Maya signed the lease in March [m5]. See [3] and [m5, m6].")])
    first = engine.ask("zzz qqq?", session=session)
    assert first["answer"] == "Maya signed the lease in March. See (3) and."
    assert first["invalid_references"] == ["m5", "m6"]
    assert session.turns[0].answer == "Maya signed the lease in March. See (3) and."

    def repeat(request: Any) -> list:
        return say([m.content for m in request.messages if m.role == "assistant"][0])

    engine, _, _ = make_engine([[call("search_messages", "s3", query="sourdough", limit=20)], repeat], archive=archive)
    again = engine.ask("say that again and check the sourdough?", session=session)
    assert again["citations"] == [] and "[1]" not in again["answer"]


def test_a_reference_that_was_cited_stays_in_the_stored_answer() -> None:
    session = ChatSession()

    def cite(request: Any) -> list:
        return say(f"Signed [{ref_of_line(last_user(request), 'the lease got signed')}] [m99].")

    engine, _, _ = make_engine([cite])
    engine.ask("when was the lease signed?", session=session)
    assert session.turns[0].answer == f"Signed [{session.turns[0].cited[0]}]."


# --- 8 and 9: budgets --------------------------------------------------------


@pytest.mark.parametrize("budget", [20, 60, 300])
def test_a_refused_call_stays_inside_its_budget(budget: int) -> None:
    box = ToolBox(tiny_archive(), tiny_labeler(), EvidenceRegistry(), ConversationAliases(), Scope(), Limits())
    calls = [
        ("search_messages", {"query": "lease", "conversations": ["\U0001f600" * 100]}),
        ("read_around", {"reference": "\U0001f600" * 100}),
        ("no_such_tool_" + "x" * 300, {}),
    ]
    for tool, arguments in calls:
        outcome = box.prepare(tool, json.dumps(arguments)).run(budget)
        assert not outcome.ok and estimate_tokens(outcome.text) <= budget


def test_lookups_past_the_cap_of_one_step_get_no_answer_of_their_own() -> None:
    steps = [[call("count_messages", f"c{i}", text=f"lease x{i}") for i in range(200)], say("done")]
    engine, fake, _ = make_engine(steps, total_tool_tokens=600, result_tokens=300, max_calls_per_step=4)
    result = engine.ask("zz?")
    request = fake.requests[1]
    answers = [m for m in request.messages if m.role == "tool"]
    called = [c.id for m in request.messages if m.role == "assistant" for c in m.tool_calls]
    assert result["usage"]["tool_calls"] == 4 and len(answers) == 4 and called == [f"c{i}" for i in range(4)]
    assert "196 more lookups of this step were not run" in answers[-1].content
    assert sum(estimate_tokens(m.content) for m in answers) <= 600


def test_a_handle_learned_while_a_result_is_written_does_not_leave_a_message_without_a_line() -> None:
    import re

    group, ann, bob = "chat55", "+13035550111", "+13035550122"
    spellings = "ask 303-555-0122 or (303) 555-0122 or 3035550122 or 303.555.0122"
    rows = [
        make_message(f"g{i:02d}", group, f"2024-01-01 10:{i:02d}:00", f"kayak note {i}: {spellings}", sender=ann)
        for i in range(12)
    ]
    rows.append(
        make_message("g99", group, "2024-01-01 10:30:00", "paddle: this is my number, I left the group", sender=bob)
    )
    described = [{"id": group, "is_group": True, "name": "Kayak club", "members": [ann]}]
    line = re.compile(r"^(m\d+) ", re.MULTILINE)
    for budget in (600, 3000):
        registry = EvidenceRegistry()
        archive = FakeArchive(rows, described)
        box = ToolBox(archive, MaskingLabeler({}), registry, ConversationAliases(), Scope(), Limits())
        first = box.prepare("search_messages", json.dumps({"query": "note 0", "limit": 1})).run(3000)
        shown = set(line.findall(first.text))
        outcome = box.prepare("read_around", json.dumps({"reference": "m1", "radius": 15})).run(budget)
        shown |= set(line.findall(outcome.text))
        assert estimate_tokens(outcome.text) <= budget
        assert [f"m{i}" for i in range(1, len(registry) + 1) if f"m{i}" not in shown] == []
        assert "3035550122" not in outcome.text and "555-0122" not in outcome.text


# --- 10 and the notes of the review -------------------------------------------


def test_the_volume_run_writes_no_error_text_to_the_log() -> None:
    from chat_engine.eval import volume

    original = FrameArchive.stats

    def failing(self: FrameArchive, conversation_id: str, **_keywords: Any) -> dict:
        row = self.messages(conversation_id, limit=1)["messages"][0]
        raise ValueError(f"cannot read row of {row['conversation']}: {row['text'][:30]!r}")

    printed: list[str] = []
    errors = io.StringIO()
    FrameArchive.stats = failing  # type: ignore[method-assign]
    try:
        with contextlib.redirect_stderr(errors):
            code = volume.main(["--invented", "600", "--repeat", "1"], out=printed.append)
    finally:
        FrameArchive.stats = original  # type: ignore[method-assign]
    assert code == 1 and any("failed: ArchiveError" in line for line in printed)
    assert errors.getvalue() == ""


def test_the_parts_of_a_quote_must_be_near_each_other() -> None:
    long = "the lease is a long story. " + "filler words go here. " * 60 + "nobody signed anything in the end."
    assert unverified_quotes('She wrote "the lease ... nobody signed anything".', [long]) == [
        "the lease ... nobody signed anything"
    ]
    near = "the lease, as you know, was signed by nobody in the end."
    assert unverified_quotes('She wrote "the lease ... signed by nobody".', [near]) == []


def test_a_flagged_quote_is_listed_without_the_references_inside_it() -> None:
    result, _ = _ask("what about the lease?", 'Maya wrote "the lease was never signed [m1] at all [3]".')
    assert result["unverified_quotes"] == ["the lease was never signed at all (3)"]


def test_a_number_in_brackets_in_a_message_is_not_shown_as_a_citation_by_the_local_answer() -> None:
    extra = make_message("x1", MAYA_PHONE, "2025-03-01 10:00:00", "see item [3] on the packing list", sender=MAYA_PHONE)
    archive = FakeArchive([*tiny_messages(), extra], CONVERSATIONS)
    result = local_answer(archive, "packing list?", labeler=tiny_labeler())
    assert '"see item (3) on the packing list" [1]' in result["answer"]


def _ask(question: str, answer: str) -> tuple[dict, Any]:
    engine, fake, _ = make_engine([say(answer)])
    return engine.ask(question), fake
