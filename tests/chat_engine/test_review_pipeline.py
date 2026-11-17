"""Regression tests for the findings of the end-to-end review."""

import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import pytest

from chat_engine.browse import browse
from chat_engine.cancel import CancelToken
from chat_engine.citations import CitationStream
from chat_engine.errors import ArchiveError, RateLimitedError
from chat_engine.fake_archive import FakeArchive, make_message
from chat_engine.fake_llm import call, say
from chat_engine.frame_archive import FrameArchive
from chat_engine.labeler import MaskingLabeler
from chat_engine.local_answer import local_answer
from chat_engine.scope import Scope
from chat_engine.session import ChatSession
from chat_engine.textnorm import query_units, surface_units
from chat_engine.verify import unverified_quotes
from tests.chat_engine.frames import frame_of
from tests.chat_engine.helpers import RAW_HANDLES, all_text, last_user, make_engine, ref_of_line, tool_results
from tests.chat_engine.tiny import (
    CONVERSATIONS,
    JONAS_PHONE,
    MAYA_PHONE,
    SAVED_NAMES,
    UNSAVED_EMAIL,
    UNSAVED_PHONE,
    tiny_labeler,
    tiny_messages,
)

RAW = [*RAW_HANDLES, UNSAVED_PHONE, UNSAVED_EMAIL]


def _ask(question: str, answer: Any, **options: Any) -> tuple[dict, Any]:
    engine, fake, _ = make_engine([answer if callable(answer) else say(answer)], archive=options.pop("archive", None))
    return engine.ask(question, **options), fake


# --- 1: the labeler knows every handle of the archive before any text goes out ---


def test_a_scoped_chat_masks_people_outside_its_scope() -> None:
    text = "the gym guy is +1 720 555 0199, or rk.unsaved@example.org"
    rows = [*tiny_messages(), make_message("x3", JONAS_PHONE, "2025-03-03 10:00:00", text, sender=JONAS_PHONE)]
    engine, fake, _ = make_engine(
        [say("ok")], archive=FakeArchive(rows, CONVERSATIONS), labeler=MaskingLabeler(SAVED_NAMES)
    )
    events: list[dict] = []
    result = engine.ask(
        "did Jonas mention the gym guy, +1 (720) 555-0199?",
        scope=Scope(conversations=(JONAS_PHONE,)),
        on_event=events.append,
    )
    seen = all_text(fake.requests[0]) + "\n".join(step["text"] for step in result["steps"])
    seen += "\n".join(event.get("text", "") for event in events if event["type"] == "status")
    assert "the gym guy is a number ending in 0199, or an unsaved email contact" in seen
    assert "Question: did Jonas mention the gym guy, a number ending in 0199?" in seen
    for raw in RAW:
        assert raw not in seen, raw


def test_a_restored_chat_masks_the_handle_in_its_earlier_question() -> None:
    session = ChatSession()
    engine, _, _ = make_engine([say("That is Dana.")], labeler=MaskingLabeler(SAVED_NAMES))
    engine.ask("who is +1 (720) 555-0199?", session=session)
    restored = ChatSession.from_dict(json.loads(json.dumps(session.to_dict())))
    engine, fake, _ = make_engine([say("No.")], labeler=MaskingLabeler(SAVED_NAMES))
    engine.ask("did Maya mention them?", session=restored, scope=Scope(conversations=(MAYA_PHONE,)))
    seen = all_text(fake.requests[0])
    assert "who is a number ending in 0199?" in seen
    for raw in RAW:
        assert raw not in seen, raw


def test_a_thread_with_an_archive_name_or_a_second_handle_is_masked_in_text() -> None:
    avery, casey, drew = "+13035550111", "+13035550133", "+13035550144"
    rows = [
        make_message(
            "a1", avery, "2025-01-01 10:00:00", "it's Avery, 303-555-0111 is my number for the kayak", sender=avery
        ),
        make_message(
            "c1", casey, "2025-01-03 10:00:00", "kayak pics are on casey.unsaved@example.org too", sender=casey
        ),
        make_message("g1", "chat9", "2025-01-04 10:00:00", "the kayak is with Drew, 303 555 0144", sender=avery),
    ]
    conversations = [
        {"id": avery, "is_group": False, "name": "Avery (gym)", "members": [avery]},
        {"id": casey, "is_group": False, "name": "", "members": [casey, "casey.unsaved@example.org"]},
        {"id": "chat9", "is_group": True, "name": "Kayak club", "members": [avery, drew]},
    ]
    engine, fake, _ = make_engine([say("ok")], archive=FakeArchive(rows, conversations), labeler=MaskingLabeler({}))
    engine.ask("who has the kayak?")
    block = last_user(fake.requests[0])
    assert "it's Avery, a number ending in 0111 is my number for the kayak" in block
    assert "kayak pics are on an unsaved email contact too" in block
    assert "the kayak is with Drew, a number ending in 0144" in block
    for raw in ("303-555-0111", "casey.unsaved", "303 555 0144"):
        assert raw not in block, raw


# --- 2: every spelling of an unsaved handle ---


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        ("Dana cell 555-0199", "Dana cell a number ending in 0199"),
        ("Dana UK cell +44 (0)7911 123456", "Dana UK cell a number ending in 3456"),
        ("Dana UK cell 07911 123456", "Dana UK cell a number ending in 3456"),
        ("Dana cell 720‑555‑0199", "Dana cell a number ending in 0199"),
        ("Dana cell 720/555/0199", "Dana cell a number ending in 0199"),
        ("Dana cell 720–555–0199", "Dana cell a number ending in 0199"),
        ("Dana cell ７２０ ５５５ ０１９９", "Dana cell a number ending in 0199"),
        ("Dana mail rk.unsaved@example.org.Thanks", "Dana mail an unsaved email contact.Thanks"),
        ("Dana mail RK.Unsaved@Example.ORG!", "Dana mail an unsaved email contact!"),
        ("see mailto:rk.unsaved@example.org", "see mailto:an unsaved email contact"),
    ],
)
def test_every_spelling_of_an_unsaved_handle_is_masked(text: str, masked: str) -> None:
    labeler = MaskingLabeler(SAVED_NAMES, known_handles=[UNSAVED_PHONE, UNSAVED_EMAIL, "+447911123456"])
    assert labeler.redact(text) == masked


@pytest.mark.parametrize(
    "text",
    [
        "Other cell 303 555 0199",
        "Maya is 415-555-0123, or 555-0123 from a landline",
        "The door code is 555-0100",
        "write to x.rk.unsaved@example.org",
        "write to rk.unsaved@example.organic",
        "on 2024-03-01 at 10:00:00 we paid 1,234,567 for 3 items",
    ],
)
def test_other_numbers_and_addresses_stay_as_they_are(text: str) -> None:
    labeler = MaskingLabeler(SAVED_NAMES, known_handles=[UNSAVED_PHONE, UNSAVED_EMAIL, "+447911123456"])
    assert labeler.redact(text) == text


def test_a_cut_never_shows_half_an_address() -> None:
    long = (
        "we talked a lot today. " * 180 + "write to rk.unsaved@example.org soon. "
        + "the keys are in the blue box. " * 17 + "also the kayak trip is on. " + "more words. " * 150
    )
    rows = [*tiny_messages(), make_message("L1", MAYA_PHONE, "2025-05-05 10:00:00", long, sender=MAYA_PHONE)]
    labeler = MaskingLabeler(SAVED_NAMES, known_handles=[UNSAVED_PHONE, UNSAVED_EMAIL])
    engine, fake, _ = make_engine(
        [[call("search_messages", "s1", query="kayak")], say("ok")], archive=FakeArchive(rows, CONVERSATIONS),
        labeler=labeler,
    )
    engine.ask("zzz?")
    out = tool_results(fake.requests[1])[0]
    assert "also the kayak trip is on" in out
    assert "ple.org" not in out and "example" not in out and "unsaved@" not in out


# --- 3: a bare number in brackets is never a citation ---


def test_a_bare_number_in_brackets_is_shown_in_round_brackets() -> None:
    stream = CitationStream(lambda ref: ref in {"m2", "m3"})
    shown = stream.feed("Signed [2]. See item [3] and [2023], or [1, 2]. Real [m3]. Mixed [m2, 3]. Kept [sic].")
    shown += stream.finish()
    assert shown == "Signed (2). See item (3) and (2023), or (1, 2). Real [1]. Mixed [2]. Kept [sic]."
    assert stream.order == ("m3", "m2") and stream.unknown == ()


def test_a_number_copied_from_a_restored_history_cites_nothing() -> None:
    history = [
        {"role": "user", "content": "who is Maya?"},
        {"role": "assistant", "content": "A friend [1]. She signed a lease [2][3]."},
    ]
    session = ChatSession.from_history(history)
    assert session.turns[0].answer == "A friend. She signed a lease."
    result, _ = _ask("and what happened with the lease?", "As said before, she signed a lease [2].", session=session)
    assert result["answer"] == "As said before, she signed a lease (2)."
    assert result["citations"] == [] and result["invalid_references"] == []


def test_a_number_in_quoted_message_text_is_not_rewritten_as_a_citation() -> None:
    extra = make_message("x1", MAYA_PHONE, "2025-03-01 10:00:00", "see item [3] on the packing list", sender=MAYA_PHONE)
    archive = FakeArchive([*tiny_messages(), extra], CONVERSATIONS)

    def answer(request: Any) -> list:
        ref = ref_of_line(last_user(request), "packing list")
        return say(f'Maya wrote "see item [3] on the packing list" [{ref}].')

    result, _ = _ask("what about the packing list?", answer, archive=archive)
    assert result["answer"] == 'Maya wrote "see item (3) on the packing list" [1].'
    assert [citation["id"] for citation in result["citations"]] == ["x1"]
    assert result["unverified_quotes"] == []


# --- 4: a turn that did not finish leaves nothing in the session ---


def test_a_cancelled_turn_leaves_the_session_as_it_was() -> None:
    cancel, session = CancelToken(), ChatSession()
    engine, fake, _ = make_engine([say("no")])
    before = json.dumps(session.to_dict(), sort_keys=True)

    def stop_at_the_search(event: dict) -> None:
        if event["type"] == "status" and event["text"] == "Looking up “maya”":
            cancel.cancel()

    result = engine.ask(
        "what did Maya say about the lease?", cancel=cancel, session=session, on_event=stop_at_the_search
    )
    assert result["cancelled"] and fake.requests == []
    assert json.dumps(session.to_dict(), sort_keys=True) == before


def test_a_failed_turn_leaves_the_session_as_it_was() -> None:
    session = ChatSession()
    engine, _, archive = make_engine([[call("search_messages", "s1", query="sourdough")], [RateLimitedError()]])
    with pytest.raises(RateLimitedError):
        engine.ask("zzz qqq?", session=session)
    assert len(session.registry) == 0 and session.aliases.snapshot() == [] and session.turns == ()
    engine, fake, _ = make_engine([say('Jonas wrote "its name is Clint Yeastwood" [m3].')], archive=archive)
    result = engine.ask("zzz qqq again?", session=session)
    assert result["citations"] == [] and result["invalid_references"] == ["m3"]
    assert result["unverified_quotes"] == ["its name is Clint Yeastwood"]


def test_a_stream_that_the_caller_closes_leaves_the_session_as_it_was() -> None:
    session = ChatSession()
    engine, _, _ = make_engine([say("The lease was signed.", chunk=4)])
    stream = engine.stream("what about the lease?", session=session)
    for event in stream:
        if event["type"] == "delta":
            break
    stream.close()
    assert len(session.registry) == 0 and session.turns == ()


def test_a_turn_that_finishes_keeps_its_evidence() -> None:
    session = ChatSession()
    engine, _, _ = make_engine([say("Signed.")])
    engine.ask("what about the lease?", session=session)
    assert len(session.registry) > 0 and len(session.turns) == 1


# --- 5: quotes ---


@pytest.mark.parametrize(
    ("answer", "flagged"),
    [
        ('Maya wrote "the lease got signed ... fifty more a month".', ["the lease got signed ... fifty more a month"]),
        ('Maya wrote "I hate him... and you... so much".', ["I hate him... and you... so much"]),
        ('마야가 "나는 절대로 월세를 내지 않을 거야"라고 썼어요.', ["나는 절대로 월세를 내지 않을 거야"]),
        ('Maya wrote _"I will never pay the rent again"_ in March.', ["I will never pay the rent again"]),
        ("마야가 '나는 절대로 월세를 내지 않을 거야'라고 썼어요.", ["나는 절대로 월세를 내지 않을 거야"]),
        ('Maya wrote "the landlord ... renew the lease" in July.', []),
        ('Maya wrote "the lease got signed btw" and "fifty more a month".', []),
        ('Maya wrote "renew the lease ... the landlord" in July.', ["renew the lease ... the landlord"]),
    ],
)
def test_a_quote_must_be_in_one_message_in_its_order(answer: str, flagged: list[str]) -> None:
    result, _ = _ask("what about the lease?", answer)
    assert result["unverified_quotes"] == flagged


def test_the_words_of_the_user_are_not_a_source_for_a_quote() -> None:
    def answer(request: Any) -> list:
        ref = ref_of_line(last_user(request), "fifty more a month")
        return say(f'Yes. Maya wrote "she hates my cooking and wants to move out" [{ref}].')

    result, _ = _ask("Did Maya ever say that she hates my cooking and wants to move out?", answer)
    assert result["unverified_quotes"] == ["she hates my cooking and wants to move out"]
    session = ChatSession()
    engine, _, _ = make_engine([say("No."), say('Earlier you asked about "the cooking and the moving out".')])
    engine.ask("what about the cooking and the moving out?", session=session)
    again = engine.ask("and now?", session=session)
    assert again["unverified_quotes"] == ["the cooking and the moving out"]


def test_a_short_quote_and_a_name_are_not_flagged() -> None:
    sources = ["the lease got signed btw", "Lisbon crew of the long summer"]
    assert unverified_quotes('She said "signed btw" in "Lisbon crew of the long summer".', sources) == []


# --- 6: words go to the archive as they are written ---


def test_surface_units_keep_the_spelling_and_fold_only_to_compare() -> None:
    written = surface_units('São Paulo "Nova Iorque" lis* LEASE lease')
    assert written == ["são", "paulo", '"nova iorque"', "lis*", "lease"]
    assert query_units("São LEASE") == ["sao", "lease"]
    assert surface_units("학교 がっこう") == ["학교", "がっこう"]


def test_korean_and_japanese_words_are_found_by_the_seed_and_by_any_search() -> None:
    rows = [
        make_message("k1", MAYA_PHONE, "2024-03-01 10:00:00", "내일 학교 도서관 에서 만나자", sender=MAYA_PHONE),
        make_message(
            "n1", MAYA_PHONE, "2024-04-01 10:00:00", "あした がっこう で パスポート を わたすね", sender=MAYA_PHONE
        ),
    ]
    cases = [
        ("학교 도서관 얘기 언제였지", "학교", "학교 도서관", "만나자"),
        ("がっこう で パスポート の はなし", "がっこう", "がっこう パスポート", "わたすね"),
    ]
    for question, plain, several, needle in cases:
        archive = FrameArchive(frame_of([*tiny_messages(), *rows], CONVERSATIONS))
        script = [
            [call("search_messages", "s1", query=plain), call("search_messages", "s2", query=several, match="any")],
            say("ok"),
        ]
        engine, fake, _ = make_engine(script, archive=archive)
        engine.ask(question)
        assert needle in last_user(fake.requests[0]), question
        found, ranked = tool_results(fake.requests[1])
        assert "1 message matches" in found and needle in found
        assert needle in ranked and "No message has" not in ranked


# --- 7: counts by sender ---


def test_two_people_with_one_mask_are_two_rows() -> None:
    group = "chat777"
    senders = (
        ["ann.unsaved@example.org"] * 3 + ["bob.unsaved@example.com"] * 2 + ["carl@example.net"] * 4
        + ["+13035550199"] * 2 + ["+12125550199"] * 5 + [MAYA_PHONE] * 6
    )
    rows = [
        make_message(f"g{i}", group, f"2024-01-01 10:{i:02d}:00", f"hello {i}", sender=sender)
        for i, sender in enumerate(senders)
    ]
    described = [{"id": group, "is_group": True, "name": "Neighbours", "members": sorted(set(senders))}]
    engine, fake, _ = make_engine(
        [[call("count_messages", "c1", group_by="sender")], say("ok")], archive=FakeArchive(rows, described)
    )
    engine.ask("who writes most in Neighbours?")
    lines = tool_results(fake.requests[1])[0].splitlines()
    assert lines[1:] == [
        "Maya Okafor: 6",
        "a number ending in 0199: 5",
        "an unsaved email contact: 4",
        "an unsaved email contact: 3",
        "a number ending in 0199: 2",
        "an unsaved email contact: 2",
    ]


# --- 8: a chat that starts again after a restart ---


def test_a_restored_history_can_carry_the_scope_of_its_chat() -> None:
    history = [
        {"role": "user", "content": "what did Jonas name his starter?"},
        {"role": "assistant", "content": "Clint Yeastwood [1]."},
    ]
    scope = Scope(conversations=(JONAS_PHONE,))
    engine, fake, _ = make_engine([say("ok")])
    engine.ask("when was that?", session=ChatSession.from_history(history, scope=scope), scope=scope)
    assert [m.content for m in fake.requests[0].messages[1:3]] == [
        "what did Jonas name his starter?", "Clint Yeastwood.",
    ]
    engine, fake, _ = make_engine([say("ok")])
    engine.ask("when was that?", session=ChatSession.from_history(history), scope=scope)
    assert "left out" in fake.requests[0].messages[2].content
    assert "changed" not in fake.requests[0].messages[2].content


# --- 9: the evidence of the newest turn comes first ---


def test_a_long_message_of_an_old_turn_does_not_push_out_the_newest_evidence() -> None:
    long = "alpha topic. " + ("filler words about the alpha plan and more filler. " * 80)
    rows = [
        make_message("a1", MAYA_PHONE, "2024-01-01 10:00:00", long, sender=MAYA_PHONE),
        make_message("b1", JONAS_PHONE, "2024-02-01 10:00:00", "bravo topic: the code is 4711", sender=JONAS_PHONE),
    ]
    seen: dict[str, str] = {}

    def first(request: Any) -> list:
        return say(f"Alpha was planned [{ref_of_line(last_user(request), 'alpha topic')}].")

    def second(request: Any) -> list:
        return say(f"The code is 4711 [{ref_of_line(last_user(request), 'bravo topic')}].")

    def third(request: Any) -> list:
        seen["block"] = last_user(request)
        return say("In February.")

    session = ChatSession()
    engine, _, _ = make_engine([first, second, third], archive=FakeArchive(rows, CONVERSATIONS))
    for question in ("what about the alpha topic?", "and the bravo topic code?", "when was that?"):
        engine.ask(question, session=session)
    assert "bravo topic: the code is 4711" in seen["block"]
    assert "alpha topic." in seen["block"] and "[cut: " in seen["block"]
    assert seen["block"].index("bravo topic") < seen["block"].index("alpha topic.")


# --- 10: a cap on the lookups of one step ---


def test_one_step_runs_a_limited_number_of_lookups() -> None:
    steps = [[call("count_messages", f"c{s}_{i}", text=f"lease x{s}q{i}") for i in range(50)] for s in range(2)]
    engine, fake, archive = make_engine([*steps, say("done")], max_calls_per_step=5)
    result = engine.ask("zz?")
    assert result["usage"]["tool_calls"] == 10
    answered = [m.content for m in fake.requests[1].messages if m.role == "tool"]
    assert len(answered) == 50 and sum("Lookup skipped" in text for text in answered) == 45
    assert Counter(name for name, _ in archive.calls)["count"] == 10


# --- 12: nothing is read after a cancel ---


@pytest.mark.parametrize("status", ["Reading your archive", "Searching your messages", "Looking up “maya”"])
def test_no_archive_call_follows_a_cancel(status: str) -> None:
    cancel = CancelToken()
    engine, fake, archive = make_engine([say("no")])
    at: dict[str, int] = {}

    def on_event(event: dict) -> None:
        if "calls" not in at and event["type"] == "status" and event["text"] == status:
            at["calls"] = len(archive.calls)
            cancel.cancel()

    result = engine.ask("what did Maya say about the lease?", cancel=cancel, on_event=on_event)
    assert result["cancelled"] and fake.requests == []
    assert archive.calls[at["calls"]:] == []


# --- 13: reading a conversation ---


def test_a_period_between_two_pages_is_empty_not_unknown() -> None:
    chat = "+15550000042"

    def clock(i: int) -> str:
        return f"10:{i // 60:02d}:{i % 60:02d}"

    rows = [make_message(f"a{i:03d}", chat, f"2020-01-01 {clock(i)}", "early words") for i in range(300)]
    rows += [make_message(f"z{i:03d}", chat, f"2022-01-01 {clock(i)}", "late words") for i in range(200)]
    archive = FakeArchive(rows)
    found = browse(archive, chat, edge="oldest", limit=5, since="2021-05-01", until="2021-05-31")
    assert found.messages == () and found.anchored and found.reached_edge
    assert len(archive.calls) < 12


def test_a_start_that_was_not_reached_is_not_called_the_first_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    from chat_engine import tool_browse

    chat = "+15550000042"
    first = datetime(2015, 1, 1, 12, 0, 0)
    rows = [
        make_message(f"d{i:04d}", chat, (first + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S"), "")
        for i in range(700)
    ]
    original = tool_browse.browse
    monkeypatch.setattr(tool_browse, "browse", lambda *args, **kwargs: original(*args, **{**kwargs, "max_hops": 1}))
    script = [
        [call("search_messages", "s1", query="", order="oldest", conversations=["a number ending in 0042"], limit=3)],
        say("ok"),
    ]
    engine, fake, _ = make_engine(script, archive=FakeArchive(rows))
    engine.ask("zz?")
    out = tool_results(fake.requests[1])[0]
    assert out.startswith("3 messages near the start of c1")
    assert "start was not reached" in out


def test_an_archive_that_reads_in_order_gives_the_first_messages_at_once() -> None:
    chat = "+15550000042"
    first = datetime(2015, 1, 1, 12, 0, 0)
    rows = [
        make_message(f"d{i:04d}", chat, (first + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S"), "")
        for i in range(700)
    ]

    class InOrder(FakeArchive):
        def iter_messages(self, conversation_id: str, *, since: Any = None, until: Any = None) -> Any:
            self._record("iter_messages", conversation_id=conversation_id, since=since, until=until)
            return iter(self._filtered([conversation_id], since, until, None))

    archive = InOrder(rows)
    found = browse(archive, chat, edge="oldest", limit=4, since="2015-03-01", max_hops=1)
    assert [m["id"] for m in found.messages] == ["d0059", "d0060", "d0061", "d0062"]
    assert found.reached_edge and [name for name, _ in archive.calls] == ["iter_messages"]
    last = browse(archive, chat, edge="newest", limit=3, until="2015-03-01", max_hops=1)
    assert [m["id"] for m in last.messages] == ["d0057", "d0058", "d0059"] and last.reached_edge and last.anchored


# --- 15: the answer without a model ---


def test_local_answer_gives_a_typed_error_when_the_archive_fails() -> None:
    class Broken(FakeArchive):
        def count(self, **_kwargs: Any) -> Any:
            raise sqlite3.OperationalError("database is locked at /private/path")

    with pytest.raises(ArchiveError) as caught:
        local_answer(Broken(tiny_messages(), CONVERSATIONS), "what about the lease?", labeler=tiny_labeler())
    assert "private" not in caught.value.message
