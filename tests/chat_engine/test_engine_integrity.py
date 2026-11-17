"""Privacy, scope, quotes, citations and follow-ups."""

from chat_engine.fake_llm import call, say
from chat_engine.scope import Scope
from chat_engine.session import ChatSession
from tests.chat_engine.helpers import (
    RAW_HANDLES,
    all_text,
    alias_of_line,
    events_of,
    last_user,
    make_engine,
    ref_of_line,
    tool_results,
)
from tests.chat_engine.tiny import GROUP, JONAS_PHONE, MAYA_PHONE, UNSAVED_EMAIL, UNSAVED_PHONE


def test_raw_handle_of_an_unsaved_person_never_reaches_the_model() -> None:
    def explore(request):
        alias = alias_of_line(last_user(request), "a number ending in 0199 · direct")
        return [
            call("search_messages", "s1", query="climbing", match="any"),
            call("search_messages", "s2", query="line", conversations=[alias]),
            call("count_messages", "s3", group_by="sender", conversations=[alias]),
            call("count_messages", "s4", group_by="conversation"),
            call("conversation_stats", "s5", conversation=alias),
            call("list_conversations", "s6", name="0199"),
            call("list_conversations", "s7"),
        ]

    def read(request):
        ref = ref_of_line(tool_results(request)[1], "my other line is")
        return [call("read_around", "s8", reference=ref, radius=10)]

    engine, fake, _ = make_engine([explore, read, say("That number is Dana from the climbing gym.")])
    events: list[dict] = []
    result = engine.ask(
        "who is +1 (720) 555-0199, and did rk.unsaved@example.org ever write?", on_event=events.append
    )

    assert len(fake.requests) == 3
    seen = "\n".join(all_text(request) for request in fake.requests)
    assert "a number ending in 0199" in seen
    assert "an unsaved email contact" in seen
    assert "my other line is a number ending in 0199 too" in seen
    assert "who is a number ending in 0199, and did an unsaved email contact ever write?" in seen
    for raw in [*RAW_HANDLES, UNSAVED_PHONE, UNSAVED_EMAIL]:
        assert raw not in seen, raw
    shown = "\n".join(event.get("text", "") for event in events if event["type"] in {"status", "delta"})
    shown += "\n".join(step["text"] for step in result["steps"])
    for raw in [*RAW_HANDLES, UNSAVED_PHONE, UNSAVED_EMAIL]:
        assert raw not in shown, raw


def test_known_handles_are_masked_even_in_a_scoped_chat() -> None:
    from chat_engine.labeler import MaskingLabeler
    from tests.chat_engine.tiny import SAVED_NAMES

    labeler = MaskingLabeler(SAVED_NAMES, known_handles=[UNSAVED_PHONE, UNSAVED_EMAIL])
    engine, fake, _ = make_engine([say("No.")], labeler=labeler)
    engine.ask("did Jonas mention 720-555-0199?", scope=Scope(conversations=(JONAS_PHONE,)))
    seen = all_text(fake.requests[0])
    assert "did Jonas mention a number ending in 0199?" in seen
    for raw in RAW_HANDLES:
        assert raw not in seen


def test_scoped_question_never_reads_outside_the_scope() -> None:
    session = ChatSession()
    outside = session.aliases.alias(MAYA_PHONE)

    def wander(request):
        return [
            call("search_messages", "w1", query="lease"),
            call("search_messages", "w2", query="lease", conversations=[outside]),
            call("count_messages", "w3", conversations=[outside], group_by="year"),
            call("conversation_stats", "w4", conversation=outside),
            call("list_conversations", "w5", name="maya"),
            call("list_conversations", "w6"),
            call("search_messages", "w7", query="lease flights lisbon", match="any"),
        ]

    engine, fake, archive = make_engine([wander, say("Nothing about a lease with Jonas.")])
    engine.ask("what about the lease?", scope=Scope(conversations=(JONAS_PHONE,)), session=session)

    results = tool_results(fake.requests[1])
    assert "no message matches" in results[0]
    for refused in results[1:4]:
        assert "outside the scope" in refused
    assert "No conversation matches" in results[4]
    assert "Jonas Weber" in results[5] and "Maya" not in results[5]
    seen = "\n".join(all_text(request) for request in fake.requests)
    for leaked in ["lease got signed", "landlord", "Lisbon", "Dana", "apartamento"]:
        assert leaked not in seen, leaked
    for name, arguments in archive.calls:
        if name in {"search", "count"}:
            assert arguments["conversations"] == [JONAS_PHONE], (name, arguments)
        elif name in {"conversation", "stats"}:
            assert arguments["conversation_id"] == JONAS_PHONE, (name, arguments)
        elif name == "context":
            assert arguments["message_id"].startswith("j"), arguments
        else:
            assert name == "summary", name


def test_made_up_quote_is_flagged_and_a_real_one_is_not() -> None:
    def answer(request):
        ref = ref_of_line(last_user(request), "the lease got signed btw")
        return say(
            f'Maya wrote "yes!! the lease got signed btw" [{ref}] and added '
            '"I have never been happier in my whole life".'
        )

    engine, _, _ = make_engine([answer])
    result = engine.ask("What did Maya say when the lease got signed?")
    assert result["unverified_quotes"] == ["I have never been happier in my whole life"]
    assert '"yes!! the lease got signed btw" [1]' in result["answer"]


def test_quoting_the_question_or_a_conversation_name_is_not_flagged() -> None:
    engine, _, _ = make_engine(
        [say('You asked about "the big lisbon reunion dinner"; nothing in "Lisbon crew" mentions it.')]
    )
    result = engine.ask("was there talk of the big lisbon reunion dinner in Lisbon crew?")
    assert result["unverified_quotes"] == []


def test_citations_are_renumbered_in_order_of_use() -> None:
    def answer(request):
        block = last_user(request)
        renew = ref_of_line(block, "the landlord wants to renew the lease")
        signed = ref_of_line(block, "the lease got signed btw")
        return say(
            f"The landlord wanted to renew [{renew}]. Before that it was signed [{signed}]. "
            f"So it was renewed [{renew}] [m999].",
            chunk=5,
        )

    engine, _, _ = make_engine([answer])
    events: list[dict] = []
    result = engine.ask("what happened with the lease, was it signed or renewed?", on_event=events.append)
    assert result["answer"] == (
        "The landlord wanted to renew [1]. Before that it was signed [2]. So it was renewed [1]."
    )
    assert [(c["number"], c["id"]) for c in result["citations"]] == [(1, "a5"), (2, "a2")]
    assert result["invalid_references"] == ["m999"]
    assert "".join(e["text"] for e in events_of("delta", events)) == result["answer"]


def test_follow_up_reuses_the_registry_and_the_earlier_turns() -> None:
    def first(request):
        ref = ref_of_line(last_user(request), "the lease got signed btw")
        return say(f"Maya said the lease got signed [{ref}].")

    seen: dict[str, str] = {}

    def second(request):
        block = last_user(request)
        seen["ref"] = ref_of_line(block, "the lease got signed btw")
        return [call("read_around", "r1", reference=seen["ref"], radius=3)]

    def third(request):
        result = tool_results(request)[-1]
        move = ref_of_line(result, "we move in on the first of April")
        return say(f"Right after [{seen['ref']}] she wrote that you move in on the first of April [{move}].")

    engine, fake, _ = make_engine([first, second, third])
    session = ChatSession()
    one = engine.ask("What did Maya say when the lease got signed?", session=session)
    first_ref = session.turns[0].cited[0]
    two = engine.ask("tell me more about that", session=session)

    assert one["citations"][0]["id"] == "a2"
    assert seen["ref"] == first_ref
    assert session.registry.get(first_ref)["id"] == "a2"
    follow_up = fake.requests[1]
    roles = [message.role for message in follow_up.messages]
    assert roles[:3] == ["system", "user", "assistant"]
    assert follow_up.messages[1].content == "What did Maya say when the lease got signed?"
    assert follow_up.messages[2].content == f"Maya said the lease got signed [{first_ref}]."
    assert "Evidence from earlier in this chat" in last_user(follow_up)
    assert [(c["number"], c["id"]) for c in two["citations"]] == [(1, "a2"), (2, "a4")]
    assert len(session.turns) == 2
    assert session.turns[1].question == "tell me more about that"


def test_session_survives_a_round_trip_through_json() -> None:
    import json

    def first(request):
        ref = ref_of_line(last_user(request), "the lease got signed btw")
        return say(f"Maya said the lease got signed [{ref}].")

    def second(request):
        ref = ref_of_line(last_user(request), "the lease got signed btw")
        return say(f"As said before [{ref}].")

    engine, _, _ = make_engine([first, second])
    session = ChatSession()
    engine.ask("What did Maya say when the lease got signed?", session=session)
    restored = ChatSession.from_dict(json.loads(json.dumps(session.to_dict())))
    again = engine.ask("and then?", session=restored)
    assert again["citations"][0]["id"] == "a2"
    assert restored.aliases.snapshot() == session.aliases.snapshot()


def test_plain_history_can_start_a_session() -> None:
    session = ChatSession.from_history(
        [
            {"role": "user", "content": "who is Maya?"},
            {"role": "assistant", "content": "A friend [1]."},
            {"role": "user", "content": ""},
            {"role": "user", "content": "unanswered"},
        ]
    )
    assert [(turn.question, turn.answer) for turn in session.turns] == [("who is Maya?", "A friend [1].")]
    engine, fake, _ = make_engine([say("Still a friend.")])
    engine.ask("and now?", session=session)
    assert fake.requests[0].messages[1].content == "who is Maya?"


def test_old_turns_are_dropped_when_the_history_is_too_long() -> None:
    session = ChatSession.from_history(
        [
            {"role": "user", "content": "first question " + "x" * 600},
            {"role": "assistant", "content": "first answer " + "y" * 600},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
        ]
    )
    engine, fake, _ = make_engine([say("ok")], history_tokens=100)
    engine.ask("third?", session=session)
    contents = [message.content for message in fake.requests[0].messages]
    assert "second question" in contents and "second answer" in contents
    assert not any(content.startswith("first question") for content in contents)


def test_group_message_citation_names_the_sender_by_label() -> None:
    def answer(request):
        ref = ref_of_line(last_user(request), "forward the lease for the Lisbon flat")
        return say(f"Someone asked for it [{ref}].")

    engine, _, _ = make_engine([answer])
    result = engine.ask("who wanted the lease for the lisbon flat forwarded?")
    citation = result["citations"][0]
    assert citation["id"] == "g4" and citation["conversation"] == GROUP
    assert citation["conversation_label"] == "Lisbon crew"
    assert citation["sender_label"] == "a number ending in 0199"
