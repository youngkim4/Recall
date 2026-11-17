"""The tool loop: steps, caps, interim text, bad calls, cancel."""

import json

from chat_engine.cancel import CancelToken
from chat_engine.fake_llm import call, done, say
from chat_engine.ports import ToolCall
from chat_engine.prompt import LIMIT_NOTE
from chat_engine.session import ChatSession
from tests.chat_engine.helpers import events_of, last_user, make_engine, ref_of_line, tool_results


def test_tool_results_go_back_to_the_model_in_order() -> None:
    def second(request):
        results = tool_results(request)
        assert len(results) == 2
        assert "sourdough starter is alive" in results[0]
        assert "Messages that match" in results[1]
        ref = ref_of_line(results[0], "sourdough starter is alive")
        return say(f"Jonas started it in 2024 [{ref}].")

    engine, fake, _ = make_engine(
        [
            [
                call("search_messages", "c1", query="sourdough", order="oldest"),
                call("count_messages", "c2", text="sourdough"),
            ],
            second,
        ]
    )
    result = engine.ask("when did the bread thing start?")
    assistant = [m for m in fake.requests[1].messages if m.role == "assistant"][-1]
    assert [c.id for c in assistant.tool_calls] == ["c1", "c2"]
    tools = [m for m in fake.requests[1].messages if m.role == "tool"]
    assert [m.tool_call_id for m in tools] == ["c1", "c2"]
    assert result["answer"] == "Jonas started it in 2024 [1]."
    assert [s["text"] for s in result["steps"] if not s["seeded"]] == [
        "Searched for “sourdough”: 2 matches",
        "Counted messages: 2",
    ]


def test_status_for_each_tool_call_is_plain_words() -> None:
    engine, _, _ = make_engine(
        [
            [call("search_messages", query="sourdough"), call("count_messages", group_by="year")],
            say("Done."),
        ]
    )
    events: list[dict] = []
    engine.ask("bread?", on_event=events.append)
    statuses = [event["text"] for event in events_of("status", events)]
    assert "Searching for “sourdough”" in statuses
    assert "Counting messages by year" in statuses


def test_text_before_a_tool_call_is_not_part_of_the_answer() -> None:
    engine, _, _ = make_engine(
        [
            [*say("Let me look that up."), call("search_messages", query="sourdough")],
            say("It started in 2024."),
        ]
    )
    events: list[dict] = []
    result = engine.ask("bread?", on_event=events.append)
    assert result["answer"] == "It started in 2024."
    resets = [event for event in events_of("status", events) if event.get("reset")]
    assert len(resets) == 1 and resets[0]["text"] == "Looking at more messages"
    after_reset = events[events.index(resets[0]) + 1:]
    assert after_reset[0] == {"type": "status", "text": "Searching for “sourdough”"}
    assert "".join(e["text"] for e in events_of("delta", after_reset)) == "It started in 2024."


def test_commentary_deltas_are_not_streamed() -> None:
    engine, _, _ = make_engine(
        [
            [*say("Thinking about bread.", commentary=True), call("search_messages", query="sourdough")],
            say("It started in 2024."),
        ]
    )
    events: list[dict] = []
    engine.ask("bread?", on_event=events.append)
    assert "".join(e["text"] for e in events_of("delta", events)) == "It started in 2024."
    assert not [event for event in events_of("status", events) if event.get("reset")]


def test_step_cap_ends_the_loop_with_an_answer() -> None:
    looping = [call("search_messages", query="sourdough")]
    engine, fake, _ = make_engine([looping, looping, say("From what I have: it started in 2024.")], max_steps=2)
    result = engine.ask("bread?")
    assert len(fake.requests) == 3
    assert [request.allow_tools for request in fake.requests] == [True, True, False]
    assert LIMIT_NOTE in last_user(fake.requests[2])
    assert LIMIT_NOTE not in last_user(fake.requests[1])
    assert result["answer"] == "From what I have: it started in 2024."
    assert result["usage"]["model_calls"] == 3


def test_tool_output_cap_ends_the_loop_with_an_answer() -> None:
    engine, fake, _ = make_engine(
        [
            [call("search_messages", query="lease", limit=20)],
            say("The lease was signed in 2021."),
        ],
        total_tool_tokens=420,
        result_tokens=400,
    )
    result = engine.ask("zzz?")
    assert [request.allow_tools for request in fake.requests] == [True, False]
    assert LIMIT_NOTE in last_user(fake.requests[1])
    assert result["answer"] == "The lease was signed in 2021."


def test_lookups_after_the_cap_in_one_step_are_skipped() -> None:
    engine, fake, archive = make_engine(
        [
            [call("search_messages", "a", query="lease", limit=20), call("search_messages", "b", query="sourdough")],
            say("Done."),
        ],
        total_tool_tokens=420,
        result_tokens=400,
    )
    engine.ask("zzz?")
    second = tool_results(fake.requests[1])[1]
    assert "limit on lookups" in second
    assert not [a for name, a in archive.calls if name == "search" and a["query"] == "sourdough"]


def test_tool_calls_in_the_forced_last_step_are_ignored() -> None:
    looping = [call("search_messages", query="sourdough")]
    engine, fake, _ = make_engine(
        [looping, [*say("It started in 2024."), call("search_messages", query="again")]], max_steps=1
    )
    result = engine.ask("bread?")
    assert result["answer"] == "It started in 2024."
    assert len(fake.requests) == 2


def test_bad_tool_calls_are_answered_and_the_loop_goes_on() -> None:
    def second(request):
        results = tool_results(request)
        assert "no tool named" in results[0]
        assert "not valid JSON" in results[1]
        assert "c99" in results[2]
        return say("I could not find it.")

    engine, _, _ = make_engine(
        [
            [
                call("delete_everything", "x1"),
                ToolCall("x2", "search_messages", "{broken"),
                call("search_messages", "x3", query="lease", conversations=["c99"]),
            ],
            second,
        ]
    )
    result = engine.ask("zzz?")
    assert result["answer"] == "I could not find it."
    assert [step["text"] for step in result["steps"] if not step["seeded"]] == [
        "Could not run delete_everything",
        "Could not run search_messages",
        "Could not run search_messages",
    ]


def test_same_lookup_twice_is_not_run_twice() -> None:
    same = call("search_messages", query="sourdough")
    engine, fake, archive = make_engine([[same], [same], say("It started in 2024.")])
    engine.ask("zzz?")
    searches = [a for name, a in archive.calls if name == "search" and a["query"] == "sourdough"]
    assert len(searches) == 1
    assert "same lookup" in tool_results(fake.requests[2])[1].lower()


def test_cancel_stops_a_stream_in_flight() -> None:
    cancel = CancelToken()
    events: list[dict] = []

    def on_event(event: dict) -> None:
        events.append(event)
        if len(events_of("delta", events)) == 2:
            cancel.cancel()

    engine, fake, _ = make_engine([say("one two three four five six seven eight", chunk=4)])
    session = ChatSession()
    result = engine.ask("lease?", on_event=on_event, cancel=cancel, session=session)
    assert result["cancelled"] is True
    assert fake.closed_early == 1 and fake.finished == 0
    assert len(fake.requests) == 1
    streamed = "".join(e["text"] for e in events_of("delta", events))
    assert result["answer"] == streamed.strip()
    assert 0 < len(result["answer"]) < len("one two three four five six seven eight")
    assert events[-1]["type"] == "done" and events[-1]["payload"]["cancelled"] is True
    assert session.turns == ()


def test_cancel_between_tool_calls_stops_the_loop() -> None:
    cancel = CancelToken()

    def on_event(event: dict) -> None:
        if event["type"] == "status" and event["text"].startswith("Searching for “sourdough”"):
            cancel.cancel()

    engine, fake, archive = make_engine(
        [[call("search_messages", "a", query="sourdough"), call("search_messages", "b", query="lease")], say("no")]
    )
    result = engine.ask("zzz?", on_event=on_event, cancel=cancel)
    assert result["cancelled"] is True and result["answer"] == ""
    assert len(fake.requests) == 1
    assert not [a for name, a in archive.calls if name == "search" and a["query"] == "lease"]


def test_cancel_before_the_start_calls_no_model() -> None:
    cancel = CancelToken()
    cancel.cancel()
    engine, fake, _ = make_engine([say("no")])
    result = engine.ask("lease?", cancel=cancel)
    assert result["cancelled"] is True
    assert fake.requests == []


def test_output_cut_at_the_cap_is_reported() -> None:
    engine, _, _ = make_engine([[*say("The lease was"), done(stop_reason="length")]])
    result = engine.ask("lease?")
    assert result["truncated"] is True
    assert result["answer"] == "The lease was"


def test_arguments_of_tool_calls_are_kept_verbatim_for_the_model() -> None:
    made = call("search_messages", "k1", query="sourdough")
    engine, fake, _ = make_engine([[made], say("ok")])
    engine.ask("zzz?")
    assistant = [m for m in fake.requests[1].messages if m.role == "assistant"][-1]
    assert json.loads(assistant.tool_calls[0].arguments) == {"query": "sourdough"}
