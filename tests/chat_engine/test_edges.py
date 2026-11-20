"""Edge paths: cancel hooks, config checks, long questions, date scopes, tight budgets."""

import json
import threading
from types import SimpleNamespace as NS

import pytest

from chat_engine.cancel import CancelToken
from chat_engine.engine import MAX_QUESTION_CHARS, EngineConfig
from chat_engine.fake_archive import FakeArchive, make_message
from chat_engine.fake_llm import call, say
from chat_engine.labeler import MaskingLabeler
from chat_engine.openai_llm import OpenAIResponsesLLM, _plain
from chat_engine.prompt import scope_line
from chat_engine.registry import ConversationAliases, EvidenceRegistry
from chat_engine.render import Limits
from chat_engine.scope import Scope
from chat_engine.search import SearchRequest, run_search
from chat_engine.tools import ToolBox
from tests.chat_engine.helpers import events_of, last_user, make_engine, system_text, tool_results
from tests.chat_engine.tiny import GROUP, JONAS_PHONE, MAYA_PHONE, tiny_archive, tiny_labeler


def _run(box: ToolBox, tool: str, budget: int = 3000, **arguments: object):
    return box.prepare(tool, json.dumps(arguments)).run(budget)


def _box(scope: Scope | None = None, archive: FakeArchive | None = None) -> tuple[ToolBox, ConversationAliases]:
    aliases = ConversationAliases()
    box = ToolBox(archive or tiny_archive(), tiny_labeler(), EvidenceRegistry(), aliases, scope or Scope(), Limits())
    return box, aliases


# --- cancel token ---


def test_cancel_runs_each_hook_once_and_hooks_can_be_removed() -> None:
    token = CancelToken()
    seen: list[str] = []
    token.on_cancel(lambda: seen.append("kept"))
    forget = token.on_cancel(lambda: seen.append("removed"))
    forget()
    assert not token.cancelled
    token.cancel()
    token.cancel()
    assert token.cancelled and seen == ["kept"]


def test_hook_added_after_the_cancel_runs_at_once_and_a_failing_hook_is_contained() -> None:
    token = CancelToken()

    def broken() -> None:
        raise RuntimeError("hook failed")

    token.on_cancel(broken)
    token.cancel()
    late: list[int] = []
    token.on_cancel(lambda: late.append(1))()
    assert late == [1]


def test_wait_returns_early_when_another_thread_cancels() -> None:
    token = CancelToken()
    assert token.wait(0.01) is False
    threading.Timer(0.05, token.cancel).start()
    assert token.wait(5.0) is True


# --- config and prompt ---


def test_config_rejects_values_that_cannot_work() -> None:
    with pytest.raises(ValueError):
        EngineConfig(model=" ", max_output_tokens=100)
    with pytest.raises(ValueError):
        EngineConfig(model="m", max_output_tokens=0)
    with pytest.raises(ValueError):
        EngineConfig(model="m", max_output_tokens=100, max_steps=0)
    with pytest.raises(ValueError):
        EngineConfig(model="m", max_output_tokens=100, history_tokens=-1)


def test_scope_line_names_conversations_and_days() -> None:
    assert scope_line(None, None, None) == "Scope: all conversations."
    assert "only c1 (Maya), only 2024-01-01 to 2024-12-31" in scope_line("c1 (Maya)", "2024-01-01", "2024-12-31")
    assert "only from 2024-01-01." in scope_line(None, "2024-01-01", None)
    assert "only up to 2024-12-31." in scope_line(None, None, "2024-12-31")


def test_very_long_question_is_cut_and_the_model_is_told() -> None:
    engine, fake, _ = make_engine([say("ok")])
    engine.ask("lease " + "x" * (MAX_QUESTION_CHARS + 500))
    block = last_user(fake.requests[0])
    assert f"[The question was cut at {MAX_QUESTION_CHARS:,} characters.]" in block
    assert len(block) < MAX_QUESTION_CHARS + 2500


def test_date_scope_is_in_the_prompt_and_limits_the_seed() -> None:
    engine, fake, archive = make_engine([say("ok")])
    engine.ask("what about the lease?", scope=Scope(since="2023-01-01", until="2023-12-31"))
    assert "only 2023-01-01 to 2023-12-31" in system_text(fake.requests[0])
    block = last_user(fake.requests[0])
    assert "the landlord wants to renew the lease" in block and "lease got signed" not in block
    for name, arguments in archive.calls:
        if name in {"search", "count"}:
            assert (arguments["since"], arguments["until"]) == ("2023-01-01", "2023-12-31")


# --- tools under scope and budget ---


def test_list_conversations_counts_inside_the_days_of_a_date_scope() -> None:
    box, _ = _box(Scope(since="2023-01-01", until="2023-12-31"))
    text = _run(box, "list_conversations", name="maya").text
    assert "1 of 1." in text
    assert "3 messages in the scope dates (you 1, them 2)" in text
    assert "0 messages" not in text and "9 messages" not in text and "Lisbon" not in text


def test_list_conversations_says_when_it_shows_only_some() -> None:
    box, _ = _box()
    text = _run(box, "list_conversations", name="maya", limit=1).text
    assert "3 of 7. Showing 1." in text and len(text.splitlines()) == 2
    cut = _run(box, "list_conversations", budget=40).text
    assert "Result cut to fit" in cut


def test_group_with_many_members_lists_a_few_and_counts_the_rest() -> None:
    members = [f"+1415555{n:04d}" for n in range(7)]
    rows = [make_message("x1", "chat-big", "2024-01-01 10:00:00", "hello all", sender=members[0])]
    archive = FakeArchive(rows, [{"id": "chat-big", "is_group": True, "name": "Big one", "members": members}])
    box, _ = _box(archive=archive)
    text = _run(box, "list_conversations").text
    assert "group of 7: " in text and ", +3 more" in text


def test_group_without_members_and_direct_thread_without_a_handle() -> None:
    rows = [
        make_message("x1", "chat-empty", "2024-01-01 10:00:00", "hello"),
        make_message("x2", "thread-77", "2024-01-02 10:00:00", "hi there"),
    ]
    conversations = [
        {"id": "chat-empty", "is_group": True, "name": "", "members": []},
        {"id": "thread-77", "is_group": False, "name": "", "members": []},
    ]
    box, _ = _box(archive=FakeArchive(rows, conversations))
    text = _run(box, "list_conversations").text
    assert "a group chat · group ·" in text
    assert "an unsaved contact · direct ·" in text


def test_count_table_is_cut_at_the_budget() -> None:
    box, _ = _box()
    text = _run(box, "count_messages", budget=120, group_by="conversation").text
    assert text.endswith("[Result cut to fit. Narrow the dates or the conversations.]")
    assert text.count("\n") < 7


def test_count_rejects_an_unknown_group_and_since_after_until() -> None:
    box, _ = _box()
    assert not _run(box, "count_messages", group_by="planet").ok
    late = _run(box, "count_messages", since="2024-05-01", until="2024-01-01")
    assert not late.ok and "since is after until" in late.text


def test_stats_needs_a_conversation_and_respects_days() -> None:
    box, aliases = _box()
    assert not _run(box, "conversation_stats").ok
    text = _run(box, "conversation_stats", conversation=aliases.alias(MAYA_PHONE), since="2023-01-01").text
    assert ", from 2023-01-01:" in text and "messages: 5 (you 2, others 3)" in text
    empty = _run(box, "conversation_stats", conversation=aliases.alias(MAYA_PHONE), until="2000-01-01").text
    assert "messages: 0 (you 0, others 0)" in empty and "busiest day" not in empty


def test_read_around_is_trimmed_to_the_budget_around_the_message() -> None:
    box, _ = _box()
    _run(box, "search_messages", query="renew")
    ref = box.context.registry.ref_of("a5")
    full = _run(box, "read_around", reference=ref, radius=4)
    tight = _run(box, "read_around", budget=150, reference=ref, radius=4)
    assert "9 messages around" in full.text
    assert "Result cut to fit" in tight.text and "the landlord wants to renew the lease" in tight.text
    assert tight.text.count("\nm") < full.text.count("\nm")


def test_reading_a_conversation_is_trimmed_to_the_budget() -> None:
    box, aliases = _box()
    text = _run(
        box, "search_messages", budget=130, query="", conversations=[aliases.alias(GROUP)], order="oldest"
    ).text
    assert "Result cut to fit" in text and "flights to Lisbon are booked" in text


def test_search_with_dates_and_sender_says_so_in_its_first_line() -> None:
    box, aliases = _box()
    out = _run(
        box, "search_messages", query="lease", conversations=[aliases.alias(MAYA_PHONE)],
        since="2023-01-01", from_me=False, order="relevance",
    )
    assert "in c1 (Maya Okafor), from 2023-01-01, sent by others: 1 message matches." in out.text
    mine = _run(box, "search_messages", query="rent", until="2023-12-31", from_me=True)
    assert "up to 2023-12-31, sent by you" in mine.text
    many = _run(
        box, "search_messages", query="the",
        conversations=[aliases.alias(c) for c in (MAYA_PHONE, JONAS_PHONE, GROUP, "+351912345678")],
    )
    assert "in 4 conversations" in many.text


def test_any_search_with_nothing_to_rank_by_says_so() -> None:
    rows = [
        make_message(f"n{i}", "+15550000001", f"2020-01-{1 + i % 28:02d} 12:00:00", "dinner plans") for i in range(300)
    ]
    box, _ = _box(archive=FakeArchive(rows))
    out = _run(box, "search_messages", query="dinner plans zzz", match="any")
    assert "Too common to rank by" in out.text and "Nothing to show." in out.text


def test_search_survives_odd_archive_values() -> None:
    class Odd:
        def search(self, query: str, **_filters: object) -> dict:
            message = make_message("o1", "+15550000001", "2024-01-01 10:00:00", "lease here")
            return {"total": 1, "hits": [{"message": {**message, "timestamp": "not a time"}, "ranges": [["x"]]}]}

        def count(self, **_filters: object) -> int:
            return 0

    outcome = run_search(Odd(), SearchRequest(query="lease"))  # type: ignore[arg-type]
    assert outcome.hits[0].focus is None and outcome.total == 1


# --- labeler corners ---


def test_labels_for_one_member_and_two_member_groups() -> None:
    labeler = MaskingLabeler({"+14155550123": "Maya Okafor", "+14155550124": "Jonas Weber"})
    base = {"is_group": True, "name": "", "message_count": 1, "sent_count": 0, "received_count": 1,
            "first_timestamp": None, "last_timestamp": None, "person_id": None}
    one = {**base, "id": "chat1", "members": ["+14155550123"]}
    two = {**base, "id": "chat2", "members": ["+14155550123", "+14155550124"]}
    four = {**base, "id": "chat4", "members": ["+14155550123", "+14155550124", "+13035550101", "+13035550102"]}
    assert labeler.conversation(one) == "a group chat with Maya Okafor"
    assert labeler.conversation(two) == "a group chat with Maya Okafor and Jonas Weber"
    assert labeler.conversation(four).endswith("and 1 other")
    assert labeler.conversation({**base, "id": "chat0", "members": []}) == "a group chat"
    saved_group = MaskingLabeler({"chat9": "Book club"})
    assert saved_group.conversation({**base, "id": "chat9", "members": []}) == "Book club"


# --- engine corners ---


def test_cancel_during_the_seeded_lookups_stops_before_the_model() -> None:
    cancel = CancelToken()

    def on_event(event: dict) -> None:
        if event["type"] == "status" and event["text"] == "Searching your messages":
            cancel.cancel()

    engine, fake, _ = make_engine([say("no")])
    result = engine.ask("what did Maya say about the lease?", on_event=on_event, cancel=cancel)
    assert result["cancelled"] is True and fake.requests == []
    assert result["steps"] == []


def test_closing_the_event_iterator_closes_the_model_stream() -> None:
    engine, fake, _ = make_engine([say("one two three four five six", chunk=4)])
    events = engine.stream("lease?")
    for event in events:
        if event["type"] == "delta":
            break
    events.close()
    assert fake.closed_early == 1


def test_each_tool_result_stays_near_its_own_budget() -> None:
    words = ["lease", "the", "you"]
    searches = [call("search_messages", f"s{i}", query=word, limit=20) for i, word in enumerate(words)]
    engine, fake, _ = make_engine([searches, say("done")], result_tokens=300)
    engine.ask("zzz?")
    assert all((len(text.encode()) + 2) // 3 <= 330 for text in tool_results(fake.requests[1]))
    assert events_of("done", list(engine.stream("zzz?"))) == []


# --- OpenAI adapter corners ---


def test_plain_turns_sdk_objects_into_plain_data() -> None:
    class Model:
        def model_dump(self, exclude_none: bool = False) -> dict:
            return {"text": "summary", "type": "summary_text"}

    assert _plain([Model(), NS(text="x", extra=None)]) == [{"text": "summary", "type": "summary_text"}, {"text": "x"}]
    assert _plain("as is") == "as is"


def test_wait_between_attempts_uses_the_cancel_token_or_a_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("chat_engine.openai_llm.time.sleep", slept.append)
    llm = OpenAIResponsesLLM(object())
    llm._pause(0.5, None)
    token = CancelToken()
    token.cancel()
    llm._pause(30.0, token)
    assert slept == [0.5]
