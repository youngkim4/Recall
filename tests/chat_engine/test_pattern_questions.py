"""A question about a person or a period is answered from patterns, not from one word match."""

import pytest

from chat_engine.engine import EngineConfig
from chat_engine.eval import runner
from chat_engine.fake_archive import FakeArchive, make_message
from chat_engine.fake_llm import say
from chat_engine.labeler import MaskingLabeler
from chat_engine.scope import Scope
from chat_engine.seed import question_period
from tests.chat_engine.helpers import RAW_HANDLES, all_text, events_of, last_user, make_engine, system_text
from tests.chat_engine.tiny import UNSAVED_EMAIL, UNSAVED_PHONE

FIGURES_2024 = 'count_messages(group_by="conversation", since="2024-01-01", until="2024-12-31")'


@pytest.mark.parametrize(
    ("question", "period"),
    [
        ("Who was my best friend from 2016-2019?", ("2016-01-01", "2019-12-31")),
        ("who did I text most between 2019 and 2016", ("2016-01-01", "2019-12-31")),
        ("what was 2021 like", ("2021-01-01", "2021-12-31")),
        ("who was I closest to 2016–2019", ("2016-01-01", "2019-12-31")),
        ("who was I closest to 2016—2019", ("2016-01-01", "2019-12-31")),
        ("what changed in 2016/2017?", ("2016-01-01", "2017-12-31")),
        ("who wrote most from 2016-06-01 to 2019-03-31", ("2016-01-01", "2019-12-31")),
        ("did she front me 10k in 2019?", ("2019-01-01", "2019-12-31")),
        ("2018년에 누구랑 제일 친했어?", ("2018-01-01", "2018-12-31")),
        ("Who do I text the most?", (None, None)),
        ("who sent 3000 messages", (None, None)),
        ("what about 20190 or 1989", (None, None)),
        ("who has the number 20162019", (None, None)),
        ("what happened in ２０２４?", (None, None)),
        ("what happened in २०२४?", (None, None)),
        ("", (None, None)),
    ],
)
def test_the_years_of_a_question_are_its_period(question: str, period: tuple[str | None, str | None]) -> None:
    assert question_period(question) == period


@pytest.mark.parametrize(
    ("question", "period"),
    [
        ("who is +1 (720) 555-2019?", (None, None)),
        ("who is +1 (720) 555-2019 and what did we do in 2021?", ("2021-01-01", "2021-12-31")),
        ("who is +82 10 2019 2020?", (None, None)),
        ("my number was 202-555-0100 2018", ("2018-01-01", "2018-12-31")),
        ("in 2018 202-555-0100 was my number", ("2018-01-01", "2018-12-31")),
        ("email jo2019@example.org", (None, None)),
        ("what did jo.2016@example.org write in 2018", ("2018-01-01", "2018-12-31")),
    ],
)
def test_digits_of_a_handle_are_not_a_year(question: str, period: tuple[str | None, str | None]) -> None:
    assert question_period(question) == period


@pytest.mark.parametrize(
    ("question", "period"),
    [
        ("I lent her 2000 dollars, when?", (None, None)),
        ("I paid 2000 in 2020", ("2020-01-01", "2020-12-31")),
        ("what will 2099 be like", (None, None)),
        ("everything from 2010 to 2030", (None, None)),
        ("who was I closest to from 2016-2019?", ("2019-01-01", "2019-12-31")),
        ("and from 2019 to 2026?", ("2019-01-01", "2026-12-31")),
    ],
)
def test_only_a_year_the_archive_holds_is_a_period(question: str, period: tuple[str | None, str | None]) -> None:
    assert question_period(question, (2019, 2026)) == period


def test_an_unsaved_number_that_reads_as_two_years_is_not_a_period() -> None:
    masked = MaskingLabeler(known_handles=["+85220192020"])
    assert masked.redact("who is 2019 2020?") != "who is 2019 2020?"
    assert question_period("who is 2019 2020?", redact=masked.redact) == (None, None)
    assert question_period("who is 2019 2020?", redact=MaskingLabeler().redact) == ("2019-01-01", "2020-12-31")
    assert question_period("what changed 2016-2019?", redact=masked.redact) == ("2016-01-01", "2019-12-31")


def test_the_seeded_figures_show_every_row_they_announce() -> None:
    people = [f"+1415555{number:04d}" for number in range(300, 345)]
    names = {handle: f"Person Number{index} Longername" for index, handle in enumerate(people)}
    conversations = [
        {"id": handle, "is_group": False, "name": "", "members": [handle], "person_id": index}
        for index, handle in enumerate(people)
    ]
    messages = [
        make_message(f"x{index}_{count}", handle, f"2024-03-{count + 1:02d} 10:00:00", "hello there", sender=handle)
        for index, handle in enumerate(people)
        for count in range(index % 5 + 1)
    ]
    engine, fake, _ = make_engine(
        [say("Many people.")], archive=FakeArchive(messages, conversations), labeler=MaskingLabeler(names)
    )
    engine.ask("who did I write with in 2024?")
    figures = last_user(fake.requests[0]).split("count_messages(")[1].split("Question:")[0]
    assert "Showing the 40 busiest of 45 rows." in figures
    assert figures.count(" · direct: ") == 40
    assert "Result cut to fit" not in figures


def test_the_last_status_before_the_model_answers_is_not_a_lookup() -> None:
    engine, _, _ = make_engine([say("Jonas and Inês.")])
    events: list[dict] = []
    engine.ask("Who did I talk to most in 2024?", on_event=events.append)
    first_delta = next(index for index, event in enumerate(events) if event["type"] == "delta")
    statuses = [event["text"] for event in events_of("status", events[:first_delta])]
    assert statuses == [
        "Reading your archive",
        "Searching your messages",
        "Counting who you wrote with in 2024",
        "Thinking it over",
    ]


def test_the_eval_runner_takes_the_steps_of_the_engine_unless_told() -> None:
    parser = runner._parser()
    assert parser.parse_args(["--scripted"]).max_steps is None
    assert runner._engine_config("m", 600, None).max_steps == EngineConfig(model="m", max_output_tokens=600).max_steps
    assert runner._engine_config("m", 600, 3).max_steps == 3


def test_a_year_the_archive_does_not_hold_runs_no_figures() -> None:
    engine, fake, _ = make_engine([say("The archive starts in 2019.")])
    result = engine.ask("I lent Maya 2000 dollars, when was that?")  # the tiny archive runs from 2019 to 2026
    assert "count_messages" not in [step["tool"] for step in result["steps"]]
    assert "count_messages(" not in last_user(fake.requests[0])


def test_the_seeded_figures_show_no_raw_handle() -> None:
    engine, fake, _ = make_engine([say("Maya, then a number ending in 0199.")])
    engine.ask("who did I write with in 2025?")
    sent = all_text(fake.requests[0])
    assert 'count_messages(group_by="conversation", since="2025-01-01", until="2025-12-31")' in sent
    assert "a number ending in 0199 · direct: 3 (you 1, others 2)" in sent
    for raw in [*RAW_HANDLES, UNSAVED_PHONE, UNSAVED_EMAIL]:
        assert raw not in sent, raw


def test_a_question_with_a_period_gets_the_figures_of_that_period_first() -> None:
    engine, fake, _ = make_engine([say("Jonas and Inês, three messages each.")])
    result = engine.ask("Who did I talk to most in 2024?")
    block = last_user(fake.requests[0])
    assert FIGURES_2024 in block
    assert "Jonas Weber · direct: 3 (you 1, others 2)" in block
    assert "Inês Duarte · direct: 3 (you 1, others 2)" in block
    assert "Maya Okafor · direct:" not in block.split(FIGURES_2024)[1].split("Question:")[0]
    assert [(step["tool"], step["seeded"]) for step in result["steps"]] == [
        ("search_messages", True),
        ("count_messages", True),
    ]
    assert result["steps"][-1]["text"] == "Counted messages by conversation: 2 rows"


def test_the_figures_come_after_the_names_of_the_question() -> None:
    engine, fake, _ = make_engine([say("Three messages.")])
    result = engine.ask("How close were Jonas and I in 2024?")
    assert [step["tool"] for step in result["steps"]] == [
        "search_messages",
        "list_conversations",
        "count_messages",
    ]
    assert FIGURES_2024 in last_user(fake.requests[0])


def test_a_question_with_no_year_gets_no_figures() -> None:
    engine, _, _ = make_engine([say("Maya.")])
    result = engine.ask("Who do I text the most?")
    assert [step["tool"] for step in result["steps"]] == ["search_messages"]


def test_a_period_outside_the_scope_of_the_user_runs_no_figures() -> None:
    engine, fake, _ = make_engine([say("That is outside the dates you chose.")])
    result = engine.ask("what happened in 2019?", scope=Scope(since="2025-01-01"))
    assert "count_messages" not in [step["tool"] for step in result["steps"]]
    assert "count_messages(" not in last_user(fake.requests[0])


def test_the_figures_stay_inside_the_scope_of_the_user() -> None:
    engine, fake, _ = make_engine([say("Three messages.")])
    engine.ask("what happened in 2024?", scope=Scope(since="2024-06-01"))
    block = last_user(fake.requests[0])
    assert 'count_messages(group_by="conversation", since="2024-06-01", until="2024-12-31")' in block
    assert "Jonas Weber · direct:" not in block.split("count_messages(")[1].split("Question:")[0]


def test_the_rules_tell_a_pattern_question_from_a_search() -> None:
    engine, fake, _ = make_engine([say("Nothing found.")])
    engine.ask("anything about zanzibar?")
    system = system_text(fake.requests[0])
    for rule in [
        "They are a first look",
        "is not a pattern question and a result holds the whole answer",
        "holds the result of a lookup for the same period, do not run it again",
        "one conversation in each call",
        "A figure from a count needs no reference",
        "is a pattern question",
        "One message never proves a pattern",
        "count_messages with group_by conversation",
        "a direct conversation tells more than a group",
        "the words people really write",
        "name the runner-up",
        "Calls and meetings in person are not in the archive",
    ]:
        assert rule in system, rule
    assert "If they answer the question, answer at once" not in system


def test_a_pattern_question_has_room_for_figures_and_reading() -> None:
    config = EngineConfig(model="m", max_output_tokens=100)
    assert config.max_steps >= 8
    assert config.total_tool_tokens >= 30_000
    assert config.total_tool_tokens >= config.max_steps * config.result_tokens
