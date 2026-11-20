"""A question that names one person also gets the first messages with that person: how it began."""

from chat_engine.fake_llm import say
from chat_engine.scope import Scope
from tests.chat_engine.helpers import last_user, make_engine
from tests.chat_engine.tiny import JONAS_PHONE, MAYA_PHONE


def _opening(question: str, **asked) -> str:
    engine, fake, _ = make_engine([say("ok")])
    engine.ask(question, **asked)
    return last_user(fake.requests[0])


def test_a_question_that_names_one_person_gets_the_start_of_that_conversation() -> None:
    prompt = _opening("Who is Jonas to me?")
    assert "one conversation: Jonas Weber)" in prompt
    assert "sourdough starter is alive" in prompt


def test_the_first_messages_come_in_the_order_of_time() -> None:
    prompt = _opening("Who is Maya?")
    first, later = prompt.index("are we still on for tonight"), prompt.index("we move in on the first of April")
    assert first < later


def test_a_question_that_names_nobody_gets_no_opening() -> None:
    assert "one conversation:" not in _opening("what did we cook last winter?")


def test_a_question_that_names_a_group_gets_no_opening_of_the_group() -> None:
    assert "one conversation:" not in _opening("what did the Lisbon crew plan?")


def test_the_opening_stays_inside_the_scope_of_the_question() -> None:
    prompt = _opening("Who is Maya?", scope=Scope(conversations=(JONAS_PHONE,)))
    assert "one conversation: Maya" not in prompt


def test_the_opening_is_only_read_when_the_scope_holds_that_conversation() -> None:
    prompt = _opening("Who is Maya?", scope=Scope(conversations=(MAYA_PHONE,)))
    assert "one conversation: Maya Okafor)" in prompt
