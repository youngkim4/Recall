"""Names in the question: a word is a name only when it is not a common word of the archive.

The owner asked "well obviously that was julia, also we dated which was not my
question", and the status line read `Looking up “that, julia, not”`: a group
name held "that" and "not". The archive below is invented: a group whose name
holds both words, an owner who writes both words often, and a saved contact
Julia whom the messages seldom name.
"""

from __future__ import annotations

from chat_engine.directory import Directory
from chat_engine.fake_archive import FakeArchive, make_message
from chat_engine.fake_llm import say
from chat_engine.labeler import MaskingLabeler
from chat_engine.question_names import COMMON_WORD_SHARE, MAX_COUNTED, names_in_question
from chat_engine.registry import ConversationAliases
from chat_engine.render import ConversationBook
from chat_engine.scope import Scope
from tests.chat_engine.helpers import events_of, last_user, make_engine

JULIA = "+14155550170"
WILL = "+14155550171"
INES = "+351912345670"
FRIEND = "+14155550172"
DRILL = "chat200"
CORRECTION = "well obviously that was julia, also we dated which was not my question"
SAVED = {JULIA: "Julia Moreno", WILL: "Will Carter", INES: "Inês Duarte", FRIEND: "Noor Haddad"}

CONVERSATIONS = [
    {"id": JULIA, "is_group": False, "name": "", "members": [JULIA], "person_id": 1},
    {"id": WILL, "is_group": False, "name": "", "members": [WILL], "person_id": 2},
    {"id": INES, "is_group": False, "name": "", "members": [INES], "person_id": 3},
    {"id": FRIEND, "is_group": False, "name": "", "members": [FRIEND], "person_id": 4},
    {"id": DRILL, "is_group": True, "name": "That Is Not A Drill", "members": [FRIEND, WILL], "person_id": None},
]


def _rows() -> list[dict]:
    """400 messages. "that" and "not" are in 60 of them, "will" in 40, "julia" in 1, "ines" in 2 (0.5 percent)."""
    rows = []
    for n in range(400):
        day = f"2022-{1 + n % 12:02d}-{1 + n % 28:02d} {n % 24:02d}:{n % 60:02d}:00"
        conversation = (JULIA, DRILL, FRIEND, WILL, INES)[n % 5]
        if n < 60:
            text = "that is not what I meant"
        elif n < 100:
            text = "I will call you later"
        elif n == 100:
            text = "julia says hi"
        elif n in (101, 102):
            text = "ines sent the photos"
        else:
            text = f"plain words number {n}"
        sender = "" if n % 2 else (FRIEND if conversation == DRILL else conversation)
        rows.append(make_message(f"x{n}", conversation, day, text, is_from_me=bool(n % 2), sender=sender))
    return rows


def owner_archive() -> FakeArchive:
    return FakeArchive(_rows(), CONVERSATIONS)


def _directory(archive: FakeArchive) -> Directory:
    labeler = MaskingLabeler(SAVED)
    book = ConversationBook(archive, labeler, ConversationAliases())
    return Directory.load(archive, book, labeler, Scope(), archive.conversations())


def _counted(archive: FakeArchive) -> list[str]:
    """The words counted to tell a name from a common word: they come before any search."""
    calls = archive.calls
    first_search = next((index for index, (name, _) in enumerate(calls) if name == "search"), len(calls))
    return [arguments["text"] for name, arguments in calls[:first_search] if name == "count" and arguments["text"]]


def test_the_threshold_is_half_a_percent_of_the_messages() -> None:
    assert COMMON_WORD_SHARE == 0.005
    assert MAX_COUNTED == 4


def test_common_words_of_a_group_name_are_not_names_and_julia_is() -> None:
    archive = owner_archive()
    directory = _directory(archive)
    assert directory.named_in(CORRECTION) == ["that", "julia", "not"]  # the candidates: words of a name
    assert names_in_question(directory, archive, CORRECTION, 400) == ["julia"]
    assert sorted(_counted(archive)) == ["julia", "not", "that"]  # one count for each candidate


def test_a_rare_real_name_is_still_found() -> None:
    archive = owner_archive()
    directory = _directory(archive)
    # "ines" is in 2 of 400 messages: 0.5 percent is not more than 0.5 percent
    assert names_in_question(directory, archive, "what did inês send me", 400) == ["ines"]
    assert _counted(archive) == ["inês"]  # the word as it was written goes to the archive
    assert names_in_question(directory, archive, "how is Noor doing", 400) == ["noor"]


def test_a_name_that_is_a_common_word_gives_way_but_the_rest_of_the_name_stays() -> None:
    archive = owner_archive()
    directory = _directory(archive)
    assert names_in_question(directory, archive, "what did Will Carter say", 400) == ["carter"]
    assert names_in_question(directory, archive, "what did Will say", 401 * 100) == ["will"]


def test_at_most_four_words_are_counted_one_time_each() -> None:
    archive = owner_archive()
    directory = _directory(archive)
    question = "julia julia noor carter ines moreno haddad duarte that not drill"
    found = names_in_question(directory, archive, question, 400)
    counted = _counted(archive)
    assert len(counted) <= MAX_COUNTED and len(counted) == len(set(counted))
    assert found == ["julia", "noor", "carter", "ines"]


def test_digits_of_a_handle_are_not_counted() -> None:
    archive = owner_archive()
    directory = _directory(archive)
    assert names_in_question(directory, archive, "who is 0170", 400) == ["0170"]
    assert _counted(archive) == []


def test_no_candidate_no_count() -> None:
    archive = owner_archive()
    directory = _directory(archive)
    assert names_in_question(directory, archive, "anything about sourdough?", 400) == []
    assert _counted(archive) == []


def test_the_status_line_of_the_owner_shows_only_julia() -> None:
    archive = owner_archive()
    engine, fake, _ = make_engine([say("Noted.")], archive=archive, labeler=MaskingLabeler(SAVED))
    events: list[dict] = []
    engine.ask(CORRECTION, on_event=events.append)
    lines = [event["text"] for event in events_of("status", events)]
    assert "Looking up “julia”" in lines
    assert not [line for line in lines if "that" in line or "not," in line]
    block = last_user(fake.requests[0])
    assert 'list_conversations(name="julia")' in block
    assert "That Is Not A Drill" not in block.split('list_conversations(name="julia")')[1].split("[lookup")[0]
    # the name check counts first; the seeded search counts the words of the question after it
    counted = _counted(archive)
    assert counted[:3] == ["that", "julia", "not"] and counted[3] == "well"


def test_when_every_candidate_is_common_there_is_no_name_lookup() -> None:
    archive = owner_archive()
    engine, fake, _ = make_engine([say("Noted.")], archive=archive, labeler=MaskingLabeler(SAVED))
    events: list[dict] = []
    engine.ask("is that not what you meant", on_event=events.append)
    assert not [event for event in events_of("status", events) if event["text"].startswith("Looking up")]
    assert "list_conversations(name=" not in last_user(fake.requests[0])
