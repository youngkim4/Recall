"""Names in the question: the name of a person is a name; a word of a group title only when it is rare.

The owner asked "well obviously that was julia, also we dated which was not my
question", and the status line read `Looking up “that, julia, not”`: a group
name held "that" and "not". The archive below is invented: a group whose name
holds both words, an owner who writes both words often, saved contacts Julia
and Will, and a book club whose title names cities.
"""

from __future__ import annotations

from chat_engine.directory import Directory
from chat_engine.fake_archive import FakeArchive, make_message
from chat_engine.fake_llm import say
from chat_engine.labeler import MaskingLabeler
from chat_engine.question_names import (
    COMMON_WORD_FLOOR,
    COMMON_WORD_SHARE,
    MAX_COUNTED,
    MAX_NAMES,
    common_limit,
    names_in_question,
)
from chat_engine.registry import ConversationAliases
from chat_engine.render import ConversationBook
from chat_engine.scope import Scope
from tests.chat_engine.helpers import events_of, last_user, make_engine

JULIA = "+14155550170"
WILL = "+14155550171"
INES = "+351912345670"
FRIEND = "+14155550172"
DRILL = "chat200"
CLUB = "chat300"
CLUB_TITLE = "Book Club Lisbon Porto Madrid Paris Rome Oslo Bern Riga Kyiv"
CORRECTION = "well obviously that was julia, also we dated which was not my question"
SAVED = {JULIA: "Julia Moreno", WILL: "Will Carter", INES: "Inês Duarte", FRIEND: "Noor Haddad"}

CONVERSATIONS = [
    {"id": JULIA, "is_group": False, "name": "", "members": [JULIA], "person_id": 1},
    {"id": WILL, "is_group": False, "name": "", "members": [WILL], "person_id": 2},
    {"id": INES, "is_group": False, "name": "", "members": [INES], "person_id": 3},
    {"id": FRIEND, "is_group": False, "name": "", "members": [FRIEND], "person_id": 4},
    {"id": DRILL, "is_group": True, "name": "That Is Not A Drill", "members": [FRIEND, WILL], "person_id": None},
    {"id": CLUB, "is_group": True, "name": CLUB_TITLE, "members": [FRIEND, INES], "person_id": None},
]


def _rows() -> list[dict]:
    """400 messages. "that" and "not" are in 60, "will" in 40, "julia" in 1, "ines" in 2.

    "lisbon" is in 15. The other cities of the club are in 25 each.
    """
    rows = []
    for n in range(400):
        day = f"2022-{1 + n % 12:02d}-{1 + n % 28:02d} {n % 24:02d}:{n % 60:02d}:00"
        conversation = (JULIA, DRILL, FRIEND, WILL, INES, CLUB)[n % 6]
        if n < 60:
            text = "that is not what I meant"
        elif n < 100:
            text = "I will call you later"
        elif n == 100:
            text = "julia says hi"
        elif n in (101, 102):
            text = "ines sent the photos"
        elif n < 118:
            text = "back from lisbon"
        elif n < 143:
            text = "porto madrid paris rome oslo bern riga kyiv were great"
        else:
            text = f"plain words number {n}"
        sender = "" if n % 2 else (FRIEND if conversation in (DRILL, CLUB) else conversation)
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


def test_the_limit_is_half_a_percent_of_the_messages_with_a_floor() -> None:
    assert (COMMON_WORD_SHARE, COMMON_WORD_FLOOR, MAX_COUNTED, MAX_NAMES) == (0.005, 20, 8, 4)
    assert common_limit(60) == 20 and common_limit(400) == 20 and common_limit(545_000) == 2725


def test_common_words_of_a_group_title_are_not_names_and_julia_is() -> None:
    archive = owner_archive()
    directory = _directory(archive)
    assert names_in_question(directory, archive, CORRECTION, 400) == ["julia"]
    assert sorted(_counted(archive)) == ["not", "that"]  # only the words of a group title are counted


def test_the_reviewer_case_julia_after_common_words_is_still_looked_up() -> None:
    archive = owner_archive()
    directory = _directory(archive)
    found = names_in_question(directory, archive, "that will not be a drill, will julia come?", 400)
    assert "julia" in found and "that" not in found and "not" not in found
    assert found == ["will", "drill", "julia"]  # Will is a saved person; "drill" is rare


def test_the_reviewer_case_a_name_said_once_in_a_small_archive_stays() -> None:
    rows = [
        make_message(f"y{n}", JULIA, f"2022-01-{1 + n % 28:02d} 10:00:00", "julia says hi" if n == 0 else f"plain {n}",
                     is_from_me=bool(n % 2), sender="" if n % 2 else JULIA)
        for n in range(60)
    ]
    tiny = FakeArchive(rows, CONVERSATIONS)
    assert names_in_question(_directory(tiny), tiny, "what did julia say", 60) == ["julia"]


def test_the_reviewer_case_the_name_of_a_saved_person_is_a_name_even_when_it_is_common() -> None:
    archive = owner_archive()
    directory = _directory(archive)
    assert names_in_question(directory, archive, "what did Will say about the trip", 400) == ["will"]
    assert names_in_question(directory, archive, "what did Will Carter say", 400) == ["will", "carter"]
    assert _counted(archive) == []


def test_a_rare_word_of_a_group_title_is_a_name_and_the_floor_holds_in_a_small_archive() -> None:
    archive = owner_archive()
    directory = _directory(archive)
    # "lisbon" is in 15 of 400 messages: more than 0.5 percent, not more than the floor of 20
    assert names_in_question(directory, archive, "photos of Lisbon", 400) == ["lisbon"]
    assert names_in_question(directory, archive, "photos of Porto", 400) == []  # 25 messages
    assert _counted(archive) == ["lisbon", "porto"]  # the word as it was written goes to the archive


def test_every_candidate_is_looked_at_the_counts_are_bounded_and_four_names_are_kept() -> None:
    archive = owner_archive()
    directory = _directory(archive)
    question = "porto madrid paris rome oslo bern riga kyiv lisbon julia"
    assert names_in_question(directory, archive, question, 400) == ["julia"]
    counted = _counted(archive)
    assert counted == ["porto", "madrid", "paris", "rome", "oslo", "bern", "riga", "kyiv"]  # lisbon: past the bound
    assert len(counted) == MAX_COUNTED
    assert names_in_question(directory, archive, "book club lisbon julia noor", 400) == [
        "book", "club", "lisbon", "julia",
    ]
    persons = names_in_question(directory, archive, "julia noor carter ines moreno haddad", 400)
    assert persons == ["julia", "noor", "carter", "ines"]


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
    assert counted[:2] == ["that", "not"] and counted[2] == "well"


def test_when_every_candidate_is_common_there_is_no_name_lookup() -> None:
    archive = owner_archive()
    engine, fake, _ = make_engine([say("Noted.")], archive=archive, labeler=MaskingLabeler(SAVED))
    events: list[dict] = []
    engine.ask("is that not what you meant", on_event=events.append)
    assert not [event for event in events_of("status", events) if event["text"].startswith("Looking up")]
    assert "list_conversations(name=" not in last_user(fake.requests[0])
