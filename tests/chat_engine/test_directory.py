from chat_engine.directory import Directory
from chat_engine.registry import ConversationAliases
from chat_engine.render import ConversationBook
from chat_engine.scope import Scope
from tests.chat_engine.tiny import (
    GROUP,
    INES_PHONE,
    JONAS_PHONE,
    MAYA_EMAIL,
    MAYA_PHONE,
    UNSAVED_EMAIL,
    UNSAVED_PHONE,
    tiny_archive,
    tiny_labeler,
)


def _directory(scope: Scope | None = None):
    archive = tiny_archive()
    labeler = tiny_labeler()
    book = ConversationBook(archive, labeler, ConversationAliases())
    listed = archive.conversations(limit=None, offset=0, order="count")
    return Directory.load(archive, book, labeler, scope or Scope(), listed), archive


def _ids(entries) -> list[str]:
    return [entry.conversation["id"] for entry in entries]


def test_find_by_name_gives_every_thread_of_the_person_and_their_groups() -> None:
    directory, _ = _directory()
    assert _ids(directory.find("maya")) == [MAYA_PHONE, GROUP, MAYA_EMAIL]
    assert _ids(directory.find("Maya Okafor")) == [MAYA_PHONE, GROUP, MAYA_EMAIL]
    assert _ids(directory.find("okafor weber")) == [GROUP]


def test_find_ignores_case_and_diacritics_and_allows_prefixes() -> None:
    directory, _ = _directory()
    assert _ids(directory.find("ines")) == [INES_PHONE]
    assert _ids(directory.find("INÊS")) == [INES_PHONE]
    assert _ids(directory.find("lisb")) == [GROUP]
    assert directory.find("zz") == []


def test_find_by_digits_matches_handles_without_showing_them() -> None:
    directory, _ = _directory()
    assert directory.find("720") == [] and directory.find("72055") == []
    found = directory.find("720", loose_digits=True)
    assert _ids(found) == [GROUP, UNSAVED_PHONE]
    assert all("720" not in entry.label for entry in found)
    assert _ids(directory.find("0199")) == [GROUP, UNSAVED_PHONE]
    assert _ids(directory.find("720 555 0199")) == []
    assert _ids(directory.find("7205550199")) == [GROUP, UNSAVED_PHONE]


def test_mask_words_are_not_names() -> None:
    directory, _ = _directory()
    assert directory.find("number") == []
    assert directory.find("unsaved") == []
    assert directory.find("ending") == []


def test_named_in_finds_names_and_handle_digits_in_a_question() -> None:
    directory, _ = _directory()
    assert directory.named_in("what did Maya say about the lease?") == ["maya"]
    assert directory.named_in("compare jonas and inês") == ["jonas", "ines"]
    assert directory.named_in("who is this 720 number") == ["720"]
    assert directory.named_in("what happened in 2021") == []
    assert directory.named_in("anything about sourdough?") == []


def test_entries_know_how_a_direct_thread_is_reached() -> None:
    directory, _ = _directory()
    via = {entry.conversation["id"]: entry.via for entry in directory.entries}
    assert via[MAYA_PHONE] == "phone" and via[MAYA_EMAIL] == "email" and via[GROUP] == ""
    assert via[UNSAVED_EMAIL] == "email"


def test_top_is_by_message_count() -> None:
    directory, _ = _directory()
    assert _ids(directory.top(2)) == [MAYA_PHONE, GROUP]
    assert len(directory) == 7


def test_scoped_directory_reads_only_the_scope() -> None:
    directory, archive = _directory(Scope(conversations=(JONAS_PHONE,)))
    assert _ids(directory.entries) == [JONAS_PHONE]
    assert directory.find("maya") == []
    assert [name for name, _ in archive.calls] == ["conversations"]
