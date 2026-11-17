"""The adapter to the real archive package.

The package is built in another branch, so these tests use a stand-in. They
prove the delegation and the lazy import, not the real package.
"""

import sys
import types

import pytest

from chat_engine.archive_adapter import ArchiveAdapter, open_archive
from chat_engine.errors import ArchiveError
from tests.chat_engine.tiny import JONAS_PHONE, MAYA_PHONE, tiny_archive


def test_adapter_delegates_every_port_call() -> None:
    inner = tiny_archive()
    plain = tiny_archive()
    adapter = ArchiveAdapter(inner)
    assert adapter.summary() == plain.summary()
    assert adapter.conversations(limit=2, offset=1, order="recent") == plain.conversations(
        limit=2, offset=1, order="recent"
    )
    assert adapter.conversation(MAYA_PHONE) == plain.conversation(MAYA_PHONE)
    assert adapter.messages(MAYA_PHONE, before="a7", limit=3) == plain.messages(MAYA_PHONE, before="a7", limit=3)
    assert adapter.context("a5", radius=1) == plain.context("a5", radius=1)
    assert adapter.search("lease", conversations=[MAYA_PHONE], order="oldest", limit=1) == plain.search(
        "lease", conversations=[MAYA_PHONE], order="oldest", limit=1
    )
    assert adapter.count(text="lease", group_by="year") == plain.count(text="lease", group_by="year")
    assert adapter.stats(JONAS_PHONE, since="2024-01-01") == plain.stats(JONAS_PHONE, since="2024-01-01")
    assert [name for name, _ in inner.calls] == [name for name, _ in plain.calls]


def test_adapter_passes_only_the_cursor_that_was_given() -> None:
    seen: list[dict] = []

    class Strict:
        def messages(self, conversation_id: str, **keywords: object) -> dict:
            seen.append(keywords)
            return {"messages": [], "has_older": False, "has_newer": False}

    adapter = ArchiveAdapter(Strict())
    adapter.messages("c", limit=5)
    adapter.messages("c", around="m1", limit=5)
    adapter.messages("c", before="m2", limit=5)
    adapter.messages("c", after="m3", limit=5)
    assert seen == [
        {"limit": 5},
        {"limit": 5, "around": "m1"},
        {"limit": 5, "before": "m2"},
        {"limit": 5, "after": "m3"},
    ]


def test_adapter_closes_the_archive_and_works_as_a_context_manager() -> None:
    class Closable:
        closed = 0

        def close(self) -> None:
            self.closed += 1

    inner = Closable()
    with ArchiveAdapter(inner) as adapter:
        assert isinstance(adapter, ArchiveAdapter)
    assert inner.closed == 1
    ArchiveAdapter(object()).close()


def test_open_archive_imports_the_package_only_when_called(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[tuple] = []

    class FakeArchiveClass:
        @classmethod
        def open(cls, path: object, *, readonly: bool = True) -> object:
            opened.append((path, readonly))
            return tiny_archive()

    module = types.ModuleType("archive")
    module.Archive = FakeArchiveClass  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "archive", module)
    adapter = open_archive("/tmp/does-not-matter.sqlite3")
    assert opened == [("/tmp/does-not-matter.sqlite3", True)]
    assert adapter.summary()["messages"] == 27


def test_open_archive_falls_back_to_the_queries_module(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeArchiveClass:
        @classmethod
        def open(cls, path: object, *, readonly: bool = True) -> object:
            return tiny_archive()

    package = types.ModuleType("archive")
    package.__path__ = []  # type: ignore[attr-defined]
    queries = types.ModuleType("archive.queries")
    queries.Archive = FakeArchiveClass  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "archive", package)
    monkeypatch.setitem(sys.modules, "archive.queries", queries)
    assert open_archive(None).summary()["conversations"] == 7


def test_missing_archive_package_is_a_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "archive", None)
    with pytest.raises(ArchiveError) as caught:
        open_archive(None)
    assert "archive" in caught.value.detail


def test_failure_to_open_is_a_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        @classmethod
        def open(cls, path: object, *, readonly: bool = True) -> object:
            raise FileNotFoundError("no archive at that path")

    module = types.ModuleType("archive")
    module.Archive = Broken  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "archive", module)
    with pytest.raises(ArchiveError) as caught:
        open_archive("/nowhere")
    assert "FileNotFoundError" in caught.value.detail
