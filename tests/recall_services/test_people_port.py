"""`people_port`: the search of people of chat over `recall_services.people`.

`recall_services/people.py` is built by another part at the same time. These
tests put a stand-in module with its signature in its place:

    status(archive, names) -> {"state": "none" | "ready", "profiles": int, ...}
    search(archive, names, description, *, kinds=None, since=None, until=None, limit=10)
        -> a list of rows, with `reason` (None when the search ran)
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from chat_engine.fake_llm import FakeLLM, call, say
from recall_services import chat, memory, people_port
from tests.recall_services import world

ROW = {
    "key": "person:1",
    "conversations": ["+15550000001"],
    "label": "Maya",
    "kind": "talking_stage",
    "status": "faded",
    "summary": "A few weeks of texts.",
    "turning_points": [{"date": "2024-01-02", "what": "They meet."}],
    "figures": {"first_day": "2024-01-01", "last_day": "2024-03-01", "messages": 40},
    "score": 0.8,
    "extra": "not passed on",
}


class Found(list):
    """The list that `people.search` gives: rows, and why the search did not run."""

    def __init__(self, rows: list, reason: str | None = None) -> None:
        super().__init__(rows)
        self.reason = reason


def stand_in(*, state: Any = None, rows: list | None = None, reason: str | None = None) -> types.ModuleType:
    module = types.ModuleType("recall_services.people")
    module.calls = []  # type: ignore[attr-defined]
    module.more = []  # type: ignore[attr-defined]

    def status(archive: Any, names: Any) -> Any:
        module.calls.append(("status", archive, names))  # type: ignore[attr-defined]
        return {"state": "ready", "profiles": 3} if state is None else state

    def search(
        archive: Any, names: Any, description: str, *, kinds=None, since=None, until=None, limit=10, **more: Any
    ) -> Found:
        module.calls.append(("search", description, kinds, since, until, limit))  # type: ignore[attr-defined]
        module.more.append(more)  # type: ignore[attr-defined]
        given = rows if rows is not None else [ROW]
        return Found([dict(row) if isinstance(row, dict) else row for row in given], reason)

    module.status = status  # type: ignore[attr-defined]
    module.search = search  # type: ignore[attr-defined]
    return module


def test_no_key_gives_no_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    module = stand_in()
    assert people_port.port("archive", "names", service=module) is None
    assert module.calls == []


def test_no_profiles_gives_no_port() -> None:
    for state in ({"state": "none", "profiles": 0}, {"state": "ready", "profiles": 0}, "broken", {"profiles": "x"}):
        assert people_port.port("archive", "names", service=stand_in(state=state)) is None


def test_a_status_that_fails_gives_no_port_and_one_line_with_no_private_text(caplog: pytest.LogCaptureFixture) -> None:
    module = stand_in()

    def broken(archive: Any, names: Any) -> Any:
        raise OSError("/private/path/people.sqlite3 for +15550000001")

    module.status = broken  # type: ignore[attr-defined]
    assert people_port.port("archive", "names", service=module) is None
    assert "OSError" in caplog.text and "+1555" not in caplog.text and "/private" not in caplog.text


def test_a_tree_without_the_service_gives_no_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "recall_services.people", None)  # an import of it fails
    assert people_port.port("archive", "names") is None


def test_the_port_reads_the_service_of_the_app_by_its_name(monkeypatch: pytest.MonkeyPatch) -> None:
    module = stand_in()
    monkeypatch.setitem(sys.modules, "recall_services.people", module)
    found = people_port.port("the archive", "the names")
    assert isinstance(found, people_port.PeoplePort)
    assert module.calls == [("status", "the archive", "the names")]


def test_search_passes_the_arguments_and_gives_the_rows_in_the_form_of_the_port() -> None:
    module = stand_in()
    port = people_port.port("archive", "names", service=module)
    assert port is not None
    rows = port.search("a fling", kinds=("talking_stage",), since="2024-01-01", until=None, limit=5)
    assert module.calls[-1] == ("search", "a fling", ["talking_stage"], "2024-01-01", None, 5)
    assert rows == [{key: value for key, value in ROW.items() if key != "extra"}]
    port.search("a fling", kinds=None, since=None, until=None, limit=5)
    assert module.calls[-1][2] is None
    # the conversations of a scope go to the service; with none, the keyword is not sent at all
    assert module.more == [{}, {}]
    port.search("a fling", kinds=None, since=None, until=None, limit=5, conversations=("+15550000001",))
    assert module.more[-1] == {"conversations": ["+15550000001"]}


def test_rows_of_another_form_are_left_out_and_missing_fields_get_empty_values() -> None:
    module = stand_in(rows=["odd", {"key": "k", "conversations": ["c"], "score": "high"}])
    port = people_port.port("archive", "names", service=module)
    assert port is not None
    (row,) = port.search("x", kinds=None, since=None, until=None, limit=5)
    assert row == {
        "key": "k", "conversations": ["c"], "label": "", "kind": "", "status": "", "summary": "",
        "turning_points": [], "figures": {}, "score": 0.0,
    }


def test_a_search_that_did_not_run_raises_so_the_engine_says_so() -> None:
    for reason in ("no_index", "no_key", "embed_failed"):
        port = people_port.port("archive", "names", service=stand_in(reason=reason))
        assert port is not None
        with pytest.raises(people_port.PeopleNotAvailable) as raised:
            port.search("x", kinds=None, since=None, until=None, limit=5)
        assert raised.value.reason == reason


def test_chat_gives_the_engine_the_people_port(archive: Any, saved: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "ports", lambda _archive, _names: {"meaning": None, "digests": None})
    module = stand_in(rows=[{**ROW, "conversations": [world.GARDEN]}])
    asked: list[tuple[Any, Any]] = []

    def given(archive_: Any, names_: Any) -> Any:
        asked.append((archive_, names_))
        return people_port.PeoplePort(archive_, names_, module)

    monkeypatch.setattr(people_port, "port", given)
    llm = FakeLLM([[call("find_people", "p1", description="a fling")], say("Nobody fits.")])
    events = list(chat.stream("Who was a fling?", archive=archive, names=saved, model="gpt-6.1-sol",
                              llm_factory=lambda _config: llm))
    assert events[-1]["type"] == "done"
    assert asked == [(archive, saved)]
    assert "find_people" in [spec.name for spec in llm.requests[0].tools]
    assert [made[1] for made in module.calls if made[0] == "search"] == ["Who was a fling?", "a fling"]


def test_a_test_can_give_its_own_people_port_through_memory_ports(
    archive: Any, saved: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory, "ports", lambda _archive, _names: {"meaning": None, "digests": None, "people": None})
    monkeypatch.setattr(people_port, "port", lambda *_: pytest.fail("not asked when memory.ports gives one"))
    llm = FakeLLM([say("Nobody.")])
    events = list(chat.stream("Who?", archive=archive, names=saved, model="gpt-6.1-sol", llm_factory=lambda _c: llm))
    assert events[-1]["type"] == "done" and "find_people" not in [spec.name for spec in llm.requests[0].tools]
