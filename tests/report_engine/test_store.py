"""Reports on disk: one folder per conversation, one file per run, never overwritten."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from report_engine.errors import ReportNotFound, StoreError
from report_engine.fake_llm import FakeLLM
from report_engine.models import SCHEMA_VERSION, Report, Scope
from report_engine.store import ReportStore, conversation_folder, write_text_atomic

from .support import SMALL, run
from .synth import CONVERSATION, GROUP, THEO
from .test_models import report as make_report


def test_two_runs_leave_two_files(reports_dir: Path) -> None:
    store = ReportStore(reports_dir)
    first = run(SMALL)
    second = run(SMALL).model_copy(update={"id": "20240701T101500Z-000002"})
    path_one, path_two = store.save(first), store.save(second)
    assert path_one != path_two
    assert path_one.parent == path_two.parent == reports_dir / conversation_folder(CONVERSATION)
    assert sorted(file.name for file in path_one.parent.iterdir()) == [
        "20240701T101500Z-000001.json",
        "20240701T101500Z-000002.json",
    ]


def test_a_saved_report_loads_back_equal(reports_dir: Path) -> None:
    store = ReportStore(reports_dir)
    report = run(SMALL, stats={"totalMessages": 40})
    store.save(report)
    assert store.load_report(CONVERSATION, report.id) == report


def test_a_new_run_never_overwrites_an_older_one(reports_dir: Path) -> None:
    store = ReportStore(reports_dir)
    report = make_report()
    path = store.save(report)
    before = path.read_bytes()
    with pytest.raises(StoreError):
        store.save(report.model_copy(update={"model": "another-model"}))
    assert path.read_bytes() == before
    assert [file.name for file in path.parent.iterdir()] == [path.name]


def test_the_file_is_plain_json_with_the_schema_version(reports_dir: Path) -> None:
    path = ReportStore(reports_dir).save(make_report(stats={"n": 1}))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["conversation"] == "+12025550143"
    assert [section["key"] for section in data["sections"]] == ["arc", "dynamics", "defining_moments", "big_picture"]
    assert data["stats"] == {"n": 1}
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_a_failed_write_leaves_no_file_behind(reports_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ReportStore(reports_dir)

    def no_link(source: Any, target: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "link", no_link)
    monkeypatch.setattr(os, "replace", no_link)
    with pytest.raises(StoreError):
        store.save(make_report())
    folder = reports_dir / conversation_folder("+12025550143")
    assert list(folder.iterdir()) == []


def test_save_works_where_hard_links_do_not(reports_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_link(source: Any, target: Any) -> None:
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(os, "link", no_link)
    store = ReportStore(reports_dir)
    report = make_report()
    path = store.save(report)
    assert store.load_report(report.conversation, report.id) == report
    with pytest.raises(StoreError):
        store.save(report)
    assert [file.name for file in path.parent.iterdir()] == [path.name]


def test_reports_are_listed_newest_first(reports_dir: Path) -> None:
    store = ReportStore(reports_dir)
    for number, stamp in enumerate(["2024-07-01T10:00:00+00:00", "2024-07-03T10:00:00+00:00", "2024-07-02T10:00:00+00:00"]):
        store.save(make_report(id=f"run-{number}", created_at=stamp))
    store.save(make_report(id="other-1", conversation=GROUP, created_at="2024-07-04T10:00:00+00:00"))
    mine = store.list_reports("+12025550143")
    assert [entry.id for entry in mine] == ["run-1", "run-2", "run-0"]
    assert [entry.id for entry in store.list_reports()] == ["other-1", "run-1", "run-2", "run-0"]
    first = mine[0]
    assert (first.conversation, first.model, first.status) == ("+12025550143", "test-model-1", "complete")
    assert first.scope == Scope()
    assert first.path.name == "run-1.json"


def test_listing_an_unknown_conversation_or_an_empty_store_is_empty(reports_dir: Path) -> None:
    store = ReportStore(reports_dir)
    assert store.list_reports() == []
    assert store.list_reports("nobody") == []
    assert store.latest_report("nobody") is None


def test_latest_report_is_the_newest_run(reports_dir: Path) -> None:
    store = ReportStore(reports_dir)
    store.save(make_report(id="old", created_at="2024-07-01T10:00:00+00:00"))
    store.save(make_report(id="new", created_at="2024-07-09T10:00:00+00:00"))
    latest = store.latest_report("+12025550143")
    assert latest is not None and latest.id == "new"


def test_loading_a_report_that_is_not_there_is_not_found(reports_dir: Path) -> None:
    store = ReportStore(reports_dir)
    store.save(make_report())
    with pytest.raises(ReportNotFound):
        store.load_report("+12025550143", "no-such-run")
    with pytest.raises(ReportNotFound):
        store.load_report("someone-else", "20240701T101500Z-abc123")


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", "", ".", "..", "x\x00y", "run 1"])
def test_a_report_id_can_not_leave_the_folder(reports_dir: Path, bad_id: str) -> None:
    store = ReportStore(reports_dir)
    with pytest.raises(StoreError):
        store.load_report("+12025550143", bad_id)
    with pytest.raises(StoreError):
        store.save(make_report(id=bad_id) if bad_id else make_report().model_copy(update={"id": ""}))


def test_a_damaged_file_is_skipped_in_lists_and_is_an_error_on_load(
    reports_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = ReportStore(reports_dir)
    good = store.save(make_report(id="good"))
    (good.parent / "bad.json").write_text("{not json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert [entry.id for entry in store.list_reports()] == ["good"]
    assert "bad.json" in caplog.text
    with pytest.raises(StoreError):
        store.load_report("+12025550143", "bad")


def test_a_report_from_a_newer_schema_is_refused(reports_dir: Path) -> None:
    store = ReportStore(reports_dir)
    path = store.save(make_report(id="future"))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(StoreError) as caught:
        store.load_report("+12025550143", "future")
    assert "newer" in str(caught.value)
    assert store.list_reports() == []


def test_folder_names_are_safe_and_do_not_collide() -> None:
    names = {conversation_folder(value) for value in ("a@b.com", "ab.com", "a/b.com", "../../etc", THEO, GROUP, "")}
    assert len(names) == 7
    for name in names:
        assert name not in {".", ".."}
        assert all(char.isalnum() or char in "+-_" for char in name)
    assert conversation_folder(GROUP).startswith("chat000000000000000042-")
    assert conversation_folder("+12025550143") == conversation_folder("+12025550143")


def test_stray_files_in_the_store_are_ignored(reports_dir: Path) -> None:
    store = ReportStore(reports_dir)
    path = store.save(make_report(id="only"))
    (path.parent / "notes.txt").write_text("not a report", encoding="utf-8")
    (path.parent / ".only.tmp").write_text("{}", encoding="utf-8")
    (reports_dir / "loose.json").write_text("{}", encoding="utf-8")
    assert [entry.id for entry in store.list_reports()] == ["only"]


def test_write_text_atomic_replaces_a_file_in_one_step(tmp_path: Path) -> None:
    target = tmp_path / "out" / "analysis.md"
    write_text_atomic(target, "first\n")
    write_text_atomic(target, "second\n")
    assert target.read_text(encoding="utf-8") == "second\n"
    assert [file.name for file in target.parent.iterdir()] == ["analysis.md"]


def test_a_run_and_a_save_work_together(reports_dir: Path) -> None:
    store = ReportStore(reports_dir)
    report = run(SMALL, llm=FakeLLM())
    store.save(report)
    latest = store.latest_report(CONVERSATION)
    assert isinstance(latest, Report)
    assert latest.sections == report.sections
