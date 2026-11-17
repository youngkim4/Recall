"""reports.py when a file, the notes of earlier runs or the model port fails in the middle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conversation import sanitize_filename
from recall_services import reports
from report_engine import LLMError, StoreError, render_markdown
from report_engine.fake_llm import FakeCall, FakeLLM, completed, default_reply
from tests.recall_services.report_world import MODEL, Desk
from tests.report_engine.synth import MINA, THEO

NAME = sanitize_filename(MINA)


def files_of(desk: Desk) -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(desk.places.out_dir.iterdir())}


def fail_on(monkeypatch: pytest.MonkeyPatch, ending: str) -> None:
    real = reports.write_text_atomic

    def write(path: Any, text: str) -> Path:
        if str(path).endswith(ending):
            raise StoreError("the disk is full")
        return real(path, text)

    monkeypatch.setattr(reports, "write_text_atomic", write)


def no_moments() -> FakeLLM:
    def responder(call: FakeCall) -> Any:
        return {**default_reply(call), "moments": []} if call.kind == "map" else None

    return FakeLLM(responder)


# --- the files of one report change together ---------------------------------------


@pytest.mark.parametrize("ending", [".csv", ".md"])
def test_a_write_that_fails_leaves_the_files_of_the_older_report_whole(
    desk: Desk, monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    """The report file and the events file are one report. A new story beside old moments is never left."""
    reports.run(MINA, MODEL, "2023-01-01", "2023-02-28", **desk.more())
    before = files_of(desk)
    fail_on(monkeypatch, ending)
    with pytest.raises(reports.ReportFailure) as caught:
        reports.run(MINA, MODEL, "2023-03-01", "2023-04-30", **desk.more())
    assert caught.value.code == "not_saved"
    assert files_of(desk) == before


def test_a_write_that_fails_does_not_bring_back_an_events_file_that_was_not_there(
    desk: Desk, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports.run(MINA, MODEL, **desk.more(no_moments()))
    before = files_of(desk)
    assert list(before) == [f"analysis_{NAME}.md"]
    fail_on(monkeypatch, ".md")
    with pytest.raises(reports.ReportFailure):
        reports.run(MINA, MODEL, **desk.more())
    assert files_of(desk) == before


def test_an_older_events_file_that_can_not_be_removed_is_a_report_that_was_not_saved(
    desk: Desk, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A report with no moment beside the moments of an older report would show those as its own."""
    reports.run(MINA, MODEL, "2023-01-01", "2023-02-28", **desk.more())
    before = files_of(desk)
    real = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name.startswith("events_timeline_"):
            raise PermissionError("held by another program")
        real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(reports.ReportFailure) as caught:
        # other months: what the first run read is not read again, and it holds moments
        reports.run(MINA, MODEL, "2023-03-01", "2023-04-30", **desk.more(no_moments()))
    assert caught.value.code == "not_saved"
    assert files_of(desk) == before


def test_an_older_page_that_can_not_be_removed_is_said_in_the_log(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    desk.places.out_dir.mkdir(parents=True, exist_ok=True)
    (desk.places.out_dir / f"analysis_{NAME}.html").write_text("<p>an older page</p>", encoding="utf-8")
    real = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self.suffix == ".html":
            raise PermissionError("held by another program")
        real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    said: list[str] = []
    reports.run(MINA, MODEL, on_progress=lambda update: said.append(update.get("log", "")), **desk.more())
    assert reports.PAGE_KEPT in said


# --- the notes of earlier runs ------------------------------------------------------


def broken_cache(desk: Desk) -> None:
    desk.places.cache_path.parent.mkdir(parents=True, exist_ok=True)
    desk.places.cache_path.write_bytes(b"this is no database " * 400)


def test_an_estimate_carries_on_when_the_notes_of_earlier_runs_can_not_be_read(desk: Desk) -> None:
    clean = reports.estimate(MINA, MODEL, **desk.estimate_more())
    broken_cache(desk)
    assert reports.estimate(MINA, MODEL, **desk.estimate_more()) == clean
    assert clean["periods_cached"] == 0


def test_a_run_says_that_the_notes_of_earlier_runs_can_not_be_opened_and_makes_no_call(desk: Desk) -> None:
    broken_cache(desk)
    model = FakeLLM()
    with pytest.raises(reports.ReportFailure) as caught:
        reports.run(MINA, MODEL, **desk.more(model))
    assert caught.value.code == "cache_unreadable"
    assert str(caught.value) == (
        "The file cache.sqlite3, which holds what earlier report runs read, can not be opened. No report was written."
    )
    assert model.calls == []


# --- the failure that is told is the one that ended the run ---------------------


@pytest.mark.parametrize("status", [401, 429])
def test_a_part_that_failed_early_is_not_blamed_for_a_report_that_was_not_written(desk: Desk, status: int) -> None:
    """One part fails with the key error, the others are read, and then the model writes no report."""
    seen = {"parts": 0}

    def responder(call: FakeCall) -> Any:
        if call.kind == "map":
            seen["parts"] += 1
            if seen["parts"] == 1:
                raise LLMError(f"x (HTTP {status})", status_code=status)
            return None
        return completed(call, None, text="", status="incomplete")

    with pytest.raises(reports.ReportFailure) as caught:
        reports.run(MINA, MODEL, **desk.more(FakeLLM(responder)))
    assert caught.value.code == "report_not_written"


def test_a_model_port_that_can_not_be_made_is_a_failure_with_a_sentence(desk: Desk) -> None:
    def factory(_info: object) -> Any:
        raise ImportError("no module named openai")

    with pytest.raises(reports.ReportFailure) as caught:
        reports.run(MINA, MODEL, **{**desk.more(), "llm_factory": factory})
    assert caught.value.code == "run_failed"
    assert str(caught.value) == "The model could not be started. No report was written."


# --- one counter for the estimate and the run ----------------------------------


def test_the_estimate_and_the_run_count_with_one_counter(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two loads can give two counters, when the vocabulary of the tokenizer loads one time and not the other."""
    loaded: list[str] = []
    real = reports.load_counter

    def load(family: str) -> Any:
        loaded.append(family)
        return real(family)

    monkeypatch.setattr(reports, "load_counter", load)
    reports.shared_counter.cache_clear()
    more = {name: value for name, value in desk.more().items() if name != "counter"}
    estimate_more = {name: value for name, value in desk.estimate_more().items() if name != "counter"}
    reports.estimate(THEO, MODEL, **estimate_more)
    reports.run(THEO, MODEL, **more)
    reports.estimate(THEO, MODEL, **estimate_more)
    assert len(loaded) == 1
    reports.shared_counter.cache_clear()


# --- what the Reports screen asks ------------------------------------------------


def test_the_story_of_a_report_file_is_shown_plain_or_not(desk: Desk) -> None:
    done = reports.run(MINA, MODEL, **desk.more())
    text = done.report_path.read_text(encoding="utf-8")
    assert reports.shown_plain(text) is False is done.shown_plain
    assert reports.shown_plain("") is False


def test_the_saved_report_of_a_report_file(desk: Desk) -> None:
    """The file holds no id. The saved report is the one of the file when the file is its rendering."""
    first = reports.run(MINA, MODEL, "2023-01-01", "2023-02-28", **desk.more())
    older_text = first.report_path.read_text(encoding="utf-8")
    assert reports.saved_report(MINA, older_text, desk.places) == first.report
    second = reports.run(MINA, MODEL, **desk.more())
    newer_text = second.report_path.read_text(encoding="utf-8")
    assert newer_text == render_markdown(second.report)
    assert reports.saved_report(MINA, newer_text, desk.places) == second.report
    assert reports.saved_report(MINA, older_text, desk.places) == first.report
    assert reports.saved_report(MINA, "# Conversation Analysis - a report of the older kind", desk.places) is None
    assert reports.saved_report(THEO, newer_text, desk.places) is None
    assert reports.saved_report("", newer_text, desk.places) is None
