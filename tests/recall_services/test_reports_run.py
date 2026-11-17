"""reports.run: a report run that saves the report and writes the files the server reads."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

import safe_html
from conversation import sanitize_filename
from recall_services import models, reports
from report_engine import CancelToken, LLMError, Report, ReportStore
from report_engine.fake_llm import FakeCall, FakeLLM, default_reply
from report_engine.markdown import MAX_SECTION_CHARS
from report_engine.models import SECTION_KEYS
from tests.integration import story_cases
from tests.recall_services.report_world import MODEL, Desk
from tests.report_engine.story_reader import story_of
from tests.report_engine.synth import MINA, THEO

EVENT_COLUMNS = ["date", "title", "detail", "category", "score", "quote"]
# the keys of a job result that the Jobs screen and the Reports screen read, and what is new
RESULT_KEYS = {"reportPath", "eventsPath", "htmlPath", "jsonPath", "reportId", "status", "warnings", "shownPlain", "usage"}


def failing(error: Exception, kind: str = "map") -> FakeLLM:
    def responder(call: FakeCall) -> Any:
        if call.kind == kind:
            raise error
        return None

    return FakeLLM(responder)


def no_moments() -> FakeLLM:
    def responder(call: FakeCall) -> Any:
        return {**default_reply(call), "moments": []} if call.kind == "map" else None

    return FakeLLM(responder)


def rows_of(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_a_run_saves_the_report_and_writes_the_files(desk: Desk) -> None:
    done = reports.run(MINA, MODEL, **desk.more())
    name = sanitize_filename(MINA)
    assert done.report_path == desk.places.out_dir / f"analysis_{name}.md"
    assert done.events_path == desk.places.out_dir / f"events_timeline_{name}.csv"
    assert done.json_path.parent.parent == desk.places.store_dir
    assert done.report_path.read_text(encoding="utf-8").splitlines()[0] == f"# Conversation Analysis - {MINA}"
    stored = ReportStore(desk.places.store_dir).load_report(MINA, done.report_id)
    assert isinstance(stored, Report) and stored == done.report
    assert (stored.status, stored.model, len(stored.periods)) == ("complete", MODEL, 4)
    assert done.shown_plain is False


def test_the_result_is_ready_for_the_job(desk: Desk) -> None:
    done = reports.run(MINA, MODEL, **desk.more())
    found = done.as_dict()
    assert set(found) == RESULT_KEYS
    assert found["reportPath"] == str(done.report_path) and found["eventsPath"] == str(done.events_path)
    assert found["htmlPath"] is None
    assert (found["reportId"], found["status"], found["shownPlain"]) == (done.report_id, "complete", False)
    assert found["usage"]["calls"] == 5  # four periods and the report itself
    json.dumps(found)


def test_the_events_file_holds_the_moments(desk: Desk) -> None:
    done = reports.run(MINA, MODEL, **desk.more())
    assert done.events_path is not None
    rows = rows_of(done.events_path)
    assert list(rows[0]) == EVENT_COLUMNS
    assert [(row["date"], row["title"], row["category"], row["quote"]) for row in rows] == [
        (moment.date, moment.title, moment.category, moment.quote) for moment in done.report.moments
    ]
    assert [float(row["score"]) for row in rows] == [moment.score for moment in done.report.moments]
    assert len(rows) == len(done.report.moments) == 4


def test_the_file_name_of_an_address_is_the_one_the_server_makes(desk: Desk) -> None:
    done = reports.run(THEO, MODEL, **desk.more())
    assert done.report_path.name == "analysis_theoexamplecom.md"
    assert done.events_path is not None and done.events_path.name == "events_timeline_theoexamplecom.csv"


def test_a_report_with_no_moment_removes_the_older_events_file(desk: Desk) -> None:
    first = reports.run(MINA, MODEL, **desk.more())
    assert first.events_path is not None and first.events_path.exists()
    again = reports.run(MINA, MODEL, **desk.more(no_moments()))
    # the notes of the first run hold its moments, so this run has them too
    assert again.events_path is not None
    desk.places.cache_path.unlink()
    without = reports.run(MINA, MODEL, **desk.more(no_moments()))
    assert without.report.moments == ()
    assert without.events_path is None and without.as_dict()["eventsPath"] is None
    assert not first.events_path.exists()


def test_an_older_page_of_the_same_conversation_is_removed(desk: Desk) -> None:
    desk.places.out_dir.mkdir(parents=True)
    page = desk.places.out_dir / f"analysis_{sanitize_filename(MINA)}.html"
    other = desk.places.out_dir / f"analysis_{sanitize_filename(THEO)}.html"
    for path in (page, other):
        path.write_text("<p>a page of an older report</p>", encoding="utf-8")
    logs: list[str] = []
    reports.run(MINA, MODEL, on_progress=lambda update: logs.append(update.get("log", "")), **desk.more())
    assert not page.exists() and other.exists()
    assert "The page of an older report was removed: this report has no page." in logs


def test_a_new_report_takes_the_place_of_the_report_file_and_keeps_both_in_the_store(desk: Desk) -> None:
    first = reports.run(MINA, MODEL, **desk.more())
    second = reports.run(MINA, MODEL, **desk.more())
    assert first.report_path == second.report_path
    assert first.json_path != second.json_path and first.json_path.exists() and second.json_path.exists()
    assert len(ReportStore(desk.places.store_dir).list_reports(MINA)) == 2


def test_the_dates_of_a_report(desk: Desk) -> None:
    done = reports.run(MINA, MODEL, "2023-02-01", "2023-02-28", **desk.more())
    assert (done.report.scope.since, done.report.scope.until) == ("2023-02-01", "2023-02-28")
    assert "- Scope: 2023-02-01 to 2023-02-28" in done.report_path.read_text(encoding="utf-8")
    assert len(done.report.periods) == 1
    # the figures of the archive for those days go into the report
    assert done.report.stats["total_messages"] == desk.archive.count(
        conversations=[MINA], since="2023-02-01", until="2023-02-28"
    )


def test_the_model_reads_saved_names_and_no_handle(desk: Desk) -> None:
    model = FakeLLM()
    reports.run(MINA, MODEL, **desk.more(model))
    sent = "\n".join(call.text for call in model.calls)
    assert "Mina" in sent and MINA not in sent and "2025550143" not in sent
    unsaved = FakeLLM()
    reports.run(THEO, MODEL, **desk.more(unsaved))
    sent = "\n".join(call.text for call in unsaved.calls)
    assert "Unsaved contact" in sent and THEO not in sent and "an unsaved" not in sent


# --- progress ----------------------------------------------------------------


def test_progress_has_the_shape_the_jobs_screen_reads(desk: Desk) -> None:
    updates: list[dict[str, Any]] = []
    reports.run(MINA, MODEL, on_progress=updates.append, **desk.more())
    steps = [update["progress"] for update in updates if "progress" in update]
    assert all(set(step) == {"step", "of", "label"} for step in steps)
    assert all(set(update) <= {"progress", "log"} for update in updates)
    # reading, four parts, writing, saving
    assert [step["step"] for step in steps] == sorted(step["step"] for step in steps)
    assert steps[0] == {"step": 1, "of": 3, "label": "Reading the conversation"}
    assert steps[-1] == {"step": 7, "of": 7, "label": "Saving the report"}
    assert {"step": 3, "of": 7, "label": "Reading part 2 of 4"} in steps
    assert {"step": 6, "of": 7, "label": "Writing the report"} in steps
    logs = [update["log"] for update in updates if "log" in update]
    assert "Reading Jan 2023 (part 1 of 4)" in logs
    assert logs[-1] == "The report is saved."
    json.dumps(updates)


def test_a_second_run_says_which_parts_it_did_not_read_again(desk: Desk) -> None:
    reports.run(MINA, MODEL, **desk.more())
    model = FakeLLM()
    logs: list[str] = []
    reports.run(MINA, MODEL, on_progress=lambda update: logs.append(update.get("log", "")), **desk.more(model))
    assert len(model.map_calls) == 0 and len(model.reduce_calls) == 1
    assert "Jan 2023 was read in an earlier run (part 1 of 4)" in logs


def test_a_progress_listener_that_fails_does_not_stop_the_run(desk: Desk) -> None:
    def broken(_update: dict[str, Any]) -> None:
        raise RuntimeError("the listener is gone")

    assert reports.run(MINA, MODEL, on_progress=broken, **desk.more()).report.status == "complete"


# --- warnings, and a story that the server shows as plain paragraphs ----------


def test_the_warnings_of_the_report_go_out_with_it(desk: Desk) -> None:
    def one_period_fails(call: FakeCall) -> Any:
        if call.kind == "map" and call.first_day.startswith("2023-02"):
            raise LLMError("the model service failed (HTTP 500)", status_code=500)
        return None

    logs: list[str] = []
    done = reports.run(
        MINA, MODEL, on_progress=lambda update: logs.append(update.get("log", "")), **desk.more(FakeLLM(one_period_fails))
    )
    assert done.report.status == "partial" and done.as_dict()["status"] == "partial"
    assert len(done.warnings) == 1 and "Feb 2023" in done.warnings[0]
    assert f"Note: {done.warnings[0]}" in logs
    assert "Feb 2023 could not be read (part 2 of 4)" in logs


def test_a_story_of_ordinary_length_is_rendered(desk: Desk) -> None:
    done = reports.run(MINA, MODEL, **desk.more())
    story = story_of(done.report_path.read_text(encoding="utf-8"))
    assert done.shown_plain is False
    assert safe_html.story_html(story) != safe_html._plain(story)


def test_a_story_that_the_server_shows_as_plain_paragraphs_is_said(desk: Desk) -> None:
    dense = story_cases.dialogue(MAX_SECTION_CHARS)

    def responder(call: FakeCall) -> Any:
        return dict.fromkeys(SECTION_KEYS, dense) if call.kind == "reduce" else None

    logs: list[str] = []
    done = reports.run(
        MINA, MODEL, on_progress=lambda update: logs.append(update.get("log", "")), **desk.more(FakeLLM(responder))
    )
    story = story_of(done.report_path.read_text(encoding="utf-8"))
    assert done.shown_plain is True and done.as_dict()["shownPlain"] is True
    # the server decides the same
    assert safe_html.story_html(story) == safe_html._plain(story)
    said = [line for line in logs if "plain paragraphs" in line]
    assert said == [
        "The story of this report is long and dense. The app shows it as plain paragraphs, with no chapters."
    ]


# --- failures ----------------------------------------------------------------


def expect_failure(desk: Desk, model: FakeLLM, code: str, contact: str = MINA, **more: Any) -> reports.ReportFailure:
    with pytest.raises(reports.ReportFailure) as caught:
        reports.run(contact, MODEL, **{**desk.more(model), **more})
    assert caught.value.code == code
    return caught.value


def test_days_with_no_message_write_no_report(desk: Desk) -> None:
    failure = expect_failure(desk, FakeLLM(), "empty", since="2020-01-01", until="2020-12-31")
    assert str(failure) == "This conversation has no messages with text in these dates. No report was written."
    assert not desk.places.out_dir.exists() or list(desk.places.out_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("error", "code", "sentence"),
    [
        (LLMError("x (HTTP 401)", status_code=401), "key_rejected", "OpenAI rejected the API key. Check the key in Settings, then run the report again."),
        (LLMError("x (HTTP 403)", status_code=403), "key_rejected", "OpenAI does not allow this API key to use the model. Check the key in Settings, then run the report again."),
        (LLMError("x (HTTP 404)", status_code=404), "unknown_model", "OpenAI does not know the model gpt-6.1-sol. Choose another level in Settings."),
        (LLMError("x (HTTP 429)", status_code=429), "rate_limit", "OpenAI is limiting requests, or the account has no credit left. Wait a moment, then run the report again."),
        (LLMError("the model service failed (APIConnectionError)"), "offline", "OpenAI could not be reached. Check the internet connection, then run the report again."),
        (LLMError("the model service failed (APITimeoutError)"), "timeout", "The model took too long to answer. Run the report again."),
        (LLMError("x (HTTP 500)", status_code=500), "model_error", "The model could not read the conversation. No report was written. Run the report again."),
    ],
)
def test_a_run_in_which_every_part_fails_says_why(desk: Desk, error: LLMError, code: str, sentence: str) -> None:
    failure = expect_failure(desk, failing(error), code)
    assert str(failure) == sentence
    assert not (desk.places.out_dir / f"analysis_{sanitize_filename(MINA)}.md").exists()


def test_a_report_that_the_model_did_not_write_keeps_what_was_read(desk: Desk) -> None:
    failure = expect_failure(desk, failing(LLMError("x (HTTP 500)", status_code=500), kind="reduce"), "report_not_written")
    assert str(failure) == (
        "The model read the conversation but did not write the report. "
        "What it read is kept: run the report again."
    )
    assert failure.usage is not None and failure.usage["calls"] == 4
    assert reports.estimate(MINA, MODEL, **desk.estimate_more())["periods_cached"] == 4


def test_a_stopped_run_writes_no_report(desk: Desk) -> None:
    token = CancelToken()
    token.cancel()
    failure = expect_failure(desk, FakeLLM(), "cancelled", cancel=token)
    assert str(failure) == "The report was stopped. What was read is kept for the next run."


def test_with_no_key_no_request_is_made_and_no_file_is_written(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    more = {name: value for name, value in desk.more().items() if name != "llm_factory"}
    with pytest.raises(reports.ReportFailure) as caught:
        reports.run(MINA, MODEL, **more)
    assert caught.value.code == "no_key"
    assert str(caught.value) == "No API key is set. Add your OpenAI API key in Settings, then run the report again."
    assert not desk.places.out_dir.exists()


def test_a_model_that_can_not_be_run_is_refused_before_any_call(desk: Desk) -> None:
    model = FakeLLM()
    with pytest.raises(models.UnknownModelIdError):
        reports.run(MINA, "gpt-7", **desk.more(model))
    with pytest.raises(models.ModelNotOfferedError):
        reports.run(MINA, "gpt-6-astra", **desk.more(model))
    assert model.calls == []


def test_a_report_that_can_not_be_saved_is_a_failure_with_its_cost(desk: Desk) -> None:
    desk.places.store_dir.parent.mkdir(parents=True, exist_ok=True)
    desk.places.store_dir.write_text("a file stands where the folder of the reports must be", encoding="utf-8")
    failure = expect_failure(desk, FakeLLM(), "not_saved")
    assert str(failure) == "The report was written but could not be saved. Run the report again."
    assert failure.usage is not None and failure.usage["calls"] == 5


@pytest.mark.parametrize("contact", ["", None])
def test_a_conversation_is_needed(desk: Desk, contact: object) -> None:
    with pytest.raises(ValueError, match="A conversation is needed"):
        reports.run(contact, MODEL, **desk.more())  # type: ignore[arg-type]


def test_the_model_port_of_the_app_goes_to_the_address_of_the_provider_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/elsewhere")
    llm = reports.default_llm(models.model_info(MODEL))
    assert str(llm._client.base_url).startswith("https://api.openai.com/")
    assert llm._model == MODEL


# --- the moments of a report, for the Reports screen ----------------------------


def test_a_moment_with_a_quote_opens_the_day_at_the_quoted_message(desk: Desk) -> None:
    done = reports.run(MINA, MODEL, **desk.more())
    found = reports.moments_for(done.report, desk.archive)
    assert len(found) == len(done.report.moments) == 4
    for shaped, moment in zip(found, done.report.moments, strict=True):
        assert set(shaped) == {*EVENT_COLUMNS, "chatId", "day", "at", "pickedId"}
        assert (shaped["date"], shaped["title"], shaped["quote"]) == (moment.date, moment.title, moment.quote)
        quoted = desk.archive.message(moment.quote_message_id)
        assert quoted is not None and moment.quote_verified
        assert (shaped["chatId"], shaped["pickedId"], shaped["at"]) == (MINA, quoted["id"], quoted["timestamp"])
        assert shaped["day"] == quoted["timestamp"][:10]
        # /api/day opens at this message
        held = desk.archive.day(shaped["chatId"], shaped["day"], around=shaped["at"])
        assert shaped["pickedId"] in [message["id"] for message in held["messages"]]
    json.dumps(found)


def test_a_moment_with_no_quote_opens_its_day(desk: Desk) -> None:
    def unquoted(call: FakeCall) -> Any:
        reply = default_reply(call)
        if call.kind != "map":
            return None
        return {**reply, "moments": [{**moment, "quote": ""} for moment in reply["moments"]]}

    done = reports.run(MINA, MODEL, **desk.more(FakeLLM(unquoted)))
    for shaped, moment in zip(reports.moments_for(done.report, desk.archive), done.report.moments, strict=True):
        assert not moment.quote_verified
        assert (shaped["chatId"], shaped["day"], shaped["at"], shaped["pickedId"]) == (MINA, moment.date, None, "")


def test_a_quoted_message_that_the_archive_no_longer_holds_opens_the_day_only(desk: Desk) -> None:
    done = reports.run(MINA, MODEL, **desk.more())
    first = done.report.moments[0]
    gone = first.model_copy(update={"quote_message_id": "no-such-guid", "message_ids": ("no-such-guid",)})
    changed = done.report.model_copy(update={"moments": (gone, *done.report.moments[1:])})
    shaped = reports.moments_for(changed, desk.archive)[0]
    assert (shaped["day"], shaped["at"], shaped["pickedId"]) == (first.date, None, "")
