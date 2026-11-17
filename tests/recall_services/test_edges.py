"""The branches of the service layer that only a fault reaches, and what the package may import."""

from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from archive import Archive, ArchiveError, ArchiveNotFoundError, SourceError
from message_text import ATTACHMENT_PLACEHOLDER
from recall_services import archive_state, files, frame, models, names, outcomes, report_failures, reports
from recall_services.archive_state import ArchiveState, PullFailed
from recall_services.names import Names
from recall_services.report_progress import WAITING, Teller
from report_engine import (
    ConfigError,
    LLMError,
    ProgressEvent,
    ReportEngineError,
    ReportOptions,
    RunError,
    StoreError,
)
from report_engine import SourceError as ReportSourceError
from tests.archive.builders import ChatDbBuilder
from tests.recall_services import state_data as data
from tests.recall_services.report_world import MODEL, Desk
from tests.report_engine.synth import MINA, THEO

PACKAGE = Path(reports.__file__).parent
# modules that read real data when they load or run, or that belong to the server
FORBIDDEN = {
    "ui_server",
    "contact_names",
    "openai_client",
    "analysis",
    "ai_analysis",
    "cli",
    "parse_imessage",
    "merge_export",
    "semantic_index",
    "retrieval_planner",
    "local_store",
    "html_report",
}


def imported_by(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {name.name.split(".")[0] for name in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("module", sorted(PACKAGE.glob("*.py")), ids=lambda path: path.name)
def test_no_service_imports_the_server_or_a_module_that_reads_real_data(module: Path) -> None:
    assert imported_by(module) & FORBIDDEN == set()


def test_no_service_names_the_function_that_reads_the_contacts_database() -> None:
    for module in PACKAGE.glob("*.py"):
        assert "resolve_contact_names" not in module.read_text(encoding="utf-8"), module.name


def test_importing_the_package_loads_no_model_sdk() -> None:
    import subprocess
    import sys

    code = (
        "import sys; import recall_services; "
        "from recall_services import chat, models, search, day, conversations, archive_state; "
        "print(sorted(name for name in ('openai', 'ui_server', 'contact_names') if name in sys.modules))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


# --- archive_state ------------------------------------------------------------


@pytest.fixture
def state(tmp_path: Path) -> Any:
    made = ArchiveState(tmp_path / "saves" / "archive.sqlite3")
    yield made
    made.close()


def test_an_archive_whose_only_message_has_no_conversation_holds_a_pull(state: ArchiveState, tmp_path: Path) -> None:
    source = tmp_path / "chat.db"
    builder = ChatDbBuilder(source)
    builder.message(None, when=data.START, text="a note with no chat", handle=builder.handle(data.PERSON))
    builder.close()
    assert state.pull_from_messages(source, None).status == "done"
    state.source_path.unlink()  # the note is lost: the content of the archive still says where it came from
    assert state.ensure_from_export(data.write_export(tmp_path / "messages.csv", ["one"])).status == "kept"


def test_a_new_archive_that_is_not_in_one_file_is_not_moved_into_place(tmp_path: Path) -> None:
    database = tmp_path / "new.sqlite3"
    writer = sqlite3.connect(database, isolation_level=None)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE notes (words TEXT)")
    writer.execute("INSERT INTO notes VALUES ('kept in the side file')")
    reader = sqlite3.connect(database, isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM notes").fetchall()
    writer.execute("INSERT INTO notes VALUES ('written while a reader looks')")
    try:
        # the reader holds the side file: it can not be emptied into the database
        with pytest.raises(ArchiveError, match="not in one file"):
            archive_state._settle(database)
    finally:
        reader.close()
        writer.close()


def test_a_load_that_can_not_be_finished_leaves_the_archive_as_it_was(
    state: ArchiveState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = data.write_export(tmp_path / "messages.csv", ["one"])
    state.ensure_from_export(export)

    def not_ready(_database: Path) -> None:
        raise ArchiveError("the new archive is not in one file")

    monkeypatch.setattr(archive_state, "_settle", not_ready)
    result = state.ensure_from_export(data.write_export(export, ["one", "two"]))
    assert (result.status, result.reason) == ("failed", "The export could not be loaded. The archive was not changed.")
    assert data.texts_of(state.archive()) == ["one"]
    assert data.kept_files(state.path.parent) == ["archive.sqlite3", "archive.sqlite3.source.json"]


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (SourceError("the copy changed three times"), "The Messages database could not be read. The archive was not changed."),
        (ArchiveError("the archive is newer than this build"), "The update failed. The archive was not changed."),
    ],
)
def test_a_pull_that_the_importer_can_not_finish(
    state: ArchiveState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: ArchiveError, reason: str
) -> None:
    database = data.write_database(tmp_path / "chat.db", ["one"])
    state.pull_from_messages(database, None)

    def fails(*_args: object, **_more: object) -> None:
        raise error

    monkeypatch.setattr(archive_state, "import_chat_db", fails)
    result = state.pull_from_messages(database, None)
    assert isinstance(result, PullFailed)
    assert result.reason == reason and type(error).__name__ in result.detail
    assert data.texts_of(state.archive()) == ["one"]


def test_the_answers_are_ready_for_json() -> None:
    built = outcomes.EnsureResult("built", counts={"messages_added": 2})
    assert built.as_dict() == {"status": "built", "changed": True, "reason": "", "counts": {"messages_added": 2}}
    assert outcomes.PullBusy().as_dict() == {"status": "busy", "changed": False, "reason": "An import is already running."}
    assert outcomes.PullCancelled().as_dict()["reason"] == "The update was stopped. The archive was not changed."
    failed = outcomes.PullFailed(reason="It failed.", detail="for the log")
    assert failed.as_dict() == {"status": "failed", "changed": False, "reason": "It failed."}
    json.dumps([built.as_dict(), failed.as_dict()])


# --- files --------------------------------------------------------------------


def test_a_text_file_that_can_not_be_put_in_place_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "note.json"
    files.write_text_atomic(target, "first")

    def refuses(*_args: object) -> None:
        raise OSError("the disk is full")

    monkeypatch.setattr(files.os, "replace", refuses)
    with pytest.raises(OSError, match="the disk is full"):
        files.write_text_atomic(target, "second")
    assert target.read_text(encoding="utf-8") == "first"
    assert [path.name for path in tmp_path.iterdir()] == ["note.json"]


def test_a_folder_that_is_not_there_is_not_synced(tmp_path: Path) -> None:
    files.fsync_folder(tmp_path / "nowhere")


# --- frame --------------------------------------------------------------------


def test_each_form_of_the_old_label_reads_as_the_placeholder(tmp_path: Path) -> None:
    source = tmp_path / "chat.db"
    builder = ChatDbBuilder(source)
    person = builder.handle(data.PERSON)
    thread = builder.chat(data.PERSON, members=[person])
    with_path = builder.message(thread, when=data.START, handle=person)
    builder.attachment(with_path, mime_type="image/jpeg", transfer_name="a.jpeg")
    no_path = builder.message(thread, when=data.START.replace(minute=1), handle=person)
    builder.attachment(no_path, mime_type="video/quicktime", transfer_name=None, filename="")
    builder.close()
    state = ArchiveState(tmp_path / "saves" / "archive.sqlite3")
    try:
        assert state.pull_from_messages(source, None).status == "done"
        assert list(frame.messages_frame(state)["text"]) == [ATTACHMENT_PLACEHOLDER, ATTACHMENT_PLACEHOLDER]
    finally:
        state.close()
    assert frame._old_label(None, None) == "[Attachment]"
    assert frame._old_label("image/png", "") == "[Attachment] image/png"


def test_with_no_archive_there_is_no_frame(tmp_path: Path) -> None:
    with pytest.raises(ArchiveNotFoundError):
        frame.messages_frame(ArchiveState(tmp_path / "saves" / "archive.sqlite3"))


# --- names --------------------------------------------------------------------


def test_a_record_with_one_value_that_is_not_in_a_list() -> None:
    saved = Names.from_records([{"name": "Petra Lund", "phones": "+15550190001", "emails": "petra@example.org"}])
    assert saved.saved("+15550190001") == saved.saved("PETRA@example.org") == "Petra Lund"


def test_a_member_with_no_handle_is_no_person_of_a_group() -> None:
    saved = Names.from_records([{"name": "Petra Lund", "phones": ["+15550190001"]}])
    assert saved.group(["", "+15550190001", "   "]) == "Petra Lund and others"


def test_a_cache_file_that_is_not_there_gives_no_names_each_time(tmp_path: Path) -> None:
    missing = tmp_path / "contact_names.json"
    assert len(names.cached(missing)) == 0
    missing.write_text(json.dumps({"contacts": [{"name": "Petra Lund", "phones": ["+15550190001"]}]}), encoding="utf-8")
    assert names.cached(missing).saved("+15550190001") == "Petra Lund"


# --- models -------------------------------------------------------------------


def test_a_family_that_is_confirmed_goes_to_the_engine_as_it_is(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = json.loads(Path(models.FACTS_PATH).read_text(encoding="utf-8"))
    sure = [{**entry, "tokenizer_family_assumed": False} for entry in raw["models"]]
    path = tmp_path / "models.json"
    path.write_text(json.dumps({**raw, "models": sure}), encoding="utf-8")
    monkeypatch.setattr(models, "_registry", lambda: models.load_registry(path))
    assert models.model_info("gpt-6.1-sol").tokenizer_family == "o200k_base"
    assert models.provider_name("openai") == "OpenAI"
    assert models.provider_name("another") == "The model provider"


# --- reports ------------------------------------------------------------------


def test_a_budget_that_can_not_hold_one_message_is_said_before_any_call(desk: Desk) -> None:
    tiny = ReportOptions(period_token_budget=1, map_max_output_tokens=500, reduce_max_output_tokens=800)
    for ask in (
        lambda: reports.estimate(MINA, MODEL, **{**desk.estimate_more(), "options": tiny}),
        lambda: reports.run(MINA, MODEL, **{**desk.more(), "options": tiny}),
    ):
        with pytest.raises(reports.ReportFailure) as caught:
            ask()
        assert caught.value.code == "too_large"
        assert str(caught.value) == "The model can not read this conversation with its limits. No report was written."


def test_an_archive_that_can_not_be_read_is_a_failure_in_plain_words(desk: Desk) -> None:
    closed = Archive.open(desk.pulled.archive_path)
    closed.close()
    for ask in (
        lambda: reports.estimate(MINA, MODEL, **{**desk.estimate_more(), "archive": closed}),
        lambda: reports.run(MINA, MODEL, **{**desk.more(), "archive": closed}),
    ):
        with pytest.raises(reports.ReportFailure) as caught:
            ask()
        assert caught.value.code == "archive_error"
        assert str(caught.value) == "The message archive could not be read. Update the archive, then try again."


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ConfigError("no room"), "too_large"),
        (ReportSourceError("the source raised ArchiveError"), "archive_error"),
        (StoreError("can not write"), "cache_unreadable"),
        (RunError("an error in the engine"), "run_failed"),
        (ReportEngineError("anything else"), "run_failed"),
    ],
)
def test_each_error_of_the_engine_has_a_code_and_a_sentence(error: ReportEngineError, code: str) -> None:
    failure = report_failures.failure_of(error, None, "OpenAI", MODEL)
    assert failure.code == code
    assert str(failure).endswith(".") and "Traceback" not in str(failure) and str(error) not in str(failure)


def test_a_failure_of_one_part_does_not_explain_another_kind_of_error() -> None:
    last = LLMError("x (HTTP 401)", status_code=401)
    assert report_failures.failure_of(RunError("an error in the engine"), last, "OpenAI", MODEL).code == "run_failed"


def test_a_file_of_an_older_report_that_can_not_be_removed_does_not_stop_the_run(
    desk: Desk, caplog: pytest.LogCaptureFixture
) -> None:
    from conversation import sanitize_filename

    blocked = desk.places.out_dir / f"analysis_{sanitize_filename(MINA)}.html"
    blocked.mkdir(parents=True)  # a folder stands where the page would be
    done = reports.run(MINA, MODEL, **desk.more())
    assert done.report_path.exists() and blocked.is_dir()
    assert "could not be removed" in caplog.text


def test_a_quote_of_a_message_that_is_now_in_another_conversation_opens_the_day_only(desk: Desk) -> None:
    done = reports.run(MINA, MODEL, **desk.more())
    elsewhere = desk.archive.messages(THEO, limit=1)["messages"][0]["id"]
    first = done.report.moments[0]
    moved = first.model_copy(update={"quote_message_id": elsewhere, "message_ids": (elsewhere,)})
    shaped = reports.moments_for(done.report.model_copy(update={"moments": (moved,)}), desk.archive)[0]
    assert (shaped["chatId"], shaped["day"], shaped["at"], shaped["pickedId"]) == (MINA, first.date, None, "")


def test_the_teller_says_when_a_run_waits_and_when_a_step_fails() -> None:
    told: list[dict[str, Any]] = []
    teller = Teller(told.append)
    teller.engine_event(ProgressEvent(step="map", state="waiting", period="Jan 2023", index=1, total=2))
    teller.engine_event(ProgressEvent(step="reduce", state="waiting", total=2))
    teller.engine_event(ProgressEvent(step="reduce", state="failed", total=2))
    teller.engine_event(ProgressEvent(step="map", state="done", period="Jan 2023", index=1, total=2))
    assert told == [{"log": WAITING}, {"log": WAITING}, {"log": "The report could not be written"}]
    Teller(None).engine_event(ProgressEvent(step="prepare", state="start"))
