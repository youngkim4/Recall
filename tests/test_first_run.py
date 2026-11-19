"""The first run of a new person: GET /api/first-run, and the `first_run` job that gets chat ready.

The archive is an invented one. The model and the embedder are the fakes of
the report engine and of the index of meanings: no request leaves this
machine. The least number of messages of the memory is left at its value of
the app (200): the first run reads every conversation all the same.
"""
import json
import threading
import time
from datetime import timedelta
from urllib.parse import quote

import pytest

import ui_server
from meaning_index import FakeEmbedder
from recall_services import first_run, key_store, keys, meaning, memory_prep, people, places
from recall_services import reports as report_service
from report_engine.fake_llm import FakeCall, FakeLLM
from tests.archive.builders import ChatDbBuilder
from tests.archive_kit import (
    data_folder,  # noqa: F401
    emptied,
    ended,
    jobs_end,  # noqa: F401
    listed,
    no_names,  # noqa: F401
    own_archives,  # noqa: F401
    pulled,
    run,
    said,
)
from tests.loopback import fetch, post
from tests.recall_services.state_data import OTHER, PERSON, START

pytestmark = pytest.mark.usefixtures(
    "fresh_bundle_cache", "no_names", "own_archives", "no_jobs", "jobs_end", "data_folder", "messages_home"
)

MARK = "kingfisher"
FIRST = [f"the {MARK} dove at noon {number}" for number in range(20)]
INVENTED = "sk-invented-" + "Q7x" * 10
ANSWER_KEYS = {"key", "archive", "steps", "estimatedCost", "ready", "jobId"}
STEP_KEYS = {"name", "state", "estimatedCost", "error", "code"}
RESULT_KEYS = {"meaning", "memory", "people", "skipped", "usd"}
FIRST_RUN_RUNS = "Recall is getting chat ready. Wait until it ends."
HELD_UP = {
    "setup_import": "An import is running. Wait until it ends, then get chat ready.",
    "export": "An import is running. Wait until it ends, then get chat ready.",
    "bridge": "An import is running. Wait until it ends, then get chat ready.",
    "meaning": "The index for the search by meaning is being built. Wait until it ends.",
    "memory": "Recall is reading the memory of chat. Wait until it ends.",
    "digest": "Recall is reading a conversation for chat. Wait until it ends.",
    "people": "The people index is being built. Wait until it ends.",
}


def reply(call: FakeCall):
    if call.kind == "map":
        return None
    return {"kind": "friend", "status": "ongoing", "summary": "You and this person talk about birds.", "turning_points": []}


@pytest.fixture
def model(monkeypatch):
    """The fake model answers the parts of the memory and the profiles; the fake embedder embeds."""
    fake = FakeLLM(reply)
    embedded = []

    def embedder(name, dimensions, **_more):
        embedded.append(name)
        return FakeEmbedder(dimensions, name)

    monkeypatch.setattr(report_service, "default_llm", lambda _info: fake)
    monkeypatch.setattr(meaning, "embedder_for", embedder)
    monkeypatch.setattr(people, "_count_tokens", lambda: lambda text: len(text.split()))
    fake.embedded = embedded
    return fake


@pytest.fixture
def no_calls(monkeypatch):
    """Ports that raise when they are called: an estimate makes no model call and no request."""

    def refused(*_args, **_more):
        raise AssertionError("an estimate called a model")

    monkeypatch.setattr(report_service, "default_llm", refused)
    monkeypatch.setattr(meaning, "embedder_for", refused)


@pytest.fixture
def archive_file(data_folder):  # noqa: F811
    return str(pulled(data_folder / "saves" / "archive.sqlite3", FIRST))


def write_two(source, other=2):
    """A Messages database of two conversations: 20 messages with one person, `other` with another.
    The guid follows the text, so a later database with more messages keeps the earlier ones."""
    source.parent.mkdir(parents=True, exist_ok=True)
    source.unlink(missing_ok=True)
    builder = ChatDbBuilder(source)
    for chat, count in ((PERSON, 20), (OTHER, other)):
        handle = builder.handle(chat)
        thread = builder.chat(chat, members=[handle])
        for place in range(count):
            text = f"{MARK} {chat} {place}"
            builder.message(thread, when=START + timedelta(minutes=place), text=text, handle=handle, guid=f"guid-{text}")
    builder.close()
    return source


@pytest.fixture
def two_people(data_folder):  # noqa: F811
    """An archive of two conversations: 20 messages with one person, 2 with another."""
    source = write_two(data_folder / "two-from" / "chat.db")
    archive_file = data_folder / "saves" / "archive.sqlite3"
    assert places.state_for(archive_file).pull_from_messages(source, None).status == "done"
    return str(places.archive_path_for(archive_file))


@pytest.fixture
def held_jobs(monkeypatch):
    """Jobs that start and do not end until they are let go. Each one stays as it was made: queued."""
    let_go = threading.Event()
    monkeypatch.setattr(ui_server, "run_job", lambda job_id, action, payload: let_go.wait(timeout=5))
    yield let_go
    let_go.set()
    with ui_server.JOBS_LOCK:
        ui_server.JOBS.clear()


def first_run_of(server, archive_file):
    status, _headers, body = fetch(server, f"/api/first-run?messagesPath={quote(archive_file)}")
    return status, json.loads(body)


def get_ready(archive_file):
    return run("first_run", {"messagesPath": archive_file})


def steps_of(answer):
    return {step["name"]: step for step in answer["steps"]}


def archive_path(archive_file):
    return places.archive_path_for(archive_file)


def lines(job):
    return [entry["message"] for entry in job["logs"]]


# --- GET /api/first-run ------------------------------------------------------------------


def test_the_answer_has_its_shape_and_sums_the_steps_and_calls_no_model(server, archive_file, no_calls):
    status, answer = first_run_of(server, archive_file)
    assert status == 200 and set(answer) == ANSWER_KEYS
    assert answer["key"] == {"present": True, "source": "environment"}
    assert answer["archive"] == {"holdsMessages": True, "conversations": 1, "messages": 20}
    assert [step["name"] for step in answer["steps"]] == ["meaning", "memory", "people"]
    assert all(set(step) == STEP_KEYS and step["error"] is None for step in answer["steps"])
    assert [step["state"] for step in answer["steps"]] == ["none", "none", "none"]
    assert all(step["estimatedCost"] > 0 for step in answer["steps"])
    assert answer["estimatedCost"] == pytest.approx(sum(step["estimatedCost"] for step in answer["steps"]))
    assert (answer["ready"], answer["jobId"]) == (False, None)


def test_the_memory_step_counts_every_conversation(server, two_people, no_calls):
    _status, answer = first_run_of(server, two_people)
    found = memory_prep.payload(places.state_for(two_people).archive(), ui_server.saved_names(), min_messages=1)
    assert steps_of(answer)["memory"]["estimatedCost"] == pytest.approx(found["estimatedCost"])
    assert found["conversations"] == 2 and answer["archive"]["conversations"] == 2


def test_the_people_bound_is_not_below_the_estimate_or_the_cost_of_the_build(server, two_people, model, monkeypatch):
    _status, before = first_run_of(server, two_people)
    bound = steps_of(before)["people"]["estimatedCost"]
    monkeypatch.setattr(memory_prep, "PREPARE_MIN_MESSAGES", 1)
    assert run("memory", {"messagesPath": two_people})["status"] == "completed"
    _status, after = first_run_of(server, two_people)
    people_step = steps_of(after)["people"]
    found = people.payload(places.state_for(two_people).archive(), ui_server.saved_names())
    assert people_step["estimatedCost"] == pytest.approx(people.TRIES * found["estimatedCost"])
    assert 0 < people_step["estimatedCost"] <= bound
    built = run("people", {"messagesPath": two_people})
    assert built["status"] == "completed" and 0 < built["spent"]["usd"] <= bound


def test_the_notes_of_a_part_are_bounded_by_the_limits_of_a_part_not_by_its_reply():
    from recall_services import first_run
    from recall_services.people_plans import INPUT_CHARS_PER_TOKEN, MOMENT_HEAD_CHARS
    from report_engine import ReportOptions, digest

    info = first_run.models.model_info(people.model())
    most_chars = digest.MAX_SUMMARY_CHARS + digest.MAX_MOMENTS_PER_PERIOD * (
        MOMENT_HEAD_CHARS + digest.MAX_TITLE_CHARS + digest.MAX_DETAIL_CHARS
    )
    roomy = ReportOptions(map_max_output_tokens=10**6)
    assert first_run.notes_tokens_per_part(info, roomy) == -(-most_chars // INPUT_CHARS_PER_TOKEN)
    assert first_run.notes_tokens_per_part(info, ReportOptions(map_max_output_tokens=100)) == 100


def test_after_the_memory_is_read_its_step_is_ready_and_not_counted(server, archive_file, model, monkeypatch):
    monkeypatch.setattr(memory_prep, "PREPARE_MIN_MESSAGES", 1)
    run("memory", {"messagesPath": archive_file})
    _status, answer = first_run_of(server, archive_file)
    assert (steps_of(answer)["memory"]["state"], steps_of(answer)["people"]["state"]) == ("ready", "none")
    waiting = [step["estimatedCost"] for step in answer["steps"] if step["state"] != "ready"]
    assert answer["estimatedCost"] == pytest.approx(sum(waiting)) and answer["ready"] is False


def test_with_no_message_the_steps_are_none_and_cost_nothing(server, data_folder, no_calls):  # noqa: F811
    archive_file = str(emptied(data_folder / "saves" / "archive.sqlite3"))
    _status, answer = first_run_of(server, archive_file)
    assert answer["archive"] == {"holdsMessages": False, "conversations": 0, "messages": 0}
    assert [(step["state"], step["estimatedCost"]) for step in answer["steps"]] == [("none", 0.0)] * 3
    assert (answer["estimatedCost"], answer["ready"]) == (0.0, False)


def test_with_no_archive_the_answer_says_there_is_nothing_yet(server, tmp_path, no_calls):
    status, answer = first_run_of(server, str(tmp_path / "none.sqlite3"))
    assert status == 200 and answer["archive"] == {"holdsMessages": False, "conversations": 0, "messages": 0}
    assert (answer["estimatedCost"], answer["ready"]) == (0.0, False)


def test_the_answer_says_where_the_key_is(server, archive_file, no_calls, monkeypatch):
    monkeypatch.delenv(keys.KEY_NAME)
    assert first_run_of(server, archive_file)[1]["key"] == {"present": False, "source": None}
    key_store.save(INVENTED)
    _status, answer = first_run_of(server, archive_file)
    assert answer["key"] == {"present": True, "source": "file"} and INVENTED not in json.dumps(answer)


def test_two_calls_work_the_answer_out_once_and_a_job_moves_it(server, archive_file, model, monkeypatch):
    calls = []
    real = first_run.estimate
    monkeypatch.setattr(first_run, "estimate", lambda *args, **more: calls.append(1) or real(*args, **more))
    first = first_run_of(server, archive_file)
    assert first_run_of(server, archive_file) == first and len(calls) == 1
    assert get_ready(archive_file)["status"] == "completed"
    _status, answer = first_run_of(server, archive_file)
    assert answer["ready"] is True and answer["estimatedCost"] == 0.0
    assert [step["state"] for step in answer["steps"]] == ["ready"] * 3


def test_the_answer_names_the_first_run_job_that_runs(server, archive_file, no_calls, held_jobs):
    job = ui_server.create_job("first_run", {"messagesPath": archive_file})
    assert first_run_of(server, archive_file)[1]["jobId"] == job["id"]


# --- the job --------------------------------------------------------------------------------


def test_the_job_runs_the_three_steps_in_order_and_chat_is_ready(two_people, model, monkeypatch):
    progress = []
    real = ui_server.update_job

    def noted(job_id, **fields):
        if "progress" in fields:
            progress.append(fields["progress"])
        real(job_id, **fields)

    monkeypatch.setattr(ui_server, "update_job", noted)
    job = get_ready(two_people)
    assert (job["status"], job["error"], job["code"], job["contact"]) == ("completed", None, None, "")
    assert all(set(one) == {"step", "of", "label", "done", "total"} and one["of"] == 3 for one in progress)
    steps = [one["step"] for one in progress]
    assert steps == sorted(steps) and set(steps) == {1, 2, 3}
    labels = [one["label"] for one in progress]
    for step, label in [(1, "Reading the meaning of your messages"), (2, "Reading your conversations"), (3, "Writing a profile of each person")]:
        assert any(one["step"] == step and one["label"] == label for one in progress)
    assert labels[-1] == "Saving the people index"
    result = job["result"]
    assert set(result) == RESULT_KEYS and result["skipped"] == []
    assert result["memory"]["conversations"] == 2 and result["people"]["built"] == 2
    assert result["meaning"]["segments"] > 0
    assert job["spent"]["usd"] == pytest.approx(result["usd"]) and result["usd"] > 0
    assert result["usd"] >= result["memory"]["usd"] + result["people"]["usd"]
    assert MARK not in said(job) and MARK not in json.dumps(result) and PERSON not in json.dumps(job)


def test_after_the_job_the_automatic_builds_after_an_import_work(two_people, model):
    assert get_ready(two_people)["status"] == "completed"
    place = archive_path(two_people)
    assert meaning.built(place)
    assert memory_prep.completed_before(place)
    assert people.completed_before(place)


def test_a_second_run_skips_the_steps_that_are_ready(archive_file, model):
    get_ready(archive_file)
    calls, embedded = len(model.calls), len(model.embedded)
    job = get_ready(archive_file)
    assert job["status"] == "completed" and job["result"]["skipped"] == ["meaning", "memory", "people"]
    assert (len(model.calls), len(model.embedded)) == (calls, embedded)
    assert [line for line in lines(job) if line in first_run.SKIPPED.values()] == [
        first_run.SKIPPED["meaning"], first_run.SKIPPED["memory"], first_run.SKIPPED["people"]
    ]
    assert job["spent"] == {"usd": 0.0, "usage": None}


def test_a_memory_that_is_read_already_gets_its_marker(archive_file, model, monkeypatch):
    monkeypatch.setattr(memory_prep, "PREPARE_MIN_MESSAGES", 1)
    run("memory", {"messagesPath": archive_file})
    memory_prep.marker_path(archive_path(archive_file)).unlink()
    job = get_ready(archive_file)
    assert job["status"] == "completed" and "memory" in job["result"]["skipped"]
    assert memory_prep.completed_before(archive_path(archive_file))


def test_a_failure_in_the_memory_step_keeps_the_index_and_tells_its_sentence(archive_file, model, monkeypatch):
    usage = {"input_tokens": 900, "cached_input_tokens": 0, "output_tokens": 100, "cost_usd": 0.0123, "calls": 3}
    real = memory_prep.run

    def paid_then_offline(*_args, **_more):
        raise memory_prep.MemoryFailure("offline", "OpenAI could not be reached.", usage)

    monkeypatch.setattr(memory_prep, "run", paid_then_offline)
    job = get_ready(archive_file)
    assert (job["status"], job["code"], job["error"]) == ("failed", "offline", "OpenAI could not be reached.")
    assert meaning.built(archive_path(archive_file)) and not memory_prep.completed_before(archive_path(archive_file))
    assert job["spent"]["usd"] > 0.0123 and job["spent"]["usage"]["calls"] == 3
    monkeypatch.setattr(memory_prep, "run", real)
    again = get_ready(archive_file)
    assert again["status"] == "completed" and again["result"]["skipped"] == ["meaning"]
    assert first_run.SKIPPED["meaning"] in lines(again)


def test_the_job_holds_counts_and_money_and_never_the_key(archive_file, model, monkeypatch):
    monkeypatch.delenv(keys.KEY_NAME)
    key_store.save(INVENTED)
    job = get_ready(archive_file)
    assert job["status"] == "completed" and INVENTED not in json.dumps(job) and INVENTED[-8:] not in json.dumps(job)


def test_a_place_with_no_message_is_not_got_ready(data_folder, model):  # noqa: F811
    job = get_ready(str(emptied(data_folder / "saves" / "archive.sqlite3")))
    assert (job["status"], job["code"], job["error"]) == ("failed", "no_messages", first_run.NO_MESSAGES)


# --- refusals ---------------------------------------------------------------------------------


def test_with_no_key_the_job_is_refused_before_it_is_made(server, archive_file, monkeypatch):
    monkeypatch.delenv(keys.KEY_NAME)
    answer = post(server, "/api/jobs", {"action": "first_run", "messagesPath": archive_file})
    assert answer == (409, {"error": first_run.NO_KEY}) and listed() == []
    assert first_run.NO_KEY == (
        "No API key is set. Add OPENAI_API_KEY to the file .env in Recall's data folder, then get chat ready again."
    )


def test_the_job_is_asked_for_as_every_job_is(server, archive_file, held_jobs):
    status, answer = post(server, "/api/jobs", {"action": "first_run", "messagesPath": archive_file})
    assert (status, answer["job"]["action"], answer["job"]["status"]) == (202, "first_run", "queued")


@pytest.mark.parametrize("other", sorted(HELD_UP))
def test_a_job_that_runs_holds_up_the_first_run(held_jobs, other):
    ui_server.create_job(other, {"conversation": PERSON} if other == "digest" else {})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job("first_run", {})
    assert (str(refused.value), refused.value.status) == (HELD_UP[other], 409)


@pytest.mark.parametrize("other", [*sorted(HELD_UP), "first_run"])
def test_the_first_run_holds_up_each_of_them(held_jobs, other):
    ui_server.create_job("first_run", {})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job(other, {"conversation": PERSON} if other == "digest" else {})
    assert (str(refused.value), refused.value.status) == (FIRST_RUN_RUNS, 409)
    assert listed() == [("first_run", "queued")]


def test_a_report_runs_beside_the_first_run(held_jobs):
    ui_server.create_job("first_run", {})
    ui_server.create_job("analyze", {"contact": PERSON})
    assert listed() == [("first_run", "queued"), ("analyze", "queued")]


# --- after a first run, the builds after an import keep chat ready ------------------------------


def jobs_of(action):
    with ui_server.JOBS_LOCK:
        return [ui_server.job_copy(job) for job in ui_server.JOBS.values() if job["action"] == action]


def one_ended(action, wait=10.0):
    """The one job of `action` that the server started by itself, once it has ended."""
    until = time.monotonic() + wait
    while not jobs_of(action) and time.monotonic() < until:
        time.sleep(0.01)
    (job,) = jobs_of(action)
    return ended(job["id"], wait)


def test_an_import_that_adds_to_a_small_conversation_is_read_and_chat_stays_ready(
    server, two_people, model, messages_home
):
    """The memory of a first run holds every conversation: the reading after an import keeps its threshold."""
    assert get_ready(two_people)["status"] == "completed"
    assert memory_prep.marker_threshold(archive_path(two_people)) == first_run.MEMORY_MIN_MESSAGES
    write_two(messages_home / "chat.db", other=6)
    assert run("export", {"messagesPath": two_people})["status"] == "completed"
    memory_job = one_ended("memory")
    assert memory_job["status"] == "completed" and memory_job["result"]["periodsBuilt"] >= 1
    assert one_ended("meaning")["status"] == "completed"
    people_job = one_ended("people")
    assert (people_job["status"], people_job["result"]["built"]) == ("completed", 1)
    _status, answer = first_run_of(server, two_people)
    assert [step["state"] for step in answer["steps"]] == ["ready"] * 3 and answer["ready"] is True
    assert memory_prep.marker_threshold(archive_path(two_people)) == first_run.MEMORY_MIN_MESSAGES


def test_a_memory_job_writes_the_threshold_it_ran_with(archive_file, model, monkeypatch):
    monkeypatch.setattr(memory_prep, "PREPARE_MIN_MESSAGES", 7)
    assert run("memory", {"messagesPath": archive_file})["status"] == "completed"
    assert memory_prep.marker_threshold(archive_path(archive_file)) == 7
    assert run("memory", {"messagesPath": archive_file, "minMessages": 1})["status"] == "completed"
    assert memory_prep.marker_threshold(archive_path(archive_file)) == 1


def test_a_memory_job_after_a_first_run_keeps_the_lower_threshold(archive_file, model):
    """A reading of the memory with the threshold of the app does not take back what a first run read."""
    get_ready(archive_file)
    assert run("memory", {"messagesPath": archive_file})["status"] == "completed"
    assert memory_prep.marker_threshold(archive_path(archive_file)) == first_run.MEMORY_MIN_MESSAGES


def test_an_old_marker_with_no_threshold_reads_with_the_threshold_of_the_app(archive_file):
    marker = memory_prep.marker_path(archive_path(archive_file))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"completedAt": "2026-01-01T00:00:00+00:00"}), encoding="utf-8")
    assert memory_prep.completed_before(archive_path(archive_file))
    assert memory_prep.marker_threshold(archive_path(archive_file)) is None


@pytest.mark.parametrize("value", ["1", 0, -3, 1.5, True])
def test_a_memory_job_refuses_a_threshold_that_is_no_count(server, archive_file, value):
    status, answer = post(server, "/api/jobs", {"action": "memory", "messagesPath": archive_file, "minMessages": value})
    assert status == 400 and answer["error"] == "minMessages must be a whole number, 1 or more."
    assert listed() == []
