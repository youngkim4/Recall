"""The insights service on the insight world: the answer of the route, the job, and the saved reading."""

from __future__ import annotations

import json
import stat

import pytest

from archive import Archive
from evals.insight_world import NOW, insight_world
from evals.world import build_world, zone
from recall_services import insights
from report_engine.ports import LLMResult, LLMUsage


class FakeLLM:
    """Says one piece of news for each thread, citing its newest message; or fails every call."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def generate(self, messages, *, schema=None, max_output_tokens, reasoning=None) -> LLMResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("down")
        user = messages[-1]["content"]
        last = [line.split(" ", 1)[0] for line in user.splitlines() if line.startswith("m")][-1]
        item = {"kind": "news", "summary": "Something big happened.", "refs": [last], "done": False}
        return LLMResult("{}", {"items": [item]}, LLMUsage(1000, 0, 100), "completed")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    folder = tmp_path_factory.mktemp("insight-service")
    with zone():
        made = build_world(folder, insight_world())
    archive = Archive.open(made.archive_path)
    yield archive, made.names, folder / "saved.json"
    archive.close()


@pytest.fixture
def saved(built):
    path = built[2]
    path.unlink(missing_ok=True)
    yield path
    path.unlink(missing_ok=True)


def test_before_any_reading_the_answer_holds_the_counted_insights(built, saved):
    archive, names, _ = built
    with zone():
        answer = insights.payload(archive, names, now=NOW, path=saved)
    assert answer["state"] == "none"
    assert answer["readAt"] is None
    assert {item["kind"] for item in answer["insights"]} == {"drift", "balance", "unanswered", "birthday"}
    assert answer["toRead"] == 12
    assert 0 < answer["estimatedCost"] < 0.05
    assert [pick["name"] for pick in answer["picks"]][0] == "Iris Calder"


def test_a_reading_is_saved_and_joins_the_answer(built, saved):
    archive, names, _ = built
    fake = FakeLLM()
    told: list[dict] = []
    with zone():
        result = insights.run(archive, names, told.append, now=NOW, llm_factory=lambda _info: fake, path=saved)
        answer = insights.payload(archive, names, now=NOW, path=saved)
    assert result["read"] == 12 and result["failed"] == 0 and result["found"] == 12
    assert result["usd"] > 0
    assert any("spent" in update for update in told)
    assert answer["state"] == "ready"
    assert answer["readAt"] == NOW.isoformat(timespec="seconds")
    assert sum(item["kind"] == "news" for item in answer["insights"]) == 12
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600


def test_the_saved_reading_holds_no_message_text(built, saved):
    archive, names, _ = built
    with zone():
        insights.run(archive, names, now=NOW, llm_factory=lambda _info: FakeLLM(), path=saved)
    body = saved.read_text()
    assert "observatory" not in body and "lisbon" not in body.lower()
    assert json.loads(body)["version"] == insights.VERSION


def test_a_reading_where_every_call_fails_saves_nothing(built, saved):
    archive, names, _ = built
    with zone(), pytest.raises(insights.InsightFailure) as raised:
        insights.run(archive, names, now=NOW, llm_factory=lambda _info: FakeLLM(fail=True), path=saved)
    assert raised.value.code == "run_failed"
    assert "provider" not in str(raised.value).lower()
    assert not saved.exists()


def test_no_key_is_a_failure_with_a_sentence(built, saved, monkeypatch):
    from recall_services import reports

    archive, names, _ = built

    def no_key(_info):
        raise reports.ReportFailure("no_key", reports.NO_KEY)

    with zone(), pytest.raises(insights.InsightFailure) as raised:
        insights.run(archive, names, now=NOW, llm_factory=no_key, path=saved)
    assert raised.value.code == "no_key"


@pytest.mark.parametrize("body", ["not json", '{"version": 99, "items": []}', '{"version": 1, "items": [{"kind": "news"}, 3]}'])
def test_a_broken_saved_file_reads_as_no_reading(built, saved, body):
    archive, names, _ = built
    saved.write_text(body)
    with zone():
        answer = insights.payload(archive, names, now=NOW, path=saved)
    assert all(item["kind"] != "news" for item in answer["insights"])


def test_the_store_file_is_one_for_each_archive(tmp_path):
    assert insights.store_file(tmp_path / "a.sqlite3") != insights.store_file(tmp_path / "b.sqlite3")
    assert insights.store_file(tmp_path / "a.sqlite3").parent.name == insights.STORE_LEAF
