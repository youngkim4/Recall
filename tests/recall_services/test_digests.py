"""digests.build and digests.estimate: the digest of a conversation for chat, wired as a report is."""

from __future__ import annotations

import dataclasses
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from recall_services import digests, models, reports
from recall_services.names import Names
from report_engine import CancelToken, PeriodCache, ProgressEvent
from report_engine.archive_source import ArchiveSource
from report_engine.chat_digest import ConversationDigest, DigestEstimate, estimate_digest
from report_engine.fake_llm import FakeLLM
from tests.recall_services.report_world import MINA_MESSAGES, MODEL, OPTIONS, PERIOD_LABELS, Desk
from tests.report_engine.synth import MINA, THEO, WordCounter

OTHER_MODEL = "gpt-6.1-terra"
ROW_COLUMNS = "key, conversation, label, model, prompt_version, payload"


def cache_rows(path: Path) -> list[tuple[Any, ...]]:
    with closing(sqlite3.connect(path)) as conn:
        return list(conn.execute(f"SELECT {ROW_COLUMNS} FROM period_results ORDER BY key"))


def estimate_more(desk: Desk) -> dict[str, Any]:
    return {"model": MODEL, **desk.estimate_more()}


def moments_of(found: ConversationDigest) -> list[Any]:
    return [moment for period in found.periods for moment in period.moments]


# --- a digest ------------------------------------------------------------------------


def test_a_digest_gives_the_periods_of_the_conversation_and_writes_no_report(desk: Desk) -> None:
    model = FakeLLM()
    found = digests.build(MINA, model=MODEL, **desk.more(model))
    assert isinstance(found, ConversationDigest) and found.conversation_id == MINA
    assert [period.label for period in found.periods] == PERIOD_LABELS
    assert sum(period.messages for period in found.periods) == MINA_MESSAGES
    assert all(period.summary and len(period.moments) == 1 for period in found.periods)
    assert (found.periods_built, found.periods_cached, found.periods_failed) == (4, 0, 0)
    assert len(model.map_calls) == 4 and model.reduce_calls == []
    assert found.usage.calls == 4 and found.usage.cost_usd > 0
    assert not desk.places.out_dir.exists() and not desk.places.store_dir.exists()
    assert desk.places.cache_path.exists()


def test_the_message_ids_of_a_moment_are_messages_of_the_conversation_in_the_archive(desk: Desk) -> None:
    found = digests.build(MINA, model=MODEL, **desk.more())
    moments = moments_of(found)
    assert len(moments) == 4
    for moment in moments:
        assert moment.message_ids and moment.quote
        held = [desk.archive.message(mid) for mid in moment.message_ids]
        assert all(message is not None and message["conversation"] == MINA for message in held)
        assert any(moment.quote in message["text"] for message in held)
        assert moment.date in {message["timestamp"][:10] for message in held}


def test_a_reply_that_cites_another_conversation_or_words_nobody_wrote_gives_no_id_and_no_quote(
    desk: Desk,
) -> None:
    """The summary stays. What the archive does not hold in this conversation is dropped."""
    own = {message["id"]: message["text"] for message in desk.archive.iter_messages(MINA)}
    # the invented talks share words: take a line of the other talk that no line of this one holds
    theirs = next(
        message
        for message in desk.archive.iter_messages(THEO)
        if not any(message["text"] in text for text in own.values())
    )
    assert theirs["id"] not in own

    def responder(call: Any) -> Any:
        first, second = call.lines[0], call.lines[1]
        claim = {"date": first.day, "detail": "A detail.", "category": "milestone", "score": 0.9}
        return {
            "summary": "The summary stays.",
            "moments": [
                {**claim, "title": "Theirs", "quote": theirs["text"], "refs": [theirs["id"]]},
                {**claim, "title": "Made up", "quote": "", "refs": ["no-such-guid", "m99999"]},
                {**claim, "title": "Wrong words", "quote": "nobody wrote these words here", "refs": [first.ref]},
                {**claim, "title": "Honest", "quote": second.text, "refs": [second.ref]},
            ],
        }

    found = digests.build(MINA, "2023-01-01", "2023-01-31", model=MODEL, **desk.more(FakeLLM(responder)))
    (period,) = found.periods
    assert period.summary == "The summary stays."
    by_title = {moment.title: moment for moment in period.moments}
    assert (by_title["Theirs"].message_ids, by_title["Theirs"].quote) == ((), "")
    assert (by_title["Made up"].message_ids, by_title["Made up"].quote) == ((), "")
    assert by_title["Wrong words"].quote == "" and len(by_title["Wrong words"].message_ids) == 1
    (honest,) = by_title["Honest"].message_ids
    assert by_title["Honest"].quote in own[honest]
    assert all(mid in own for moment in period.moments for mid in moment.message_ids)


def test_the_model_reads_saved_names_and_no_handle(desk: Desk) -> None:
    model = FakeLLM()
    digests.build(MINA, model=MODEL, **desk.more(model))
    sent = "\n".join(call.text for call in model.calls)
    assert "Mina" in sent and MINA not in sent and "2025550143" not in sent
    unsaved = FakeLLM()
    digests.build(THEO, model=MODEL, **desk.more(unsaved))
    sent = "\n".join(call.text for call in unsaved.calls)
    assert "Unsaved contact" in sent and THEO not in sent and "an unsaved" not in sent


# --- one cache for reports and digests, in the app ---------------------------------------


def test_a_report_first_makes_the_digest_cost_no_call(desk: Desk) -> None:
    done = reports.run(MINA, MODEL, **desk.more())
    rows = cache_rows(desk.places.cache_path)
    model = FakeLLM()
    found = digests.build(MINA, model=MODEL, **desk.more(model))
    assert model.calls == []
    assert (found.periods_built, found.periods_cached) == (0, 4)
    assert cache_rows(desk.places.cache_path) == rows
    assert [period.summary for period in found.periods] == [period.summary for period in done.report.periods]
    assert [[moment.message_ids for moment in period.moments] for period in found.periods] == [
        [moment.message_ids for moment in period.moments] for period in done.report.periods
    ]


def test_a_digest_first_makes_the_report_read_no_part_again(desk: Desk) -> None:
    digests.build(MINA, model=MODEL, **desk.more())
    rows = cache_rows(desk.places.cache_path)
    model = FakeLLM()
    done = reports.run(MINA, MODEL, **desk.more(model))
    assert model.map_calls == [] and len(model.reduce_calls) == 1
    assert all(period.from_cache for period in done.report.periods)
    assert {period.cache_key for period in done.report.periods} == {row[0] for row in rows}
    assert cache_rows(desk.places.cache_path) == rows


def test_a_digest_and_a_report_write_the_same_cache_entries(desk: Desk, tmp_path: Path) -> None:
    other = dataclasses.replace(desk.places, cache_path=tmp_path / "other" / "cache.sqlite3")
    reports.run(MINA, MODEL, **desk.more())
    digests.build(MINA, model=MODEL, **{**desk.more(), "places": other})
    assert cache_rows(other.cache_path) == cache_rows(desk.places.cache_path)
    assert len(cache_rows(other.cache_path)) == 4


def test_a_handle_typed_into_a_message_is_masked_as_a_report_masks_it(tmp_path: Path) -> None:
    """The digest reads the text that a report reads: the same mask, so the same cache keys."""
    from tests.integration.report_data import HANDLE_PARTS, MINA_CONTACT, Pulled
    from tests.report_engine.synth import RAVI, conversation, message

    pulled = Pulled(tmp_path / "world")
    typed = message("typed", "2024-06-02 12:00:00", f"Call {THEO} or (202) 555-0171 when you arrive at the pier")
    pulled.write(MINA, [*conversation("2024-06-01", days=3, per_day=4), typed])
    pulled.write(THEO, conversation("2024-06-01", days=2, per_day=2, others=(THEO,)))
    pulled.write(RAVI, conversation("2024-06-01", days=2, per_day=2, others=(RAVI,)))
    pulled.pull([MINA_CONTACT])
    places = reports.ReportPlaces(tmp_path / "out", tmp_path / "saves" / "reports", tmp_path / "saves" / "cache.sqlite3")
    first, second = FakeLLM(), FakeLLM()
    try:
        with pulled.open() as archive:
            more = {
                "archive": archive,
                "names": Names.from_records([MINA_CONTACT]),
                "places": places,
                "options": OPTIONS,
                "counter": WordCounter(),
            }
            found = digests.build(MINA, model=MODEL, llm_factory=lambda _info: first, **more)
            planned = digests.estimate(MINA, model=MODEL, **more)
            reports.run(MINA, MODEL, llm_factory=lambda _info: second, **more)
    finally:
        pulled.close()
    sent = "\n".join(call.text for call in first.calls)
    assert "when you arrive at the pier" in sent
    unsaved = [part for part in HANDLE_PARTS if "0171" in part or part == THEO]
    assert unsaved and not [part for part in unsaved if part in sent]
    assert found.periods_built == planned.periods == planned.periods_cached >= 1
    assert second.map_calls == [] and len(second.reduce_calls) == 1


# --- the model ---------------------------------------------------------------------------


def test_with_no_model_named_the_digest_uses_the_default_model_as_a_report_does(desk: Desk) -> None:
    assert models.default_id() == MODEL
    reports.run(MINA, "", **desk.more())
    model = FakeLLM()
    found = digests.build(MINA, **desk.more(model))
    assert model.calls == [] and found.periods_cached == 4
    assert digests.estimate(MINA, **desk.estimate_more()).periods_cached == 4


def test_the_cache_is_shared_only_with_reports_of_the_same_model(desk: Desk) -> None:
    reports.run(MINA, MODEL, **desk.more())
    model = FakeLLM()
    found = digests.build(MINA, model=OTHER_MODEL, **desk.more(model))
    assert len(model.map_calls) == 4 and found.periods_built == 4
    assert digests.estimate(MINA, model=OTHER_MODEL, **desk.estimate_more()).periods_cached == 4
    assert {row[3] for row in cache_rows(desk.places.cache_path)} == {MODEL, OTHER_MODEL}


def test_a_model_that_can_not_be_run_is_refused_before_any_call(desk: Desk) -> None:
    model = FakeLLM()
    with pytest.raises(models.UnknownModelIdError):
        digests.build(MINA, model="gpt-7", **desk.more(model))
    with pytest.raises(models.ModelNotOfferedError):
        digests.build(MINA, model="gpt-6-astra", **desk.more(model))
    with pytest.raises(models.UnknownModelIdError):
        digests.estimate(MINA, model="gpt-7", **desk.estimate_more())
    assert model.calls == []


def test_the_port_of_the_app_is_the_port_of_a_report(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no factory the digest asks `reports.default_llm`, which reads the key and pins the address."""
    asked: list[str] = []
    model = FakeLLM()

    def port(info: Any) -> Any:
        asked.append(info.id)
        return model

    monkeypatch.setattr(reports, "default_llm", port)
    more = {name: value for name, value in desk.more().items() if name != "llm_factory"}
    digests.build(MINA, model=MODEL, **more)
    assert asked == [MODEL] and len(model.map_calls) == 4


# --- the dates -----------------------------------------------------------------------------


def test_the_dates_select_the_parts_that_overlap_them(desk: Desk) -> None:
    model = FakeLLM()
    found = digests.build(MINA, "2023-02-10", "2023-03-05", model=MODEL, **desk.more(model))
    assert [period.label for period in found.periods] == ["Feb 2023", "Mar 2023"]
    # a part is read whole, as a report on all dates reads it
    assert [(call.first_day, call.last_day) for call in model.map_calls] == [
        ("2023-02-01", "2023-02-28"),
        ("2023-03-01", "2023-03-31"),
    ]
    again = FakeLLM()
    reports.run(MINA, MODEL, **desk.more(again))
    assert [call.first_day[:7] for call in again.map_calls] == ["2023-01", "2023-04"]


@pytest.mark.parametrize(
    ("since", "until", "said"),
    [
        ("2023-02-30", None, "Not a day"),
        (None, "last week", "Not a day"),
        ("2023-03-01", "2023-02-01", "The first day is after the last day."),
    ],
)
def test_dates_that_are_no_dates_are_refused(desk: Desk, since: Any, until: Any, said: str) -> None:
    model = FakeLLM()
    with pytest.raises(ValueError, match=said):
        digests.build(MINA, since, until, model=MODEL, **desk.more(model))
    with pytest.raises(ValueError, match=said):
        digests.estimate(MINA, since, until, **estimate_more(desk))
    assert model.calls == []


@pytest.mark.parametrize("conversation", ["", None, "   "])
def test_a_conversation_is_needed(desk: Desk, conversation: Any) -> None:
    with pytest.raises(ValueError, match="A conversation is needed"):
        digests.build(conversation, model=MODEL, **desk.more())
    with pytest.raises(ValueError, match="A conversation is needed"):
        digests.estimate(conversation, **estimate_more(desk))


# --- the estimate ------------------------------------------------------------------------


def test_the_estimate_has_the_numbers_of_the_engine(desk: Desk) -> None:
    found = digests.estimate(MINA, **estimate_more(desk))
    engine = estimate_digest(
        source=ArchiveSource(desk.archive),
        conversation=MINA,
        model_info=models.model_info(MODEL),
        labeler=desk.names.report_labeler(desk.archive.conversation(MINA)),
        cache=None,
        options=OPTIONS,
        counter=WordCounter(),
    )
    assert isinstance(found, DigestEstimate) and found == engine
    assert (found.periods, found.periods_cached) == (4, 0)
    assert found.input_tokens > 0 and found.output_tokens == 4 * 500 and found.usd > 0


def test_the_estimate_is_the_map_part_of_the_estimate_of_a_report(desk: Desk) -> None:
    found = digests.estimate(MINA, **estimate_more(desk))
    report = reports.estimate(MINA, MODEL, **desk.estimate_more())
    assert (found.periods, found.periods_cached) == (report["periods"], report["periods_cached"])
    assert 0 < found.input_tokens < report["input_tokens"]
    assert 0 < found.usd < report["estimated_cost"]


def test_the_estimate_follows_what_was_read(desk: Desk) -> None:
    before = digests.estimate(MINA, **estimate_more(desk))
    assert not desk.places.cache_path.exists()
    reports.run(MINA, MODEL, "2023-01-01", "2023-01-31", **desk.more())
    one = digests.estimate(MINA, **estimate_more(desk))
    assert (one.periods, one.periods_cached) == (4, 1) and one.usd < before.usd
    digests.build(MINA, model=MODEL, **desk.more())
    after = digests.estimate(MINA, **estimate_more(desk))
    assert after == DigestEstimate(periods=4, periods_cached=4, input_tokens=0, output_tokens=0, usd=0.0)


def test_the_estimate_of_dates(desk: Desk) -> None:
    whole = digests.estimate(MINA, **estimate_more(desk))
    part = digests.estimate(MINA, "2023-02-10", "2023-03-05", **estimate_more(desk))
    none = digests.estimate(MINA, "2020-01-01", "2020-12-31", **estimate_more(desk))
    assert (part.periods, none.periods) == (2, 0)
    assert 0 < part.input_tokens < whole.input_tokens
    assert none == DigestEstimate(periods=0, periods_cached=0, input_tokens=0, output_tokens=0, usd=0.0)


def test_the_estimate_says_how_many_calls_the_digest_makes(desk: Desk) -> None:
    reports.run(MINA, MODEL, "2023-01-01", "2023-01-31", **desk.more())
    planned = digests.estimate(MINA, None, "2023-03-15", **estimate_more(desk))
    model = FakeLLM()
    found = digests.build(MINA, None, "2023-03-15", model=MODEL, **desk.more(model))
    assert planned.periods - planned.periods_cached == found.periods_built == len(model.calls) == 2
    assert planned.periods_cached == found.periods_cached == 1


# --- progress and cancel -----------------------------------------------------------------


def test_progress_is_the_events_of_the_engine(desk: Desk) -> None:
    events: list[ProgressEvent] = []
    digests.build(MINA, on_progress=events.append, model=MODEL, **desk.more())
    assert all(isinstance(event, ProgressEvent) for event in events)
    assert events[0] == ProgressEvent(step="prepare", state="start")
    assert [(event.period, event.index, event.total) for event in events if event.state == "start"][1:] == [
        (label, index, 4) for index, label in enumerate(PERIOD_LABELS, start=1)
    ]
    assert events[-1] == ProgressEvent(step="done", state="done", total=4)


def test_a_progress_listener_that_fails_does_not_stop_the_digest(desk: Desk) -> None:
    def broken(_event: ProgressEvent) -> None:
        raise RuntimeError("the listener is gone")

    assert digests.build(MINA, on_progress=broken, model=MODEL, **desk.more()).periods_built == 4


def test_a_stop_between_two_parts_keeps_what_was_read(desk: Desk) -> None:
    token = CancelToken()

    def responder(call: Any) -> Any:
        if call.first_day.startswith("2023-02"):
            token.cancel()
        return None

    model = FakeLLM(responder)
    with pytest.raises(digests.ReportFailure) as caught:
        digests.build(MINA, cancel=token, model=MODEL, **desk.more(model))
    assert caught.value.code == "cancelled"
    assert str(caught.value) == "The report was stopped. What was read is kept for the next run."
    assert caught.value.usage is not None and caught.value.usage["calls"] == 2
    assert len(model.calls) == 2 and PeriodCache(desk.places.cache_path).count() == 2
    resumed = FakeLLM()
    found = digests.build(MINA, model=MODEL, **desk.more(resumed))
    assert [call.first_day[:7] for call in resumed.map_calls] == ["2023-03", "2023-04"]
    assert (found.periods_built, found.periods_cached) == (2, 2)
