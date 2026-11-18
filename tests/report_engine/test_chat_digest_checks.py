"""What a digest claims is checked as a report checks it, and no private text leaves it."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from report_engine.cache import PeriodCache
from report_engine.chat_digest import ConversationDigest, MomentNote
from report_engine.errors import AllPeriodsFailed, SourceError
from report_engine.fake_llm import FakeCall, FakeLLM
from report_engine.ports import SourceMessage

from .chat_digest_support import Talks, digest, tagged
from .support import LARGE, SMALL, run
from .synth import MINA, THEO, message

OTHER_ID = "other-talk-7"
OTHER_TEXT = "the harbour key is under the blue flowerpot"
OTHER = [message(OTHER_ID, "2024-06-01 09:30:00", OTHER_TEXT, sender=THEO)]
OWN = tagged(SMALL)
TEXT_OF = {item["id"]: item["text"] for item in OWN}
SUMMARY = "The summary stays."
NEVER_WRITTEN = "words that nobody ever wrote in this talk"


def claim(title: str, day: str, quote: str, refs: list[str]) -> dict[str, Any]:
    return {
        "date": day,
        "title": title,
        "detail": "A detail.",
        "category": "milestone",
        "score": 0.9,
        "quote": quote,
        "refs": refs,
    }


def hostile(call: FakeCall) -> Any:
    """A reply that claims more than the conversation holds."""
    if call.kind != "map":
        return None
    first, second = call.lines[0], call.lines[1]
    return {
        "summary": SUMMARY,
        "carry_over": "",
        "moments": [
            claim("Another talk", first.day, OTHER_TEXT, [OTHER_ID]),
            claim("Made up", first.day, "", ["no-such-guid", "m99999", OWN[2]["id"]]),
            claim("Wrong words", first.day, NEVER_WRITTEN, [first.ref]),
            claim("Honest", second.day, second.text, [second.ref]),
            claim("Wrong day", "2019-01-01", "", []),
        ],
    }


def moments_of(llm: FakeLLM) -> dict[str, MomentNote]:
    source = Talks({MINA: OWN, THEO: OTHER})
    (period,) = digest((), source=source, llm=llm).periods
    assert period.summary == SUMMARY
    return {moment.title: moment for moment in period.moments}


def test_a_reference_to_a_message_of_another_conversation_is_dropped() -> None:
    found = moments_of(FakeLLM(hostile))["Another talk"]
    assert found.message_ids == () and found.quote == ""


def test_an_id_that_was_made_up_is_dropped() -> None:
    """A reference is a short reference of the period. An id of the archive is not one."""
    found = moments_of(FakeLLM(hostile))["Made up"]
    assert found.message_ids == () and found.quote == ""


def test_a_quote_that_is_not_in_the_message_is_dropped_and_the_reference_stays() -> None:
    found = moments_of(FakeLLM(hostile))["Wrong words"]
    assert found.quote == ""
    assert found.message_ids == (OWN[0]["id"],)


def test_a_true_quote_stays_with_the_id_of_its_message() -> None:
    found = moments_of(FakeLLM(hostile))["Honest"]
    assert found.message_ids == (OWN[1]["id"],)
    assert found.quote == TEXT_OF[OWN[1]["id"]]


def test_a_moment_with_no_day_of_the_period_and_no_reference_is_dropped() -> None:
    assert set(moments_of(FakeLLM(hostile))) == {"Another talk", "Made up", "Wrong words", "Honest"}


def test_every_message_id_is_a_message_of_the_conversation() -> None:
    known = set(TEXT_OF)
    for moment in moments_of(FakeLLM(hostile)).values():
        assert set(moment.message_ids) <= known
        assert OTHER_ID not in moment.message_ids


def test_a_hostile_reply_gives_what_a_report_gives() -> None:
    source = Talks({MINA: OWN, THEO: OTHER})
    report = run((), source=source, llm=FakeLLM(hostile))
    (period,) = digest((), source=source, llm=FakeLLM(hostile)).periods
    assert [
        (moment.date, moment.title, moment.detail, moment.quote, moment.message_ids)
        for moment in report.periods[0].moments
    ] == [
        (moment.date, moment.title, moment.detail, moment.quote, moment.message_ids)
        for moment in period.moments
    ]


def test_a_hostile_reply_that_is_in_the_cache_is_checked_again(cache_path: Path) -> None:
    """The cache holds the reply as the model gave it. The check runs on each read."""
    cache = PeriodCache(cache_path)
    source = Talks({MINA: OWN, THEO: OTHER})
    digest((), source=source, llm=FakeLLM(hostile), cache=cache)
    llm = FakeLLM()
    (period,) = digest((), source=source, llm=llm, cache=cache).periods
    assert llm.calls == []
    found = {moment.title: moment for moment in period.moments}
    assert found["Another talk"].message_ids == () and found["Another talk"].quote == ""
    assert found["Wrong words"].quote == ""


# --- the messages of a moment are messages of its period --------------------------------

DATED = tagged(LARGE)


def on_day(day: str) -> SourceMessage:
    """The first message of a day of the long conversation."""
    return next(item for item in DATED if item["timestamp"].startswith(day))


def one_claim(month: str, day: str, quote: str, cite: str | None) -> Any:
    """A model that gives the period of `month` one moment. It cites the first line of `cite`."""

    def responder(call: FakeCall) -> Any:
        if call.kind != "map" or not call.first_day.startswith(month):
            return None
        refs = [line.ref for line in call.lines if line.day == cite][:1]
        return {"summary": SUMMARY, "moments": [claim("Across", day, quote, refs)]}

    return responder


def test_the_id_of_a_message_outside_the_period_is_not_given_and_its_quote_is_dropped() -> None:
    """A report looks for a quote up to three days from the moment, also in the next period.
    A digest gives only messages of the period of the moment."""
    inside, outside = on_day("2023-01-31"), on_day("2023-02-01")
    responder = one_claim("2023-01", "2023-01-31", outside["text"], "2023-01-31")
    found = digest(DATED, llm=FakeLLM(responder), since="2023-01-01", until="2023-01-31")
    (period,) = found.periods
    (moment,) = period.moments
    assert (period.summary, moment.title, moment.date) == (SUMMARY, "Across", "2023-01-31")
    assert moment.message_ids == (inside["id"],) and moment.quote == ""
    # a report is what it was
    told = run(DATED, llm=FakeLLM(responder)).periods[0].moments[0]
    assert told.message_ids == (inside["id"], outside["id"]) and told.quote == outside["text"]


def test_a_quote_that_only_a_message_before_the_period_holds_is_dropped() -> None:
    outside = on_day("2023-01-31")
    responder = one_claim("2023-02", "2023-02-01", outside["text"], None)
    found = digest(DATED, llm=FakeLLM(responder), since="2023-02-01", until="2023-02-28")
    (period,) = found.periods
    (moment,) = period.moments
    assert (period.label, moment.title, moment.date) == ("Feb 2023", "Across", "2023-02-01")
    assert moment.message_ids == () and moment.quote == ""
    told = run(DATED, llm=FakeLLM(responder)).periods[1].moments[0]
    assert told.message_ids == (outside["id"],) and told.quote == outside["text"]


def test_a_quote_in_a_message_of_the_period_that_was_not_cited_stays() -> None:
    cited, quoted = on_day("2023-01-15"), on_day("2023-01-16")
    responder = one_claim("2023-01", "2023-01-15", quoted["text"], "2023-01-15")
    (period,) = digest(DATED, llm=FakeLLM(responder), until="2023-01-31").periods
    (moment,) = period.moments
    assert moment.message_ids == (cited["id"], quoted["id"]) and moment.quote == quoted["text"]


def test_a_quote_in_a_cited_message_of_the_period_stays() -> None:
    cited = on_day("2023-01-31")
    responder = one_claim("2023-01", "2023-01-31", cited["text"], "2023-01-31")
    (period,) = digest(DATED, llm=FakeLLM(responder), until="2023-01-31").periods
    (moment,) = period.moments
    assert moment.message_ids == (cited["id"],) and moment.quote == cited["text"]


def test_every_message_id_of_a_digest_is_a_message_of_its_period() -> None:
    """Each period gets a moment on its last day that quotes the first message after the period."""
    day_of = {item["id"]: item["timestamp"][:10] for item in DATED}

    def responder(call: FakeCall) -> Any:
        if call.kind != "map":
            return None
        after = [item for item in DATED if item["timestamp"][:10] > call.last_day][:1]
        quote = after[0]["text"] if after else call.lines[-1].text
        return {
            "summary": SUMMARY,
            "moments": [claim("Edge", call.last_day, quote, [call.lines[-1].ref])],
        }

    found = digest(DATED, llm=FakeLLM(responder))
    assert len(found.periods) == 4
    for period in found.periods:
        (moment,) = period.moments
        assert len(moment.message_ids) == 1
        assert all(period.first_day <= day_of[mid] <= period.last_day for mid in moment.message_ids)
    quoted = [bool(period.moments[0].quote) for period in found.periods]
    assert quoted == [False, False, False, True]


# --- no private text in a log line or an error -------------------------------------

MARK = "zebramark"
NAME = "Quillon Zebraname"
HANDLE = "+12025550143"
MARKS = (MARK, "Quillon", "Zebraname", HANDLE, "2025550143", "555-0143")


def marked_messages() -> list[SourceMessage]:
    return [
        {**item, "text": f"{MARK} {item['text']}"}  # type: ignore[misc]
        for item in LARGE
    ]


def marked_labeler(_sender: str) -> str:
    return NAME


def marked_digest(**more: Any) -> ConversationDigest:
    """A digest of a conversation whose id is a handle and whose every line and name is marked."""
    values = {"conversation_id": HANDLE, "labeler": marked_labeler, **more}
    return digest(marked_messages(), **values)


class Listening:
    """Every log record of a block, at every level, as the text a handler writes."""

    def __init__(self, caplog: pytest.LogCaptureFixture) -> None:
        self._caplog = caplog
        caplog.set_level(logging.DEBUG)

    def text(self) -> str:
        lines = [record.getMessage() for record in self._caplog.records]
        return "\n".join([*lines, self._caplog.text])


def assert_clean(*texts: str) -> None:
    for text in texts:
        assert not [mark for mark in MARKS if mark in text]


def test_a_run_in_which_every_period_fails_says_no_word_of_a_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    heard = Listening(caplog)

    def responder(call: FakeCall) -> Any:
        raise RuntimeError(f"the reply was about: {call.text}")

    with pytest.raises(AllPeriodsFailed) as caught:
        marked_digest(llm=FakeLLM(responder))
    assert_clean(str(caught.value), repr(caught.value), heard.text())
    assert "RuntimeError" in str(caught.value)


def test_a_run_with_a_failed_period_and_hostile_replies_logs_no_word_of_a_message(
    caplog: pytest.LogCaptureFixture, cache_path: Path
) -> None:
    heard = Listening(caplog)

    def responder(call: FakeCall) -> Any:
        if call.first_day.startswith("2023-02"):
            raise RuntimeError(call.text)
        line = call.lines[0]
        return {
            "summary": f"A summary with {MARK} and {NAME}.",
            "carry_over": f"{NAME} wrote {MARK}.",
            "moments": [
                claim(f"{NAME} said {MARK}", line.day, f"{MARK} {NEVER_WRITTEN}", [line.ref]),
                claim(f"{MARK} again", "1999-01-01", line.text, ["m99999"]),
            ],
        }

    cache = PeriodCache(cache_path)
    found = marked_digest(llm=FakeLLM(responder), cache=cache)
    assert found.periods_failed == 1 and MARK in found.periods[0].summary
    # a second run reads the cache and runs the failed period again
    marked_digest(cache=cache)
    assert_clean(heard.text())
    assert heard.text() != ""


class BrokenSource:
    def messages(self, conversation: str, since: str | None, until: str | None) -> Iterator[Any]:
        raise ValueError(f"row of {conversation} holds {MARK} from {NAME}")


def test_a_source_or_a_labeler_that_fails_is_named_by_its_type_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    heard = Listening(caplog)

    def broken_labeler(sender: str) -> str:
        raise KeyError(f"{sender} {NAME} {MARK}")

    with pytest.raises(SourceError) as of_source:
        digest((), source=BrokenSource(), conversation_id=HANDLE)
    with pytest.raises(SourceError) as of_labeler:
        marked_digest(labeler=broken_labeler)
    assert_clean(str(of_source.value), str(of_labeler.value), heard.text())
    assert "ValueError" in str(of_source.value)


def test_a_cache_entry_that_does_not_parse_is_logged_by_the_label_of_its_period(
    caplog: pytest.LogCaptureFixture, cache_path: Path
) -> None:
    cache = PeriodCache(cache_path)
    marked_digest(cache=cache)
    with closing(sqlite3.connect(cache_path)) as conn, conn:
        broken = json.dumps({"summary": "", "note": MARK})
        conn.execute("UPDATE period_results SET payload = ?", (broken,))
    heard = Listening(caplog)
    llm = FakeLLM()
    found = marked_digest(cache=cache, llm=llm)
    assert len(llm.map_calls) == 4 and found.periods_built == 4
    assert "Jan 2023" in heard.text()
    assert_clean(heard.text())
