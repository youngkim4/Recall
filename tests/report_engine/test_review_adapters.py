"""Sources, the store and the model adapter: cases found in review."""

import email.utils
import os
import stat
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import openai
import pandas as pd
import pytest

from report_engine.archive_source import ArchiveSource
from report_engine.errors import LLMError, ReportNotFound, SourceError, StoreError
from report_engine.frame_source import FrameSource
from report_engine.openai_llm import OFFICIAL_BASE_URL, OpenAIResponsesLLM, build_client
from report_engine.store import ReportStore, conversation_folder

from .synth import CONVERSATION, MINA
from .test_models import report as make_report
from .test_openai_llm import MESSAGES, REQUEST, Client, adapter, reply


def frame(**columns: Any) -> pd.DataFrame:
    base: dict[str, Any] = {
        "message_id": [1, 2, 3],
        "timestamp": ["2024-06-01 09:00:00", "2024-06-01 09:01:00", "2024-06-02 10:00:00"],
        "sender": [MINA, MINA, MINA],
        "text": ["a", "b", "c"],
        "chat_id": [CONVERSATION] * 3,
        "is_from_me": [0, 1, 0],
    }
    base.update(columns)
    return pd.DataFrame(base)


def ids(
    source: Any, since: str | None = None, until: str | None = None, chat: str = CONVERSATION
) -> list[str]:
    return [row["id"] for row in source.messages(chat, since, until)]


def test_a_frame_with_repeated_index_labels_yields_each_row_once() -> None:
    doubled = pd.concat([frame(), frame(message_id=[4, 5, 6])])
    assert list(doubled.index) == [0, 1, 2, 0, 1, 2]
    assert sorted(ids(FrameSource(doubled))) == ["1", "2", "3", "4", "5", "6"]


def test_a_numeric_chat_id_column_still_matches() -> None:
    numbers = frame(chat_id=[22000.0, 22000.0, float("nan")])
    assert ids(FrameSource(numbers), chat="22000") == ["1", "2"]
    assert ids(FrameSource(frame(chat_id=[22000, 22000, 7])), chat="22000") == ["1", "2"]


def test_timestamps_with_different_offsets_are_read_as_instants_in_local_time() -> None:
    mixed = frame(
        timestamp=[
            "2024-06-01T10:00:00+02:00",
            "2024-06-01T09:00:00+00:00",
            "2024-06-01T08:30:00+00:00",
        ]
    )
    rows = list(FrameSource(mixed).messages(CONVERSATION, None, None))
    assert [row["id"] for row in rows] == ["1", "3", "2"]
    expected = datetime.fromtimestamp(datetime(2024, 6, 1, 8, 0, tzinfo=UTC).timestamp())
    assert rows[0]["timestamp"] == expected.strftime("%Y-%m-%d %H:%M:%S")


def test_a_zone_aware_column_is_converted_to_local_time() -> None:
    stamps = pd.to_datetime(
        ["2024-06-01 16:00:00", "2024-12-01 16:00:00", "2024-06-02 16:00:00"]
    ).tz_localize("UTC")
    rows = list(FrameSource(frame(timestamp=stamps)).messages(CONVERSATION, None, None))
    expected = sorted(
        datetime.fromtimestamp(datetime(2024, month, day, 16, tzinfo=UTC).timestamp()).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        for month, day in ((6, 1), (12, 1), (6, 2))
    )
    assert [row["timestamp"] for row in rows] == expected


def test_iso_text_with_a_t_separator_is_read() -> None:
    assert ids(
        FrameSource(frame(timestamp=["2024-06-01T09:00:00", "2024-06-01T09:01:00", "junk"]))
    ) == ["1", "2"]


def test_messages_within_one_second_keep_their_true_order() -> None:
    stamps = pd.to_datetime(
        ["2024-06-01 09:00:00.900", "2024-06-01 09:00:00.100", "2024-06-01 09:00:00.500"]
    )
    assert ids(FrameSource(frame(timestamp=stamps))) == ["2", "3", "1"]


def test_a_row_without_an_id_gets_the_same_name_under_any_scope() -> None:
    source = FrameSource(frame(message_id=[None, None, None]))
    assert ids(source) == ["row-0", "row-1", "row-2"]
    assert ids(source, since="2024-06-02") == ["row-2"]


class Rows:
    def __init__(self, *rows: dict[str, Any]) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    def iter_messages(self, conversation_id: str, **bounds: Any) -> Iterator[dict[str, Any]]:
        self.calls.append(bounds)
        yield from self.rows


def row(**fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": 1,
        "timestamp": "2024-06-30 10:00:00",
        "is_from_me": 0,
        "sender": MINA,
        "text": "hi",
        "kind": "message",
    }
    base.update(fields)
    return base


@pytest.mark.parametrize(
    "fields",
    [
        {"timestamp": "2024-06-30T10:00:00"},
        {"timestamp": "2024-06-30 23:59:59.5"},
        {"timestamp": None},
        {"is_from_me": "false"},
        {"is_from_me": "0"},
        {"is_from_me": 2},
        {"id": None},
    ],
)
def test_an_archive_row_that_breaks_the_contract_is_an_error_not_a_guess(
    fields: dict[str, Any],
) -> None:
    source = ArchiveSource(Rows(row(**fields)))
    with pytest.raises(SourceError):
        list(source.messages(CONVERSATION, None, "2024-06-30"))


def test_archive_flags_and_ids_are_read_strictly() -> None:
    source = ArchiveSource(
        Rows(row(is_from_me=True, sender=None, text=None), row(id="guid-2", is_from_me=1))
    )
    found = list(source.messages(CONVERSATION, None, None))
    assert [(item["id"], item["is_from_me"], item["sender"], item["text"]) for item in found] == [
        ("1", True, "", ""),
        ("guid-2", True, MINA, "hi"),
    ]


def test_an_until_at_the_end_of_the_calendar_asks_the_archive_for_no_upper_day() -> None:
    archive = Rows(row(timestamp="9999-12-31 10:00:00"))
    assert len(list(ArchiveSource(archive).messages(CONVERSATION, None, "9999-12-31"))) == 1
    assert archive.calls == [{"since": None, "until": None}]


def test_a_report_file_is_for_its_owner_only(reports_dir: Path) -> None:
    path = ReportStore(reports_dir).save(make_report())
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_a_write_that_fails_half_way_leaves_no_temp_file(
    reports_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_sync(handle: int) -> None:
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "fsync", no_sync)
    with pytest.raises(StoreError):
        ReportStore(reports_dir).save(make_report())
    folder = reports_dir / conversation_folder("+12025550143")
    assert list(folder.iterdir()) == []


def test_an_id_that_differs_only_by_letter_case_does_not_open_another_report(
    reports_dir: Path,
) -> None:
    store = ReportStore(reports_dir)
    store.save(make_report(id="Run1"))
    with pytest.raises(ReportNotFound):
        store.load_report("+12025550143", "run1")
    assert store.load_report("+12025550143", "Run1").id == "Run1"


def test_a_report_filed_under_the_wrong_conversation_is_not_returned(reports_dir: Path) -> None:
    store = ReportStore(reports_dir)
    path = store.save(make_report(id="mine"))
    other = reports_dir / conversation_folder("someone-else")
    other.mkdir()
    (other / "mine.json").write_bytes(path.read_bytes())
    with pytest.raises(ReportNotFound):
        store.load_report("someone-else", "mine")


def test_two_writers_of_one_id_can_not_both_win_without_hard_links(
    reports_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_link(source: Any, target: Any) -> None:
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(os, "link", no_link)
    store = ReportStore(reports_dir)
    first = make_report(id="same", model="first-model")
    path = store.save(first)
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: False if self == path else real_exists(self))
    with pytest.raises(StoreError):
        store.save(make_report(id="same", model="second-model"))
    monkeypatch.setattr(Path, "exists", real_exists)
    assert store.load_report("+12025550143", "same").model == "first-model"
    assert [file.name for file in path.parent.iterdir()] == ["same.json"]


def test_reports_list_in_time_order_whatever_offset_they_were_given(reports_dir: Path) -> None:
    store = ReportStore(reports_dir)
    store.save(make_report(id="a", created_at="2024-07-01T10:00:00+02:00"))
    store.save(make_report(id="b", created_at="2024-07-01T09:00:00+00:00"))
    store.save(make_report(id="c", created_at="2024-07-01T10:00:00+00:00"))
    assert [entry.id for entry in store.list_reports()] == ["c", "b", "a"]
    latest = store.latest_report("+12025550143")
    assert latest is not None and latest.id == "c"


def limited(headers: dict[str, str]) -> Exception:
    return openai.RateLimitError(
        "slow down", response=httpx.Response(429, headers=headers, request=REQUEST), body=None
    )


@pytest.mark.parametrize(
    "headers, waits",
    [
        ({"retry-after": "nan"}, [2.0]),
        ({"retry-after": "inf"}, [2.0]),
        ({"retry-after-ms": "soon", "retry-after": "3"}, [3.0]),
        ({"retry-after": "-5"}, [0.0]),
    ],
)
def test_odd_retry_headers_are_read_one_by_one(headers: dict[str, str], waits: list[float]) -> None:
    llm, slept = adapter(Client(limited(headers), reply()))
    llm.generate(MESSAGES, max_output_tokens=10)
    assert slept == waits


def test_a_retry_date_with_no_zone_is_read_as_utc() -> None:
    later = email.utils.formatdate(time.time() + 20, usegmt=True).replace("GMT", "-0000")
    llm, slept = adapter(Client(limited({"retry-after": later}), reply()))
    llm.generate(MESSAGES, max_output_tokens=10)
    assert len(slept) == 1 and 17.0 < slept[0] <= 20.0


def test_an_error_carries_facts_only_and_no_chain_to_the_request() -> None:
    response = httpx.Response(400, headers={"x-request-id": "req_abc123"}, request=REQUEST)
    echo = openai.BadRequestError(
        "Invalid input: 'Read this.' is too long",
        response=response,
        body={"code": "string_above_max_length"},
    )
    llm, _ = adapter(Client(echo))
    with pytest.raises(LLMError) as caught:
        llm.generate(MESSAGES, max_output_tokens=10)
    assert (
        str(caught.value)
        == "the model service failed (HTTP 400, string_above_max_length, request req_abc123)"
    )
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_a_request_needs_a_user_message() -> None:
    llm, _ = adapter(Client(reply()))
    with pytest.raises(ValueError):
        llm.generate([{"role": "system", "content": "only rules"}], max_output_tokens=10)


def test_a_failed_reply_logs_its_code_and_nothing_else(caplog: pytest.LogCaptureFixture) -> None:
    failed = reply(status="failed").model_copy(
        update={"error": type("E", (), {"code": "server_error", "message": "secret text"})()}
    )
    llm, _ = adapter(Client(failed))
    with caplog.at_level("WARNING"):
        assert llm.generate(MESSAGES, max_output_tokens=10).status == "failed"
    assert "server_error" in caplog.text
    assert "secret text" not in caplog.text


def test_the_client_address_is_always_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://elsewhere.invalid/v1")
    client = build_client("sk-test-dummy-not-a-real-key")
    assert str(client.base_url).rstrip("/") == OFFICIAL_BASE_URL
    assert OpenAIResponsesLLM(client, "test-model-1") is not None
