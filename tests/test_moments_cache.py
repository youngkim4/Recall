"""The moments of the Overview are worked out once for one archive, one set of names and one day."""

import pandas as pd
import pytest

import ui_server
from tests.export_kit import (  # noqa: F401 - archives and contacts_cache are fixtures
    AVERY,
    archives,
    contacts_cache,
    loaded,
    row,
)

pytestmark = pytest.mark.usefixtures("archives", "contacts_cache")

DAY = pd.Timestamp("2026-11-16 12:00:00")


@pytest.fixture
def counted(monkeypatch: pytest.MonkeyPatch) -> list[pd.Timestamp]:
    calls: list[pd.Timestamp] = []
    real = ui_server.memories_payload

    def counting(messages_path, now=None):
        calls.append(now)
        return real(messages_path, now=now)

    monkeypatch.setattr(ui_server, "memories_payload", counting)
    ui_server.forget_moments()
    return calls


def test_the_moments_of_one_day_are_worked_out_once(tmp_path, counted) -> None:
    csv = loaded(tmp_path / "messages.csv", [row(AVERY, "2024-11-16 09:00:00", "the lake again this year?")])
    first = ui_server.cached_moments(csv, now=DAY)
    second = ui_server.cached_moments(csv, now=DAY + pd.Timedelta(hours=3))
    assert first == second
    assert len(counted) == 1


def test_a_new_day_works_the_moments_out_again(tmp_path, counted) -> None:
    csv = loaded(tmp_path / "messages.csv", [row(AVERY, "2024-11-16 09:00:00", "the lake again this year?")])
    ui_server.cached_moments(csv, now=DAY)
    ui_server.cached_moments(csv, now=DAY + pd.Timedelta(days=1))
    assert len(counted) == 2


def test_another_content_of_the_archive_works_the_moments_out_again(tmp_path, counted, monkeypatch) -> None:
    csv = loaded(tmp_path / "messages.csv", [row(AVERY, "2024-11-16 09:00:00", "the lake again this year?")])
    ui_server.cached_moments(csv, now=DAY)
    real = ui_server.bundle_signature
    monkeypatch.setattr(ui_server, "bundle_signature", lambda path: ("another", real(path)))
    ui_server.cached_moments(csv, now=DAY)
    assert len(counted) == 2


def test_the_route_answers_from_the_cache(tmp_path, counted, monkeypatch) -> None:
    csv = loaded(tmp_path / "messages.csv", [row(AVERY, "2024-11-16 09:00:00", "the lake again this year?")])
    monkeypatch.setattr(ui_server, "local_now", lambda: DAY)
    ui_server.cached_moments(csv)
    ui_server.cached_moments(csv)
    assert counted == [DAY]
