"""Period cache: one small SQLite file, keyed by what the model was shown."""

import sqlite3
import threading
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from report_engine.cache import CACHE_SCHEMA_VERSION, CacheEntry, PeriodCache, period_key
from report_engine.errors import StoreError
from report_engine.ports import LLMUsage

PAYLOAD = {"summary": "A calm month.", "carry_over": "A trip is planned.", "moments": []}


def entry(key: str = "k1", **overrides: object) -> CacheEntry:
    values: dict[str, object] = {
        "key": key,
        "payload": PAYLOAD,
        "usage": LLMUsage(input_tokens=900, cached_input_tokens=100, output_tokens=50),
        "cost_usd": 0.0123,
    }
    values.update(overrides)
    return CacheEntry(**values)  # type: ignore[arg-type]


def execute(path: Path, statement: str) -> None:
    """Change the file behind the cache's back, and close the connection after."""
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(statement)


def put(cache: PeriodCache, item: CacheEntry, label: str = "Jun 2024") -> None:
    cache.put(item, conversation="conv-1", label=label, model="test-model-1", prompt_version="v1")


def test_key_is_stable_and_hex() -> None:
    first = period_key("v1", "model-a", ("Mina", "You"), "m1 09:00 Mina: hi")
    again = period_key("v1", "model-a", ("Mina", "You"), "m1 09:00 Mina: hi")
    assert first == again
    assert len(first) == 64
    assert int(first, 16) >= 0


@pytest.mark.parametrize(
    "changed",
    [
        ("v2", "model-a", ("Mina", "You"), "m1 09:00 Mina: hi"),
        ("v1", "model-b", ("Mina", "You"), "m1 09:00 Mina: hi"),
        ("v1", "model-a", ("Mina", "Theo", "You"), "m1 09:00 Mina: hi"),
        ("v1", "model-a", ("Mina", "You"), "m1 09:00 Mina: hi!"),
    ],
)
def test_key_changes_with_each_of_its_four_parts(changed: tuple) -> None:
    base = period_key("v1", "model-a", ("Mina", "You"), "m1 09:00 Mina: hi")
    assert period_key(*changed) != base


def test_key_parts_can_not_run_into_each_other() -> None:
    assert period_key("v1", "ab", ("c",), "d") != period_key("v1", "a", ("bc",), "d")
    assert period_key("v1", "m", ("a", "b"), "t") != period_key("v1", "m", ("a,b",), "t")


def test_a_missing_key_is_a_miss(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    assert cache.get("nope") is None
    assert cache.get_many(["nope", "nada"]) == {}
    assert cache.known(["nope"]) == frozenset()


def test_put_then_get_returns_the_same_entry(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    put(cache, entry())
    found = cache.get("k1")
    assert found is not None
    assert found.payload == PAYLOAD
    assert found.usage == LLMUsage(input_tokens=900, cached_input_tokens=100, output_tokens=50)
    assert found.cost_usd == pytest.approx(0.0123)
    assert found.created_at


def test_entries_outlive_the_object_that_wrote_them(cache_path: Path) -> None:
    put(PeriodCache(cache_path), entry())
    assert PeriodCache(cache_path).get("k1") is not None


def test_get_many_returns_only_the_hits(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    put(cache, entry("a"))
    put(cache, entry("b"))
    assert set(cache.get_many(["a", "x", "b"])) == {"a", "b"}
    assert cache.known(["a", "x", "b"]) == frozenset({"a", "b"})


def test_get_many_handles_more_keys_than_one_query_takes(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    keys = [f"key-{number}" for number in range(1300)]
    for key in keys[::100]:
        put(cache, entry(key))
    assert set(cache.get_many(keys)) == set(keys[::100])


def test_put_replaces_an_entry_with_the_same_key(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    put(cache, entry())
    put(cache, entry(payload={"summary": "Second take.", "carry_over": "", "moments": []}))
    found = cache.get("k1")
    assert found is not None and found.payload["summary"] == "Second take."
    assert cache.count() == 1


def test_the_cache_does_not_keep_a_reference_to_the_payload(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    payload = {"summary": "s", "carry_over": "", "moments": [{"title": "t"}]}
    put(cache, entry(payload=payload))
    payload["moments"].append({"title": "added later"})
    found = cache.get("k1")
    assert found is not None and len(found.payload["moments"]) == 1


def test_a_broken_payload_is_a_miss_and_is_removed(
    cache_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cache = PeriodCache(cache_path)
    put(cache, entry())
    execute(cache_path, "UPDATE period_results SET payload = '{not json'")
    with caplog.at_level("WARNING"):
        assert cache.get("k1") is None
    assert cache.count() == 0
    assert "k1"[:8] in caplog.text


def test_prune_removes_entries_that_were_not_used_lately(cache_path: Path) -> None:
    now = [datetime(2024, 1, 1, tzinfo=UTC)]
    cache = PeriodCache(cache_path, now=lambda: now[0])
    put(cache, entry("old"))
    put(cache, entry("used"))
    now[0] += timedelta(days=100)
    assert set(cache.get_many(["used"], touch=True)) == {"used"}
    put(cache, entry("new"))
    assert cache.prune(unused_for_days=30) == 1
    assert cache.known(["old", "used", "new"]) == frozenset({"used", "new"})


def test_reading_without_touch_does_not_reset_the_age(cache_path: Path) -> None:
    now = [datetime(2024, 1, 1, tzinfo=UTC)]
    cache = PeriodCache(cache_path, now=lambda: now[0])
    put(cache, entry("old"))
    now[0] += timedelta(days=100)
    assert cache.get("old") is not None
    assert cache.known(["old"]) == frozenset({"old"})
    assert cache.prune(unused_for_days=30) == 1


def test_clear_removes_everything(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    put(cache, entry("a"))
    put(cache, entry("b"))
    assert cache.clear() == 2
    assert cache.count() == 0


def test_the_cache_makes_its_folder(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "er" / "cache.sqlite"
    put(PeriodCache(path), entry())
    assert path.exists()


def test_a_file_from_a_newer_version_is_refused(cache_path: Path) -> None:
    PeriodCache(cache_path)
    execute(cache_path, f"PRAGMA user_version = {CACHE_SCHEMA_VERSION + 1}")
    with pytest.raises(StoreError):
        PeriodCache(cache_path)


def test_a_file_that_is_not_a_database_is_a_store_error(cache_path: Path) -> None:
    cache_path.write_text("this is not sqlite", encoding="utf-8")
    with pytest.raises(StoreError):
        PeriodCache(cache_path)


def test_two_threads_can_write_at_once(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    errors: list[BaseException] = []

    def work(prefix: str) -> None:
        try:
            for number in range(25):
                put(cache, entry(f"{prefix}-{number}"))
        except BaseException as exc:  # noqa: BLE001 - collected for the assert below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(name,)) for name in ("a", "b", "c")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert cache.count() == 75
