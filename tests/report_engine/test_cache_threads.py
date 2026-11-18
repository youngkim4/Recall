"""Threads that open one new period cache at the same moment: each one opens it.

Several conversations are read at once, one thread each, and each run opens
the cache file. The first opening makes the file.
"""

from __future__ import annotations

import threading
from pathlib import Path

from report_engine.cache import PeriodCache

THREADS = 6
ROUNDS = 25


def opened_at_once(path: Path) -> list[Exception]:
    barrier, failed = threading.Barrier(THREADS, timeout=10), []

    def work() -> None:
        barrier.wait()
        try:
            PeriodCache(path)
        except Exception as error:  # noqa: BLE001 - the test counts them
            failed.append(error)

    threads = [threading.Thread(target=work) for _ in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    return failed


def test_threads_that_open_a_new_cache_at_once_all_open_it(tmp_path: Path) -> None:
    failed = [error for round_ in range(ROUNDS) for error in opened_at_once(tmp_path / f"{round_}.sqlite")]
    assert failed == []


# --- which conversations a model read ------------------------------------------------


def test_the_cache_says_which_conversations_a_model_read(tmp_path: Path) -> None:
    from report_engine.cache import CacheEntry
    from report_engine.ports import LLMUsage

    cache = PeriodCache(tmp_path / "read.sqlite")
    assert cache.conversations_read("model-a") == frozenset()
    for conversation, model in (("one", "model-a"), ("two", "model-b"), ("one", "model-a"), ("three", "model-a")):
        entry = CacheEntry(key=f"{conversation}-{model}-{len(conversation)}", payload={}, usage=LLMUsage(), cost_usd=0.0)
        cache.put(entry, conversation=conversation, label="Jan 2023", model=model, prompt_version="v")
    assert cache.conversations_read("model-a") == frozenset({"one", "three"})
    assert cache.conversations_read("model-b") == frozenset({"two"})
