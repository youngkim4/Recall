from chat_engine.fake_archive import FakeArchive, make_message
from chat_engine.search import SearchRequest, dedupe_moments, run_search
from tests.chat_engine.tiny import GROUP, MAYA_EMAIL, MAYA_PHONE, tiny_archive


def _ids(outcome) -> list[str]:
    return [found.message["id"] for found in outcome.hits]


def _noise(count: int, word: str = "dinner") -> list[dict]:
    return [
        make_message(f"n{i}", "+15550000001", f"2020-01-{1 + i % 28:02d} 12:{i % 60:02d}:00", f"{word} plans {i}")
        for i in range(count)
    ]


def test_all_mode_passes_the_query_and_reports_the_exact_total() -> None:
    archive = tiny_archive()
    outcome = run_search(archive, SearchRequest(query="lease", order="oldest", limit=3))
    assert outcome.total == 4
    assert _ids(outcome) == ["g4", "a2", "b1"]
    name, arguments = archive.calls[0]
    assert name == "search" and arguments["query"] == "lease" and arguments["order"] == "oldest"


def test_all_mode_focus_points_at_the_match() -> None:
    outcome = run_search(tiny_archive(), SearchRequest(query="lease", conversations=(MAYA_EMAIL,)))
    found = outcome.hits[0]
    assert found.message["text"][found.focus:found.focus + 5] == "lease"


def test_all_mode_with_no_hit_counts_each_word() -> None:
    outcome = run_search(tiny_archive(), SearchRequest(query="lease zanzibar"))
    assert outcome.total == 0 and not outcome.hits
    assert dict(outcome.word_counts) == {"lease": 4, "zanzibar": 0}


def test_one_long_exchange_does_not_take_all_the_slots() -> None:
    burst = [
        make_message(f"s{i}", "+15550000002", f"2024-05-05 10:{i:02d}:00", f"the wedding plan part {i}")
        for i in range(10)
    ]
    other = [
        make_message("o1", "+15550000003", "2022-01-01 09:00:00", "wedding invite arrived"),
        make_message("o2", "+15550000004", "2021-01-01 09:00:00", "whose wedding was that"),
    ]
    outcome = run_search(FakeArchive([*burst, *other]), SearchRequest(query="wedding", limit=4))
    assert outcome.total == 12
    ids = _ids(outcome)
    assert len([i for i in ids if i.startswith("s")]) == 2
    assert {"o1", "o2"} <= set(ids)


def test_dedupe_moments_counts_per_conversation_and_time() -> None:
    rows = [
        make_message("a", "c1", "2024-01-01 10:00:00", "x"),
        make_message("b", "c1", "2024-01-01 10:05:00", "x"),
        make_message("c", "c1", "2024-01-01 10:10:00", "x"),
        make_message("d", "c2", "2024-01-01 10:10:00", "x"),
        make_message("e", "c1", "2024-01-03 10:10:00", "x"),
    ]
    kept = dedupe_moments(rows, per_moment=2)
    assert [m["id"] for m in kept] == ["a", "b", "d", "e"]


def test_any_mode_weighs_rare_words_above_common_ones() -> None:
    planted = [
        make_message("r1", "+15550000002", "2023-03-03 09:00:00", "the notary wants the deed on monday"),
        make_message("r2", "+15550000002", "2023-03-03 09:30:00", "dinner after we see the notary"),
    ]
    archive = FakeArchive([*_noise(300), *planted])
    outcome = run_search(archive, SearchRequest(query="notary dinner deed", match="any", limit=5))
    assert _ids(outcome)[:2] == ["r1", "r2"] or _ids(outcome)[:2] == ["r2", "r1"]
    counts = dict(outcome.word_counts)
    assert counts["notary"] == 2 and counts["deed"] == 1 and counts["dinner"] == 301
    assert "dinner" in outcome.common
    assert set(outcome.used) == {"notary", "deed"}
    assert outcome.total is None


def test_any_mode_reports_absent_words_and_finds_the_rest() -> None:
    outcome = run_search(tiny_archive(), SearchRequest(query="zanzibar sourdough", match="any"))
    assert "zanzibar" in outcome.absent
    assert set(_ids(outcome)) == {"j1", "j4"}


def test_any_mode_with_only_common_words_finds_nothing() -> None:
    archive = FakeArchive(_noise(400))
    outcome = run_search(archive, SearchRequest(query="dinner plans", match="any"))
    assert not outcome.hits
    assert set(outcome.common) == {"dinner", "plans"}


def test_any_mode_boost_prefers_the_named_conversation() -> None:
    archive = tiny_archive()
    plain = run_search(archive, SearchRequest(query="lease flat", match="any", limit=3))
    boosted = run_search(
        archive, SearchRequest(query="lease flat", match="any", limit=3, boost=(MAYA_EMAIL,))
    )
    assert _ids(plain)[0] == "g4"
    assert _ids(boosted)[0] in {"g4", "b1"}
    assert "b1" in _ids(boosted)[:2]


def test_any_mode_gives_a_burst_bonus_to_neighbours() -> None:
    rows = [
        make_message("f1", "+15550000002", "2024-04-01 10:00:00", "can you front me"),
        make_message("f2", "+15550000002", "2024-04-01 10:01:00", "like 10k"),
        make_message("f3", "+15550000003", "2021-04-01 10:00:00", "front door is open"),
        *_noise(200),
    ]
    outcome = run_search(FakeArchive(rows), SearchRequest(query="front 10k", match="any", limit=3))
    assert set(_ids(outcome)[:2]) == {"f1", "f2"}


def test_search_filters_reach_the_archive() -> None:
    archive = tiny_archive()
    run_search(
        archive,
        SearchRequest(
            query="lease", conversations=(MAYA_PHONE,), since="2023-01-01", until="2023-12-31", from_me=False
        ),
    )
    _, arguments = archive.calls[0]
    assert arguments["conversations"] == [MAYA_PHONE]
    assert (arguments["since"], arguments["until"], arguments["from_me"]) == ("2023-01-01", "2023-12-31", False)


def test_any_mode_never_reads_outside_the_given_conversations() -> None:
    archive = tiny_archive()
    run_search(archive, SearchRequest(query="lease lisbon flights", match="any", conversations=(GROUP,)))
    for name, arguments in archive.calls:
        assert name in {"search", "count"}
        assert arguments["conversations"] == [GROUP]
