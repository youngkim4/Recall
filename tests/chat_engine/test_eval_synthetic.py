"""The synthetic archive: stable, varied, and its facts are true."""

import time

from chat_engine.eval import synthetic_data as data
from chat_engine.eval.synthetic import TODAY, build_synthetic_archive
from chat_engine.textnorm import words

SYNTHETIC = build_synthetic_archive()


def test_same_seed_gives_the_same_archive() -> None:
    again = build_synthetic_archive()
    assert again.messages == SYNTHETIC.messages
    assert again.facts == SYNTHETIC.facts
    assert build_synthetic_archive(seed=8).messages != SYNTHETIC.messages


def test_archive_has_the_required_variety() -> None:
    archive = SYNTHETIC.archive()
    summary = archive.summary()
    assert 4000 < summary["messages"] < 9000
    assert summary["first_timestamp"][:4] == "2019" and summary["last_timestamp"][:4] == "2026"
    conversations = {c["id"]: c for c in archive.conversations(order="count")}
    assert len(conversations) == 11
    assert sum(1 for c in conversations.values() if c["is_group"]) == 3
    assert conversations[data.MAYA_PHONE]["person_id"] == conversations[data.MAYA_EMAIL]["person_id"]
    assert data.UNSAVED_PHONE not in SYNTHETIC.saved_names and data.UNSAVED_EMAIL not in SYNTHETIC.saved_names
    assert len(data.HANDOVER) > 2000
    assert archive.search("cozinha")["total"] == 1
    assert SYNTHETIC.today == TODAY
    ids = [m["id"] for m in SYNTHETIC.messages]
    assert len(ids) == len(set(ids))


def test_filler_holds_none_of_the_words_the_questions_ask_about() -> None:
    asked = {
        "sourdough", "starter", "lease", "marathon", "cabin", "wifi", "password", "badge", "ferry", "rye",
        "baked", "tent", "climbing", "gym", "flights", "booked", "blood", "zanzibar", "apartamento",
        "cozinha", "taberna", "landlord", "signed", "name", "named", "lisbon", "hostel", "octopus",
    }
    for pool in data.POOLS.values():
        for line in pool:
            assert not asked & set(words(line)), line


def test_facts_agree_with_an_independent_count_through_the_archive() -> None:
    archive = SYNTHETIC.archive()
    facts = SYNTHETIC.facts
    assert facts["messages"] == archive.count()
    assert facts["sourdough_total"] == archive.count(text="sourdough") == 12
    assert facts["sourdough_jonas"] == archive.count(text="sourdough", conversations=[data.JONAS]) == 9
    assert (facts["sourdough_first_day"], facts["sourdough_last_day"]) == ("2024-02-10", "2026-08-01")
    assert facts["priya_2024"] == archive.count(conversations=[data.PRIYA], since="2024-01-01", until="2024-12-31")
    assert facts["priya_last_month"] == archive.count(
        conversations=[data.PRIYA], since="2026-10-01", until="2026-10-31"
    )
    assert facts["priya_last_month"] > 0
    assert facts["top_conversation"] == "priya"
    assert facts["maya_total"] > facts["jonas_total"]
    assert facts["maya_busiest_month_2023"] == "2023-09"
    assert (facts["marathon_maya"], facts["marathon_priya"]) == (6, 2)
    assert facts["maya_sent"] + facts["maya_received"] == facts["maya_phone_total"]
    assert facts["maya_sent"] != facts["maya_received"]


def test_busiest_month_of_2023_with_maya_is_clear() -> None:
    rows = SYNTHETIC.archive().count(
        conversations=[data.MAYA_PHONE, data.MAYA_EMAIL], since="2023-01-01", until="2023-12-31", group_by="month"
    )
    totals = sorted((row["total"] for row in rows), reverse=True)
    assert totals[0] == SYNTHETIC.facts["maya_busiest_month_2023_total"]
    assert totals[0] > totals[1] + 10


def test_planted_recall_facts_are_unique() -> None:
    archive = SYNTHETIC.archive()
    for query in ['"Clint Yeastwood"', "pinecone", '"Taberna do Largo"', "Bainbridge", "rye", "badge", "Dana"]:
        assert archive.search(query)["total"] == 1, query
    assert archive.search('"its name is Clint Yeastwood"')["hits"][0]["message"]["id"] == SYNTHETIC.planted[
        "starter_name"
    ]


def test_search_on_the_synthetic_archive_is_fast() -> None:
    archive = SYNTHETIC.archive()
    started = time.perf_counter()
    for _ in range(200):
        archive.search("sourdough", order="oldest", limit=5)
        archive.count(text="sounds good", group_by="year")
    assert time.perf_counter() - started < 2.0
