"""No line reference of the report engine leaves in a digest.

A map reply cites lines of its transcript as m12. Chat has references of the
same form for other messages, so a digest gives its text without them. The
cache keeps the reply as the model wrote it: a report needs the references.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from report_engine import transcript
from report_engine.cache import PeriodCache
from report_engine.chat_digest_refs import without_refs
from report_engine.fake_llm import FakeCall, FakeLLM
from report_engine.transcript import Line, render_period, resolve_ref

from .chat_digest_support import cache_rows, digest, tagged
from .support import SMALL, run

REFERENCE = re.compile(r"(?<!\w)[mM][0-9]+(?!\w)")
# a reply of a real model on an invented archive, and the sentences around it
REAL = (
    "Mira Castellan initiated contact after getting your number from Jo, and you welcomed "
    "the message (m1–m2). The talk moved to the harbour plan and to the keys "
    "(m3–m18, m353–m358). By the end of the month you agreed on a date (m928–m931)"
)
REAL_CLEAN = (
    "Mira Castellan initiated contact after getting your number from Jo, and you welcomed "
    "the message. The talk moved to the harbour plan and to the keys. "
    "By the end of the month you agreed on a date"
)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (REAL, REAL_CLEAN),
        ("You welcomed the message (m1–m2).", "You welcomed the message."),
        (
            "You welcomed the message (m1-m2). Then silence.",
            "You welcomed the message. Then silence.",
        ),
        ("You welcomed the message (m1–2).", "You welcomed the message."),
        ("You welcomed the message [m12].", "You welcomed the message."),
        ("You welcomed the message m12.", "You welcomed the message."),
        ("The plan changed (m3, m7 and m9) in May.", "The plan changed in May."),
        (
            "The plan changed m3, m7 and m9, and nobody minded.",
            "The plan changed, and nobody minded.",
        ),
        ("The plan changed (see m12).", "The plan changed."),
        ("The plan changed, see m12.", "The plan changed."),
        ("The plan changed (lines m3–m18).", "The plan changed."),
        ("The plan changed (see lines m3 to m18; also m40).", "The plan changed."),
        ("The plan changed (refs: m3, m7).", "The plan changed."),
        ("She apologised in m12 and you accepted.", "She apologised and you accepted."),
        (
            "She apologised (on 3 May, m12) and you accepted.",
            "She apologised (on 3 May) and you accepted.",
        ),
        (
            "She apologised (m12, on 3 May) and you accepted.",
            "She apologised (on 3 May) and you accepted.",
        ),
        ("They met, m12, and left.", "They met, and left."),
        ("A quiet week (m4 – m9); then the call (m10).", "A quiet week; then the call."),
        (
            "First part (m1).\n\nSecond part (m2–m5).\nThird line.",
            "First part.\n\nSecond part.\nThird line.",
        ),
        ("The apology (m12)", "The apology"),
        ("m12: the apology", "the apology"),
        ("(m12) The apology", "The apology"),
        ("In m12, she says sorry.", "she says sorry."),
        ("Was it a joke (m7)? Yes!", "Was it a joke? Yes!"),
        ("m12", ""),
        ("(m3–m18)", ""),
        ("It ended ((m12)) well.", "It ended well."),
    ],
)
def test_references_go_and_the_sentence_reads_cleanly(before: str, after: str) -> None:
    cleaned = without_refs(before)
    assert cleaned == after
    assert not REFERENCE.search(cleaned)
    assert "  " not in cleaned and "()" not in cleaned and "[]" not in cleaned
    assert not re.search(r" [.,;:?!)]", cleaned)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Nothing to remove here.",
        "She sent an m4a file and an mp3.",
        "You drove the M25 motorway to the M4.",
        "The flat has 40 m2 and the room 12 m2.",
        "Model m1234567890 is not a reference: it has ten digits.",
        "The code was am12 or m12b or x-m12.",
        "Find it at example.com/m12 or #m12 or @m12.",
        "Bonjour !  Deux  espaces , et ( une parenthèse ) .",
        "A list: (a) first, (b) second [sic].",
    ],
)
def test_text_with_no_reference_is_not_touched(text: str) -> None:
    assert without_refs(text) == text


def test_the_form_is_the_form_that_the_engine_writes_and_reads() -> None:
    """`render_period` writes m1..mN before each line. `resolve_ref` reads m and 1 to 9 digits."""
    lines = [
        Line(id=f"id-{n}", day="2024-06-01", clock="09:00", label="Mina", text="hello")
        for n in range(12)
    ]
    written = [row.split(" ", 1)[0] for row in render_period(lines).text.splitlines()[1:]]
    assert written[0] == "m1" and written[-1] == "m12"
    for ref in (*written, "m999999999"):
        assert transcript._REF.fullmatch(ref)
        assert without_refs(f"It happened ({ref}).") == "It happened."
        assert without_refs(f"It happened in {ref}.") == "It happened."
    assert resolve_ref("m12", lines) is lines[11]
    # a reference outside the period is not a line of the engine, and it must go all the same
    assert resolve_ref("m931", lines) is None
    assert without_refs("It happened (m931).") == "It happened."
    assert not transcript._REF.fullmatch("m1234567890")


def test_text_made_to_be_slow_is_read_in_one_pass() -> None:
    """Long runs of spaces after a reference, a bracket or a word that can hold one."""
    blank = " " * 20_000
    hostile = f"(m1{blank}x (see{blank}y m2,{blank}and{blank}z (lines{blank}m3{blank}–{blank}w"
    cleaned = without_refs(hostile)
    assert "m1" not in cleaned and "m2" not in cleaned and "m3" not in cleaned
    assert without_refs("m1, " * 5_000) == ""
    assert without_refs("(see " * 5_000) == "(see " * 5_000


# --- in a digest ---------------------------------------------------------------------

MESSAGES = [
    *tagged(SMALL)[:-1],
    {**tagged(SMALL)[-1], "text": "take the m25 bus then call me from gate m12"},
]


def cites(call: FakeCall) -> Any:
    """A map reply that cites lines in each text field, as a real model does."""
    if call.kind != "map":
        return None
    first, last = call.lines[0], call.lines[-1]
    moment = {"date": first.day, "category": "milestone", "score": 0.9}
    return {
        "summary": REAL,
        "carry_over": "Open: the date (m40).",
        "moments": [
            {
                **moment,
                "title": f"First contact ({first.ref})",
                "detail": f"You welcomed the message, see {first.ref}–m2. It set the tone (m3).",
                "quote": first.text,
                "refs": [first.ref],
            },
            {
                **moment,
                "date": last.day,
                "title": "The bus",
                "detail": "A plan for the day.",
                "quote": last.text,
                "refs": [last.ref],
            },
            {**moment, "title": first.ref, "detail": "Only a reference.", "refs": [first.ref]},
        ],
    }


def texts_of(found: Any) -> list[str]:
    period = found.periods[0]
    moments = [text for moment in period.moments for text in (moment.title, moment.detail)]
    return [period.summary, *moments]


def test_a_digest_holds_no_reference_in_a_summary_a_title_or_a_detail() -> None:
    found = digest(MESSAGES, llm=FakeLLM(cites))
    (period,) = found.periods
    assert period.summary == REAL_CLEAN
    first = period.moments[0]
    assert first.title == "First contact"
    assert first.detail == "You welcomed the message. It set the tone."
    assert not [text for text in texts_of(found) if REFERENCE.search(text)]
    assert first.message_ids == (MESSAGES[0]["id"],)


def test_a_quote_is_message_text_and_stays_as_the_check_left_it() -> None:
    (period,) = digest(MESSAGES, llm=FakeLLM(cites)).periods
    assert period.moments[0].quote == MESSAGES[0]["text"]
    assert period.moments[1].quote == "take the m25 bus then call me from gate m12"


def test_a_moment_whose_title_is_only_a_reference_is_not_given() -> None:
    (period,) = digest(MESSAGES, llm=FakeLLM(cites)).periods
    assert [moment.title for moment in period.moments] == ["First contact", "The bus"]


def test_the_cache_keeps_the_reply_as_the_model_wrote_it(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    fresh = digest(MESSAGES, llm=FakeLLM(cites), cache=cache)
    rows = cache_rows(cache_path)
    assert "(m1–m2)" in rows[0][5] and "First contact (m1)" in rows[0][5]
    llm = FakeLLM()
    again = digest(MESSAGES, llm=llm, cache=cache)
    assert llm.calls == [] and again.periods == fresh.periods
    assert again.periods[0].summary == REAL_CLEAN
    assert cache_rows(cache_path) == rows


def test_a_report_made_from_that_cache_is_what_it_was(tmp_path: Path) -> None:
    """The report reads the references in the summary. A digest before it changes nothing."""
    read_by_a_digest, untouched = tmp_path / "digest.sqlite3", tmp_path / "report.sqlite3"
    digest(MESSAGES, llm=FakeLLM(cites), cache=PeriodCache(read_by_a_digest))
    llm = FakeLLM(cites)
    after = run(MESSAGES, llm=llm, cache=PeriodCache(read_by_a_digest))
    alone = run(MESSAGES, llm=FakeLLM(cites), cache=PeriodCache(untouched))
    assert llm.map_calls == [] and after.periods[0].from_cache
    assert after.periods[0].summary == alone.periods[0].summary == REAL
    assert [moment.title for moment in after.periods[0].moments] == [
        moment.title for moment in alone.periods[0].moments
    ]
    assert after.periods[0].moments[0].title == "First contact (m1)"
    assert after.sections == alone.sections and after.moments == alone.moments
    assert "(m3–m18, m353–m358)" in llm.reduce_calls[0].text
    assert cache_rows(read_by_a_digest) == cache_rows(untouched)
