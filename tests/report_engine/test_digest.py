"""The map reply is outside data. It is checked before anything trusts it."""

from typing import Any

import pytest

from report_engine.digest import (
    MAX_CARRY_CHARS,
    MAX_MOMENTS_PER_PERIOD,
    MAX_SUMMARY_CHARS,
    DigestError,
    PeriodDigest,
    digest_payload,
    parse_digest,
)


def moment(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "date": "2024-06-03",
        "title": "A plan",
        "detail": "You two set a date.",
        "category": "milestone",
        "score": 0.7,
        "quote": "see you at the harbor",
        "refs": ["m4", "m5"],
    }
    values.update(overrides)
    return values


def reply(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "summary": "A calm month.",
        "carry_over": "A trip is planned.",
        "moments": [moment()],
    }
    values.update(overrides)
    return values


def test_a_good_reply_parses() -> None:
    digest = parse_digest(reply())
    assert digest.summary == "A calm month."
    assert digest.carry_over == "A trip is planned."
    assert len(digest.moments) == 1
    first = digest.moments[0]
    assert (first.date, first.title, first.category, first.score) == ("2024-06-03", "A plan", "milestone", 0.7)
    assert first.refs == ("m4", "m5")


@pytest.mark.parametrize("bad", [None, [], "text", 7])
def test_a_reply_that_is_not_an_object_is_refused(bad: Any) -> None:
    with pytest.raises(DigestError):
        parse_digest(bad)


@pytest.mark.parametrize("summary", ["", "   \n", None, 12])
def test_a_reply_without_a_summary_is_refused(summary: Any) -> None:
    with pytest.raises(DigestError):
        parse_digest(reply(summary=summary))


def test_a_reply_without_a_summary_key_is_refused() -> None:
    data = reply()
    del data["summary"]
    with pytest.raises(DigestError):
        parse_digest(data)


def test_moments_must_be_a_list_when_present() -> None:
    with pytest.raises(DigestError):
        parse_digest(reply(moments="none"))


def test_missing_moments_and_notes_mean_none() -> None:
    digest = parse_digest({"summary": "Quiet."})
    assert digest.moments == ()
    assert digest.carry_over == ""


def test_items_that_are_not_objects_are_skipped() -> None:
    digest = parse_digest(reply(moments=[moment(), "junk", None, 3]))
    assert len(digest.moments) == 1


def test_loose_values_are_coerced() -> None:
    digest = parse_digest(
        reply(moments=[moment(score="0.8", refs="m9", title=None, quote=None, detail=5, date=None)])
    )
    first = digest.moments[0]
    assert first.score == 0.8
    assert first.refs == ("m9",)
    assert (first.title, first.quote, first.detail, first.date) == ("", "", "5", "")


@pytest.mark.parametrize("score, expected", [(None, 0.0), ("high", 0.0), (float("nan"), 0.0), (7, 7.0), (True, 0.0)])
def test_a_score_that_is_not_a_number_is_zero(score: Any, expected: float) -> None:
    assert parse_digest(reply(moments=[moment(score=score)])).moments[0].score == expected


def test_refs_keep_only_strings() -> None:
    digest = parse_digest(reply(moments=[moment(refs=["m1", 2, None, "m3"])]))
    assert digest.moments[0].refs == ("m1", "m3")


def test_too_many_moments_keeps_the_strongest_in_their_order() -> None:
    scores = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.95, 0.05]
    digest = parse_digest(reply(moments=[moment(title=f"t{n}", score=s) for n, s in enumerate(scores)]))
    assert len(digest.moments) == MAX_MOMENTS_PER_PERIOD
    kept = [m.score for m in digest.moments]
    assert sorted(kept) == sorted(scores)[-MAX_MOMENTS_PER_PERIOD:]
    order = [scores.index(score) for score in kept]
    assert order == sorted(order)


def test_long_text_is_cut_to_a_limit() -> None:
    digest = parse_digest(reply(summary="s" * (MAX_SUMMARY_CHARS + 50), carry_over="c" * (MAX_CARRY_CHARS + 50)))
    assert len(digest.summary) == MAX_SUMMARY_CHARS
    assert len(digest.carry_over) == MAX_CARRY_CHARS


def test_the_cache_form_parses_back_to_the_same_digest() -> None:
    digest = parse_digest(reply(moments=[moment(), moment(title="Second", refs=[])]))
    payload = digest_payload(digest)
    assert parse_digest(payload) == digest
    assert isinstance(payload["moments"], list)
    assert isinstance(payload["moments"][0]["refs"], list)


def test_parse_does_not_change_the_reply() -> None:
    data = reply()
    snapshot = {**data, "moments": [dict(item) for item in data["moments"]]}
    parse_digest(data)
    assert data == snapshot


def test_digest_is_immutable() -> None:
    digest = parse_digest(reply())
    assert isinstance(digest, PeriodDigest)
    with pytest.raises(Exception):
        digest.summary = "changed"  # type: ignore[misc]
