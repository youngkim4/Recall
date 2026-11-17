"""Check what the model claims against the messages.

References map back to message ids. A date must be a day the period holds; a
wrong date moves to the earliest cited day, and a moment with no usable date
is dropped. A quote must be part of a cited message, or of a message within
three days of the moment's date. A quote that is not found is cleared.

A quote matches when it is equal after white space, quote marks and Unicode
composition are evened out. Letter case must match. In scripts that put
spaces between words, a quote can not start or end in the middle of a word:
"ok" is not a quote from "tokens".

A quote is one run of words of one message. The mark of a cut at the end of
a quote is taken off: a model can copy it from a line that was cut. A mark
inside a quote is not accepted: "A [...] B" is two places put together.

A verified quote keeps the id of its message and the name of its writer.
Text inside a message can claim another writer; the name comes from the
message itself, never from the text.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from report_engine.dates import is_day, shift_day
from report_engine.digest import MAX_DETAIL_CHARS, MAX_QUOTE_CHARS, MAX_TITLE_CHARS, RawMoment
from report_engine.models import CATEGORIES, Moment
from report_engine.transcript import (
    ELISION_MARK,
    Line,
    is_word_char,
    resolve_ref,
    squash,
    word_cut,
)

QUOTE_WINDOW_DAYS = 3
MIN_WINDOW_QUOTE_CHARS = 8
MAX_MESSAGE_IDS = 8

_SINGLE_MARKS = (0x2018, 0x2019, 0x201A, 0x201B, 0x2032)
_DOUBLE_MARKS = (0x201C, 0x201D, 0x201E, 0x201F, 0x2033, 0xAB, 0xBB)
_VARIATION_SELECTORS = (0xFE0E, 0xFE0F)
_ELLIPSIS = chr(0x2026)
_EVEN_OUT = {
    **dict.fromkeys(_SINGLE_MARKS, "'"),
    **dict.fromkeys(_DOUBLE_MARKS, '"'),
    **dict.fromkeys(_VARIATION_SELECTORS),
    0x2026: "...",
}
_QUOTE_MARKS = frozenset("'\"") | frozenset(chr(code) for code in _SINGLE_MARKS + _DOUBLE_MARKS)
_EDGE_DOTS = ("...", _ELLIPSIS)
_CUT_MARKS = (ELISION_MARK, f"[{_ELLIPSIS}]")


def normalise(text: str) -> str:
    """The form two texts are compared in."""
    return squash(unicodedata.normalize("NFC", text).translate(_EVEN_OUT))


def clean_quote(quote: str) -> str:
    """Drop what a model adds around a quote: wrapping marks and dots at the edges."""
    text = squash(quote)
    if len(text) >= 2 and text[0] in _QUOTE_MARKS and text[-1] in _QUOTE_MARKS:
        text = text[1:-1].strip()
    for dots in _EDGE_DOTS:
        if text.startswith(dots):
            text = text[len(dots) :].strip()
        if text.endswith(dots):
            text = text[: -len(dots)].strip()
    return text


def quoted_words(quote: str) -> str:
    """The words that a quote claims: with no wrapping marks, no dots at the
    edges and no mark of a cut at the end. A line that was cut ends with the
    mark, and a model can copy it; the mark is not words of the message. A
    mark anywhere else stays: words on both sides of a mark were never one
    run of words in the message."""
    words = clean_quote(quote)
    for mark in _CUT_MARKS:
        if words.endswith(mark):
            return clean_quote(words[: -len(mark)])
    return words


def quote_text(quote: str) -> str:
    """A quote in the form that it is compared in."""
    return normalise(quoted_words(quote))


def _splits_word(text: str, index: int) -> bool:
    """True when `index` falls between two letters or digits of one word."""
    return 0 < index < len(text) and is_word_char(text[index - 1]) and is_word_char(text[index])


def _occurrences(text: str, part: str, start: int) -> Iterator[int]:
    at = text.find(part, start)
    while at >= 0:
        yield at
        at = text.find(part, at + 1)


def holds(text: str, quote: str) -> bool:
    """True when `text` holds `quote` as one run, on word boundaries at both ends."""
    if not quote:
        return False
    return any(
        not _splits_word(text, begin) and not _splits_word(text, begin + len(quote))
        for begin in _occurrences(text, quote, 0)
    )


class MessageIndex:
    """Every line of the conversation by day, for the quote window."""

    def __init__(self, lines: Sequence[Line]) -> None:
        by_day: dict[str, list[Line]] = {}
        for line in lines:
            by_day.setdefault(line.day, []).append(line)
        self._by_day = {day: tuple(found) for day, found in by_day.items()}
        self._compare_form: dict[int, str] = {}

    def compare_form(self, line: Line) -> str:
        """`normalise(line.text)`, worked out once for each line."""
        key = id(line)
        if key not in self._compare_form:
            self._compare_form[key] = normalise(line.text)
        return self._compare_form[key]

    def _day_at(self, day: str, distance: int) -> tuple[Line, ...]:
        try:
            return self._by_day.get(shift_day(day, distance), ())
        except ValueError:
            return ()

    def near(self, day: str, radius: int) -> Iterator[Line]:
        """Lines within `radius` days of `day`, the nearest day first."""
        if not is_day(day):
            return
        yield from self._by_day.get(day, ())
        for distance in range(1, radius + 1):
            yield from self._day_at(day, -distance)
            yield from self._day_at(day, distance)


@dataclass(frozen=True)
class Checked:
    """The moments of one period after checks, and what the checks changed."""

    moments: tuple[Moment, ...] = ()
    quotes_cleared: int = 0
    dates_repaired: int = 0
    dropped: int = 0


@dataclass(frozen=True)
class _Outcome:
    moment: Moment | None
    quote_cleared: bool = False
    date_repaired: bool = False


def _cited_lines(refs: Sequence[str], period_lines: Sequence[Line]) -> tuple[Line, ...]:
    """The lines that the references name, each line one time. Two rows of an
    export can have one id: they are two lines here."""
    cited: list[Line] = []
    for ref in refs:
        line = resolve_ref(ref, period_lines)
        if line is not None and not any(line is known for known in cited):
            cited.append(line)
    return tuple(cited[:MAX_MESSAGE_IDS])


def _settle_date(
    date: str, cited: Sequence[Line], period_days: frozenset[str]
) -> tuple[str | None, bool]:
    if date in period_days:
        return date, False
    if cited:
        return min(line.day for line in cited), True
    return None, False


def _find_quote(wanted: str, cited: Sequence[Line], date: str, index: MessageIndex) -> Line | None:
    for line in cited:
        if holds(index.compare_form(line), wanted):
            return line
    if len(wanted) < MIN_WINDOW_QUOTE_CHARS:
        return None
    for line in index.near(date, QUOTE_WINDOW_DAYS):
        if holds(index.compare_form(line), wanted):
            return line
    return None


def _check_one(
    raw: RawMoment,
    period_lines: Sequence[Line],
    period_days: frozenset[str],
    index: MessageIndex,
) -> _Outcome:
    title = squash(raw.title)[:MAX_TITLE_CHARS]
    if not title or raw.category not in CATEGORIES:
        return _Outcome(None)
    cited = _cited_lines(raw.refs, period_lines)
    date, repaired = _settle_date(raw.date, cited, period_days)
    if date is None:
        return _Outcome(None)

    given = quoted_words(raw.quote)
    quote = word_cut(given, MAX_QUOTE_CHARS)
    wanted = normalise(quote)
    source = _find_quote(wanted, cited, date, index) if wanted else None
    ids = list(dict.fromkeys(line.id for line in cited))
    if source is not None and source.id not in ids:
        ids.append(source.id)

    moment = Moment(
        date=date,
        title=title,
        detail=squash(raw.detail)[:MAX_DETAIL_CHARS],
        category=raw.category,
        score=min(1.0, max(0.0, raw.score)),
        quote=quote if source is not None else "",
        message_ids=tuple(ids),
        quote_verified=source is not None,
        quote_message_id=source.id if source is not None else "",
        quote_speaker=source.label if source is not None else "",
    )
    # a quote that can not be cut to length between words counts as not found too
    cleared = bool(normalise(given)) and source is None
    return _Outcome(moment, quote_cleared=cleared, date_repaired=repaired)


def check_moments(
    raw_moments: Sequence[RawMoment], period_lines: Sequence[Line], index: MessageIndex
) -> Checked:
    """Verify the moments of one period. `index` covers the whole conversation."""
    period_days = frozenset(line.day for line in period_lines)
    outcomes = [_check_one(raw, period_lines, period_days, index) for raw in raw_moments]
    return Checked(
        moments=tuple(outcome.moment for outcome in outcomes if outcome.moment is not None),
        quotes_cleared=sum(outcome.quote_cleared for outcome in outcomes),
        dates_repaired=sum(outcome.date_repaired for outcome in outcomes),
        dropped=sum(outcome.moment is None for outcome in outcomes),
    )
