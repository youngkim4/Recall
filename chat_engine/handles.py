"""What a handle is, and when two spellings are the same phone number.

A handle is a phone number, an email address or a short code. The same phone
number has many spellings: "+351 912 345 678", "912345678", "(912) 345-678".
Two spellings with a country code are the same number only when all digits
are the same. Without a country code on one side, the national digits decide.

In free text a number can also stand in its local form (the last seven
digits alone), with a 0 after the country code, with digits of another
script, and with any marks between its groups.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Generic, Iterator, Optional, Sequence, TypeVar

EMAIL_MASK = "an unsaved email contact"
UNKNOWN_MASK = "an unsaved contact"
PHONE_MASK = "a number ending in {tail}"
SHORT_CODE_MASK = "the short code {digits}"
GROUP_MASK = "a group chat"

MIN_PHONE_DIGITS = 7
_MIN_SHORT_CODE_DIGITS = 3
_MAX_COUNTRY_DIGITS = 3
_INDEX_DIGITS = 7
_LOCAL_DIGITS = 7  # the form without an area code
_NATIONAL_DIGITS = 10  # a number this long has a shorter local form
_TRUNK = "0"  # written after a country code in some countries: +44 (0)7911 ...
_REMEMBERED = 65_536  # parsed handles kept: every question reads the handles of the whole archive again

_PREFIX_RE = re.compile(r"^(?:(?:imessage|sms|rcs|tel|mailto|p|e)(?:;[-+];|:))+", re.IGNORECASE)
# An address in text is ASCII only. So a letter of another script next to it is not part of it:
# "name@example.org입니다" holds the address "name@example.org".
_EMAIL_BODY = r"[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9\-]{1,63}(?:\.[A-Za-z0-9\-]{1,63}){1,8}"
EMAIL_RE = re.compile(rf"(?<![A-Za-z0-9.%+\-]){_EMAIL_BODY}")
_EMAIL_ONLY_RE = re.compile(_EMAIL_BODY)
# A handle of the archive can be an address in another form: with an apostrophe, with letters of another script.
_LOOSE_EMAIL_RE = re.compile(r"[^\s@]{1,64}@[^\s@]{1,255}\.[^\s@.]{2,63}")
_NUMBER_MARKS_RE = re.compile(r"^[\d\s+().\-]+$")
# digits of any script, with up to five marks between two digits: a space, a dash of any kind, a slash, a line break
PHONE_SPAN_RE = re.compile(r"(?:\+\s?)?\(?\d(?:(?:[^\w]|_){0,5}\d){6,}")
_DASHES = "-\u2010\u2011\u2012\u2013\u2014\u2212"
_INSIDE_A_NUMBER = frozenset(" \t\n\r\u00a0." + _DASHES)  # marks between the groups of one number
_LOCAL_JOINTS = frozenset(["", " ", "\u00a0", ".", *_DASHES])  # between the three and the four digits of a local form

Value = TypeVar("Value")


def strip_prefix(handle: object) -> str:
    """The handle without a service prefix such as "tel:" or "iMessage;-;"."""
    return _stripped(str(handle or ""))


@lru_cache(maxsize=_REMEMBERED)
def _stripped(text: str) -> str:
    return _PREFIX_RE.sub("", text.strip()).strip("; \t")


@dataclass(frozen=True)
class Phone:
    digits: str
    international: bool  # written with a leading "+"

    @property
    def tail(self) -> str:
        return self.digits[-4:]


def parse_phone(handle: object) -> Optional[Phone]:
    """The phone number a handle spells, or None when it is not one."""
    return _parsed(str(handle or ""))


@lru_cache(maxsize=_REMEMBERED)
def _parsed(raw: str) -> Optional[Phone]:
    text = _stripped(raw)
    digits = re.sub(r"\D", "", text)
    if len(digits) < MIN_PHONE_DIGITS or not _NUMBER_MARKS_RE.fullmatch(text):
        return None
    return Phone(digits, text.startswith("+"))


def _national(phone: Phone) -> str:
    return phone.digits if phone.international else phone.digits.lstrip("0")


def same_phone(one: Phone, other: Phone) -> bool:
    if one.international and other.international:
        return one.digits == other.digits
    longer, shorter = (one, other) if len(one.digits) >= len(other.digits) else (other, one)
    national = _national(shorter)
    extra = len(longer.digits) - len(national)
    return len(national) >= MIN_PHONE_DIGITS and extra <= _MAX_COUNTRY_DIGITS and longer.digits.endswith(national)


def handle_kind(handle: object) -> str:
    """ "email", "phone", "short_code" or "other" (a group id, a name, nothing)."""
    return _kind(str(handle or ""))


@lru_cache(maxsize=_REMEMBERED)
def _kind(raw: str) -> str:
    text = _stripped(raw)
    if _EMAIL_ONLY_RE.fullmatch(text) or _LOOSE_EMAIL_RE.fullmatch(text):
        return "email"
    if parse_phone(text) is not None:
        return "phone"
    digits = re.sub(r"\D", "", text)
    if len(digits) >= _MIN_SHORT_CODE_DIGITS and _NUMBER_MARKS_RE.fullmatch(text):
        return "short_code"
    return "other"


def email_key(handle: object) -> str:
    return strip_prefix(handle).lower()


def is_plain_address(key: str) -> bool:
    """True when the address has the usual ASCII form, so the scan of a text finds it."""
    return _EMAIL_ONLY_RE.fullmatch(key) is not None


def looks_like_handle(text: object) -> bool:
    """True when the whole text is a phone number or an email address."""
    return handle_kind(text) in ("email", "phone")


def mask_handle(handle: object) -> str:
    """The words that stand for a handle nobody saved. Never the handle itself.

    A short code is a public number of a sender, so it is shown.
    """
    kind = handle_kind(handle)
    if kind == "email":
        return EMAIL_MASK
    phone = parse_phone(handle)
    if phone is not None:
        return PHONE_MASK.format(tail=phone.tail)
    if kind == "short_code":
        return SHORT_CODE_MASK.format(digits=re.sub(r"\D", "", strip_prefix(handle)))
    return UNKNOWN_MASK


def is_mask(label: object) -> bool:
    """True when a label is one of the masks above, not a real name."""
    text = str(label or "")
    starts = (PHONE_MASK.format(tail=""), SHORT_CODE_MASK.format(digits=""), GROUP_MASK)
    return text in (EMAIL_MASK, UNKNOWN_MASK) or text.startswith(starts)


def handle_spans(text: object) -> list[tuple[int, int]]:
    """Where a phone number or an email address stands in free text."""
    source = str(text or "")
    spans = [match.span() for match in EMAIL_RE.finditer(source)]
    spans.extend(match.span() for match in PHONE_SPAN_RE.finditer(source))
    return sorted(spans)


class PhoneIndex(Generic[Value]):
    """Phone numbers with a value each, found by any spelling of the number."""

    def __init__(self) -> None:
        self._by_ending: dict[str, list[tuple[Phone, Value]]] = {}

    def __bool__(self) -> bool:
        return bool(self._by_ending)

    def add(self, phone: Phone, value: Value) -> None:
        bucket = self._by_ending.setdefault(phone.digits[-_INDEX_DIGITS:], [])
        if not any(known == phone for known, _ in bucket):
            bucket.append((phone, value))

    def get(self, phone: Phone) -> Optional[Value]:
        for known, value in self._by_ending.get(phone.digits[-_INDEX_DIGITS:], []):
            if same_phone(known, phone):
                return value
        return None

    def has_ending(self, digits: str) -> bool:
        """True when a known number ends with these digits (seven or more)."""
        return len(digits) >= _INDEX_DIGITS and bool(self._by_ending.get(digits[-_INDEX_DIGITS:]))

    def ending_at(self, digits: str, end: int) -> Iterator[tuple[Phone, Value]]:
        """Known numbers whose last digits are the digits that end at `end`."""
        if end >= _INDEX_DIGITS:
            yield from self._by_ending.get(digits[end - _INDEX_DIGITS:end], [])


def breaks_before(gaps: Sequence[str], index: int) -> bool:
    """True when the digit at `index` starts something new: it does not go on the number before it."""
    return index == 0 or any(char not in _INSIDE_A_NUMBER and char != ")" for char in gaps[index])


def _breaks_after(gaps: Sequence[str], index: int) -> bool:
    return index == len(gaps) or any(char not in _INSIDE_A_NUMBER for char in gaps[index])


def match_in_run(
    index: PhoneIndex[Value], digits: str, international: bool, gaps: Sequence[str] = ()
) -> list[tuple[int, int, Value]]:
    """Known numbers inside one run of digits, left to right, never overlapping.

    Each match is (start, end, value) in digit positions. `international` says
    that the run starts with "+": then digits before a match are a country
    code and must agree with the known number. `gaps[i]` is the text between
    digit i-1 and digit i: it tells one number from two numbers side by side.
    """
    gaps = gaps or ("",) * len(digits)
    found: list[tuple[int, int, Value]] = []
    floor = 0
    for end in range(_INDEX_DIGITS, len(digits) + 1):
        best: Optional[tuple[int, Value]] = None
        for phone, value in index.ending_at(digits, end):
            start = _match_start(phone, digits, end, floor, international and not found, gaps)
            if start is not None and (best is None or start < best[0]):
                best = (start, value)
        if best is not None:
            found.append((best[0], end, best[1]))
            floor = end
    return found


def _spellings(phone: Phone) -> list[tuple[str, str]]:
    """Digit runs that can stand for the number in text, with the country digits each one leaves out."""
    full = phone.digits
    spellings = [(full, "")]
    if phone.international:
        for dropped in range(1, _MAX_COUNTRY_DIGITS + 1):
            if len(full) - dropped >= MIN_PHONE_DIGITS:
                spellings.append((full[dropped:], full[:dropped]))
    elif full.startswith("0") and len(full.lstrip("0")) >= MIN_PHONE_DIGITS:
        spellings.append((full.lstrip("0"), ""))
    return spellings


def _match_start(
    phone: Phone, digits: str, end: int, floor: int, strict_country: bool, gaps: Sequence[str]
) -> Optional[int]:
    """Where `phone` starts when it ends at `end`, or None when it is not there."""
    for spelled, country in _spellings(phone):
        start = end - len(spelled)
        if start < floor or digits[start:end] != spelled:
            continue
        lead = "" if breaks_before(gaps, start) else digits[floor:start]
        if strict_country and country and lead and lead not in (country, country + _TRUNK):
            continue  # a "+" and another country code: a different number
        return start
    start = end - _LOCAL_DIGITS
    if start >= floor and _is_local_form(phone, digits, start, end, gaps):
        return start
    return None


def _is_local_form(phone: Phone, digits: str, start: int, end: int, gaps: Sequence[str]) -> bool:
    """True when digits[start:end] are the last seven digits of the number, written as a number of their own.

    They stand alone (no digit of the same number before or after them), and
    as one group or as three and four: 5550199, 555-0199. So "303 555 0199"
    is another number, and "3/14/2024" is a date.
    """
    if len(phone.digits) < _NATIONAL_DIGITS or digits[start:end] != phone.digits[-_LOCAL_DIGITS:]:
        return False
    if not (breaks_before(gaps, start) and _breaks_after(gaps, end)):
        return False
    return all(
        gaps[i] == "" or (i == start + 3 and gaps[i] in _LOCAL_JOINTS) for i in range(start + 1, end)
    )
