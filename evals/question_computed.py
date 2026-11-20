"""Questions whose answers code counts from the world: counts, patterns over time, spans of days.

No number here is typed by hand. A change of a scene or of the filler
changes the gold answer with it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from evals.question_types import CountQuery, Question
from evals.world import World
from evals.world.people import conversation_id
from evals.world.script import Line

# a count question cites a handful of the messages it counts, not all of them
SAMPLE_GUIDS = 6


def spoken(day: str) -> str:
    """ "2023-02-18" as "February 18, 2023"."""
    value = date.fromisoformat(day)
    return f"{value:%B} {value.day}, {value.year}"


def pick(
    made: World,
    chat: str | None = None,
    *,
    mine: bool | None = None,
    since: str | None = None,
    until: str | None = None,
    holding: str | None = None,
) -> tuple[Line, ...]:
    """The lines of the world that fit, oldest first. `holding` is matched without case."""
    wanted = holding.casefold() if holding else None
    return tuple(
        line
        for line in made.lines
        if (chat is None or line.conversation == chat)
        and (mine is None or line.from_me == mine)
        and (since is None or line.day >= since)
        and (until is None or line.day <= until)
        and (wanted is None or wanted in line.text.casefold())
    )


def sample(lines: tuple[Line, ...]) -> tuple[str, ...]:
    """A few guids from the start and the end of a run of lines."""
    if len(lines) <= SAMPLE_GUIDS:
        return tuple(line.guid for line in lines)
    half = SAMPLE_GUIDS // 2
    return tuple(line.guid for line in (*lines[:half], *lines[-half:]))


def _count(
    made: World, qid: str, text: str, chat: str, what: str, *, mine: bool | None = None,
    since: str | None = None, until: str | None = None, holding: str | None = None,
) -> Question:  # fmt: skip
    found = pick(made, chat, mine=mine, since=since, until=until, holding=holding)
    query = CountQuery((conversation_id(chat),), mine, since, until, holding)
    number = len(found)
    return Question(
        id=qid, kind="count", text=text, gold=f"{number} {what}.", key_facts=(f"{number} {what}",),
        gold_guids=sample(found), expected_count=number, count_query=query,
    )  # fmt: skip


def count_questions(made: World) -> tuple[Question, ...]:
    return (
        _count(made, "count_goose", "How many goose updates has Theo sent me?", "theo", "goose updates",
               mine=False, holding="goose update"),
        _count(made, "count_wren_may", "How many messages did Wren and I exchange in May 2024?", "wren",
               "messages in May 2024", since="2024-05-01", until="2024-05-31"),
        _count(made, "count_juni_2024", "How many messages did Juniper send me in 2024?", "juni",
               "messages from Juniper in 2024", mine=False, since="2024-01-01", until="2024-12-31"),
        _count(made, "count_recruiter", "How many messages did the Kestrel & Finch recruiter send me?", "ada",
               "messages from the recruiter", mine=False),
        _count(made, "count_to_marisol", "How many messages have I sent Marisol in total?", "mari",
               "messages sent to Marisol", mine=True),
        _count(made, "count_family_2024", "How many messages were in the Fennimore Fam group chat in 2024?",
               "family", "messages in the family chat in 2024", since="2024-01-01", until="2024-12-31"),
        _count(made, "count_dash", "How many messages did Dash and I exchange in all?", "dash",
               "messages with Dash"),
    )  # fmt: skip


def _per_month(made: World, chat: str, since: str, until: str, months: int) -> int:
    return round(len(pick(made, chat, since=since, until=until)) / months)


def _wren_after_move(made: World) -> Question:
    before = _per_month(made, "wren", "2024-01-01", "2024-12-31", 12)
    after = _per_month(made, "wren", "2025-03-01", "2025-12-31", 10)
    return Question(
        id="pattern_wren_move", kind="pattern",
        text="How did my texting with Wren change after we moved in together?",
        gold=f"We texted much less after moving in on 2025-03-01: about {before} messages a month in 2024, "
        f"about {after} a month from March 2025.",
        key_facts=("much less texting after moving in together in March 2025",
                   f"about {before} a month in 2024 against about {after} a month after the move"),
        gold_guids=made.guids(("apartment_found", "move_in")),
    )  # fmt: skip


def _by_year(made: World, chat: str, mine: bool | None = None) -> dict[int, int]:
    return {year: len(pick(made, chat, mine=mine, since=f"{year}-01-01", until=f"{year}-12-31")) for year in (2023, 2024, 2025)}


def _juniper_fade(made: World) -> Question:
    years = _by_year(made, "juni")
    return Question(
        id="pattern_juniper", kind="pattern", text="How did my friendship with Juniper change from 2023 to 2025?",
        gold=f"It faded: {years[2023]} messages in 2023, {years[2024]} in 2024 and {years[2025]} in 2025 "
        "(one birthday exchange, the last on 2025-02-02).",
        key_facts=("very frequent in 2023", "much less in 2024", "only a birthday exchange in 2025"),
        gold_guids=made.guids(("camera_returned", "juni_forever", "juni_last")),
    )  # fmt: skip


def _theo_move(made: World) -> Question:
    before = _per_month(made, "theo", "2023-01-01", "2023-07-31", 7)
    after = _per_month(made, "theo", "2023-09-01", "2025-11-30", 27)
    return Question(
        id="pattern_theo", kind="pattern", text="How did my texting with Theo change after he moved?",
        gold=f"Less often but steady: about {before} messages a month before the move on 2023-08-19, about "
        f"{after} a month after it. The goose jokes went on.",
        key_facts=("texted less after he moved to Denver in August 2023", "still in touch after the move",
                   f"about {before} a month before against about {after} after"),
        gold_guids=made.guids(("theo_moves", "theo_denver", "goose_last")),
    )  # fmt: skip


def _dash_slowdown(made: World) -> Question:
    months = {name: len(pick(made, "dash", since=f"2023-{number}-01", until=f"2023-{number}-31"))
              for name, number in (("January", "01"), ("February", "02"), ("March", "03"))}  # fmt: skip
    return Question(
        id="pattern_dash", kind="pattern", text="How did my texting with Dash change over time?",
        gold=f"Busy at the start, then it slowed: {months['January']} messages in January 2023, "
        f"{months['February']} in February, {months['March']} in March, and it ended on 2023-03-10.",
        key_facts=("most messages in January 2023", "slowed in February 2023", "ended in March 2023"),
        gold_guids=made.guids(("dash_first", "dash_ask", "dash_last")),
    )  # fmt: skip


def _family_peak(made: World) -> Question:
    years = _by_year(made, "family")
    busiest = max(years, key=lambda year: years[year])
    return Question(
        id="pattern_family_year", kind="pattern", text="Which year was the family group chat busiest?",
        gold=f"{busiest}: {years[2023]} messages in 2023, {years[2024]} in 2024, {years[2025]} in 2025.",
        key_facts=(f"{busiest} was the busiest year",),
        gold_guids=made.guids(("mom60_plan", "mom60_dinner", "family_offer")),
    )  # fmt: skip


def _marisol_after(made: World) -> Question:
    before = len(pick(made, "mari", until="2024-09-27"))
    after = len(pick(made, "mari", since="2024-09-28"))
    return Question(
        id="pattern_marisol", kind="pattern", text="How did my messages with Marisol change after I left Bellweather?",
        gold=f"They almost stopped: {before} messages up to my last day on 2024-09-27, {after} after it.",
        key_facts=("almost no messages after my last day in September 2024",),
        gold_guids=made.guids(("mari_last_day", "mari_after")),
    )  # fmt: skip


def _last_goose(made: World) -> Question:
    last = pick(made, "theo", mine=False, holding="goose update")[-1]
    return Question(
        id="last_goose", kind="first_last", text="When did Theo send his most recent goose update?",
        gold=f"{last.day} ({spoken(last.day)}).", key_facts=(spoken(last.day),), gold_guids=(last.guid,),
    )  # fmt: skip


def _visit_span(made: World) -> Question:
    moved, visited = made.labelled("theo_moves"), made.labelled("theo_visit")
    days = (date.fromisoformat(visited.day) - date.fromisoformat(moved.day)).days
    months = round(days / 30.44)
    return Question(
        id="hop_theo_visit", kind="multi_hop", text="How long had Theo lived in Denver when I visited him?",
        gold=f"About {months} months ({days} days): he moved on {moved.day} and I arrived on {visited.day}.",
        key_facts=(f"about {months} months", f"moved {spoken(moved.day)}", f"visited {spoken(visited.day)}"),
        gold_guids=(moved.guid, visited.guid),
    )  # fmt: skip


PATTERNS: tuple[Callable[[World], Question], ...] = (
    _wren_after_move, _juniper_fade, _theo_move, _dash_slowdown, _family_peak, _marisol_after,
)


def computed_questions(made: World) -> tuple[Question, ...]:
    return (
        *count_questions(made),
        *(make(made) for make in PATTERNS),
        _last_goose(made),
        _visit_span(made),
    )
