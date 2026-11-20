"""The questions of the hard world: the written ones of `hard_static`, and the ones code answers.

Code answers every number here: counts across conversations, who was texted
most in a period, spans between days, and the figures of each year in a long
story. A change of the traffic changes the gold answer with it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import date

from evals.hard_static import HARD_STATIC
from evals.question_computed import pick, sample, spoken
from evals.question_types import CountQuery, Question, from_spec
from evals.world import World
from evals.world.hard.cast import CODE_SENDERS, GROUPS, SERVICES
from evals.world.people import Chat

YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
BIG_GROUP = 1000
_SERVICE_KEYS = frozenset(key for key, _handle in SERVICES)


def _name(made: World, key: str) -> str:
    person = next(person for person in made.people if person.key == key)
    return person.saved_as or "a number nobody saved"


def _chat_name(chat: Chat) -> str:
    return chat.title or "an untitled group"


def _people_chats(made: World) -> tuple[Chat, ...]:
    """One-to-one chats with a person: no group, no sender of codes or deliveries."""
    return tuple(chat for chat in made.chats if not chat.is_group and chat.key not in _SERVICE_KEYS)


def _ranked(made: World, chats: tuple[Chat, ...], since: str, until: str) -> list[tuple[Chat, int]]:
    counts = Counter(line.conversation for line in pick(made, since=since, until=until))
    return sorted(((chat, counts[chat.key]) for chat in chats), key=lambda pair: -pair[1])


def _top_person(made: World, qid: str, text: str, since: str, until: str, anchor: tuple[str, ...]) -> Question:
    (top, most), (second, next_most) = _ranked(made, _people_chats(made), since, until)[:2]
    name = _name(made, top.members[0])
    return Question(
        id=qid, kind="temporal", text=text,
        gold=f"{name}: {most} messages from {since} to {until}; next was {_name(made, second.members[0])} with "
        f"{next_most}.",
        key_facts=(name,), gold_guids=anchor or sample(pick(made, top.key, since=since, until=until)),
    )  # fmt: skip


def _month_of(made: World, label: str) -> tuple[str, str]:
    day = date.fromisoformat(made.labelled(label).day)
    following = date(day.year + day.month // 12, day.month % 12 + 1, 1)
    return day.replace(day=1).isoformat(), date.fromordinal(following.toordinal() - 1).isoformat()


def _temporal(made: World) -> tuple[Question, ...]:
    since, until = _month_of(made, "linden_movein")
    moved_in, moved_out = made.labelled("fulton_movein"), made.labelled("fulton_out")
    days = (date.fromisoformat(moved_out.day) - date.fromisoformat(moved_in.day)).days
    return (
        _top_person(made, "temp_move_month", "Which person did I text the most in the month I moved into Linden St?",
                    since, until, made.guids(("linden_movein",))),
        _top_person(made, "temp_2021_top", "Which person did I text the most in 2021?", "2021-01-01", "2021-12-31", ()),
        Question(
            id="temp_fulton_span", kind="temporal", text="How long did I live at 88 Fulton Ave?",
            gold=f"About {days / 365.25:.1f} years ({days} days): from {moved_in.day} to {moved_out.day}.",
            key_facts=(f"about {round(days / 365.25 * 2) / 2:g} years", f"from {spoken(moved_in.day)}",
                       f"to {spoken(moved_out.day)}"),
            gold_guids=(moved_in.guid, moved_out.guid),
        ),
    )  # fmt: skip


def _aggregate_count(made: World, qid: str, text: str, what: str, query: CountQuery, found: tuple) -> Question:
    return Question(
        id=qid, kind="aggregate", text=text, gold=f"{len(found)} {what}.", key_facts=(f"{len(found)} {what}",),
        gold_guids=sample(found), expected_count=len(found), count_query=query,
    )  # fmt: skip


def _sent_by(made: World, keys: tuple[str, ...], since: str | None = None, until: str | None = None) -> dict[str, int]:
    lines = pick(made, since=since, until=until)
    return {key: sum(1 for line in lines if line.sender == key) for key in keys}


def _aggregate(made: World) -> tuple[Question, ...]:
    totals = Counter(line.conversation for line in made.lines)
    big = [chat for chat in GROUPS if totals[chat.key] > BIG_GROUP]
    in_2023 = Counter(line.conversation for line in pick(made, since="2023-01-01", until="2023-12-31"))
    busiest = max(GROUPS, key=lambda chat: in_2023[chat.key])
    by_2024 = _sent_by(made, ("raff", "odalys"), "2024-01-01", "2024-12-31")
    more = max(by_2024, key=lambda key: by_2024[key])
    mayas = _sent_by(made, ("maya", "mayao"))
    move_day = made.labelled("linden_movein").day
    senders = sorted({line.sender for line in pick(made, since=move_day, until=move_day)} - {"me"} - _SERVICE_KEYS)
    codes = tuple(line for key in CODE_SENDERS for line in pick(made, key, since="2025-01-01", until="2025-12-31"))
    code_ids = tuple(made.chat_id(key) for key in CODE_SENDERS)
    return (
        Question(
            id="agg_big_groups", kind="aggregate", text=f"How many of my group chats have more than {BIG_GROUP:,} messages?",
            gold=f"{len(big)}: " + ", ".join(f"{_chat_name(chat)} ({totals[chat.key]:,})" for chat in big) + ".",
            key_facts=(f"{len(big)} group chats",), gold_guids=sample(made.of(big[0].key)),
        ),
        _aggregate_count(made, "agg_boba", "How many messages in all my chats mention boba?", "messages mention boba",
                         CountQuery(text="boba"), pick(made, holding="boba")),
        Question(
            id="agg_raff_odalys_2024", kind="aggregate",
            text="Who sent me more messages in 2024, counting every chat: Rafferty or Odalys?",
            gold=f"{_name(made, more)}: Rafferty sent {by_2024['raff']}, Odalys sent {by_2024['odalys']}.",
            key_facts=(_name(made, more).split()[0],), gold_guids=sample(pick(made, more, since="2024-01-01", until="2024-12-31")),
        ),
        _aggregate_count(made, "agg_codes_2025", "How many sign-in or verification codes did I get by text in 2025?",
                         "codes", CountQuery(code_ids, None, "2025-01-01", "2025-12-31"), codes),
        Question(
            id="agg_busiest_group_2023", kind="aggregate", text="Which group chat was the busiest in 2023?",
            gold=f"{_chat_name(busiest)}, with {in_2023[busiest.key]:,} messages in 2023.",
            key_facts=(_chat_name(busiest),), gold_guids=sample(pick(made, busiest.key, since="2023-01-01", until="2023-12-31")),
        ),
        Question(
            id="agg_movein_people", kind="aggregate", text="How many different people texted me on the day I moved into Linden St?",
            gold=f"{len(senders)} people on {move_day}.", key_facts=(f"{len(senders)} people",),
            gold_guids=made.guids(("linden_movein",)),
        ),
        Question(
            id="agg_mayas", kind="aggregate", text="How many messages has each Maya sent me, counting every chat?",
            gold=f"Maya Lindqvist sent {mayas['maya']:,}; Maya Ostrowski sent {mayas['mayao']:,}.",
            key_facts=(f"Maya Lindqvist: {mayas['maya']:,}", f"Maya Ostrowski: {mayas['mayao']:,}"),
            gold_guids=made.guids(("maya_half", "mayao_intro")),
        ),
    )  # fmt: skip


def _by_year(made: World, chat: str) -> str:
    counts = {year: len(pick(made, chat, since=f"{year}-01-01", until=f"{year}-12-31")) for year in YEARS}
    return ", ".join(f"{year}: {count}" for year, count in counts.items())


def _story(made: World, qid: str, text: str, chat: str, summary: str, facts: tuple[str, ...], labels: tuple[str, ...]) -> Question:
    return Question(
        id=qid, kind="long_story", text=text, gold=f"{summary} Messages by year: {_by_year(made, chat)}.",
        key_facts=facts, gold_guids=made.guids(labels),
    )  # fmt: skip


STORIES: tuple[Callable[[World], Question], ...] = (
    lambda made: _story(
        made, "story_raff", "How did my friendship with Rafferty change from 2020 to 2025?", "raff",
        "Very close and constant until he moved to Lisbon in November 2022; rare while he was abroad; close again "
        "after he came back to Oakland in April 2025.",
        ("constant contact until late 2022", "quiet while he lived in Lisbon", "close again after April 2025"),
        ("van", "raff_lisbon", "raff_back")),
    lambda made: _story(
        made, "story_petra", "How did things with Petra go over the years?", "petra",
        "Roommates at 88 Fulton Ave from March 2020 to September 2022, with friction over the dishes in June 2021 "
        "that a chore chart solved; after I moved out we texted rarely but stayed friendly.",
        ("roommates at 88 Fulton Ave 2020 to 2022", "friction over the dishes in 2021", "rare but friendly after 2022"),
        ("fulton_movein", "dishes", "fulton_out")),
    lambda made: _story(
        made, "story_crew", "How did the chicago crew group chat change over the years?", "crew",
        "Busiest in 2020 and 2021 (the Chicago trip), quieter from 2022 to 2024, and busy again in 2025 around "
        "my 30th and the reunion.",
        ("busiest in 2020 and 2021", "quieter from 2022 to 2024", "busy again in 2025"),
        ("chi21_plan", "maya_half", "tavola")),
    lambda made: _story(
        made, "story_kip", "How did my texting with Kip change over time?", "kip",
        "Steady but light from 2020 to 2022, nothing at all in 2023, then he came back in January 2024 saying he "
        "had had a rough year, and we have texted since.",
        ("steady from 2020 to 2022", "silent for all of 2023", "back in touch from January 2024"),
        ("chi21_plan", "kip_back")),
)  # fmt: skip


def hard_questions(made: World) -> tuple[Question, ...]:
    """Every question that only the hard world asks."""
    return (
        *(from_spec(made, spec) for spec in HARD_STATIC),
        *_temporal(made),
        *_aggregate(made),
        *(make(made) for make in STORIES),
    )
