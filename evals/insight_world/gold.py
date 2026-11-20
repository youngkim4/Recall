"""What the engine must find in the insight world, and the traps it must not fall into.

A `Want` is found when an insight of that kind and person cites at least one
of its labels (any message, when it has none). A `Trap` is sprung when an
insight of that kind and person cites one of its labels (any message, when it
has none). Every other insight is an extra: it is counted, and a person reads
it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Want:
    kind: str
    person: str
    labels: tuple[str, ...] = ()
    why: str = ""


@dataclass(frozen=True)
class Trap:
    kind: str
    person: str
    labels: tuple[str, ...] = ()
    why: str = ""


WANTS: tuple[Want, ...] = (
    Want("drift", "nora", why="weekly for a year and a half, silent since July"),
    Want("balance", "hugo", why="I start five of every six conversations"),
    Want("balance", "quinn", why="Quinn starts nine of every ten"),
    Want("birthday", "iris", why="wished on December 28 two years running"),
    Want("unanswered", "jonah", ("jonah_show", "jonah_ticket"), "asked about the show four days ago"),
    Want("news", "felix", ("felix_job", "felix_move"), "new job and a move to Tucson"),
    Want("news", "mira", ("mira_lisbon",), "moving to Lisbon in January"),
    Want("news", "sol", ("sol_dev", "sol_done"), "a hard time with Dev"),
    Want("plan", "kaya", ("kaya_dinner", "kaya_dinner_push"), "dinner that never got a day"),
    Want("plan", "sol", ("sol_facetime",), "the FaceTime that did not happen"),
    Want("promise", "kaya", ("kaya_recipe",), "the recipe I never sent"),
    Want("promise", "rosa", ("rosa_intro",), "the introduction I never made"),
)

TRAPS: tuple[Trap, ...] = (
    Trap("drift", "greta", why="every two months is her normal"),
    Trap("drift", "pia", why="gone for years: not a nudge"),
    Trap("drift", "felix", why="talked three days ago"),
    Trap("plan", "felix", ("felix_climb", "felix_climb_done"), "the climbing happened"),
    Trap("plan", "leo", ("leo_hike_off",), "the hike was called off"),
    Trap("promise", "leo", ("leo_photos", "leo_photos_sent"), "I sent the photos"),
    Trap("promise", "otto", ("otto_tickets",), "Otto's promise, not mine"),
    Trap("plan", "leo", ("leo_brunch",), "a set day: it happened"),
    Trap("plan", "kaya", ("kaya_call",), "a call time is logistics"),
    Trap("promise", "kaya", ("kaya_call",), "a call time is logistics"),
    Trap("promise", "rosa", ("rosa_call",), "a call happens off the messages"),
    Trap("news", "mira", ("mira_ramen",), "small talk"),
    Trap("news", "tess", ("tess_cat",), "small talk"),
    Trap("unanswered", "tess", why="I answered"),
    Trap("balance", "felix", why="they take turns"),
    *(Trap(kind, "fernbank", why="a sender of codes") for kind in ("drift", "balance", "unanswered", "birthday", "plan", "promise", "news")),
)
