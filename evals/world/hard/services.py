"""Messages from senders nobody saves: sign-in codes, deliveries, reminders and transit alerts.

Every line of `fernbank` and `quill` is a sign-in code. Sam answers only the
dental reminders ("C" to confirm). The draw is seeded per sender.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import date, timedelta

from evals.world.script import Item, Line, guid_for, local

START, END = date(2020, 1, 1), date(2025, 12, 19)


def _code(draw: random.Random) -> str:
    return f"{draw.randrange(0, 1_000_000):06d}"


def _parcel(draw: random.Random) -> tuple[str, ...]:
    order = draw.randrange(100_000, 999_999)
    if draw.random() < 0.5:
        return (f"Parcel Pal: order {order} is out for delivery today",)
    return (f"Parcel Pal: order {order} was delivered to your front door. Photo: https://example.org/p/{order}",)


def _fernbank(draw: random.Random) -> tuple[str, ...]:
    return (f"Fernbank: your sign-in code is {_code(draw)}. It expires in 10 minutes. Do not share it.",)


def _quill(draw: random.Random) -> tuple[str, ...]:
    return (f"{_code(draw)} is your Quill verification code.",)


def _dental(draw: random.Random) -> tuple[str, ...]:
    hour = draw.choice(("9:00", "10:30", "2:15", "4:00"))
    return (f"Brightsmile Dental: reminder of your cleaning on {draw.choice(('monday', 'tuesday', 'thursday'))} "
            f"at {hour}. Reply C to confirm", "C")  # fmt: skip


def _pharmacy(draw: random.Random) -> tuple[str, ...]:
    return (draw.choice(("Corner Pharmacy: your prescription is ready for pickup",
                         "Corner Pharmacy: your refill is due. Reply R to refill")),)  # fmt: skip


def _transit(draw: random.Random) -> tuple[str, ...]:
    line = draw.choice(("blue", "green", "orange"))
    return (f"BayLink alert: {draw.randint(5, 25)} min delays on the {line} line",)


SENDERS: dict[str, tuple[float, Callable[[random.Random], tuple[str, ...]]]] = {
    "parcel": (0.09, _parcel),
    "fernbank": (0.07, _fernbank),
    "quill": (0.06, _quill),
    "dental": (0.012, _dental),
    "pharmacy": (0.03, _pharmacy),
    "transit": (0.05, _transit),
}


def service_lines() -> tuple[Item, ...]:
    items: list[Item] = []
    for sender, (rate, make) in SENDERS.items():
        draw = random.Random(f"service|{sender}")
        day = START
        while day <= END:
            if draw.random() < rate:
                clock = local(f"{day.isoformat()} {draw.randint(7, 21):02d}:{draw.randint(0, 59):02d}")
                for place, text in enumerate(make(draw)):
                    who = "me" if text in ("C",) else sender
                    when = clock + timedelta(minutes=place * draw.randint(2, 90))
                    items.append(Line(sender, when, who, text, guid_for("service", sender, day, place)))
            day += timedelta(days=1)
    return tuple(items)
