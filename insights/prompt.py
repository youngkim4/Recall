"""What the model reads to find plans, promises and news: the rules, the schema and the transcript.

The transcript names each message with a short ref (m1, m2, ...). The model
cites refs, never message ids. The other person is called by the thread name,
and the owner is "Me".
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from insights.thread import Note

SCHEMA_NAME = "relationship_items"
READ_KINDS = ("plan", "promise", "news")

SYSTEM = """You read the recent messages between me and one person. Find what a caring friend should not forget.

Give items of three kinds:

- plan: we both said we wanted to do something together (a meal, a visit, a trip, a long call), but it stayed vague and never got a set day: "we should...", "soon", "sometime", "next week maybe", "let me check and text you". Then nothing more about it.
  Leave out:
  - a plan with a set day or time. People do not text that it happened, so it did. This holds when the day is past, and when it is still ahead.
  - an invite for today, tonight, tomorrow or this weekend that the other person said yes to ("tacos?" "always tacos"). It happened.
  - a plan that happened or was called off.
  - quick logistics: when to call, running late, where to meet, "call you at 8".
- promise: I said I would do something for this person that takes effort: send them a thing, share a file or a recipe, introduce them to someone, return or pay back something, look into something for them. Then no later message shows that I did it.
  Leave out:
  - calls and call times ("i'll call you at 8", "call you later"). A call happens off the messages.
  - a promise of the other person. Only my own promises count.
- news: something big in this person's life that a good friend would ask about later: health, a family member, a job, a move, school or an exam, a relationship, a loss, a baby, a big trip. Leave out small talk and daily plans.

Rules:
- Use only what the messages say. Do not guess.
- One item for each thing. Do not give the same thing twice.
- refs: the refs of the messages that show the item. Cite the message where it was said first, then any later message that keeps it open.
- done: true when a later message shows that it happened, was settled or was called off. For news, true when it is over and needs no check-in.
- summary: one short sentence to me, under 20 words. Use the person's name. Plain words, no quotes.
- An empty list is a good answer when nothing fits."""

SCHEMA: dict[str, Any] = {
    "title": SCHEMA_NAME,
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "summary", "refs", "done"],
                "properties": {
                    "kind": {"type": "string", "enum": list(READ_KINDS)},
                    "summary": {"type": "string"},
                    "refs": {"type": "array", "items": {"type": "string"}},
                    "done": {"type": "boolean"},
                },
            },
        }
    },
}


def ref(place: int) -> str:
    return f"m{place + 1}"


def transcript(name: str, notes: Sequence[Note]) -> str:
    """One line for each note, with a day header when the day changes."""
    lines: list[str] = []
    day = ""
    for place, note in enumerate(notes):
        stamp = note.when.strftime("%Y-%m-%d %a")
        if stamp != day:
            lines.append(f"--- {stamp} ---")
            day = stamp
        who = "Me" if note.from_me else name
        text = " ".join(note.text.split())
        lines.append(f"{ref(place)} {note.when:%H:%M} {who}: {text}")
    return "\n".join(lines)


def user_text(name: str, notes: Sequence[Note], now: datetime) -> str:
    return (
        f"Today is {now:%Y-%m-%d %A}. The person is {name}.\n\n"
        f"<messages>\n{transcript(name, notes)}\n</messages>"
    )
