"""A small invented message history, used by importer and query tests.

Three conversations:
- a one-to-one thread with Rowan's phone number
- a one-to-one thread with Rowan's email address (the same person)
- a group "Lake Weekend" with Rowan, Mira and Tessa

Times are UTC. The test time zone is America/Los_Angeles (UTC-8 in winter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from tests.archive.builders import ChatDbBuilder, utc
from tests.archive.typedstream_fixtures import blob

ROWAN_PHONE = "+15550100123"
ROWAN_EMAIL = "rowan@example.com"
MIRA_PHONE = "+15550100456"
TESSA_EMAIL = "tessa@example.com"
GROUP_ID = "chat100200300"
GROUP_TITLE = "Lake Weekend"

ROWAN_CONTACT = {"name": "Rowan Vale", "phones": ["(555) 010-0123"], "emails": ["Rowan@Example.com"]}
MIRA_CONTACT = {"name": "Mira Okafor", "phones": ["555 010 0456"], "emails": []}

SOURCE_ROWS = 18
MESSAGE_ROWS = 12
REACTION_ROWS = 4
SYSTEM_ROWS = 2


@dataclass
class Lake:
    handles: dict[str, int] = field(default_factory=dict)
    chats: dict[str, int] = field(default_factory=dict)
    rowids: dict[str, int] = field(default_factory=dict)
    guids: dict[str, str] = field(default_factory=dict)

    def add(self, builder: ChatDbBuilder, label: str, rowid: int) -> None:
        self.rowids[label] = rowid
        self.guids[label] = builder.guid_of(rowid)


def build_lake(b: ChatDbBuilder) -> Lake:
    """Write the history into `b` and commit. 18 source rows: 12 messages, 4 tapbacks, 2 group events."""
    lake = Lake()
    rowan = lake.handles["rowan_phone"] = b.handle(ROWAN_PHONE)
    rowan_mail = lake.handles["rowan_email"] = b.handle(ROWAN_EMAIL)
    mira = lake.handles["mira"] = b.handle(MIRA_PHONE)
    tessa = lake.handles["tessa"] = b.handle(TESSA_EMAIL)
    phone = lake.chats["phone"] = b.chat(ROWAN_PHONE, members=[rowan])
    email = lake.chats["email"] = b.chat(ROWAN_EMAIL, members=[rowan_mail])
    group = lake.chats["group"] = b.chat(GROUP_ID, display_name=GROUP_TITLE, members=[rowan, mira, tessa])

    # one-to-one, phone number
    lake.add(
        b,
        "saturday",
        b.message(phone, when=utc(2024, 1, 15, 18, 0), text="Are we still on for Saturday?", handle=rowan),
    )
    lake.add(
        b,
        "dinner",
        b.message(phone, when=utc(2024, 1, 15, 18, 1), body=blob("part_attribute"), from_me=True, handle=rowan),
    )
    photo = b.message(phone, when=utc(2024, 1, 15, 18, 2), body=blob("attachment_only"), handle=rowan)
    b.attachment(photo, mime_type="image/heic", transfer_name="IMG_0042.HEIC", total_bytes=2_048_000)
    lake.add(b, "photo", photo)
    view = b.message(
        phone, when=utc(2024, 1, 15, 18, 3), body=blob("attachment_with_caption"), from_me=True, handle=rowan
    )
    b.attachment(view, mime_type="image/jpeg", transfer_name="view.jpeg", total_bytes=512_000)
    lake.add(b, "view", view)
    lake.add(b, "love_dinner", b.tapback(phone, lake.guids["dinner"], 2000, when=utc(2024, 1, 15, 18, 4), handle=rowan))
    lake.add(
        b,
        "like_saturday",
        b.tapback(phone, lake.guids["saturday"], 2001, when=utc(2024, 1, 15, 18, 5), from_me=True, handle=rowan),
    )
    lake.add(
        b,
        "unlike_saturday",
        b.tapback(phone, lake.guids["saturday"], 3001, when=utc(2024, 1, 15, 18, 6), from_me=True, handle=rowan),
    )
    lake.add(b, "long", b.message(phone, when=utc(2024, 1, 16, 2, 0), body=blob("ascii_300"), handle=rowan))

    # one-to-one, email address
    lake.add(b, "laptop", b.message(email, when=utc(2024, 2, 1, 20, 0), text="Sent from my laptop", handle=rowan_mail))
    lake.add(
        b,
        "gotit",
        b.message(
            email,
            when=utc(2024, 2, 1, 20, 5),
            text="Got it",
            from_me=True,
            handle=rowan_mail,
            reply_to=lake.guids["laptop"],
        ),
    )

    # group
    lake.add(
        b, "renamed", b.message(group, when=utc(2024, 3, 1, 17, 0), handle=rowan, item_type=2, group_title=GROUP_TITLE)
    )
    lake.add(b, "canoe", b.message(group, when=utc(2024, 3, 1, 17, 1), text="Who brings the canoe?", handle=mira))
    lake.add(b, "ican", b.message(group, when=utc(2024, 3, 1, 17, 2), text="I can", handle=tessa))
    lake.add(b, "great", b.message(group, when=utc(2024, 3, 1, 17, 3), text="Great", from_me=True))
    lake.add(
        b,
        "fire_great",
        b.tapback(
            group,
            lake.guids["great"],
            2006,
            when=utc(2024, 3, 1, 17, 4),
            handle=tessa,
            prefix="bp:",
            emoji="🔥",
            text=None,
        ),
    )
    lake.add(
        b,
        "jackets",
        b.message(
            group, when=utc(2024, 3, 1, 17, 10), text="Bring life jackets", handle=mira, edited=utc(2024, 3, 1, 17, 12)
        ),
    )
    lake.add(
        b, "unsent", b.message(group, when=utc(2024, 3, 1, 17, 15), handle=tessa, retracted=utc(2024, 3, 1, 17, 16))
    )
    lake.add(
        b,
        "added",
        b.message(
            group, when=utc(2024, 3, 1, 17, 20), handle=rowan, item_type=1, group_action_type=0, other_handle=tessa
        ),
    )
    b.commit()
    return lake


# ---------------------------------------------------------------------------
# "Harbor": the history with hand-computed figures for the query tests.
# Times below are local time in America/Los_Angeles. The clocks change on
# 2024-03-10 at 02:00 (to 03:00).
# ---------------------------------------------------------------------------

HARBOR_ZONE = ZoneInfo("America/Los_Angeles")
HARBOR_GROUP_ID = "chat777000111"
HARBOR_GROUP_TITLE = "Harbor Crew"
HARBOR_EMPTY_ID = "+15550100777"


def la(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=HARBOR_ZONE)


def build_harbor(b: ChatDbBuilder) -> Lake:
    """19 messages in 3 conversations, 2 group events, 7 tapbacks, and one chat with no message."""
    w = Lake()
    rowan = w.handles["rowan_phone"] = b.handle(ROWAN_PHONE)
    rowan_mail = w.handles["rowan_email"] = b.handle(ROWAN_EMAIL)
    mira = w.handles["mira"] = b.handle(MIRA_PHONE)
    tessa = w.handles["tessa"] = b.handle(TESSA_EMAIL)
    nobody = b.handle(HARBOR_EMPTY_ID)
    a = w.chats["a"] = b.chat(ROWAN_PHONE, members=[rowan])
    mail = w.chats["b"] = b.chat(ROWAN_EMAIL, members=[rowan_mail])
    crew = w.chats["g"] = b.chat(HARBOR_GROUP_ID, display_name=HARBOR_GROUP_TITLE, members=[rowan, mira, tessa])
    b.chat(HARBOR_EMPTY_ID, members=[nobody])

    def say(label: str, chat: int, when: datetime, text: str | None, sender: int | None, **more: object) -> int:
        rowid = b.message(chat, when=when, text=text, from_me=sender is None, handle=sender or 0, **more)  # type: ignore[arg-type]
        w.add(b, label, rowid)
        return rowid

    # group "Harbor Crew": 7 messages (g13..g19) and one rename event
    say("g_renamed", crew, la(2023, 12, 31, 23, 0), None, rowan, item_type=2, group_title=HARBOR_GROUP_TITLE)
    say("g13", crew, la(2023, 12, 31, 23, 59, 59), "Happy new year, almost", mira)
    say("g14", crew, la(2024, 1, 1, 0, 0, 0), "Happy new year!", tessa)
    say("g15", crew, la(2024, 1, 1, 0, 0, 30), "Happy new year 🎉", None)
    w.add(b, "g15_like", b.tapback(crew, w.guids["g15"], 2001, when=la(2024, 1, 1, 0, 1), handle=mira))
    w.add(b, "g15_laugh", b.tapback(crew, w.guids["g15"], 2003, when=la(2024, 1, 1, 0, 2), handle=mira))
    w.add(
        b,
        "g15_fire",
        b.tapback(
            crew, w.guids["g15"], 2006, when=la(2024, 1, 1, 0, 3), handle=tessa, prefix="bp:", emoji="🔥", text=None
        ),
    )
    w.add(b, "g15_love", b.tapback(crew, w.guids["g15"], 2000, when=la(2024, 1, 1, 0, 4), from_me=True))
    say("g18", crew, la(2024, 2, 14, 18, 0), "今日は寿司を食べました", mira)
    say("g19", crew, la(2024, 2, 14, 18, 1), "오늘 학교에서 만나요", None)

    # one-to-one with Rowan's phone number: 10 messages (a1..a10) and one event
    say("a1", a, la(2024, 3, 8, 9, 0), "Morning. Coffee later?", rowan)
    say("a2", a, la(2024, 3, 8, 9, 5), "Yes, at ten.", None)
    w.add(b, "a2_love", b.tapback(a, w.guids["a2"], 2000, when=la(2024, 3, 8, 9, 6), handle=rowan))
    w.add(b, "a1_like", b.tapback(a, w.guids["a1"], 2001, when=la(2024, 3, 8, 9, 7), from_me=True))
    w.add(b, "a1_unlike", b.tapback(a, w.guids["a1"], 3001, when=la(2024, 3, 8, 9, 8), from_me=True))
    say("a3", a, la(2024, 3, 8, 21, 15), "That was fun.", rowan)
    say("a4", a, la(2024, 3, 9, 23, 30), "Clocks change tonight.", None)
    say("a5", a, la(2024, 3, 10, 0, 30), "Good night.", rowan)
    say("g16", crew, la(2024, 3, 10, 1, 59, 59), "Last second before the clocks change", mira)
    say("g17", crew, la(2024, 3, 10, 3, 0, 0), "First second after", tessa, edited=la(2024, 3, 10, 3, 1))
    say("a6", a, la(2024, 3, 10, 3, 30), "Can't sleep.", None)
    say("a7", a, la(2024, 3, 10, 12, 0), "Brunch?", rowan)
    say("a8", a, la(2024, 3, 10, 12, 0), "On my way", None)
    photo = say("a9", a, la(2024, 3, 10, 12, 1), None, rowan, body=blob("attachment_only"))
    b.attachment(photo, mime_type="image/heic", transfer_name="IMG_0101.HEIC")
    b.attachment(photo, mime_type="video/quicktime", transfer_name="IMG_0102.MOV")

    # one-to-one with Rowan's email address: 2 messages (b11, b12)
    say("b11", mail, la(2024, 3, 10, 12, 30), "Email thread: sent you the café menu ☕", rowan_mail)
    say("b12", mail, la(2024, 3, 11, 7, 45), "Thanks, déjà vu from last year", None, reply_to=w.guids["b11"])

    say("a10", a, la(2024, 3, 25, 8, 0), "Long time.", None)
    say("a_event", a, la(2024, 3, 25, 8, 5), None, rowan, item_type=4)
    b.commit()
    return w
