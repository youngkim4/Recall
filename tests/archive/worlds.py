"""A small invented message history, used by importer and query tests.

Three conversations:
- a one-to-one thread with Rowan's phone number
- a one-to-one thread with Rowan's email address (the same person)
- a group "Lake Weekend" with Rowan, Mira and Tessa

Times are UTC. The test time zone is America/Los_Angeles (UTC-8 in winter).
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    lake.add(b, "saturday", b.message(phone, when=utc(2024, 1, 15, 18, 0), text="Are we still on for Saturday?", handle=rowan))
    lake.add(b, "dinner", b.message(phone, when=utc(2024, 1, 15, 18, 1), body=blob("part_attribute"), from_me=True, handle=rowan))
    photo = b.message(phone, when=utc(2024, 1, 15, 18, 2), body=blob("attachment_only"), handle=rowan)
    b.attachment(photo, mime_type="image/heic", transfer_name="IMG_0042.HEIC", total_bytes=2_048_000)
    lake.add(b, "photo", photo)
    view = b.message(phone, when=utc(2024, 1, 15, 18, 3), body=blob("attachment_with_caption"), from_me=True, handle=rowan)
    b.attachment(view, mime_type="image/jpeg", transfer_name="view.jpeg", total_bytes=512_000)
    lake.add(b, "view", view)
    lake.add(b, "love_dinner", b.tapback(phone, lake.guids["dinner"], 2000, when=utc(2024, 1, 15, 18, 4), handle=rowan))
    lake.add(b, "like_saturday", b.tapback(phone, lake.guids["saturday"], 2001, when=utc(2024, 1, 15, 18, 5), from_me=True, handle=rowan))
    lake.add(b, "unlike_saturday", b.tapback(phone, lake.guids["saturday"], 3001, when=utc(2024, 1, 15, 18, 6), from_me=True, handle=rowan))
    lake.add(b, "long", b.message(phone, when=utc(2024, 1, 16, 2, 0), body=blob("ascii_300"), handle=rowan))

    # one-to-one, email address
    lake.add(b, "laptop", b.message(email, when=utc(2024, 2, 1, 20, 0), text="Sent from my laptop", handle=rowan_mail))
    lake.add(b, "gotit", b.message(email, when=utc(2024, 2, 1, 20, 5), text="Got it", from_me=True, handle=rowan_mail, reply_to=lake.guids["laptop"]))

    # group
    lake.add(b, "renamed", b.message(group, when=utc(2024, 3, 1, 17, 0), handle=rowan, item_type=2, group_title=GROUP_TITLE))
    lake.add(b, "canoe", b.message(group, when=utc(2024, 3, 1, 17, 1), text="Who brings the canoe?", handle=mira))
    lake.add(b, "ican", b.message(group, when=utc(2024, 3, 1, 17, 2), text="I can", handle=tessa))
    lake.add(b, "great", b.message(group, when=utc(2024, 3, 1, 17, 3), text="Great", from_me=True))
    lake.add(b, "fire_great", b.tapback(group, lake.guids["great"], 2006, when=utc(2024, 3, 1, 17, 4), handle=tessa, prefix="bp:", emoji="🔥", text=None))
    lake.add(b, "jackets", b.message(group, when=utc(2024, 3, 1, 17, 10), text="Bring life jackets", handle=mira, edited=utc(2024, 3, 1, 17, 12)))
    lake.add(b, "unsent", b.message(group, when=utc(2024, 3, 1, 17, 15), handle=tessa, retracted=utc(2024, 3, 1, 17, 16)))
    lake.add(b, "added", b.message(group, when=utc(2024, 3, 1, 17, 20), handle=rowan, item_type=1, group_action_type=0, other_handle=tessa))
    b.commit()
    return lake
