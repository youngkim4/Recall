"""A small hand-written archive with invented people, for unit tests."""

from __future__ import annotations

from chat_engine.fake_archive import FakeArchive, make_message
from chat_engine.labeler import MaskingLabeler

MAYA_PHONE = "+14155550123"
MAYA_EMAIL = "maya.okafor@example.org"
JONAS_PHONE = "+14155550124"
INES_PHONE = "+351912345678"
UNSAVED_PHONE = "+17205550199"
UNSAVED_EMAIL = "rk.unsaved@example.org"
GROUP = "chat100"

SAVED_NAMES = {
    MAYA_PHONE: "Maya Okafor",
    MAYA_EMAIL: "Maya Okafor",
    JONAS_PHONE: "Jonas Weber",
    INES_PHONE: "Inês Duarte",
}

LONG_TEXT = (
    "Okay here is the whole plan for the move. "
    + " ".join(f"Step {n}: carry box number {n} down to the van and tick it off the list." for n in range(1, 31))
    + " The spare key stays under the blue flowerpot."
)

CONVERSATIONS = [
    {"id": MAYA_PHONE, "is_group": False, "name": "", "members": [MAYA_PHONE], "person_id": 1},
    {"id": MAYA_EMAIL, "is_group": False, "name": "", "members": [MAYA_EMAIL], "person_id": 1},
    {"id": JONAS_PHONE, "is_group": False, "name": "", "members": [JONAS_PHONE], "person_id": 2},
    {"id": INES_PHONE, "is_group": False, "name": "", "members": [INES_PHONE], "person_id": 3},
    {"id": UNSAVED_PHONE, "is_group": False, "name": "", "members": [UNSAVED_PHONE], "person_id": None},
    {"id": UNSAVED_EMAIL, "is_group": False, "name": "", "members": [UNSAVED_EMAIL], "person_id": None},
    {
        "id": GROUP,
        "is_group": True,
        "name": "Lisbon crew",
        "members": [MAYA_PHONE, JONAS_PHONE, UNSAVED_PHONE],
        "person_id": None,
    },
]

_ROWS = [
    # Maya, phone thread
    ("a1", MAYA_PHONE, "2021-03-04 18:20:00", "are we still on for tonight", True, ""),
    ("a2", MAYA_PHONE, "2021-03-04 18:22:00", "yes!! the lease got signed btw", False, MAYA_PHONE),
    ("a3", MAYA_PHONE, "2021-03-04 18:23:00", "omg congrats", True, ""),
    ("a4", MAYA_PHONE, "2021-03-04 18:24:00", "we move in on the first of April", False, MAYA_PHONE),
    ("a5", MAYA_PHONE, "2023-07-11 09:00:00", "the landlord wants to renew the lease", False, MAYA_PHONE),
    ("a6", MAYA_PHONE, "2023-07-11 09:02:00", "same rent?", True, ""),
    ("a7", MAYA_PHONE, "2023-07-11 09:05:00", "fifty more a month", False, MAYA_PHONE),
    ("a8", MAYA_PHONE, "2025-12-24 20:00:00", "merry christmas you", True, ""),
    ("a9", MAYA_PHONE, "2025-12-24 20:01:00", LONG_TEXT, False, MAYA_PHONE),
    # Maya, email thread
    ("b1", MAYA_EMAIL, "2022-05-02 08:00:00", "sending the lease scan from my laptop", False, MAYA_EMAIL),
    ("b2", MAYA_EMAIL, "2022-05-02 08:10:00", "got it, thanks", True, ""),
    # Jonas
    ("j1", JONAS_PHONE, "2024-02-10 12:00:00", "sourdough starter is alive", False, JONAS_PHONE),
    ("j2", JONAS_PHONE, "2024-02-10 12:01:00", "you named it??", True, ""),
    ("j3", JONAS_PHONE, "2024-02-10 12:02:00", "its name is Clint Yeastwood", False, JONAS_PHONE),
    ("j4", JONAS_PHONE, "2026-08-01 10:00:00", "sourdough again this weekend?", True, ""),
    # Inês, Portuguese
    ("p1", INES_PHONE, "2024-06-01 21:00:00", "Você já viu o apartamento novo?", False, INES_PHONE),
    ("p2", INES_PHONE, "2024-06-01 21:03:00", "Ainda não, amanhã passo lá", True, ""),
    ("p3", INES_PHONE, "2024-06-02 10:00:00", "O apartamento é ótimo, a cozinha é enorme", False, INES_PHONE),
    # unsaved phone
    ("u1", UNSAVED_PHONE, "2025-01-15 14:00:00", "hey this is Dana from the climbing gym", False, UNSAVED_PHONE),
    ("u2", UNSAVED_PHONE, "2025-01-15 14:05:00", "oh hi! saving your number now", True, ""),
    ("u3", UNSAVED_PHONE, "2025-01-16 08:00:00", "my other line is (720) 555-0199 too", False, UNSAVED_PHONE),
    # unsaved email
    ("e1", UNSAVED_EMAIL, "2025-02-01 11:00:00", "your order of climbing chalk has shipped", False, UNSAVED_EMAIL),
    # group
    ("g1", GROUP, "2019-06-01 09:00:00", "flights to Lisbon are booked", False, MAYA_PHONE),
    ("g2", GROUP, "2019-06-01 09:01:00", "window seat for me please", False, JONAS_PHONE),
    ("g3", GROUP, "2019-06-01 09:02:00", "I call the aisle", True, ""),
    ("g4", GROUP, "2019-06-01 09:03:00", "can someone forward the lease for the Lisbon flat", False, UNSAVED_PHONE),
    ("g5", GROUP, "2019-06-02 10:00:00", "", False, MAYA_PHONE),
]


def tiny_messages() -> list[dict]:
    messages = []
    for message_id, conversation, timestamp, text, from_me, sender in _ROWS:
        kind = "attachment" if message_id == "g5" else "text"
        attachments = ["IMG_0042.HEIC"] if message_id == "g5" else []
        messages.append(
            make_message(
                message_id,
                conversation,
                timestamp,
                text,
                is_from_me=from_me,
                sender=sender,
                kind=kind,
                attachments=attachments,
            )
        )
    return messages


def tiny_archive() -> FakeArchive:
    return FakeArchive(tiny_messages(), CONVERSATIONS)


def tiny_labeler() -> MaskingLabeler:
    return MaskingLabeler(SAVED_NAMES)
