"""The cast and the planted messages of the synthetic archive.

Every person, number and address here is invented. The filler lines hold none
of the words the questions ask about, so each planted fact stands alone.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

MAYA_PHONE = "+14155550123"
MAYA_EMAIL = "maya.okafor@example.org"
JONAS = "+14155550124"
PRIYA = "+14155550125"
TOMAS = "+14155550126"
ARUN = "+14155550127"
INES = "+351912345678"
UNSAVED_PHONE = "+17205550199"
UNSAVED_PHONE_TWO = "+13035550142"
UNSAVED_EMAIL = "rk.unsaved@example.org"
LISBON = "chat-lisbon-crew"
FAMILY = "chat-raman-family"
NONAME = "chat-work-two"

SAVED_NAMES = {
    MAYA_PHONE: "Maya Okafor",
    MAYA_EMAIL: "Maya Okafor",
    JONAS: "Jonas Weber",
    PRIYA: "Priya Raman",
    TOMAS: "Tomás Varga",
    ARUN: "Arun Raman",
    INES: "Inês Duarte",
}

# key -> conversation id, as the questions and the facts name them
KEYS = {
    "maya_phone": MAYA_PHONE,
    "maya_email": MAYA_EMAIL,
    "jonas": JONAS,
    "priya": PRIYA,
    "tomas": TOMAS,
    "ines": INES,
    "unsaved_phone": UNSAVED_PHONE,
    "unsaved_email": UNSAVED_EMAIL,
    "lisbon": LISBON,
    "family": FAMILY,
    "noname": NONAME,
}

CONVERSATIONS = [
    {"id": MAYA_PHONE, "is_group": False, "name": "", "members": [MAYA_PHONE], "person_id": 1},
    {"id": MAYA_EMAIL, "is_group": False, "name": "", "members": [MAYA_EMAIL], "person_id": 1},
    {"id": JONAS, "is_group": False, "name": "", "members": [JONAS], "person_id": 2},
    {"id": PRIYA, "is_group": False, "name": "", "members": [PRIYA], "person_id": 3},
    {"id": TOMAS, "is_group": False, "name": "", "members": [TOMAS], "person_id": 4},
    {"id": INES, "is_group": False, "name": "", "members": [INES], "person_id": 5},
    {"id": UNSAVED_PHONE, "is_group": False, "name": "", "members": [UNSAVED_PHONE], "person_id": None},
    {"id": UNSAVED_EMAIL, "is_group": False, "name": "", "members": [UNSAVED_EMAIL], "person_id": None},
    {
        "id": LISBON,
        "is_group": True,
        "name": "Lisbon crew",
        "members": [MAYA_PHONE, JONAS, UNSAVED_PHONE_TWO],
        "person_id": None,
    },
    {"id": FAMILY, "is_group": True, "name": "Raman family", "members": [PRIYA, ARUN], "person_id": None},
    {"id": NONAME, "is_group": True, "name": "", "members": [TOMAS, JONAS], "person_id": None},
]


class Chatter(NamedTuple):
    """Background messages of one conversation."""

    conversation: str
    start: str
    end: str
    active: float  # chance that a day has messages
    low: int
    high: int
    pool: str
    senders: tuple[str, ...]


CHATTER = [
    Chatter(PRIYA, "2019-01-05", "2026-11-14", 0.18, 2, 6, "en", (PRIYA,)),
    Chatter(MAYA_PHONE, "2019-02-01", "2026-11-10", 0.12, 2, 6, "en", (MAYA_PHONE,)),
    Chatter(MAYA_PHONE, "2023-09-08", "2023-09-21", 1.0, 3, 5, "en", (MAYA_PHONE,)),
    Chatter(JONAS, "2019-03-01", "2026-11-12", 0.07, 2, 5, "en", (JONAS,)),
    Chatter(TOMAS, "2022-03-01", "2026-11-13", 0.06, 2, 4, "en", (TOMAS,)),
    Chatter(INES, "2023-01-10", "2026-11-09", 0.05, 2, 4, "pt", (INES,)),
    Chatter(MAYA_EMAIL, "2021-01-01", "2026-06-30", 0.006, 1, 2, "mail", (MAYA_EMAIL,)),
    Chatter(UNSAVED_PHONE, "2025-01-20", "2026-10-30", 0.03, 1, 3, "en", (UNSAVED_PHONE,)),
    Chatter(LISBON, "2019-05-02", "2019-09-15", 0.35, 2, 5, "en", (MAYA_PHONE, JONAS, UNSAVED_PHONE_TWO)),
    Chatter(LISBON, "2023-06-01", "2023-06-20", 0.5, 2, 4, "en", (MAYA_PHONE, JONAS, UNSAVED_PHONE_TWO)),
    Chatter(FAMILY, "2019-01-10", "2026-11-11", 0.05, 2, 4, "en", (PRIYA, ARUN)),
    Chatter(NONAME, "2024-10-01", "2024-10-31", 0.4, 2, 3, "en", (TOMAS, JONAS)),
]

POOLS = {
    "en": [
        "ok sounds good", "running a bit late", "haha no way", "did you eat yet", "call me when you can",
        "on my way", "that was so funny", "see you soon", "can you send the photo", "just got home",
        "what time works for you", "I am so tired today", "thanks for earlier", "let me check my calendar",
        "movie night on friday?", "the traffic is terrible", "good morning!", "sleep well", "coffee tomorrow?",
        "I will bring snacks", "left my umbrella at yours", "lol exactly", "how was the meeting", "miss you",
        "save me a seat", "phone was on silent sorry", "be there in ten", "that sounds perfect",
        "need anything from the shop?", "happy friday", "long day at work", "the weather is so nice",
        "did you see the game", "text me the address", "I owe you one", "what are you up to", "home by seven",
        "sure, why not", "no worries at all", "talk later",
    ],
    "pt": [
        "bom dia!", "tudo bem contigo?", "já almoçaste?", "até logo", "estou a caminho", "que saudades",
        "obrigada pela ajuda", "amanhã falamos", "boa noite, dorme bem", "vou chegar atrasada",
        "que dia tão longo", "combinado!", "liga-me quando puderes", "hoje está muito calor",
    ],
    "mail": [
        "forwarding the tickets for saturday", "see the note below", "can you print this for me",
        "here is the address again", "calendar invite is on its way",
    ],
}

HANDOVER = (
    "Hi, here is the full handover before I go on leave. "
    + " ".join(
        f"Item {n}: the weekly report for region {n} goes out on Thursday, copy the numbers from the shared "
        "sheet, check the totals twice and send it to the regional lead before noon."
        for n in range(1, 13)
    )
    + " Two more things. The projector in room four needs the long cable. "
    "And the spare badge is in the grey cabinet, drawer three. Thanks for covering!"
)


class Planted(NamedTuple):
    key: str
    conversation: str
    timestamp: str
    text: str
    sender: Optional[str]  # None: the owner wrote it
    kind: str = "text"


def _from(sender: str, key: str, conversation: str, timestamp: str, text: str) -> Planted:
    return Planted(key, conversation, timestamp, text, sender)


def _me(key: str, conversation: str, timestamp: str, text: str) -> Planted:
    return Planted(key, conversation, timestamp, text, None)


PLANTED = [
    # Maya, phone: the lease, the cabin, Lisbon, the marathon
    _me("tonight", MAYA_PHONE, "2021-03-04 18:20:00", "are we still on for tonight"),
    _from(MAYA_PHONE, "lease_signed", MAYA_PHONE, "2021-03-04 18:22:00", "yes!! the lease got signed btw"),
    _me("congrats", MAYA_PHONE, "2021-03-04 18:23:00", "omg congrats"),
    _from(MAYA_PHONE, "move_in", MAYA_PHONE, "2021-03-04 18:24:00", "we move in on the first of April"),
    _from(MAYA_PHONE, "lease_renew", MAYA_PHONE, "2023-07-11 09:00:00", "the landlord wants to renew the lease"),
    _me("same_rent", MAYA_PHONE, "2023-07-11 09:02:00", "same rent?"),
    _from(MAYA_PHONE, "fifty_more", MAYA_PHONE, "2023-07-11 09:05:00", "fifty more a month"),
    _from(MAYA_PHONE, "wifi", MAYA_PHONE, "2022-08-19 16:05:00",
          "the wifi password at the cabin is pinecone-42-lantern"),
    _me("wifi_thanks", MAYA_PHONE, "2022-08-19 16:06:00", "saved, thank you"),
    _from(MAYA_PHONE, "restaurant", MAYA_PHONE, "2019-05-28 12:00:00",
          "when we are in Lisbon we have to eat at Taberna do Largo, best grilled octopus ever"),
    _from(MAYA_PHONE, "marathon_1", MAYA_PHONE, "2025-03-02 08:00:00", "I registered for the marathon in october"),
    _from(MAYA_PHONE, "marathon_2", MAYA_PHONE, "2025-06-09 07:10:00", "marathon training starts monday"),
    _from(MAYA_PHONE, "marathon_3", MAYA_PHONE, "2025-08-24 11:30:00", "ran 18 miles, marathon prep is brutal"),
    _from(MAYA_PHONE, "marathon_4", MAYA_PHONE, "2025-09-28 19:00:00", "the marathon is in three weeks!!"),
    _from(MAYA_PHONE, "marathon_5", MAYA_PHONE, "2025-10-19 14:45:00", "finished the marathon in 4:12"),
    _from(MAYA_PHONE, "marathon_6", MAYA_PHONE, "2025-10-20 09:00:00", "never running a marathon again lol"),
    _me("sourdough_maya", MAYA_PHONE, "2024-03-01 13:00:00", "Jonas will not stop talking about his sourdough"),
    # Maya, email
    _from(MAYA_EMAIL, "lease_email", MAYA_EMAIL, "2022-05-02 08:00:00",
          "sending the lease scan from my laptop, the PDF is attached"),
    _me("lease_email_reply", MAYA_EMAIL, "2022-05-02 08:10:00", "got it, thanks"),
    # Jonas: sourdough comes back over the years; what he baked
    _from(JONAS, "sourdough_first", JONAS, "2024-02-10 12:00:00", "sourdough starter is alive"),
    _me("named_it", JONAS, "2024-02-10 12:01:00", "you named it??"),
    _from(JONAS, "starter_name", JONAS, "2024-02-10 12:02:00", "its name is Clint Yeastwood"),
    _from(JONAS, "sourdough_loaf", JONAS, "2024-02-11 09:00:00", "first sourdough loaf tomorrow, wish me luck"),
    _from(JONAS, "sourdough_pancakes", JONAS, "2024-06-15 08:30:00", "sourdough pancakes this morning"),
    _me("sourdough_save", JONAS, "2024-06-15 08:32:00", "save me some sourdough please"),
    _from(JONAS, "sourdough_back", JONAS, "2025-01-05 17:00:00", "Clint survived the holidays, sourdough is back"),
    _me("sourdough_long_live", JONAS, "2025-01-05 17:02:00", "long live the sourdough"),
    _from(JONAS, "sourdough_class", JONAS, "2025-11-20 18:00:00", "teaching a sourdough class at the library"),
    _me("sourdough_guy", JONAS, "2025-11-20 18:03:00", "of course you are, sourdough guy"),
    _me("sourdough_last", JONAS, "2026-08-01 10:00:00", "sourdough again this weekend?"),
    _from(JONAS, "bring_butter", JONAS, "2026-08-01 10:05:00", "always. bring butter"),
    _from(JONAS, "baked_focaccia", JONAS, "2025-07-06 16:00:00", "baked focaccia for the picnic"),
    _from(JONAS, "baked_banana", JONAS, "2026-03-14 15:00:00", "baked banana bread, it is gone already"),
    _from(JONAS, "baked_rye", JONAS, "2026-07-12 16:20:00",
          "baked a rye loaf for the lake trip, crust came out perfect"),
    # Priya: the ferry, and two words about the marathon
    _from(PRIYA, "ferry_vashon", PRIYA, "2022-08-13 10:00:00", "ferry to Vashon is delayed again"),
    _from(PRIYA, "ferry_bainbridge", PRIYA, "2024-08-17 15:30:00", "the ferry to Bainbridge was so worth it"),
    _from(PRIYA, "marathon_priya_1", PRIYA, "2025-03-03 09:00:00", "are you really doing a marathon?"),
    _from(PRIYA, "marathon_priya_2", PRIYA, "2025-10-19 15:00:00", "marathon girl! so proud"),
    # Tomás: one long message
    _from(TOMAS, "handover", TOMAS, "2025-05-30 17:45:00", HANDOVER),
    _me("handover_thanks", TOMAS, "2025-05-30 17:50:00", "all clear, enjoy the time off"),
    # Inês: Portuguese
    _from(INES, "apartment_question", INES, "2024-06-01 21:00:00", "Você já viu o apartamento novo?"),
    _me("apartment_reply", INES, "2024-06-01 21:03:00", "Ainda não, amanhã passo lá"),
    _from(INES, "kitchen", INES, "2024-06-02 10:00:00", "O apartamento é ótimo, a cozinha é enorme"),
    # an unsaved number
    _from(UNSAVED_PHONE, "dana_intro", UNSAVED_PHONE, "2025-01-15 14:00:00",
          "hey this is Dana from the climbing gym"),
    _me("dana_reply", UNSAVED_PHONE, "2025-01-15 14:05:00", "oh hi! saving your number now"),
    _from(UNSAVED_PHONE, "dana_comp", UNSAVED_PHONE, "2025-01-16 08:00:00", "the comp is on saturday at nine"),
    _from(UNSAVED_PHONE, "dana_line", UNSAVED_PHONE, "2025-06-01 12:00:00", "my other line is (720) 555-0199 too"),
    # an unsaved email address
    _from(UNSAVED_EMAIL, "rafael_intro", UNSAVED_EMAIL, "2025-02-01 11:00:00",
          "Hello, this is Rafael from Northwind Outfitters. Your tent order has shipped."),
    _from(UNSAVED_EMAIL, "rafael_arrive", UNSAVED_EMAIL, "2025-02-03 09:00:00",
          "The tent should arrive on Thursday."),
    _me("rafael_thanks", UNSAVED_EMAIL, "2025-02-03 09:30:00", "thank you Rafael"),
    _from(UNSAVED_EMAIL, "rafael_done", UNSAVED_EMAIL, "2025-02-06 16:00:00",
          "Your tent was delivered. Enjoy the trip!"),
    # the Lisbon group
    _from(MAYA_PHONE, "flights", LISBON, "2019-06-01 09:00:00", "flights to Lisbon are booked, I paid for all three"),
    _from(JONAS, "window", LISBON, "2019-06-01 09:01:00", "window seat for me please"),
    _me("aisle", LISBON, "2019-06-01 09:02:00", "I call the aisle"),
    _from(UNSAVED_PHONE_TWO, "hostel", LISBON, "2019-06-01 09:03:00", "can someone forward the hostel address"),
    _from(JONAS, "sourdough_reunion", LISBON, "2023-06-10 10:00:00", "bringing sourdough to the reunion"),
    _me("sourdough_nobody", LISBON, "2023-06-10 10:02:00", "nobody asked for sourdough Jonas"),
    Planted("lisbon_photo", LISBON, "2019-06-21 22:00:00", "", MAYA_PHONE, "attachment"),
]
