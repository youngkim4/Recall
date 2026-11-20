"""The people and conversations that the hard world adds to the small one. All invented.

Numbers are +1 <area> 555-01xx; no real line uses them. Two people share the
first name Maya on purpose: Maya Lindqvist (a college friend) and Maya
Ostrowski (a coworker from October 2024). Nobody else shares a first name.

Not saved: the Larkspur recruiter (Delphine), the two older landlords
(Harlan, Okoro), every third acquaintance, and the senders of codes and
deliveries.
"""

from __future__ import annotations

from evals.world.people import CHATS, CONTACTS, PEOPLE, Chat, Person

# people who speak in the written scenes: their keys are letters only
NAMED: tuple[tuple[str, str, str], ...] = (
    ("maya", "+16285550101", "Maya Lindqvist"),
    ("mayao", "+16285550102", "Maya Ostrowski"),
    ("raff", "+16285550103", "Rafferty Dunmore"),
    ("odalys", "+16285550104", "Odalys Brenner"),
    ("kip", "+16285550105", "Kip Achterberg"),
    ("petra", "+16285550106", "Petra Vossberg"),
    ("benedikt", "+16285550107", "Benedikt Saar"),
    ("bram", "+16285550108", "Bram Toller"),
    ("yusuf", "+16285550109", "Yusuf Delacroix-Hale"),
    ("wendell", "+16285550110", "Wendell Ambrose"),
    ("noor", "+16285550111", "Noor Haddadin"),
    ("calloway", "+16285550112", "Calloway Pike"),
    ("imelda", "+16285550113", "Imelda Sorrento"),
    ("tobias", "+16285550114", "Tobias Grell"),
    ("lottie", "+16285550115", "Lottie Fennimore"),
    ("ansel", "+16285550116", "Ansel Fennimore"),
    ("greer", "+16285550117", "Greer Fennimore"),
    ("delphine", "+13415550120", ""),
    ("harlan", "+13415550121", ""),
    ("okoro", "+13415550122", ""),
)

ACQUAINTANCES: tuple[str, ...] = (
    "Anouk Pellerin", "Basil Thorne", "Cosima Veldt", "Darius Quenneville", "Elodie Marchbank",
    "Fenwick Osei", "Galen Rusk", "Hedda Lindgren", "Ignatius Moe", "Jasper Velloso",
    "Kaia Brandvold", "Leopold Arkwright", "Mirabel Stoake", "Nico Fairweather", "Oona Ballantyne",
    "Percival Hask", "Quincy Albright", "Rhiannon Tufte", "Silas Kerrigan", "Tamsin Oakhurst",
    "Ulric Penhallow", "Vesna Corrigan", "Wilhelmina Strand", "Xavi Montclair", "Yara Danforth",
    "Zeno Halvorsen", "Agathe Rimbault", "Bertil Ashgrove", "Clio Ferrante", "Dorian Vale",
)  # fmt: skip

# senders of codes, deliveries and reminders; nobody saves them
SERVICES: tuple[tuple[str, str], ...] = (
    ("parcel", "+18885550130"),
    ("fernbank", "+18885550131"),
    ("quill", "+18885550132"),
    ("dental", "+17075550133"),
    ("pharmacy", "+17075550134"),
    ("transit", "+18885550135"),
)
CODE_SENDERS = ("fernbank", "quill")


def acquaintance_key(place: int) -> str:
    return f"acq{place:02d}"


def _acquaintances() -> tuple[Person, ...]:
    return tuple(
        Person(acquaintance_key(place), f"+19255550{140 + place:03d}", "" if place % 3 == 2 else name)
        for place, name in enumerate(ACQUAINTANCES)
    )


NEW_PEOPLE: tuple[Person, ...] = (
    *(Person(key, handle, saved) for key, handle, saved in NAMED),
    *_acquaintances(),
    *(Person(key, handle, "") for key, handle in SERVICES),
)
NEW_BY_KEY = {person.key: person for person in NEW_PEOPLE}

GROUPS: tuple[Chat, ...] = (
    Chat("crew", "chat520101", ("maya", "raff", "odalys", "kip"), "chicago crew"),
    Chat("bwdesign", "chat520102", ("benedikt", "acq11", "acq17"), "bellweather design"),
    Chat("kestrel", "chat520103", ("mayao", "bram", "yusuf"), "kestrel design team"),
    Chat("building", "chat520104", ("wendell", "acq28", "acq20"), "linden st building"),
    Chat("climb", "chat520105", ("noor", "calloway", "raff", "acq19"), "send train"),
    Chat("books", "chat520106", ("imelda", "tobias", "odalys"), "book club"),
    Chat("cousins", "chat520107", ("lottie", "ansel", "greer"), "cousin crew"),
    Chat("trivia", "chat520108", ("acq06", "acq07", "acq10", "acq13"), "trivia night"),
    Chat("hoops", "chat520109", ("acq01", "acq03", "acq05", "acq18", "acq25"), "sunday hoops"),
    Chat("plants", "chat520110", ("acq00", "acq12", "acq14"), ""),
)

DIRECT_KEYS: tuple[str, ...] = (
    "maya", "mayao", "raff", "odalys", "kip", "petra", "benedikt", "delphine", "harlan", "okoro",
    *(acquaintance_key(place) for place in range(len(ACQUAINTANCES))),
    *(key for key, _handle in SERVICES),
)  # fmt: skip

NEW_CHATS: tuple[Chat, ...] = (
    *(Chat(key, NEW_BY_KEY[key].handle, (key,)) for key in DIRECT_KEYS),
    *GROUPS,
)

def _record(person: Person) -> dict[str, object]:
    """A Contacts record, in the shape `import_chat_db` reads."""
    return {"name": person.saved_as, "phones": [person.handle], "emails": []}


HARD_PEOPLE: tuple[Person, ...] = (*PEOPLE, *NEW_PEOPLE)
HARD_CHATS: tuple[Chat, ...] = (*CHATS, *NEW_CHATS)
HARD_CONTACTS: tuple[dict[str, object], ...] = (
    *CONTACTS,
    *(_record(person) for person in NEW_PEOPLE if person.saved_as),
)


def first_name(person: Person) -> str:
    return person.saved_as.split()[0] if person.saved_as else ""
