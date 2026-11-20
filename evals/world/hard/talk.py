"""Seeded templates for the everyday traffic of the hard world.

`line(pool, draw)` fills one template of a pool from the slot words. No slot
word and no template holds a word of `HARD_RESERVED`, so traffic never
answers a hard question by chance; a test checks every generated line.
`typo` swaps two letters of one word now and then, as thumbs do.
"""

from __future__ import annotations

import random
import string

from evals.world.filler_lines import RESERVED

HARD_RESERVED: tuple[str, ...] = (
    "fernhollow", "4417", "vet school", "9th floor", "tavola", "salt orchard", "garage", "biscuit", "gertrude",
    "pothos", "molar", "chipmunk", "licensed", "dmv", "cello", "chello", "fractured", "cast ", "xray", "van ",
    "lisbon", "halcyon", "logan square", "larkspur", "big sur", "thirty", "linden", "fulton", "marathon",
    "13.1", "chicago", "wrist", "10k", "kitten", "cat ", "teeth", "dentist", "license", "driving", "instrument",
    "lesson", "plant", "roommate", "apartment", "moving", "helped", "bone",
)  # fmt: skip

SLOTS: dict[str, tuple[str, ...]] = {
    "food": ("tacos", "pho", "pizza", "sushi", "curry", "burritos", "noodles", "banh mi", "thai", "falafel"),
    "when": ("tonight", "tmrw", "saturday", "sunday", "this weekend", "later", "after work", "friday"),
    "n": ("5", "10", "15", "20"),
    "time": ("6", "6:30", "7", "8pm", "noon", "10am", "9"),
    "thing": ("video", "article", "meme", "thread", "recipe", "trailer", "playlist"),
    "topic": ("taxes", "the group gift", "rent", "the playoffs", "the potluck", "the reading", "plans",
              "the launch", "onboarding", "the roadmap", "the heat wave", "the election"),
    "doc": ("spec", "mocks", "notes", "spreadsheet", "brief", "research doc", "slides"),
    "place": ("the park", "the beach", "the farmers market", "the lake", "tahoe", "the coast", "the library",
              "the bar on 4th", "the taqueria", "the museum", "the bookstore"),
    "item": ("charger", "umbrella", "hoodie", "tupperware", "sunglasses", "water bottle", "jacket"),
    "month": ("january", "february", "march", "april", "may", "june", "july", "august", "september",
              "october", "november", "december"),
}  # fmt: skip

OPENERS: dict[str, tuple[str, ...]] = {
    "group": (
        "anyone up for {food} {when}?", "who's around {when}", "{when} still on?", "running {n} min late",
        "omw", "just saw this {thing} {link}", "look at this {link}", "this is so us {link}",
        "can someone resend the {doc}", "back from {place}, so tired", "{food} was elite", "ok who took my {item}",
        "rain check on {when}?", "i'm at {place} if anyone wants to join", "need recs for {topic}",
        "song of the day {link}", "how's everyone's week", "send help, {topic} is chaos", "boba run?",
        "who wants boba {when}", "remember {place}?? classic", "i miss {place}", "free {item} on my stoop",
        "game {when}?", "i'm so behind on {topic}", "happy friday!!", "good morning chat", "ugh mondays",
        "can't believe it's already {month}", "who's bringing {food}", "did anyone find a {item}",
    ),
    "friend": (
        "hey", "u up?", "how was {place}", "{food} {when}?", "call later?", "i need to tell you something",
        "thinking of u", "look {link}", "this {thing} made me think of you {link}", "how's {topic} going",
        "are you free {when}", "i'm outside", "back from {place}", "ok i tried the boba place, life changing",
        "running {n} late sorry", "did you see the {thing} i sent", "miss u", "how's your week",
    ),
    "work": (
        "can you review the {doc} when you get a sec", "standup at {time}", "pushing our sync to {time}",
        "{doc} is in the shared folder", "quick q on {topic}", "shipping {topic} {when}", "nice work today",
        "heads up, {topic} slipped a week", "lunch {when}?", "in the office {when}?", "can you send the {doc}",
    ),
}  # fmt: skip

REPLIES: dict[str, tuple[str, ...]] = {
    "group": (
        "yes", "lol", "same", "i'm in", "can't {when} :(", "omg", "LMAO", "wait what", "ok", "k", "bet", "ty!!",
        "no way", "so good", "i'll be there at {time}", "sounds good", "count me out", "maybe", "true", "haha",
        "noo", "agreed", "yesss", "ok fine", "save me a seat", "who has the car", "i can bring {food}",
        "be there in {n}", "lmk", "on it", "nah", "huh", "fair", "so real", "i'm crying", "boba after?",
    ),
    "friend": (
        "yes", "lol", "omg", "haha", "ok", "k", "yesss", "same", "aw", "i'm free {when}", "can't {when}",
        "call u at {time}", "LMAO", "love that", "wait tell me", "no way", "ty", "miss u too", "deal",
    ),
    "work": ("on it", "will do", "looks good", "ty", "sounds good", "sure", "done", "can we push to {time}", "lgtm",
             "ok", "thanks!", "yep"),
}  # fmt: skip

TYPO_RATE = 0.03


def link(draw: random.Random) -> str:
    """A link to example.org. A code that happens to spell a reserved word is drawn again."""
    while True:
        code = "".join(draw.choice(string.ascii_lowercase + string.digits) for _ in range(7))
        if not any(word.strip() in code for word in (*HARD_RESERVED, *RESERVED)):
            return f"https://example.org/{draw.choice('vrsp')}/{code}"


def fill(template: str, draw: random.Random) -> str:
    """One template with each slot filled."""
    values = {name: draw.choice(words) for name, words in SLOTS.items() if "{" + name + "}" in template}
    if "{link}" in template:
        values["link"] = link(draw)
    return template.format(**values)


def typo(text: str, draw: random.Random) -> str:
    """Now and then, two letters of one word change places."""
    if draw.random() >= TYPO_RATE:
        return text
    words = text.split(" ")
    long = [place for place, word in enumerate(words) if len(word) >= 4 and word.isalpha()]
    if not long:
        return text
    place = draw.choice(long)
    word = words[place]
    cut = draw.randrange(len(word) - 1)
    swapped = word[:cut] + word[cut + 1] + word[cut] + word[cut + 2 :]
    return " ".join([*words[:place], swapped, *words[place + 1 :]])


def opener(pool: str, draw: random.Random) -> str:
    return typo(fill(draw.choice(OPENERS[pool]), draw), draw)


def reply(pool: str, draw: random.Random) -> str:
    return typo(fill(draw.choice(REPLIES[pool]), draw), draw)
