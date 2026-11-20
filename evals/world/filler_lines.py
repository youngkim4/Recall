"""Short everyday exchanges for the filler days. Each is a tuple of (speaker, text).

"them" is the other person of a one-to-one chat. In the family chat the
speakers are "mom", "dad", "clem" and "me". No exchange holds a word of
`RESERVED`: filler must never answer a question by chance.
"""

from __future__ import annotations

Exchange = tuple[tuple[str, str], ...]

RESERVED: tuple[str, ...] = (
    "goose", "camera", "pentax", "film", "montreal", "bagel", "gallery", "hollis", "offer", "kestrel",
    "dumpling", "japan", "tokyo", "kyoto", "osaka", "dog", "puppy", "brother", "married", "wedding", "wife",
    "husband", "promotion", "promoted", "$", "dollar", "birthday", "bday", "lease", "keys", "plumber",
    "denver", "alcott", "deck", "lupa", "pottery", "starling", "ritual", "menya", "ramen", "airbnb",
    "official", "the app", "per night", "magpie", "sink",
)

PARTNER: tuple[Exchange, ...] = (
    (("them", "did you eat"), ("me", "not yet, you?"), ("them", "making pasta, come over")),
    (("me", "omw"), ("them", "drive safe")),
    (("them", "can you grab oat milk on the way"), ("me", "yep"), ("them", "ur the best")),
    (("me", "how was climbing"), ("them", "sent the purple v4 finally"), ("me", "WHAT"), ("them", "i'm unstoppable")),
    (("them", "thinking about you"), ("me", "thinking about you more")),
    (("me", "running 10 late sorry"), ("them", "all good i'm early anyway")),
    (("them", "movie tonight?"), ("me", "yes. you pick"), ("them", "something with a heist")),
    (("me", "love you"), ("them", "love you too")),
    (("them", "the cat next door is in our yard again"), ("me", "pics or it didn't happen"), ("them", "he left")),
    (("me", "do we need anything from the store"), ("them", "lemons and the good bread"), ("me", "on it")),
    (("them", "long day. tacos?"), ("me", "always tacos")),
    (("me", "night night"), ("them", "night. sleep well")),
    (("them", "what time are you home"), ("me", "like 6:30"), ("them", "ok i'll start dinner")),
    (("me", "did you see that sunset"), ("them", "stopped on the sidewalk for it")),
    (("them", "my mom says hi"), ("me", "hi to her!!")),
    (("me", "farmers market sunday?"), ("them", "yes peaches are back")),
)

CLOSE_FRIEND: tuple[Exchange, ...] = (
    (("them", "brunch tmrw?"), ("me", "yes. usual spot?"), ("them", "11")),
    (("me", "i need to tell you something so dumb"), ("them", "obviously call me")),
    (("them", "are u watching the finale"), ("me", "NO SPOILERS"), ("them", "my lips are sealed")),
    (("me", "thrifting saturday?"), ("them", "yes!! i need a lamp")),
    (("them", "i'm so tired of my landlord"), ("me", "ugh what now"), ("them", "the heater again")),
    (("me", "running club thursday?"), ("them", "if it's not raining")),
    (("them", "lol look at this"), ("me", "STOP"), ("them", "i cried")),
    (("me", "miss u"), ("them", "miss u more. this week?")),
    (("them", "karaoke friday"), ("me", "only if you do the duet with me")),
    (("me", "what was that book you said"), ("them", "the one about the lighthouse"), ("me", "ty")),
)

FRIEND: tuple[Exchange, ...] = (
    (("them", "did you see the game"), ("me", "i saw the last 5 min"), ("them", "that's the only part that mattered")),
    (("me", "new album is so good"), ("them", "track 4"), ("me", "track 4!!")),
    (("them", "lol"), ("me", "lmao")),
    (("me", "how's work"), ("them", "busy but good. you?"), ("me", "same")),
    (("them", "sending you a video"), ("me", "this is the best thing i've seen all week")),
    (("me", "call later?"), ("them", "yes after 7")),
    (("them", "i made sourdough"), ("me", "proud of you"), ("them", "it's a brick but it's mine")),
    (("me", "bike ride sunday?"), ("them", "only if there's coffee after")),
    (("them", "what are you up to this weekend"), ("me", "nothing and i love it")),
)

WORK: tuple[Exchange, ...] = (
    (("them", "standup moved to 10 today"), ("me", "thanks for the heads up")),
    (("them", "can you send me the latest mocks"), ("me", "sending now")),
    (("me", "running a few min late to the 1:1"), ("them", "no worries")),
    (("them", "great work in the review today"), ("me", "thank you!")),
    (("them", "are you in the office tomorrow"), ("me", "yes from 9")),
    (("me", "out sick today, will check messages"), ("them", "feel better")),
    (("them", "quick q about the onboarding flow"), ("me", "sure, call me")),
)

TALKING: tuple[Exchange, ...] = (
    (("them", "what are you up to"), ("me", "just got home. you?"), ("them", "cooking badly")),
    (("me", "good morning"), ("them", "morning!! sleep ok?")),
    (("them", "what's your go to karaoke song"), ("me", "dancing queen, no shame")),
    (("me", "that playlist was good"), ("them", "told you")),
)

FAMILY: tuple[Exchange, ...] = (
    (("mom", "did everyone see the photos from the garden"), ("clem", "the tomatoes!!"), ("dad", "record year")),
    (("dad", "who's coming sunday"), ("me", "me"), ("clem", "me +1")),
    (("mom", "call your grandmother"), ("me", "calling her tonight")),
    (("clem", "lol dad's new phone case"), ("dad", "it's practical"), ("mom", "it's very practical")),
    (("mom", "love you all"), ("me", "love you mom")),
    (("dad", "the roof guy came. all good"), ("mom", "thank goodness")),
    (("clem", "anyone want to split a costco run"), ("me", "yes pls")),
    (("mom", "made the lemon cake"), ("clem", "save me a piece"), ("dad", "no promises")),
    (("me", "made it home safe"), ("mom", "good. thanks for telling us")),
)
