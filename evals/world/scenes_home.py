"""Scenes of the family group chat, of work, and of two numbers nobody saved.

Facts the questions use:
- "Fennimore Fam": Mom, Dad and Clem (Sam's sister). Mom's birthday is March 14; she turned 60 in 2024.
  Dad picked Osteria Lupa; the dinner was on 2024-03-16. Sam's gift was a pottery class.
- Marisol, Sam's manager at Bellweather Labs, asked Sam to stay for the Q3 deck on 2024-08-02. Sam told
  her about leaving on 2024-09-09. The last day was 2024-09-27.
- Ada, a recruiter at Kestrel & Finch (not saved), wrote first on 2024-08-20. The offer came on
  2024-09-03: product designer, 128k base, start October 7. Sam accepted on 2024-09-06.
- Gus, the landlord of 412 Alcott St (not saved): lease 2025-01-15, keys 2025-02-27, plumber for the
  kitchen sink on 2025-06-03.
"""

from __future__ import annotations

from evals.world.script import Item, scene

FAMILY = """
@2023-03-14 08:10
me: happy birthday mom!!  #mom_bday_2023
mom: thank you sweetheart
dad: we're doing tacos tonight
@2024-02-20 19:00
clem: ok sibling summit. mom turns 60 on march 14. dinner the saturday after?  #mom60_plan
dad: she wants osteria lupa  #mom60_place
me: i'm in. i'll get her that pottery class she keeps talking about  #mom_gift
clem: perfect. you're the best little sibling  #clem_sibling
@2024-03-14 08:00
me: happy 60th mom!!!  #mom60_wish
dad: happy birthday rosie
clem: HAPPY BIRTHDAY MOM
mom: thank you my loves. 60!!!
@2024-03-16 22:10
mom: what a night. thank you for lupa and the pottery class. i'm going to make you all mugs  #mom60_dinner
clem ^love
@2024-09-06 18:30
me: news: i got an offer from kestrel & finch and i said yes!! product designer  #family_offer
mom: SO PROUD
dad: knew it
clem: when do u start
me: october 7
@2025-03-14 08:20
me: happy birthday mom!! love you  #mom_bday_2025
mom: thank you! come for dinner sunday?
"""

MARISOL = """
@2023-01-09 08:40
mari: welcome back! standup moved to 9:30 this week  #mari_first
me: got it, thanks
@2024-08-02 17:30
mari: any chance you can stay and finish the q3 deck? board meeting moved to monday  #q3_deck_ask
me: ugh ok. how long
mari: hour tops
@2024-08-02 21:05
mari: thank you for staying. sorry it ran so late  #q3_deck_done
@2024-09-09 09:00
me: hey marisol, can we chat today? i got an offer and i'm going to take it  #tell_marisol
mari: oh! congrats, truly. 10am?
me: 10 works
@2024-09-27 17:00
mari: last day!! we'll miss you at bellweather. keep in touch  #mari_last_day
me: thank you for everything
@2025-04-02 12:30
mari: saw the kestrel launch, looks great!  #mari_after
"""

ADA = """
@2024-08-20 11:15
ada: hi sam, this is ada from kestrel & finch. are you open to chatting about a product designer role?  #ada_first
me: hi ada! yes, happy to
@2024-08-27 16:00
ada: thanks for coming in today, the team loved your portfolio  #interview
@2024-09-03 14:00
ada: great news, we'd like to offer you the role. 128k base, start date october 7  #offer
me: wow thank you! can i have until friday?
ada: of course
@2024-09-06 10:00
me: i'd love to accept  #accept
ada: amazing!! welcome to kestrel & finch
"""

GUS = """
@2025-01-15 09:30
gus: hi this is gus. lease for 2B at 412 alcott st is ready to sign  #lease
me: great, we'll sign tonight
@2025-02-27 15:00
gus: keys ready friday the 28th, i'll leave them at the front desk  #keys
@2025-06-03 08:00
gus: plumber coming thursday 10am for the kitchen sink  #plumber
me: thanks gus
"""


def home_scenes() -> tuple[Item, ...]:
    return (
        *scene("family", "family", FAMILY),
        *scene("mari", "marisol", MARISOL),
        *scene("ada", "ada", ADA),
        *scene("gus", "gus", GUS),
    )
