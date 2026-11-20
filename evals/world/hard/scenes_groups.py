"""Written scenes of the hard world in group chats: trips, races, and the needles said once.

Facts the hard questions use:
- chicago crew: the Chicago trip of 2021-07-09 to 07-12 at Kip's cousin's place in Logan Square; Sam's
  license news is in Maya's chat, the Linden St move day is here (2022-09-17); Maya Lindqvist finished a
  half marathon on 2023-10-08; Sam's 30th in Big Sur on 2025-07-24 (said as "thirty"); Kip booked Tavola
  Rosa on 18th for the reunion on 2025-08-15.
- kestrel design team: Maya Ostrowski introduces herself on 2024-10-07; Bram says the team moves to the
  9th floor in January (2024-11-20); Maya Ostrowski signs up for a 10k (2025-03-02).
- linden st building: the lobby wifi password fernhollow42 (2023-02-10); Wendell's spare garage spot
  (2024-01-08).
- send train: the gym door code 4417 (2022-01-12).
- book club: the June 2023 pick, The Salt Orchard (2023-05-28).
- cousin crew: Lottie got into vet school (2024-03-21).
"""

from __future__ import annotations

from evals.world.script import Item, scene

CREW = """
@2021-06-20 20:00
kip: chicago july 9 to 12. we're staying at my cousin's place in logan square  #chi21_plan
maya: IN
raff: in
odalys: in, booking my flight now
@2021-07-10 11:30
kip: deep dish at 9 tonight, don't be late
raff: i was born late
@2022-09-17 18:00
me: officially a linden st resident. unit 5, come visit  #linden_movein
odalys: housewarming!!!
maya: so proud of u
@2023-10-08 12:15
maya: FINISHED. 13.1 miles in 2:04  #maya_half
raff: maya the machine
me: legend
@2025-07-24 21:00
me: thirty and flirty in big sur!! the cabin is unreal  #big_sur
odalys: happy thirty!!!
@2025-08-15 09:00
kip: for the reunion dinner: tavola rosa on 18th, i booked 8pm  #tavola
maya: perfect
"""

KESTREL = """
@2024-10-07 09:30
mayao: welcome sam! i'm maya, i run research here  #mayao_intro
me: hi maya! excited to be here
@2024-11-20 15:00
bram: heads up, we're moving up to the 9th floor in january  #ninth_floor
yusuf: finally, windows
@2025-03-02 10:00
mayao: signed up for a 10k in may, someone hold me to it  #mayao_10k
bram: we will be insufferable about it
"""

BUILDING = """
@2023-02-10 19:00
wendell: lobby wifi pw is fernhollow42 if anyone's guests need it  #wifi
me: lifesaver, thanks
@2024-01-08 08:30
wendell: my spot in the garage is free till march if anyone needs it. unit 2  #parking
"""

CLIMB = """
@2022-01-12 17:00
noor: door code at the gym is 4417 now  #door_code
calloway: noted
"""

BOOKS = """
@2023-05-28 20:00
imelda: june pick is the salt orchard. 300 pages, you have time  #salt_orchard
tobias: finally something short
"""

COUSINS = """
@2024-03-21 12:00
greer: lottie got into vet school!!!  #vet
ansel: LOTTIE
me: so proud of you lottie
lottie: thank youuu
"""


def group_scenes() -> tuple[Item, ...]:
    return (
        *scene("crew", "crew", CREW),
        *scene("kestrel", "kestrel", KESTREL),
        *scene("building", "building", BUILDING),
        *scene("climb", "climb", CLIMB),
        *scene("books", "books", BOOKS),
        *scene("cousins", "cousins", COUSINS),
    )
