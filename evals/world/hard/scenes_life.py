"""Written scenes of the hard world in one-to-one chats: homes, work, injuries, the long friendships.

Facts the hard questions use:
- 88 Fulton Ave, apt 3, with the roommate Petra: in on 2020-03-01, out on 2022-09-17. Friction over the
  dishes in June 2021. Rafferty's van moved Sam to 1550 Linden St, unit 5, on 2022-09-17.
- Bellweather from 2020-06-01. Driver's license on 2020-08-14 (said as "LICENSED", never "test").
- Wrist fractured on 2021-03-20 at the climbing gym (said as "fractured", never "broke" or "bone").
- Larkspur Health offered a senior designer role on 2021-04-12; Sam said no on 2021-04-15 (the commute).
- Cello from 2022-10-05, typed "chello". Wisdom teeth on 2023-03-08, said as "molars".
- Rafferty moved to Lisbon on 2022-11-12 and came back on 2025-04-05. Kip went quiet for all of 2023.
- Chicago twice: with the college crew 2021-07-09 to 07-12 (Kip's cousin's place in Logan Square), and
  for work with Benedikt 2023-04-18 to 04-21 (the Halcyon Hotel, the Prairie Design Summit).
- Odalys: a pothos named Gertrude (2023-01-15) and a kitten named Biscuit (2023-11-04).
"""

from __future__ import annotations

from evals.world.script import Item, scene

PETRA = """
@2020-02-02 18:00
petra: saw the place on fulton. two bedrooms, the light is insane
me: let's apply tonight
@2020-03-01 12:10
petra: welcome to apt 3, roomie. 88 fulton ave is officially ours  #fulton_movein
me: home sweet home
@2021-06-14 21:30
petra: can we talk about the dishes situation  #dishes
me: yeah that's fair. i'll make a chore chart
petra: thank u. i love living with you i just hate the sink tower
+2
me: ok chart is on the fridge
@2022-09-17 20:45
petra: the fulton place is so quiet without you  #fulton_out
me: i miss it already. come see linden st soon
@2024-06-30 10:00
petra: coffee soon? it's been a minute
me: yes!! next week
"""

RAFF = """
@2021-03-20 18:05
raff: where are u, we said 6  #wrist_where
me: urgent care. landed wrong off the top of the purple problem
+150
me: xray says it's fractured. cast for 6 weeks  #wrist
raff: NOOO. which arm
me: left wrist. typing with one thumb  #wrist_which
raff: i'm bringing you soup
@2022-09-14 19:20
raff: i can bring my van saturday for the linden st thing. how many boxes  #van
me: like 30 boxes and the couch. you're a lifesaver  #van_reply
@2022-11-12 07:30
raff: flight to lisbon is booked. one way  #raff_lisbon
me: i'm happy for you and also devastated
@2025-04-05 16:00
raff: i'm back in oakland for good!!  #raff_back
me: RAFFERTY. dinner this week, no excuses
"""

WORK = """
@2020-05-31 21:00
me: first day at bellweather tomorrow. so nervous  #bw_nervous
maya: you're going to be great
"""

BENEDIKT = """
@2020-06-01 09:05
benedikt: welcome to bellweather! desk by the window is yours  #bw_first_day
me: thank you!! see you at standup
@2023-03-30 14:00
benedikt: prairie design summit is booked. april 18 to 21, we're at the halcyon hotel downtown  #chi23_hotel
me: perfect. chicago in april, bring a coat
@2023-04-19 22:10
benedikt: the keynote was long but the deep dish made up for it
"""

DELPHINE = """
@2021-04-12 10:30
delphine: hi sam, delphine from larkspur health. we'd love to extend an offer: senior designer, start may 3  #larkspur_offer
me: thank you! let me think it over
@2021-04-15 09:15
me: thank you so much, but i'm going to pass. the commute to the south bay would be brutal  #larkspur_no
delphine: understood, best of luck sam
"""

MAYA = """
@2020-08-14 16:40
me: I'M LICENSED. the dmv lady said my parallel parking was "fine" lol  #license
maya: a menace on the roads. proud of u
@2021-04-14 20:00
me: larkspur wants me but the commute would kill me  #larkspur_vent
maya: then don't. you like bellweather
@2023-04-19 08:00
me: chicago round two. colder than july 2021 lol  #chi23_maya
maya: say hi to the bean for me
"""

ODALYS = """
@2022-10-05 19:30
me: first chello lesson today. my fingers are destroyed  #cello
odalys: a musician!!
@2023-01-15 13:00
odalys: dropping off a pothos for the new place
me: omg thank you
odalys: what are you naming her  #plant_ask
me: gertrude, obviously  #plant_name
@2023-03-08 17:20
me: the dentist yanked all four back molars. i look like a chipmunk  #molars
odalys: pics or it didn't happen
@2023-11-04 11:00
odalys: meet biscuit!! 9 weeks old and already runs the place  #biscuit
me: I NEED TO MEET HER
"""

KIP = """
@2024-01-20 19:00
kip: sorry i vanished. rough year. i'm ok now  #kip_back
me: kip!! no apology needed. call this week?
"""

OKORO = """
@2022-08-20 10:00
okoro: hello sam, unit 5 at 1550 linden st is yours from september 17  #linden_lease
me: wonderful, thank you
"""

HARLAN = """
@2020-02-20 15:00
harlan: application approved for 88 fulton ave apt 3. move in march 1  #fulton_lease
me: thank you harlan!
"""


def life_scenes() -> tuple[Item, ...]:
    return (
        *scene("petra", "petra", PETRA),
        *scene("raff", "raff", RAFF),
        *scene("maya", "work", WORK),
        *scene("benedikt", "benedikt", BENEDIKT),
        *scene("delphine", "delphine", DELPHINE),
        *scene("maya", "maya", MAYA),
        *scene("odalys", "odalys", ODALYS),
        *scene("kip", "kip", KIP),
        *scene("okoro", "okoro", OKORO),
        *scene("harlan", "harlan", HARLAN),
    )
