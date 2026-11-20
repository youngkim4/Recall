"""Scenes with Wren, the partner: how they met, the trip, the fight, the move.

Facts the questions use:
- first text 2023-02-18; first date at Menya Hoshi (ramen) on 2023-02-25
- official on 2023-04-15; Wren's birthday is May 18
- Montreal trip for her birthday: idea 2024-01-20, booked 2024-02-11, away 2024-05-16 to 2024-05-21, Airbnb in Mile End
- fight 2024-08-02 (Sam missed her first show at Hollis Gallery, working late on the Q3 deck), silence for two days,
  make-up 2024-08-05 with soup dumplings at Dumpling Moon
- they moved in together at 412 Alcott St on 2025-03-01; they text less after that
"""

from __future__ import annotations

from evals.world.script import Item, scene

MEETING = """
@2023-02-18 19:12
wren: hey! it's wren from the bouldering gym  #wren_first
wren: you said you'd send me that ramen place
me: omg hi. yes its menya hoshi on 9th
me: the spicy miso is unreal
wren: ok bold claim
wren: prove it?
me: saturday?? 7pm
wren: deal  #first_date_plan
@2023-02-25 18:20
wren: running 5 min late sorry!!
me: no rush i got us on the list
+150
wren: ok you were right about the miso at menya hoshi  #first_date_after
wren: i had a really nice time
me: me too. like really
wren ^love
@2023-04-15 22:40
wren: can i ask a dumb question
me: always
wren: are we like. official? bc i deleted the app lol  #official_ask
me: wren. yes. i deleted it two weeks ago  #official_yes
wren: ok good bc i told my mom about you
me ^love
@2023-05-18 08:05
me: happy birthday wren!! dinner is on me tonight  #wren_bday_2023
wren: !!! where are we going
me: secret. wear the green jacket
"""

MONTREAL = """
@2024-01-20 21:05
wren: ok hear me out. montreal in may  #montreal_idea
me: for your birthday??
wren: for my birthday!!!
me: im in. bagels and that tiny crepe place you keep sending me
wren: i'll make a doc
@2024-02-11 13:30
me: flights booked!! may 16 out, may 21 back  #flights_booked
me: sfo to yul, the 7am one sorry
wren: 7AM
wren: i love you but 7am
me: airbnb in mile end is confirmed too  #airbnb_booked
wren ^love
@2024-05-16 05:10
me: im outside
wren: coming coming. do we have the passports
me: both in my bag
@2024-05-17 09:40
wren: lost you in the bagel line lol where are u  #bagel_line
me: by the wood oven. got sesame and everything  #bagel_order
wren: first montreal meal and it's perfect
@2024-05-18 00:01
me: happy birthday wren. best trip ever  #wren_bday_2024
wren: best birthday ever
@2024-05-21 22:40
wren: home. already miss montreal  #montreal_home
me: same. we're going back
"""

FIGHT = """
@2024-08-02 16:52
wren: doors open at 7 at hollis. you're still coming right?  #opening_reminder
me: wouldn't miss it
@2024-08-02 19:40
wren: where are you
+25
wren: sam??
@2024-08-02 21:15
me: im so sorry. still at the office, marisol needed the q3 deck tonight  #fight_excuse
wren: it was my first show  #fight_first_show
wren: you promised. you literally promised
me: i know. i messed up
wren: i don't want to talk right now  #fight_end
@2024-08-05 08:30
me: i'm really sorry about friday. i should have left. no deck is worth missing your show  #makeup_apology
+40
wren: thank you for saying that
wren: i was really hurt
me: i know. can i take you to dumpling moon tonight? and i want to see your pieces at the gallery
wren: ok. yes. i missed you  #makeup_yes
+10h
wren: ok the soup dumplings fixed like 80% of it  #makeup_dinner
me: i'll fix the other 20
wren ^laugh
@2024-08-10 14:00
me: went back to hollis gallery today. the blue bowl is my favorite piece  #gallery_visit
wren: that's the one i almost didn't put in
"""

MOVING = """
@2025-01-12 20:00
wren: ok the place on alcott st. two bedroom. light everywhere  #apartment_found
me: let's do it
wren: i'll tell gus we want it
@2025-02-28 17:30
me: got the keys!!
wren: AHHH
@2025-03-01 22:10
wren: first night in our place  #move_in
me: our place!!
wren ^love
@2025-05-18 07:30
me: happy birthday love. breakfast in bed in 5 min  #wren_bday_2025
wren: best roommate
"""

MAIL = """
@2023-06-07 12:00
wrenmail: sending you the gym waiver from my laptop  #wren_mail
me: got it, signed
"""


def wren_scenes() -> tuple[Item, ...]:
    return (
        *scene("wren", "meeting", MEETING),
        *scene("wren", "montreal", MONTREAL),
        *scene("wren", "fight", FIGHT),
        *scene("wren", "moving", MOVING),
        *scene("wrenmail", "mail", MAIL),
    )


# days in Wren's thread with no filler: the scenes, and the two silent days after the fight
WREN_QUIET_DAYS = frozenset({"2024-08-03", "2024-08-04"})
