"""Scenes with friends: Dash (a talking stage), Juniper (a friend who drifted) and Theo (moved to Denver).

Facts the questions use:
- Dash, from a dating app, is not saved. Drinks at the Starling on Grand on 2023-01-14. It ends on
  2023-03-10, when Sam says they are seeing someone.
- Juniper: Sam left the Pentax K1000 film camera at her party (asked 2023-04-23). She found it under the
  couch on 2023-04-24 and gave it back at Ritual on 2023-04-27. Her birthday is February 2. The last
  message is hers on 2025-02-02.
- Theo: the running joke is the goose at the lake (first on 2023-03-19, "goose update: ..."). He got a job
  in Denver on 2023-06-10 and moved on 2023-08-19. Sam visited him 2025-06-13 to 2025-06-16.
"""

from __future__ import annotations

from evals.world.script import Item, scene

DASH = """
@2023-01-08 20:14
dash: hey it's dash from the app  #dash_first
dash: your profile said you shoot film?
me: haha yes guilty. i have an old pentax
dash: ok that's actually cool
@2023-01-12 18:30
dash: drinks saturday? the starling on grand  #dash_drinks
me: yes! 8?
dash: 8
@2023-01-14 23:30
dash: that was fun. you're funny
me: you too!! thanks for the drinks
@2023-01-27 19:00
dash: bowling tmrw still on?
me: yes but i'm going to be terrible
@2023-02-09 12:10
dash: u free this weekend?
me: ahh this weekend is packed. next one?
dash: sure
@2023-02-26 16:45
dash: hey stranger
+300
me: hey! sorry crazy week
@2023-03-10 21:02
dash: so are we still doing this or  #dash_ask
me: hey honestly i've started seeing someone. i'm sorry i didn't say sooner  #dash_end
dash: all good. take care  #dash_last
"""

JUNIPER = """
@2023-02-02 09:00
me: HAPPY BIRTHDAY JUNI  #juni_bday_2023
juni: thank uuu. drinks tonight at mine??
me: obviously
@2023-04-20 17:45
juni: party at mine saturday! bring the film camera  #juni_party
me: yesss
@2023-04-23 11:02
me: wait did i leave my camera at yours?? the pentax k1000  #camera_lost
juni: lemme look
juni: not seeing it in the living room :(
me: noooo it has a half finished roll in it
@2023-04-24 18:40
juni: FOUND IT  #camera_found
juni: under the couch next to a very sad lime
me: JUNI. you're a hero
@2023-04-27 09:15
juni: ritual at 10? i'll bring your baby  #camera_return_plan
me: yes!!!
+120
me: camera is back with its person. thank u  #camera_returned
juni ^love
@2024-02-02 10:15
me: happy birthday juni!! miss you  #juni_bday_2024
juni: thank u love!
@2024-03-09 20:20
juni: omg it's been forever  #juni_forever
me: i know!! dinner soon?
juni: yes next week?
me: yes!!
@2024-07-21 15:05
juni: saw a pentax at the thrift store and thought of u
me: haha did u buy it
@2025-02-02 10:40
me: happy birthday juni!! hope ur doing well  #juni_bday_2025
juni: thank you!! we should catch up soon  #juni_last
"""

THEO = """
@2023-03-19 17:20
theo: the goose at the lake chased u for like 40 feet  #goose_origin
me: it was personal
theo: goose update: he's back by the boathouse
@2023-04-30 11:00
theo: goose update: he has a girlfriend now
me: good for him honestly
@2023-06-10 19:30
theo: i got the denver job!!!  #theo_job
me: THEO. congrats. also no
theo: i leave mid august
@2023-06-25 13:10
theo: goose update: he hissed at a toddler. a menace
@2023-08-19 08:45
theo: last box in the truck. bye oakland  #theo_moves
me: crying in the club
@2023-08-22 20:00
theo: denver update: the sky is huge and i'm out of breath on stairs  #theo_denver
theo: goose update: no geese here. only magpies. i'm in mourning
me: any goose update from the lake? i'll check for u
@2024-01-14 10:30
theo: goose update: a denver goose exists and it looked at me
@2024-06-02 18:00
theo: goose update: i think about him every day
@2024-11-10 12:00
theo: goose update: dreamt about him. we made up
theo: also when are you visiting
me: summer. promise
@2025-06-13 14:20
me: landed!!  #theo_visit
theo: welcome to altitude. drink water
@2025-06-14 11:00
theo: goose update: there's one at sloan's lake and it's coming for you  #goose_visit
me: i'm not going near that lake
@2025-06-16 09:00
me: thank you for the best weekend. denver is unfair
theo: come back soon
@2025-11-30 16:00
theo: goose update: thanksgiving goose has entered the chat  #goose_last
me: lmaooo
"""


def friend_scenes() -> tuple[Item, ...]:
    return (
        *scene("dash", "dash", DASH),
        *scene("juni", "juniper", JUNIPER),
        *scene("theo", "theo", THEO),
    )
