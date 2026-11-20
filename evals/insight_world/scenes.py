"""The written scenes of the insight world. Each one holds an insight to find, or a trap.

Labels name the messages that the gold list (`gold.py`) expects an insight
to cite. A trap is a thing that looks like an insight but is not one: a plan
that happened, a promise that was kept, a promise of the other person, small
talk.
"""

from __future__ import annotations

from evals.world.script import Item, scene

SCENES: dict[str, str] = {
    "felix": """
@2025-10-18 10:00
felix: climbing saturday at 11  #felix_climb
me: yes!! i'll bring the chalk
@2025-10-25 16:30
felix: that was so fun, my arms are dead  #felix_climb_done
me: same lol
@2025-12-05 19:10
felix: GOT THE JOB at the observatory!!! i start in january  #felix_job
me: FELIX that's huge, so proud of you
felix: still can't believe it, i have to move to tucson  #felix_move
""",
    "jonah": """
@2025-12-16 17:20
jonah: are you still coming to my show on the 3rd  #jonah_show
jonah: no pressure but i saved you a ticket?  #jonah_ticket
""",
    "kaya": """
@2025-10-12 21:00
kaya: we NEED to get dinner soon i miss u  #kaya_dinner
me: omg yes!! next week
kaya: maybe! my week is chaos, i'll text you  #kaya_dinner_push
@2025-11-30 18:40
kaya: that soup you made at the party was unreal
me: it's my grandma's recipe, i'll send it to you tonight  #kaya_recipe
kaya: yesss
@2025-12-02 17:00
kaya: call you at 8  #kaya_call
me: yes call me at 8, i'll be home
""",
    "leo": """
@2025-11-08 11:00
leo: hike sunday
me: down
@2025-11-08 19:30
leo: actually let's skip it, the trail is closed all winter  #leo_hike_off
me: ah ok, agreed, too cold anyway
@2025-11-15 13:00
me: i'll send you the photos from the wedding later  #leo_photos
@2025-11-15 22:10
me: here they are, the one of your mom is so good  #leo_photos_sent
leo: thank you!! these are perfect
@2025-11-29 10:00
leo: brunch sunday 11 at fig  #leo_brunch
me: perfect, see you there
""",
    "mira": """
@2025-11-25 20:15
mira: ok big news. i'm moving to lisbon in january  #mira_lisbon
me: WHAT. tell me everything
mira: new job, flat with a view of the river, i'm terrified and excited
@2025-12-09 13:00
mira: i had the best ramen of my life today  #mira_ramen
me: jealous
""",
    "otto": """
@2025-11-20 18:00
otto: i'll send you the concert tickets this week  #otto_tickets
me: thank you!!
""",
    "rosa": """
@2025-11-10 12:30
rosa: the internship deadline is in two weeks and i don't know anyone there
me: wait my manager runs that team. i'll introduce you to her  #rosa_intro
rosa: you would do that?? thank you so much
@2025-12-05 13:10
rosa: free to talk today
me: in meetings, i'll call you after work  #rosa_call
""",
    "sol": """
@2025-12-07 23:10
sol: things with dev are not good rn  #sol_dev
sol: we might be done honestly  #sol_done
me: oh sol. i'm here if you want to talk
@2025-12-11 19:00
sol: we should facetime soon, i miss your face  #sol_facetime
me: yes def, i'll text you when work calms down
""",
    "tess": """
@2025-12-14 23:40
tess: did you get home ok  #tess_home
me: yes! thanks for tonight
@2025-12-14 23:45
tess: my cat knocked my plant off the shelf again lol  #tess_cat
""",
    "iris": """
@2023-12-28 09:00
me: happy birthday iris!!  #iris_bday_2023
iris: thank you!!
@2024-12-28 08:40
me: happy bday!! hope it's a good one  #iris_bday_2024
iris: love you
""",
    "fernbank": """
@2025-11-02 08:00
fernbank: Your Fernbank code is 448210. Do not share it.
@2025-12-03 08:00
fernbank: Your Fernbank code is 902114. Do not share it.
""",
}


def scenes() -> tuple[Item, ...]:
    return tuple(item for key, script in SCENES.items() for item in scene(key, "story", script))
