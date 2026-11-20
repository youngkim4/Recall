"""The questions written by hand. Their answers are fixed facts of the scenes.

Counts, figures over time and spans between days are in
`question_computed.py`: code counts them from the world.
"""

from __future__ import annotations

from evals.question_types import Spec

FACT: tuple[Spec, ...] = (
    Spec("fact_first_date", "fact", "Where did Wren and I go on our first date?",
         "Menya Hoshi, a ramen place on 9th.", ("Menya Hoshi", "ramen"), ("first_date_plan", "first_date_after")),
    Spec("fact_lost_item", "fact", "What did I leave at Juniper's party, and where did she find it?",
         "The Pentax K1000 film camera; she found it under her couch.",
         ("Pentax K1000 film camera", "under the couch"), ("camera_lost", "camera_found")),
    Spec("fact_theo_city", "fact", "Which city did Theo move to?", "Denver.", ("Denver",), ("theo_job", "theo_denver")),
    Spec("fact_mom60_place", "fact", "Where did we have Mom's 60th birthday dinner?",
         "Osteria Lupa.", ("Osteria Lupa",), ("mom60_place", "mom60_dinner")),
    Spec("fact_mom60_gift", "fact", "What did I give Mom for her 60th birthday?",
         "A pottery class.", ("pottery class",), ("mom_gift", "mom60_dinner")),
    Spec("fact_salary", "fact", "What base salary was in the job offer I got in 2024?",
         "128k base.", ("128k",), ("offer",)),
    Spec("fact_airbnb_area", "fact", "What neighborhood was our Airbnb in Montreal in?",
         "Mile End.", ("Mile End",), ("airbnb_booked",)),
    Spec("fact_gallery_name", "fact", "What was the name of the gallery where Wren had her show?",
         "Hollis Gallery.", ("Hollis",), ("opening_reminder", "gallery_visit")),
    Spec("fact_address", "fact", "What is the address of the apartment Wren and I moved into?",
         "Unit 2B at 412 Alcott St.", ("412 Alcott St", "2B"), ("lease", "apartment_found")),
    Spec("fact_dash_bar", "fact", "Where did Dash suggest we go for drinks?",
         "The Starling on Grand.", ("the Starling on Grand",), ("dash_drinks",)),
    Spec("fact_job_title", "fact", "What role was I offered at Kestrel & Finch?",
         "Product designer.", ("product designer",), ("ada_first", "offer")),
    Spec("fact_plumber", "fact", "What did the plumber come to fix at the new apartment?",
         "The kitchen sink.", ("kitchen sink",), ("plumber",)),
)

FIRST_LAST: tuple[Spec, ...] = (
    Spec("first_wren", "first_last", "When did Wren first text me?",
         "2023-02-18 (February 18, 2023).", ("February 18, 2023",), ("wren_first",)),
    Spec("last_juniper", "first_last", "When was the last time I heard from Juniper?",
         "2025-02-02 (February 2, 2025), when she thanked me for the birthday text.",
         ("February 2, 2025",), ("juni_last",)),
    Spec("last_dash", "first_last", "When did Dash last message me?",
         "2023-03-10 (March 10, 2023): \"all good. take care\".", ("March 10, 2023",), ("dash_last",)),
    Spec("first_goose", "first_last", "When did the goose joke with Theo start?",
         "2023-03-19 (March 19, 2023), when the goose at the lake chased me.",
         ("March 19, 2023",), ("goose_origin",)),
    Spec("first_wren_bday", "first_last", "When did I first wish Wren a happy birthday?",
         "2023-05-18 (May 18, 2023).", ("May 18, 2023",), ("wren_bday_2023",)),
    Spec("first_recruiter", "first_last", "When did the recruiter from Kestrel & Finch first reach out?",
         "2024-08-20 (August 20, 2024).", ("August 20, 2024",), ("ada_first",)),
    Spec("last_mom_bday", "first_last", "When did I last wish Mom a happy birthday in the family chat?",
         "2025-03-14 (March 14, 2025).", ("March 14, 2025",), ("mom_bday_2025",)),
)

RELATIONSHIP: tuple[Spec, ...] = (
    Spec("who_wren", "relationship", "Who is Wren to me?",
         "My partner: we met at the bouldering gym in February 2023, became official in April 2023, "
         "and have lived together since March 2025. Ongoing.",
         ("partner", "met at the bouldering gym in 2023", "living together since March 2025", "ongoing"),
         ("wren_first", "official_yes", "move_in")),
    Spec("who_dash", "relationship", "Who was Dash, and what happened between us?",
         "A talking stage from a dating app in early 2023. It ended on 2023-03-10 when I said I had started "
         "seeing someone.",
         ("talking stage from a dating app", "early 2023", "ended in March 2023 because I was seeing someone"),
         ("dash_first", "dash_end", "dash_last")),
    Spec("who_juniper", "relationship", "Is Juniper still a close friend?",
         "No: she was a close friend in 2023, we drifted in 2024, and the last exchange was a birthday text "
         "on 2025-02-02.",
         ("close friend in 2023", "drifted apart", "last contact February 2025"),
         ("camera_returned", "juni_forever", "juni_last")),
    Spec("who_marisol", "relationship", "Who is Marisol?",
         "My manager at Bellweather Labs, my old job; I left in September 2024.",
         ("manager at work", "Bellweather", "I left that job in September 2024"),
         ("q3_deck_ask", "tell_marisol", "mari_last_day")),
    Spec("who_clem", "relationship", "Who is Clem?",
         "My older sibling (Clem calls me \"the best little sibling\"), in the family chat with Mom and Dad.",
         ("my sibling", "in the family group chat"), ("clem_sibling", "mom60_plan")),
    Spec("who_theo", "relationship", "Who is Theo, and are we still in touch?",
         "A close friend who moved from Oakland to Denver in August 2023. Still in touch: the goose jokes go on "
         "and I visited him in June 2025.",
         ("close friend", "moved to Denver in 2023", "still in touch"), ("theo_moves", "theo_visit", "goose_last")),
    Spec("faded_friend", "relationship", "Which close friendship of mine faded over the last few years?",
         "Juniper: very close in 2023, rare in 2024, only a birthday text in 2025.",
         ("Juniper",), ("juni_forever", "juni_last")),
    Spec("who_gus", "relationship", "Who is Gus?",
         "The landlord of the apartment at 412 Alcott St.", ("landlord",), ("lease", "keys")),
)

MULTI_HOP: tuple[Spec, ...] = (
    Spec("hop_show_night", "multi_hop",
         "The night of Wren's first gallery show, who was I working late for, and on what?",
         "Marisol, my manager: the Q3 deck for a board meeting.", ("Marisol", "Q3 deck"),
         ("opening_reminder", "fight_excuse", "q3_deck_ask")),
    Spec("hop_makeup_days", "multi_hop",
         "How many days after the fight about Wren's show did we make up?",
         "3 days: the fight was on 2024-08-02 and we made up on 2024-08-05.",
         ("3 days", "August 2 to August 5, 2024"), ("fight_end", "makeup_yes")),
    Spec("hop_accept_days", "multi_hop", "How many days after I got the job offer did I accept it?",
         "3 days: the offer came on 2024-09-03 and I accepted on 2024-09-06.",
         ("3 days", "September 3 to September 6, 2024"), ("offer", "accept")),
    Spec("hop_camera_return", "multi_hop",
         "What did Juniper find under her couch, and when and where did she give it back?",
         "My Pentax K1000 film camera; she gave it back at Ritual on 2023-04-27.",
         ("Pentax K1000 film camera", "April 27, 2023", "Ritual"), ("camera_found", "camera_return_plan")),
    Spec("hop_trip_bday", "multi_hop", "Whose birthday was the Montreal trip for, and what date is that birthday?",
         "Wren's birthday, May 18.", ("Wren", "May 18"), ("montreal_idea", "wren_bday_2024")),
    Spec("hop_new_job", "multi_hop",
         "When I told my family about the new job, which company was it, and when did I start?",
         "Kestrel & Finch; I started on October 7, 2024.", ("Kestrel & Finch", "October 7, 2024"),
         ("family_offer", "offer")),
)

ABSENT: tuple[Spec, ...] = (
    Spec("absent_japan", "absent", "When did I go to Japan?", "The messages do not mention a trip to Japan.",
         ("no trip to Japan in the messages",), absent_terms=("japan", "tokyo", "kyoto", "osaka")),
    Spec("absent_dog", "absent", "What is my dog's name?", "The messages do not mention a dog.",
         ("no dog in the messages",), absent_terms=("dog", "puppy")),
    Spec("absent_brother", "absent", "What did Wren's brother say at Thanksgiving?",
         "The messages do not mention a brother of Wren.", ("no brother of Wren in the messages",),
         absent_terms=("brother",)),
    Spec("absent_airbnb_price", "absent", "How much did the Montreal Airbnb cost?",
         "The messages do not say what the Airbnb cost.", ("the price is not in the messages",),
         absent_terms=("$", "dollar", "per night", "cost")),
    Spec("absent_theo_wedding", "absent", "When did Theo get married?",
         "The messages do not mention Theo getting married.", ("no wedding of Theo in the messages",),
         absent_terms=("married", "wedding", "wife", "husband")),
    Spec("absent_promotion", "absent", "What did Marisol say about my promotion?",
         "The messages do not mention a promotion.", ("no promotion in the messages",),
         absent_terms=("promotion", "promoted")),
)

FOLLOW_UP: tuple[Spec, ...] = (
    Spec("follow_montreal_food", "follow_up", "What did we eat first there?",
         "Bagels (sesame and everything) from the wood oven place.", ("bagels",),
         ("bagel_line", "bagel_order"), prior="Who did I go to Montreal with?"),
    Spec("follow_theo_move", "follow_up", "When did he move?",
         "2023-08-19 (August 19, 2023).", ("August 19, 2023",), ("theo_moves",), prior="Who is Theo?"),
    Spec("follow_tell_manager", "follow_up", "When did I tell my manager at the old job?",
         "2024-09-09 (September 9, 2024), I told Marisol.", ("Marisol", "September 9, 2024"),
         ("tell_marisol",), prior="What job offer did I get in 2024?"),
    Spec("follow_who_found", "follow_up", "Who found it?",
         "Juniper; it was under her couch.", ("Juniper",), ("camera_found",), prior="What did I lose in April 2023?"),
    Spec("follow_makeup", "follow_up", "How did we make up?",
         "I apologized on 2024-08-05 and took Wren to Dumpling Moon for soup dumplings that night.",
         ("I apologized", "Dumpling Moon", "August 5, 2024"), ("makeup_apology", "makeup_yes", "makeup_dinner"),
         prior="When did Wren and I have our big fight in 2024?"),
    Spec("follow_who_picked", "follow_up", "Who picked that restaurant?",
         "Dad said Mom wanted Osteria Lupa.", ("Dad",), ("mom60_place",),
         prior="Where was Mom's 60th birthday dinner?"),
)

STATIC: tuple[Spec, ...] = (*FACT, *FIRST_LAST, *RELATIONSHIP, *MULTI_HOP, *ABSENT, *FOLLOW_UP)
