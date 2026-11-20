"""The hard questions written by hand. Their answers are fixed facts of the hard scenes.

- paraphrase: no content word of the question is in a gold message; a search by words alone misses.
- disambiguate: the same first name, or the same topic, in two places; time or context decides.
- temporal: what was going on in a period of the owner's life.
- needle: one fact said once in a busy group chat.
"""

from __future__ import annotations

from evals.question_types import Spec

PARAPHRASE: tuple[Spec, ...] = (
    Spec("para_teeth", "paraphrase", "When did I get my wisdom teeth removed?",
         "2023-03-08 (March 8, 2023): the dentist took out all four back molars.", ("March 8, 2023",), ("molars",)),
    Spec("para_cat", "paraphrase", "Which friend adopted a cat, and what is its name?",
         "Odalys; the kitten is Biscuit (November 2023).", ("Odalys", "Biscuit"), ("biscuit",)),
    Spec("para_bone", "paraphrase", "Did I ever break a bone?",
         "Yes: I fractured my left wrist at the climbing gym on 2021-03-20 and had a cast for 6 weeks.",
         ("fractured my wrist", "March 2021"), ("wrist_where", "wrist", "wrist_which")),
    Spec("para_license", "paraphrase", "When did I pass my driving test?",
         "2020-08-14 (August 14, 2020).", ("August 14, 2020",), ("license",)),
    Spec("para_cello", "paraphrase", "What instrument did I start learning, and when?",
         "The cello, from 2022-10-05 (October 2022).", ("cello", "October 2022"), ("cello",)),
    Spec("para_van", "paraphrase", "Who helped me move apartments in 2022?",
         "Rafferty, with his van, on 2022-09-17.", ("Rafferty", "his van"), ("van", "van_reply")),
    Spec("para_30th", "paraphrase", "Where did I go for my 30th?", "Big Sur (July 2025), in a cabin.",
         ("Big Sur",), ("big_sur",)),
    Spec("para_plant", "paraphrase", "What did I name the plant Odalys gave me?",
         "Gertrude (a pothos, January 2023).", ("Gertrude",), ("plant_ask", "plant_name")),
)

DISAMBIGUATE: tuple[Spec, ...] = (
    Spec("dis_maya_work", "disambiguate", "Which Maya do I work with, and what is her last name?",
         "Maya Ostrowski, who runs research at Kestrel & Finch.", ("Maya Ostrowski",), ("mayao_intro",)),
    Spec("dis_maya_race", "disambiguate", "Which Maya ran a half marathon, and when?",
         "Maya Lindqvist, on 2023-10-08 (13.1 miles in 2:04).", ("Maya Lindqvist", "October 8, 2023"),
         ("maya_half",)),
    Spec("dis_chicago_who", "disambiguate", "Who did I go to Chicago with in 2021?",
         "The chicago crew: Maya Lindqvist, Rafferty, Odalys and Kip.", ("Maya", "Rafferty", "Odalys", "Kip"),
         ("chi21_plan",)),
    Spec("dis_chicago_hotel", "disambiguate", "Where did I stay on my work trip to Chicago?",
         "The Halcyon Hotel downtown, for the Prairie Design Summit in April 2023.", ("Halcyon Hotel",),
         ("chi23_hotel",)),
    Spec("dis_chicago_2021_stay", "disambiguate", "Where did we stay in Chicago in 2021?",
         "At Kip's cousin's place in Logan Square.", ("Kip's cousin's place", "Logan Square"), ("chi21_plan",)),
    Spec("dis_address_before", "disambiguate", "What was my address before Linden St?",
         "88 Fulton Ave, apt 3 (March 2020 to September 2022).", ("88 Fulton Ave", "apt 3"),
         ("fulton_movein", "fulton_lease")),
    Spec("dis_offer_declined", "disambiguate", "Which job offer did I turn down, and why?",
         "Larkspur Health's senior designer offer in April 2021: the commute to the South Bay.",
         ("Larkspur Health", "2021", "the commute"), ("larkspur_offer", "larkspur_no")),
)

TEMPORAL: tuple[Spec, ...] = (
    Spec("temp_fall22", "temporal", "What was going on in my life in fall 2022?",
         "I moved to 1550 Linden St (2022-09-17), started cello lessons (2022-10-05), and Rafferty moved to "
         "Lisbon (2022-11-12).",
         ("moved to 1550 Linden St in September 2022", "started cello in October 2022",
          "Rafferty moved to Lisbon in November 2022"), ("linden_movein", "cello", "raff_lisbon")),
    Spec("temp_summer20", "temporal", "What was going on in my life in summer 2020?",
         "I started at Bellweather (2020-06-01) and got my driver's license (2020-08-14).",
         ("started at Bellweather in June 2020", "got my driver's license in August 2020"),
         ("bw_first_day", "license")),
    Spec("temp_spring21", "temporal", "What happened in my life in spring 2021?",
         "I fractured my wrist (2021-03-20) and turned down a job offer from Larkspur Health (2021-04-15).",
         ("fractured my wrist in March 2021", "turned down Larkspur Health in April 2021"),
         ("wrist", "larkspur_no")),
    Spec("temp_april23", "temporal", "What was I doing the week of April 18, 2023?",
         "A work trip to Chicago with Benedikt for the Prairie Design Summit, at the Halcyon Hotel.",
         ("work trip to Chicago", "Prairie Design Summit"), ("chi23_hotel", "chi23_maya")),
)

NEEDLE: tuple[Spec, ...] = (
    Spec("needle_wifi", "needle", "What is the lobby wifi password of my old Linden St building?",
         "fernhollow42.", ("fernhollow42",), ("wifi",)),
    Spec("needle_door", "needle", "What was the door code at the climbing gym?", "4417 (from January 2022).",
         ("4417",), ("door_code",)),
    Spec("needle_vet", "needle", "Which cousin got into vet school?", "Lottie (March 2024).", ("Lottie",), ("vet",)),
    Spec("needle_floor", "needle", "Which floor did my team at Kestrel move to?", "The 9th floor, in January.",
         ("9th floor",), ("ninth_floor",)),
    Spec("needle_reunion", "needle", "Where was the chicago crew reunion dinner booked?",
         "Tavola Rosa on 18th, at 8pm.", ("Tavola Rosa",), ("tavola",)),
    Spec("needle_parking", "needle", "Who in my building offered a spare parking spot?",
         "Wendell, in unit 2 (January 2024).", ("Wendell",), ("parking",)),
    Spec("needle_book", "needle", "What was the book club pick for June 2023?", "The Salt Orchard.",
         ("The Salt Orchard",), ("salt_orchard",)),
)

LONG_STORY_JOB = Spec(
    "story_work", "long_story", "How did my work life change from 2020 to 2025?",
    "I started at Bellweather in June 2020, turned down Larkspur Health in April 2021, worked under Marisol "
    "until September 2024, and started at Kestrel & Finch on October 7, 2024.",
    ("started at Bellweather in June 2020", "turned down Larkspur Health in 2021",
     "moved to Kestrel & Finch in October 2024"),
    ("bw_first_day", "larkspur_no", "accept", "mayao_intro"),
)  # fmt: skip

HARD_STATIC: tuple[Spec, ...] = (*PARAPHRASE, *DISAMBIGUATE, *TEMPORAL, *NEEDLE, LONG_STORY_JOB)
