"""Tests for the list of people who wrote in a group chat."""
import pandas as pd

from group_people import SPEAKER_LIMIT, group_speakers

AVERY = "+15550000001"
AVERY_EMAIL = "avery@example.com"
BLAKE = "+15550000002"
UNSAVED = "+15550000009"
NAMES = {AVERY: "Avery Stone", AVERY_EMAIL: "Avery Stone", BLAKE: "Blake Reyes"}


def senders(*handles):
    return pd.Series(list(handles), dtype="string")


def test_people_are_listed_busiest_first_with_their_share():
    count, people = group_speakers(senders(BLAKE, AVERY, BLAKE, BLAKE), NAMES)

    assert count == 2
    assert people == [
        {"handle": BLAKE, "name": "Blake Reyes", "count": 3, "share": 0.75},
        {"handle": AVERY, "name": "Avery Stone", "count": 1, "share": 0.25},
    ]


def test_one_person_writing_from_two_handles_is_one_person():
    count, people = group_speakers(senders(AVERY, AVERY_EMAIL, AVERY_EMAIL, BLAKE), NAMES)

    assert count == 2
    # the handle kept is the one they wrote from most
    assert people[0] == {"handle": AVERY_EMAIL, "name": "Avery Stone", "count": 3, "share": 0.75}


def test_someone_not_in_contacts_is_listed_with_no_name():
    _, people = group_speakers(senders(UNSAVED, BLAKE, BLAKE), NAMES)

    assert people[1] == {"handle": UNSAVED, "name": "", "count": 1, "share": 0.333}


def test_two_unsaved_numbers_stay_two_people():
    count, people = group_speakers(senders("+15550000008", UNSAVED), {})

    assert count == 2
    assert [person["handle"] for person in people] == ["+15550000008", UNSAVED]


def test_people_with_the_same_count_keep_one_order():
    first = group_speakers(senders(BLAKE, AVERY), NAMES)
    second = group_speakers(senders(AVERY, BLAKE), NAMES)

    assert first == second


def test_blank_senders_are_nobody():
    count, people = group_speakers(senders("", "  ", BLAKE), NAMES)

    assert count == 1
    assert [person["handle"] for person in people] == [BLAKE]


def test_a_very_large_group_is_cut_but_still_counted_in_full():
    handles = [f"+1555000{index:04d}" for index in range(SPEAKER_LIMIT + 3)]

    count, people = group_speakers(senders(*handles), {})

    assert count == SPEAKER_LIMIT + 3
    assert len(people) == SPEAKER_LIMIT


def test_nobody_wrote():
    assert group_speakers(senders(), NAMES) == (0, [])
