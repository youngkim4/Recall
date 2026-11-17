"""Which handles are one person. Pure rules, no database."""

from __future__ import annotations

import pytest

from archive.handles import handle_match_key
from archive.people import (
    ContactGroup,
    HandleRow,
    PersonPlan,
    contact_groups,
    plan_people,
)

PHONE = "+15550100123"
EMAIL = "rowan@example.com"
OTHER_PHONE = "+15550100456"
OTHER_EMAIL = "mira@example.com"


def handle(handle_id: int, text: str, person_id: int | None = None) -> HandleRow:
    return HandleRow(id=handle_id, match_key=handle_match_key(text), person_id=person_id)


def groups(*records: dict[str, object]) -> tuple[ContactGroup, ...]:
    return contact_groups(list(records))[0]


def by_handles(plans: tuple[PersonPlan, ...]) -> dict[tuple[int, ...], PersonPlan]:
    return {plan.handle_ids: plan for plan in plans}


class TestContactGroups:
    def test_phone_and_email_of_one_record_share_a_group(self) -> None:
        made = groups({"name": "Rowan Vale", "phones": ["(555) 010-0123"], "emails": ["Rowan@Example.com"]})
        assert made == (ContactGroup("Rowan Vale", frozenset({"p:5550100123", "e:rowan@example.com"})),)

    def test_two_records_with_the_same_name_stay_apart(self) -> None:
        made = groups(
            {"name": "Sam Lee", "phones": ["555-010-0123"], "emails": []},
            {"name": "Sam Lee", "phones": ["555-010-0456"], "emails": []},
        )
        assert [group.name for group in made] == ["Sam Lee", "Sam Lee"]
        assert made[0].keys.isdisjoint(made[1].keys)

    def test_duplicate_cards_with_one_name_and_a_shared_number_are_one_group(self) -> None:
        made = groups(
            {"name": "Sam Lee", "phones": ["555-010-0123"], "emails": []},
            {"name": "sam  lee", "phones": ["+1 555 010 0123"], "emails": ["sam@example.com"]},
        )
        assert len(made) == 1
        assert made[0].name == "Sam Lee"
        assert made[0].keys == frozenset({"p:5550100123", "e:sam@example.com"})

    def test_a_card_that_links_two_earlier_cards_joins_all_three_under_the_first(self) -> None:
        for _ in range(20):  # set order must not change the result
            made = groups(
                {"name": "Mira Okafor", "phones": ["555-010-0999"], "emails": []},
                {"name": "Sam Lee", "phones": ["555-010-0123"], "emails": []},
                {"name": "SAM LEE", "phones": ["555-010-0456"], "emails": []},
                {"name": "sam lee", "phones": ["555-010-0123", "555-010-0456"], "emails": ["sam@example.com"]},
            )
            assert [group.name for group in made] == ["Mira Okafor", "Sam Lee"]
            assert made[1].keys == frozenset({"p:5550100123", "p:5550100456", "e:sam@example.com"})

    def test_different_names_with_a_shared_number_stay_apart(self) -> None:
        made = groups(
            {"name": "Tessa Lind", "phones": ["555-010-0123"], "emails": []},
            {"name": "Jonas Brandt", "phones": ["555-010-0123"], "emails": []},
        )
        assert [group.name for group in made] == ["Tessa Lind", "Jonas Brandt"]

    def test_bad_records_are_counted_and_left_out(self) -> None:
        made, skipped = contact_groups(
            [
                "not a record",
                {"name": "No Handles"},
                {"name": "Odd Lists", "phones": "5550100123", "emails": None},
                {"name": "Mixed", "phones": [None, 5550100456, "555-010-0789"], "emails": ["no-at-sign"]},
            ]
        )
        assert skipped == 1
        assert made == (ContactGroup("Mixed", frozenset({"p:5550100789"})),)

    def test_contacts_must_be_a_list(self) -> None:
        with pytest.raises(TypeError):
            contact_groups({"name": "Rowan Vale"})  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            contact_groups("Rowan Vale")  # type: ignore[arg-type]


class TestPlanWithContacts:
    def test_phone_and_email_become_one_person(self) -> None:
        plans = plan_people(
            [handle(1, PHONE), handle(2, EMAIL), handle(3, OTHER_PHONE)],
            groups({"name": "Rowan Vale", "phones": ["555-010-0123"], "emails": [EMAIL]}),
        )
        found = by_handles(plans)
        assert found[(1, 2)].name == "Rowan Vale"
        assert found[(3,)].name == ""
        assert len(plans) == 2

    def test_same_named_records_are_two_people(self) -> None:
        plans = plan_people(
            [handle(1, PHONE), handle(2, OTHER_PHONE)],
            groups(
                {"name": "Sam Lee", "phones": [PHONE], "emails": []},
                {"name": "Sam Lee", "phones": [OTHER_PHONE], "emails": []},
            ),
        )
        assert sorted(plan.handle_ids for plan in plans) == [(1,), (2,)]
        assert {plan.name for plan in plans} == {"Sam Lee"}

    def test_handle_shared_by_two_names_goes_to_the_first_record(self) -> None:
        plans = plan_people(
            [handle(1, PHONE), handle(2, OTHER_EMAIL)],
            groups(
                {"name": "Tessa Lind", "phones": [PHONE], "emails": []},
                {"name": "Jonas Brandt", "phones": [PHONE], "emails": [OTHER_EMAIL]},
            ),
        )
        found = by_handles(plans)
        assert found[(1,)].name == "Tessa Lind"
        assert found[(2,)].name == "Jonas Brandt"

    def test_empty_contact_list_makes_every_handle_its_own_unnamed_person(self) -> None:
        plans = plan_people([handle(1, PHONE, 7), handle(2, EMAIL, 7)], ())
        assert sorted((plan.handle_ids, plan.name) for plan in plans) == [((1,), ""), ((2,), "")]

    def test_every_handle_is_in_exactly_one_plan(self) -> None:
        rows = [handle(index, f"+1555010{index:04d}") for index in range(1, 40)]
        plans = plan_people(rows, groups({"name": "Rowan Vale", "phones": ["+15550100005", "+15550100006"]}))
        seen = sorted(handle_id for plan in plans for handle_id in plan.handle_ids)
        assert seen == list(range(1, 40))


class TestPersonIdsStayStable:
    def test_unchanged_people_keep_their_ids(self) -> None:
        book = groups({"name": "Rowan Vale", "phones": [PHONE], "emails": [EMAIL]})
        plans = plan_people([handle(1, PHONE, 4), handle(2, EMAIL, 4), handle(3, OTHER_PHONE, 9)], book)
        found = by_handles(plans)
        assert found[(1, 2)].person_id == 4
        assert found[(3,)].person_id == 9

    def test_merge_keeps_the_lower_id(self) -> None:
        book = groups({"name": "Rowan Vale", "phones": [PHONE], "emails": [EMAIL]})
        plans = plan_people([handle(1, PHONE, 9), handle(2, EMAIL, 5)], book)
        assert plans == (PersonPlan(person_id=5, name="Rowan Vale", handle_ids=(1, 2)),)

    def test_merge_prefers_the_person_with_more_of_the_handles(self) -> None:
        book = groups({"name": "Rowan Vale", "phones": [PHONE, OTHER_PHONE], "emails": [EMAIL]})
        plans = plan_people([handle(1, PHONE, 9), handle(2, OTHER_PHONE, 9), handle(3, EMAIL, 5)], book)
        assert plans[0].person_id == 9

    def test_split_keeps_the_id_on_the_first_handle(self) -> None:
        plans = plan_people([handle(1, PHONE, 4), handle(2, EMAIL, 4)], ())
        found = by_handles(plans)
        assert found[(1,)].person_id == 4
        assert found[(2,)].person_id is None

    def test_new_handle_gets_a_new_person(self) -> None:
        plans = plan_people([handle(1, PHONE, 4), handle(2, OTHER_PHONE)], ())
        assert by_handles(plans)[(2,)].person_id is None


class TestPlanWithoutContacts:
    def test_existing_people_are_kept_and_names_are_not_touched(self) -> None:
        plans = plan_people([handle(1, PHONE, 4), handle(2, EMAIL, 4), handle(3, OTHER_PHONE)], None)
        found = by_handles(plans)
        assert found[(1, 2)] == PersonPlan(person_id=4, name=None, handle_ids=(1, 2))
        assert found[(3,)] == PersonPlan(person_id=None, name="", handle_ids=(3,))
