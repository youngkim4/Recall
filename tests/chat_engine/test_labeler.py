import time

from chat_engine.handles import Phone, handle_spans, is_mask, parse_phone, same_phone
from chat_engine.labeler import (
    EMAIL_MASK,
    UNKNOWN_MASK,
    MaskingLabeler,
    handle_kind,
    looks_like_handle,
    mask_handle,
)

UNSAVED_PHONE = "+17205550199"
UNSAVED_EMAIL = "rk.unsaved@example.org"
SAVED_PHONE = "+14155550123"


def _conversation(**fields: object) -> dict:
    base = {
        "id": UNSAVED_PHONE,
        "is_group": False,
        "name": "",
        "message_count": 3,
        "sent_count": 1,
        "received_count": 2,
        "first_timestamp": "2024-01-01 10:00:00",
        "last_timestamp": "2024-01-02 10:00:00",
        "members": [UNSAVED_PHONE],
        "person_id": None,
    }
    return {**base, **fields}


def test_handle_kind() -> None:
    assert handle_kind("+1 (720) 555-0199") == "phone"
    assert handle_kind("tel:+17205550199") == "phone"
    assert handle_kind("Mailto:RK.Unsaved@Example.org") == "email"
    assert handle_kind("chat8812345") == "other"
    assert handle_kind("Maya Okafor") == "other"
    assert handle_kind("262966") == "short_code"
    assert handle_kind("") == "other"


def test_one_number_in_many_spellings_is_the_same_number() -> None:
    usa = parse_phone("+1 (720) 555-0199")
    assert usa == Phone("17205550199", True)
    assert same_phone(usa, parse_phone("720-555-0199"))
    assert same_phone(usa, parse_phone("17205550199"))
    assert same_phone(parse_phone("+351912345678"), parse_phone("912 345 678"))
    assert same_phone(parse_phone("+33612345678"), parse_phone("06 12 34 56 78"))
    assert not same_phone(usa, parse_phone("+44 720 555 0199"))
    assert not same_phone(usa, parse_phone("720-555-0198"))
    assert not same_phone(usa, parse_phone("555-0199"))
    assert parse_phone("Maya") is None and parse_phone("262966") is None


def test_chat_guid_prefixes_are_understood() -> None:
    assert handle_kind("iMessage;-;+14155550123") == "phone"
    assert handle_kind("SMS;-;+14155550123") == "phone"
    assert handle_kind("iMessage;-;someone@example.org") == "email"
    assert MaskingLabeler({"+14155550123": "Ann"}).sender("iMessage;-;+14155550123") == "Ann"
    labeler = MaskingLabeler(known_handles=["iMessage;-;+17205550199"])
    assert labeler.redact("n +17205550199") == "n a number ending in 0199"


def test_looks_like_handle() -> None:
    assert looks_like_handle("+1 720 555 0199")
    assert looks_like_handle("someone@example.org")
    assert not looks_like_handle("Flat 24 crew")
    assert not looks_like_handle("Maya Okafor")


def test_mask_never_shows_the_handle() -> None:
    assert mask_handle(UNSAVED_PHONE) == "a number ending in 0199"
    assert mask_handle(UNSAVED_EMAIL) == EMAIL_MASK
    assert mask_handle("chat8812345") == UNKNOWN_MASK
    assert mask_handle("") == UNKNOWN_MASK


def test_short_code_is_shown_as_a_short_code() -> None:
    assert mask_handle("262966") == "the short code 262966"
    assert MaskingLabeler().sender("262966") == "the short code 262966"
    assert is_mask("the short code 262966") and is_mask("a number ending in 0199") and is_mask(EMAIL_MASK)
    assert not is_mask("Maya Okafor")
    assert MaskingLabeler(known_handles=["262966"]).redact("text STOP to 262966") == "text STOP to 262966"


def test_saved_name_is_used_as_it_is() -> None:
    labeler = MaskingLabeler({SAVED_PHONE: "Maya Okafor"})
    assert labeler.sender("+1 (415) 555-0123") == "Maya Okafor"
    assert labeler.sender(UNSAVED_PHONE) == "a number ending in 0199"
    assert labeler.sender(UNSAVED_EMAIL) == EMAIL_MASK


def test_saved_name_that_is_a_handle_is_not_trusted() -> None:
    labeler = MaskingLabeler({UNSAVED_PHONE: "+1 720 555 0199", UNSAVED_EMAIL: UNSAVED_EMAIL})
    assert labeler.sender(UNSAVED_PHONE) == "a number ending in 0199"
    assert labeler.sender(UNSAVED_EMAIL) == EMAIL_MASK


def test_direct_conversation_label() -> None:
    labeler = MaskingLabeler({SAVED_PHONE: "Maya Okafor"})
    saved = _conversation(id=SAVED_PHONE, members=[SAVED_PHONE])
    assert labeler.conversation(saved) == "Maya Okafor"
    assert labeler.conversation(_conversation()) == "a number ending in 0199"
    email = _conversation(id=UNSAVED_EMAIL, members=[UNSAVED_EMAIL])
    assert labeler.conversation(email) == EMAIL_MASK


def test_direct_conversation_uses_archive_name_only_when_safe() -> None:
    labeler = MaskingLabeler()
    named = _conversation(name="Dana Whitlock")
    assert labeler.conversation(named) == "Dana Whitlock"
    raw = _conversation(name="+1 (720) 555-0199")
    assert labeler.conversation(raw) == "a number ending in 0199"


def test_direct_conversation_with_opaque_id_uses_member_handle() -> None:
    labeler = MaskingLabeler({SAVED_PHONE: "Maya Okafor"})
    opaque = _conversation(id="conv-17", members=[SAVED_PHONE])
    assert labeler.conversation(opaque) == "Maya Okafor"
    unknown = _conversation(id="conv-18", members=[UNSAVED_PHONE])
    assert labeler.conversation(unknown) == "a number ending in 0199"


def test_group_label_uses_its_name_or_its_members() -> None:
    labeler = MaskingLabeler({SAVED_PHONE: "Maya Okafor", "+14155550124": "Jonas Weber"})
    named = _conversation(id="chat1", is_group=True, name="Lisbon crew", members=[SAVED_PHONE])
    assert labeler.conversation(named) == "Lisbon crew"
    unnamed = _conversation(
        id="chat2",
        is_group=True,
        members=[SAVED_PHONE, "+14155550124", UNSAVED_PHONE, "+13035550111", "+13035550112"],
    )
    label = labeler.conversation(unnamed)
    assert label == "a group chat with Maya Okafor, Jonas Weber, a number ending in 0199 and 2 others"
    assert UNSAVED_PHONE not in label


def test_group_name_that_is_a_handle_is_masked() -> None:
    labeler = MaskingLabeler()
    group = _conversation(id="chat3", is_group=True, name=UNSAVED_PHONE, members=[UNSAVED_PHONE])
    assert "7205550199" not in labeler.conversation(group)


def test_redact_masks_unsaved_handles_in_free_text() -> None:
    labeler = MaskingLabeler({SAVED_PHONE: "Maya Okafor"}, known_handles=[UNSAVED_PHONE, UNSAVED_EMAIL])
    text = "new phone: (720) 555-0199, or mail RK.Unsaved@example.org. Maya is +1 415 555 0123."
    out = labeler.redact(text)
    assert "555-0199" not in out
    assert "720" not in out
    assert "example.org" not in out
    assert "a number ending in 0199" in out
    assert EMAIL_MASK in out
    # a saved person's number is not masked
    assert "+1 415 555 0123" in out


def test_redact_learns_handles_it_has_labelled() -> None:
    labeler = MaskingLabeler()
    assert labeler.redact("call 720-555-0199") == "call 720-555-0199"
    labeler.sender(UNSAVED_PHONE)
    assert labeler.redact("call 720-555-0199") == "call a number ending in 0199"
    assert labeler.redact("call +1 720 555 0199 now") == "call a number ending in 0199 now"


def test_redact_leaves_other_numbers_alone() -> None:
    labeler = MaskingLabeler(known_handles=[UNSAVED_PHONE])
    text = "order 4420019912, 2024-03-04 18:22, $1,250,000 and 911"
    assert labeler.redact(text) == text


def test_redact_handles_number_glued_to_longer_digit_runs() -> None:
    labeler = MaskingLabeler(known_handles=[UNSAVED_PHONE])
    out = labeler.redact("ref 99-7205550199-55 ok")
    assert "7205550199" not in out
    assert "a number ending in 0199" in out


def test_redact_empty_and_plain_text() -> None:
    labeler = MaskingLabeler(known_handles=[UNSAVED_PHONE])
    assert labeler.redact("") == ""
    assert labeler.redact("see you at eight") == "see you at eight"


def test_two_unsaved_numbers_in_one_run_are_both_masked() -> None:
    labeler = MaskingLabeler(known_handles=["+14155550123", "+14155550124"])
    both = "a number ending in 0123 a number ending in 0124"
    assert labeler.redact("call 4155550123 4155550124 now") == f"call {both} now"
    assert labeler.redact("415 555 0123 415 555 0124") == both
    assert labeler.redact("4155550123-4155550124") == "a number ending in 0123-a number ending in 0124"
    assert labeler.redact("+1 415 555 0123, +1 (415) 555-0124") == "a number ending in 0123, a number ending in 0124"


def test_national_spelling_of_an_international_number_is_masked() -> None:
    assert MaskingLabeler(known_handles=["+351912345678"]).redact("912 345 678") == "a number ending in 5678"
    assert MaskingLabeler(known_handles=["+33612345678"]).redact("06 12 34 56 78") == "a number ending in 5678"
    assert MaskingLabeler(known_handles=["+34612345678"]).redact("tel 612 34 56 78.") == "tel a number ending in 5678."
    assert MaskingLabeler(known_handles=["07205550199"]).redact("7205550199") == "a number ending in 0199"


def test_saved_name_is_found_by_any_spelling_of_the_number() -> None:
    labeler = MaskingLabeler({"+351912345678": "Inês Duarte"})
    assert labeler.sender("912345678") == "Inês Duarte"
    assert labeler.sender("+351 912 345 678") == "Inês Duarte"
    assert labeler.redact("ligue 912 345 678") == "ligue 912 345 678"


def test_same_ending_in_two_countries_is_two_people() -> None:
    labeler = MaskingLabeler({"+14155550123": "Ann"}, ["+444155550123"])
    assert labeler.sender("+444155550123") == "a number ending in 0123"
    assert labeler.sender("+14155550123") == "Ann"
    assert labeler.redact("uk +444155550123 now") == "uk a number ending in 0123 now"
    assert labeler.redact("Ann is +1 415 555 0123") == "Ann is +1 415 555 0123"
    assert labeler.redact("Ann is 415 555 0123") == "Ann is 415 555 0123"


def test_archive_names_that_hold_a_handle_are_scrubbed() -> None:
    labeler = MaskingLabeler()
    group = _conversation(id="chat1", is_group=True, name="Dinner call +1 415 555 0123", members=[])
    assert labeler.conversation(group) == "Dinner call a number ending in 0123"
    mail = _conversation(id="chat2", is_group=True, name="bob@example.com and friends", members=[])
    assert labeler.conversation(mail) == "an unsaved email contact and friends"
    direct = _conversation(id="+14155550123", name="Chris 415-555-0123", members=["+14155550123"])
    assert labeler.conversation(direct) == "Chris a number ending in 0123"
    plain = _conversation(id="chat3", is_group=True, name="Flat 24, since 2019", members=[])
    assert labeler.conversation(plain) == "Flat 24, since 2019"


def test_redact_is_fast_on_long_runs_with_no_handle() -> None:
    labeler = MaskingLabeler(known_handles=["a@b.com", "+17205550199"])
    started = time.perf_counter()
    labeler.redact("a" * 50_000)
    labeler.redact("1 " * 25_000)
    labeler.redact("1-" * 25_000)
    handle_spans("a" * 50_000)
    assert time.perf_counter() - started < 1.5


def test_email_addresses_are_matched_whole() -> None:
    labeler = MaskingLabeler(known_handles=["rk.unsaved@example.org"])
    assert labeler.redact("mail RK.Unsaved@example.org.") == f"mail {EMAIL_MASK}."
    assert labeler.redact("xrk.unsaved@example.org") == "xrk.unsaved@example.org"
