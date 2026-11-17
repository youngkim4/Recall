from chat_engine.labeler import (
    EMAIL_MASK,
    UNKNOWN_MASK,
    MaskingLabeler,
    handle_key,
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


def test_handle_kind_and_key() -> None:
    assert handle_kind("+1 (720) 555-0199") == "phone"
    assert handle_kind("tel:+17205550199") == "phone"
    assert handle_kind("Mailto:RK.Unsaved@Example.org") == "email"
    assert handle_kind("chat8812345") == "other"
    assert handle_kind("Maya Okafor") == "other"
    assert handle_key("+1 (720) 555-0199") == "7205550199"
    assert handle_key("720-555-0199") == "7205550199"
    assert handle_key("RK.Unsaved@Example.org") == "rk.unsaved@example.org"
    assert handle_key("Maya Okafor") == ""


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
