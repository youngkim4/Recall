import pytest

from chat_engine.registry import ConversationAliases, EvidenceRegistry, normalize_ref
from chat_engine.scope import Scope


def _message(message_id: str, text: str = "hello") -> dict:
    return {
        "id": message_id,
        "conversation": "c-a",
        "timestamp": "2024-01-01 10:00:00",
        "sent_at": 1,
        "is_from_me": False,
        "sender": "+17205550199",
        "text": text,
        "kind": "text",
        "attachments": [],
        "reply_to": None,
        "edited": False,
        "unsent": False,
    }


def test_registry_gives_one_stable_reference_per_message() -> None:
    registry = EvidenceRegistry()
    first = registry.add(_message("a"))
    second = registry.add(_message("b"))
    again = registry.add(_message("a"))
    assert (first, second, again) == ("m1", "m2", "m1")
    assert len(registry) == 2
    assert registry.get("m2")["id"] == "b"
    assert registry.get("m9") is None
    assert registry.ref_of("b") == "m2"
    assert registry.ref_of("zz") is None


def test_registry_does_not_alias_the_caller_message() -> None:
    registry = EvidenceRegistry()
    message = _message("a", "first")
    ref = registry.add(message)
    message["text"] = "changed later"
    assert registry.get(ref)["text"] == "first"


def test_registry_round_trips_through_a_snapshot() -> None:
    registry = EvidenceRegistry()
    registry.add(_message("a"))
    registry.add(_message("b"))
    restored = EvidenceRegistry.restore(registry.snapshot())
    assert restored.ref_of("b") == "m2"
    assert restored.add(_message("c")) == "m3"


def test_normalize_ref_accepts_common_spellings() -> None:
    assert normalize_ref(" M3 ") == "m3"
    assert normalize_ref("[m12]") == "m12"
    assert normalize_ref("m03") == "m3"
    assert normalize_ref("3") == ""
    assert normalize_ref("message three") == ""


def test_aliases_hide_conversation_ids() -> None:
    aliases = ConversationAliases()
    assert aliases.alias("+17205550199") == "c1"
    assert aliases.alias("chat99") == "c2"
    assert aliases.alias("+17205550199") == "c1"
    assert aliases.resolve("C2") == "chat99"
    assert aliases.resolve("c7") is None
    assert aliases.resolve("+17205550199") is None
    restored = ConversationAliases.restore(aliases.snapshot())
    assert restored.alias("chat99") == "c2"
    assert restored.alias("new") == "c3"


def test_scope_all_allows_everything() -> None:
    scope = Scope()
    assert not scope.is_limited
    assert scope.allows("anything")
    assert scope.narrow(None) is None
    assert scope.narrow(["a", "b"]) == ("a", "b")
    assert scope.clamp("2024-01-01", None) == ("2024-01-01", None)


def test_scope_never_widens_conversations() -> None:
    scope = Scope(conversations=("a", "b"))
    assert scope.is_limited
    assert scope.allows("a") and not scope.allows("z")
    assert scope.narrow(None) == ("a", "b")
    assert scope.narrow(["b", "z"]) == ("b",)
    assert scope.narrow(["z"]) == ()


def test_scope_never_widens_dates() -> None:
    scope = Scope(since="2024-01-01", until="2024-12-31")
    assert scope.is_limited
    assert scope.clamp(None, None) == ("2024-01-01", "2024-12-31")
    assert scope.clamp("2023-06-01", "2025-06-01") == ("2024-01-01", "2024-12-31")
    assert scope.clamp("2024-03-01", "2024-04-01") == ("2024-03-01", "2024-04-01")
    assert scope.allows_time("2024-05-05 09:00:00")
    assert not scope.allows_time("2025-01-01 00:00:00")
    assert not scope.allows_time("2023-12-31 23:59:59")


def test_scope_rejects_bad_days() -> None:
    with pytest.raises(ValueError):
        Scope(since="2024-13-40")
    with pytest.raises(ValueError):
        Scope(since="2024-05-01", until="2024-04-01")
