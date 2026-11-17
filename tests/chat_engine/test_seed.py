from chat_engine.labeler import MaskingLabeler
from chat_engine.seed import question_units

UNSAVED = "+17205550199"


def _units(question: str, labeler: MaskingLabeler | None = None) -> list[str]:
    return question_units(question, (labeler or MaskingLabeler()).redact)


def test_short_words_and_years_are_not_search_words() -> None:
    assert _units("Who do I text the most?") == ["who", "text", "the", "most"]
    assert _units("did she front me 10k in 2019?") == ["did", "she", "front", "10k"]
    assert _units("já vi o apartamento, é ótimo") == ["apartamento", "ótimo"]
    assert _units("학교 도서관 얘기") == ["학교", "도서관", "얘기"]
    assert _units("да, это он") == ["да", "это", "он"]
    assert _units("??") == []


def test_only_the_longest_words_of_a_long_question_are_kept() -> None:
    question = " ".join(f"word{'x' * n}" for n in range(1, 21))
    units = _units(question)
    assert len(units) == 12
    assert units[0] == "word" + "x" * 9 and units[-1] == "word" + "x" * 20


def test_handle_of_an_unsaved_person_is_not_a_search_word() -> None:
    labeler = MaskingLabeler(known_handles=[UNSAVED, "rk.unsaved@example.org"])
    units = _units("who is +1 (720) 555-0199 or rk.unsaved@example.org really?", labeler)
    assert units == ["who", "really"]
    saved = MaskingLabeler({UNSAVED: "Dana Whitlock"})
    assert "720" in _units("who is +1 (720) 555-0199 really?", saved)
