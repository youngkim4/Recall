from chat_engine.textnorm import fold, query_units, unit_matches, words


def test_fold_ignores_case_and_diacritics() -> None:
    assert fold("São Paulo") == "sao paulo"
    assert fold("CAFÉ") == fold("cafe")
    assert fold("Straße") == "strasse"


def test_words_are_unicode_aware() -> None:
    assert words("Você já viu o contrato?") == ["voce", "ja", "viu", "o", "contrato"]
    assert words("don't stop") == ["don", "t", "stop"]
    assert words("10k up front!") == ["10k", "up", "front"]
    assert words("") == []


def test_query_units_keep_phrases_and_prefixes() -> None:
    assert query_units('lease "new flat" sign*') == ["lease", '"new flat"', "sign*"]


def test_query_units_survive_bad_syntax() -> None:
    assert query_units('lease "new flat') == ["lease", "new", "flat"]
    assert query_units("  ") == []
    assert query_units('""') == []


def test_unit_matches_word_phrase_and_prefix() -> None:
    text = "We signed the new flat lease on Friday"
    assert unit_matches("lease", text)
    assert unit_matches("LEASE", text)
    assert not unit_matches("leas", text)
    assert unit_matches("leas*", text)
    assert unit_matches('"new flat"', text)
    assert not unit_matches('"flat new"', text)


def test_unit_matches_ignores_diacritics() -> None:
    assert unit_matches("apartamento", "o apartaménto é ótimo")
    assert unit_matches("otimo", "o apartamento é ótimo")
