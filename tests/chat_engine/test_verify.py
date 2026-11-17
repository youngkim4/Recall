from chat_engine.verify import quoted_spans, unverified_quotes

SOURCES = [
    "yes!! the lease got signed btw",
    "we move in on the first of April",
    "don’t worry about the deposit, it’s covered",
    "O apartamento é ótimo, a cozinha é enorme",
]


def test_quoted_spans_finds_the_usual_marks() -> None:
    answer = (
        'She wrote "the lease got signed btw" and “we move in on the first of April”, '
        "then «a cozinha é enorme» and 'short'."
    )
    assert quoted_spans(answer) == [
        "the lease got signed btw",
        "we move in on the first of April",
        "a cozinha é enorme",
        "short",
    ]


def test_single_quotes_do_not_break_on_apostrophes() -> None:
    spans = quoted_spans("He said 'don't worry about the deposit' and it's fine.")
    assert spans == ["don't worry about the deposit"]
    assert quoted_spans("the Smiths' house and the Jones' car") == []


def test_blockquote_lines_count_as_quotes() -> None:
    assert quoted_spans("Here it is:\n> we move in on the first of April\nDone.") == [
        "we move in on the first of April"
    ]


def test_real_quotes_pass() -> None:
    answer = 'She said "the lease got signed btw" [m2] and “We move in on the first of April.”'
    assert unverified_quotes(answer, SOURCES) == []


def test_made_up_quote_is_flagged() -> None:
    answer = 'She said "the lease got signed btw" and "the landlord cried with joy".'
    assert unverified_quotes(answer, SOURCES) == ["the landlord cried with joy"]


def test_short_quotes_are_not_checked() -> None:
    assert unverified_quotes('She called it "the flat" and said "no way".', SOURCES) == []


def test_spaces_quote_marks_and_case_are_normalised() -> None:
    answer = 'He wrote "Don\'t  worry about the\ndeposit, it\'s covered".'
    assert unverified_quotes(answer, SOURCES) == []


def test_elided_quote_is_checked_piece_by_piece() -> None:
    good = 'She said "the lease got signed ... first of April".'
    assert unverified_quotes(good, SOURCES) == []
    bad = 'She said "the lease got signed […] and the dog ate it all".'
    assert unverified_quotes(bad, SOURCES) == ["the lease got signed […] and the dog ate it all"]


def test_reworded_quote_is_flagged() -> None:
    answer = 'She said "the lease was signed yesterday".'
    assert unverified_quotes(answer, SOURCES) == ["the lease was signed yesterday"]


def test_citation_markers_inside_a_quote_are_ignored() -> None:
    answer = 'She said "the lease got signed btw [m2]".'
    assert unverified_quotes(answer, SOURCES) == []


def test_each_flagged_quote_is_listed_once() -> None:
    answer = '"totally invented line" and again "totally invented line".'
    assert unverified_quotes(answer, SOURCES) == ["totally invented line"]
