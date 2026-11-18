"""How a digest that is too long is cut: every period stays, the summaries share the cut, weak moments go."""

from chat_engine.digest_fit import Fit, cut_summary, fit_digest
from chat_engine.render import estimate_tokens


def test_a_digest_that_fits_is_not_cut() -> None:
    fit = fit_digest(fixed=50, summaries=[100, 200], moments=[[30, 30], [40]], room=450)
    assert fit == Fit(share=1.0, kept=(2, 1))
    assert not fit.cut(moments=[[30, 30], [40]])


def test_moments_go_from_the_weakest_end_of_the_period_that_has_the_most() -> None:
    moments = [[30, 30, 30], [40, 40], [50]]
    assert fit_digest(fixed=0, summaries=[10, 10, 10], moments=moments, room=30 + 219).kept == (2, 2, 1)
    assert fit_digest(fixed=0, summaries=[10, 10, 10], moments=moments, room=30 + 189).kept == (1, 2, 1)
    assert fit_digest(fixed=0, summaries=[10, 10, 10], moments=moments, room=30 + 149).kept == (1, 1, 1)
    assert fit_digest(fixed=0, summaries=[10, 10, 10], moments=moments, room=30 + 119).kept == (0, 1, 1)
    assert fit_digest(fixed=0, summaries=[10, 10, 10], moments=moments, room=30).kept == (0, 0, 0)
    fit = fit_digest(fixed=0, summaries=[10, 10, 10], moments=moments, room=30 + 149)
    assert fit.share == 1.0 and fit.dropped(moments) == 3 and fit.cut(moments)


def test_summaries_that_do_not_fit_are_cut_by_the_same_share_and_no_moment_is_shown() -> None:
    fit = fit_digest(fixed=100, summaries=[300, 100, 600], moments=[[20], [], [20, 20]], room=600)
    assert fit.kept == (0, 0, 0)
    assert fit.share == 0.5
    assert [fit.summary_tokens(count) for count in (300, 100, 600)] == [150, 50, 300]


def test_fixed_lines_that_fill_the_room_leave_nothing_for_the_summaries() -> None:
    fit = fit_digest(fixed=500, summaries=[100, 100], moments=[[10], [10]], room=400)
    assert fit.share == 0.0 and fit.kept == (0, 0) and fit.summary_tokens(100) == 0


def test_no_period_is_no_cut() -> None:
    assert fit_digest(fixed=10, summaries=[], moments=[], room=5) == Fit(share=1.0, kept=())


def test_a_summary_is_cut_to_its_tokens_with_a_mark() -> None:
    text = "They talk about the move every day. " * 20
    assert cut_summary(text, estimate_tokens(text), estimate_tokens) == text
    half = cut_summary(text, estimate_tokens(text) // 2, estimate_tokens)
    assert half.endswith(" [cut]") and text.startswith(half[:-6].rstrip())
    assert estimate_tokens(half) <= estimate_tokens(text) // 2
    assert cut_summary(text, 2, estimate_tokens) == "[cut]"
    assert cut_summary(text, 0, estimate_tokens) == "[cut]"
