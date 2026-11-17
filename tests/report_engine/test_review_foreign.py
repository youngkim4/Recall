"""Foreign code at the seams: the source, the labeler, and engine errors that come late."""

from collections.abc import Iterator
from typing import Any

import pytest

from report_engine import pipeline
from report_engine.errors import ConfigError, RunError, SourceError
from report_engine.estimate import estimate
from report_engine.fake_llm import FakeLLM
from report_engine.ports import SourceMessage

from .support import SMALL, SMALL_OPTIONS, run
from .synth import CONVERSATION, MINA, WordCounter, labeler, message, model_info


class BrokenAtCall:
    """Fails before it gives a message. Its error text holds a handle."""

    def messages(
        self, conversation: str, since: str | None, until: str | None
    ) -> Iterator[SourceMessage]:
        raise OSError(f"can not read the thread of {MINA}")


class BrokenMidway:
    """Gives two messages, then fails as a database can. Its error text holds
    words of a message."""

    def messages(
        self, conversation: str, since: str | None, until: str | None
    ) -> Iterator[SourceMessage]:
        yield message(1, "2024-06-01 09:00:00", "first")
        yield message(2, "2024-06-01 09:01:00", "second")
        raise RuntimeError("row 3 holds: a private line of text")


class NotMessages:
    """Gives values that are not messages."""

    def messages(self, conversation: str, since: str | None, until: str | None) -> Iterator[Any]:
        yield 17


@pytest.mark.parametrize(
    "source, kind",
    [(BrokenAtCall(), "OSError"), (BrokenMidway(), "RuntimeError"), (NotMessages(), "TypeError")],
)
def test_an_error_of_the_source_is_a_source_error_that_names_only_its_type(
    source: Any, kind: str
) -> None:
    llm = FakeLLM()
    with pytest.raises(SourceError) as caught:
        run([], llm=llm, source=source)
    text = str(caught.value)
    assert "source" in text and kind in text
    assert "private" not in text and MINA not in text
    assert type(caught.value.__cause__).__name__ == kind
    assert llm.calls == []


def test_an_error_of_the_labeler_is_a_source_error_that_names_only_its_type() -> None:
    names: dict[str, str] = {}
    llm = FakeLLM()
    with pytest.raises(SourceError) as caught:
        run(SMALL, llm=llm, labeler=lambda sender: names[sender])
    text = str(caught.value)
    assert "labeler" in text and "KeyError" in text
    assert MINA not in text
    assert isinstance(caught.value.__cause__, KeyError)
    assert llm.calls == []


def test_a_labeler_that_gives_no_name_is_not_an_error() -> None:
    report = run(SMALL, labeler=lambda sender: None)
    assert report.status == "complete"
    assert "Unsaved contact" in report.model_dump_json()


def test_estimate_reports_a_broken_source_the_same_way() -> None:
    with pytest.raises(SourceError) as caught:
        estimate(
            BrokenAtCall(),
            CONVERSATION,
            None,
            model_info(),
            None,
            labeler=labeler,
            options=SMALL_OPTIONS,
            counter=WordCounter(),
        )
    assert MINA not in str(caught.value)


def test_an_engine_error_after_the_first_call_is_a_run_error_with_the_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the first call a run ends only with a RunError: the caller reads
    what was spent from one place."""

    def found_late(*args: Any, **kwargs: Any) -> Any:
        raise ConfigError("found late")

    monkeypatch.setattr(pipeline, "select_moments", found_late)
    llm = FakeLLM()
    with pytest.raises(RunError) as caught:
        run(SMALL, llm=llm)
    assert isinstance(caught.value.__cause__, ConfigError)
    assert "ConfigError" in str(caught.value)
    assert caught.value.usage is not None
    assert caught.value.usage.calls == 1
    assert len(llm.map_calls) == 1
    assert llm.reduce_calls == []
