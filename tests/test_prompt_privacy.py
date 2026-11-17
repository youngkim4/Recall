"""Tests that a handle with no saved name is never sent whole to the model, and
is never written out in a report page. The model's part is replaced, so
nothing is sent anywhere."""
from pathlib import Path

import pandas as pd
import pytest

import ai_analysis
import analysis
import contact_names

COLUMNS = ["message_id", "timestamp", "sender", "text", "chat_id", "chat_display_name", "is_from_me"]
AVERY = "+15550000001"
UNSAVED = "+15550000009"
ADDRESS = "sam99@example.com"
GROUP = "chat4471"
# every way the digits of the unsaved number could be written whole
WHOLE_NUMBER = (UNSAVED, "15550000009", "5550000009")


def export(tmp_path, chat_id, senders, group_name=""):
    """A made-up conversation on two days a year apart, written in turn by `senders`."""
    stamps = ["2023-03-01 09:00:00", "2023-03-01 09:05:00", "2024-06-10 20:00:00", "2024-06-10 20:01:00"]
    rows = [
        (index + 1, stamp, senders[index % len(senders)], f"line {index}", chat_id, group_name, 1 if index == 3 else 0)
        for index, stamp in enumerate(stamps)
    ]
    path = tmp_path / "messages.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


@pytest.fixture
def sent(monkeypatch):
    """Every text a report run hands to the model."""
    prompts: list[str] = []

    def record(_client, messages, **kwargs):
        prompts.extend(str(message["content"]) for message in messages)
        return '{"events": []}' if kwargs.get("text_format") else "A story."

    monkeypatch.setattr(ai_analysis, "_call_openai", record)
    monkeypatch.setattr(ai_analysis, "OpenAI", lambda *args, **kwargs: object())
    monkeypatch.setattr(contact_names, "resolve_contact_names", lambda handles, **_: {})
    return prompts


@pytest.fixture(params=["whole", "in parts"])
def length(request, monkeypatch):
    """A report is written from the whole conversation, or from it in parts
    when it is too long. The two build their prompts in different places."""
    if request.param == "in parts":
        monkeypatch.setattr(analysis, "get_token_budget", lambda model=None: 1)
        monkeypatch.setattr(ai_analysis, "get_token_budget", lambda model=None: 1)
    return request.param


@pytest.mark.parametrize("chat_id, senders, label", [
    pytest.param(UNSAVED, [UNSAVED], "a number ending in 0009", id="a number"),
    pytest.param(ADDRESS, [ADDRESS], "an unsaved email contact", id="an address"),
    pytest.param(GROUP, [UNSAVED, ADDRESS], "a group chat", id="a group"),
])
def test_no_prompt_holds_a_whole_handle_that_is_not_saved(tmp_path, sent, length, chat_id, senders, label):
    csv = export(tmp_path, chat_id, senders)

    analysis.run_cli(str(csv), chat_id, str(tmp_path / "out"))

    # events and the story, and in parts one of each for a period and the synthesis
    assert len(sent) >= (4 if length == "whole" else 10)
    for prompt in sent:
        assert not any(number in prompt for number in WHOLE_NUMBER)
        assert ADDRESS not in prompt
    assert any(label in prompt for prompt in sent)


def test_the_model_is_told_a_saved_name(tmp_path, sent, length, monkeypatch):
    monkeypatch.setattr(contact_names, "resolve_contact_names", lambda handles, **_: {AVERY: "Avery Stone"})
    csv = export(tmp_path, AVERY, [AVERY])

    analysis.run_cli(str(csv), AVERY, str(tmp_path / "out"))

    assert any("Avery Stone" in prompt for prompt in sent)
    assert not any(AVERY in prompt for prompt in sent)


def test_the_model_is_told_the_name_a_group_was_given(tmp_path, sent, length):
    csv = export(tmp_path, GROUP, [UNSAVED, ADDRESS], group_name="Road Trip")

    analysis.run_cli(str(csv), GROUP, str(tmp_path / "out"))

    assert any("Road Trip" in prompt for prompt in sent)


def page_of(tmp_path, chat_id, senders, group_name=""):
    csv = export(tmp_path, chat_id, senders, group_name)
    _, report_path, _events, html_path = analysis.run_cli(str(csv), chat_id, str(tmp_path / "out"), html=True)
    return Path(html_path).read_text(encoding="utf-8"), Path(report_path).read_text(encoding="utf-8")


def test_a_report_page_never_writes_out_a_number_that_is_not_saved(tmp_path, sent):
    page, report = page_of(tmp_path, UNSAVED, [UNSAVED])

    assert "<title>Conversation Analysis — Number ending in 0009</title>" in page
    assert "<h1>Number ending in 0009</h1>" in page
    assert not any(number in page for number in WHOLE_NUMBER)
    # the report in markdown keeps the id in its heading: the app reads it back from there
    assert report.startswith(f"# Conversation Analysis - {UNSAVED}\n")


def test_a_report_page_is_titled_with_a_saved_name(tmp_path, sent, monkeypatch):
    monkeypatch.setattr(contact_names, "resolve_contact_names", lambda handles, **_: {AVERY: "Avery Stone"})

    page, _report = page_of(tmp_path, AVERY, [AVERY])

    assert "<h1>Avery Stone</h1>" in page
    assert AVERY not in page


def test_a_report_page_is_titled_with_the_name_a_group_was_given(tmp_path, sent):
    named, _report = page_of(tmp_path, GROUP, [UNSAVED, ADDRESS], group_name="Road Trip")

    assert "<h1>Road Trip</h1>" in named
    assert GROUP not in named


def test_a_report_page_for_a_group_with_no_name_says_so(tmp_path, sent):
    unnamed, _report = page_of(tmp_path, GROUP, [UNSAVED, ADDRESS])

    assert "<h1>Unnamed group</h1>" in unnamed
    assert GROUP not in unnamed
