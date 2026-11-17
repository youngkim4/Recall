"""The fake model, and the package's import surface."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import report_engine
from report_engine.errors import LLMError
from report_engine.fake_llm import FakeCall, FakeLLM, completed, default_reply, fake_usage
from report_engine.models import SECTION_KEYS
from report_engine.ports import LLMResult, LLMUsage
from report_engine.prompts import MAP_SCHEMA, REDUCE_SCHEMA, map_messages, reduce_messages

TRANSCRIPT = (
    "--- 2024-06-01 Saturday ---\n"
    "m1 09:00 Mina: short\n"
    "m2 09:05 You: this is the longest line here\n"
    "--- 2024-06-02 Sunday ---\n"
    "m3 21:30 Theo: night"
)
NOTES = [("May 2024", "A trip is planned: by train."), ("Late May", "Nothing new.")]


def map_request(notes: Any = ()) -> list:
    return map_messages(TRANSCRIPT, ("Mina", "Theo", "You"), notes)


def test_a_map_call_is_read_back_from_the_prompt() -> None:
    llm = FakeLLM()
    llm.generate(map_request(NOTES), schema=MAP_SCHEMA, max_output_tokens=100, reasoning="low")
    (call,) = llm.calls
    assert call.kind == "map"
    assert call.transcript == TRANSCRIPT
    assert [(line.ref, line.day, line.clock, line.label, line.text) for line in call.lines] == [
        ("m1", "2024-06-01", "09:00", "Mina", "short"),
        ("m2", "2024-06-01", "09:05", "You", "this is the longest line here"),
        ("m3", "2024-06-02", "21:30", "Theo", "night"),
    ]
    assert (call.first_day, call.last_day) == ("2024-06-01", "2024-06-02")
    assert call.notes == (("May 2024", "A trip is planned: by train."), ("Late May", "Nothing new."))
    assert (call.max_output_tokens, call.reasoning) == (100, "low")
    assert llm.map_calls == [call] and llm.reduce_calls == []


def test_the_default_map_reply_quotes_the_longest_line_of_the_transcript() -> None:
    result = FakeLLM().generate(map_request(), schema=MAP_SCHEMA, max_output_tokens=100)
    assert result.status == "completed"
    assert result.parsed is not None
    (moment,) = result.parsed["moments"]
    assert moment["quote"] == "this is the longest line here"
    assert moment["refs"] == ["m2"]
    assert moment["date"] == "2024-06-01"
    assert result.parsed["summary"] == "Fake summary: 3 lines from 2024-06-01 to 2024-06-02."
    assert json.loads(result.text) == result.parsed


def test_the_default_reduce_reply_has_the_four_sections() -> None:
    llm = FakeLLM()
    request = reduce_messages(labels=("Mina", "You"), figures="Messages with text: 3.", periods=(), moments=())
    result = llm.generate(request, schema=REDUCE_SCHEMA, max_output_tokens=100)
    assert result.parsed is not None
    assert list(result.parsed) == list(SECTION_KEYS)
    assert all(text.strip() for text in result.parsed.values())
    assert llm.reduce_calls[0].kind == "reduce"
    assert llm.reduce_calls[0].lines == ()
    assert (llm.reduce_calls[0].first_day, llm.reduce_calls[0].last_day) == ("", "")


def test_usage_is_one_token_per_word() -> None:
    llm = FakeLLM()
    request = map_request()
    result = llm.generate(request, schema=MAP_SCHEMA, max_output_tokens=100)
    words_in = sum(len(message["content"].split()) for message in request)
    assert result.usage == LLMUsage(
        input_tokens=words_in, cached_input_tokens=0, output_tokens=len(result.text.split())
    )
    assert fake_usage(llm.calls[0], "three small words").output_tokens == 3


def test_a_responder_can_answer_with_a_parsed_reply_a_result_or_nothing() -> None:
    answers: list[Any] = [
        {"summary": "mine", "carry_over": "", "moments": []},
        LLMResult(text="", parsed=None, usage=LLMUsage(input_tokens=9), status="incomplete"),
        None,
    ]
    llm = FakeLLM(lambda call: answers.pop(0))
    first = llm.generate(map_request(), schema=MAP_SCHEMA, max_output_tokens=10)
    second = llm.generate(map_request(), schema=MAP_SCHEMA, max_output_tokens=10)
    third = llm.generate(map_request(), schema=MAP_SCHEMA, max_output_tokens=10)
    assert first.parsed == {"summary": "mine", "carry_over": "", "moments": []}
    assert (second.status, second.usage.input_tokens) == ("incomplete", 9)
    assert third.parsed == default_reply(llm.calls[2])
    assert len(llm.calls) == 3


def test_a_responder_can_fail_a_call_and_the_call_is_still_recorded() -> None:
    def responder(call: FakeCall) -> Any:
        raise LLMError("down")

    llm = FakeLLM(responder)
    with pytest.raises(LLMError):
        llm.generate(map_request(), schema=MAP_SCHEMA, max_output_tokens=10)
    assert len(llm.calls) == 1


def test_a_call_with_no_schema_is_neither_map_nor_reduce() -> None:
    llm = FakeLLM()
    result = llm.generate([{"role": "user", "content": "hello there"}], max_output_tokens=10)
    assert llm.calls[0].kind == "other"
    assert (llm.calls[0].transcript, llm.calls[0].notes) == ("", ())
    assert result.parsed == {}


def test_completed_can_shape_a_reply() -> None:
    llm = FakeLLM()
    llm.generate(map_request(), schema=MAP_SCHEMA, max_output_tokens=10)
    call = llm.calls[0]
    cut = completed(call, None, text="half", status="incomplete", usage=LLMUsage(input_tokens=5, output_tokens=1))
    assert (cut.text, cut.parsed, cut.status, cut.usage.input_tokens) == ("half", None, "incomplete", 5)


def test_the_fake_keeps_its_own_copy_of_the_messages() -> None:
    request = map_request()
    llm = FakeLLM()
    llm.generate(request, schema=MAP_SCHEMA, max_output_tokens=10)
    request[1]["content"] = "changed after the call"
    assert "changed after the call" not in llm.calls[0].text


def test_the_public_names_can_be_imported() -> None:
    for name in report_engine.__all__:
        assert getattr(report_engine, name) is not None
    assert report_engine.render_markdown is report_engine.store.render_markdown


def test_importing_the_package_loads_no_heavy_or_private_module() -> None:
    root = Path(report_engine.__file__).resolve().parent.parent
    code = (
        "import sys, report_engine;"
        "names = ('pandas', 'numpy', 'openai', 'tiktoken', 'archive', 'dotenv', 'ui_server', "
        "'conversation', 'analysis', 'openai_client');"
        "print(','.join(name for name in names if name in sys.modules))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True, timeout=60
    )
    assert done.stdout.strip() == ""
