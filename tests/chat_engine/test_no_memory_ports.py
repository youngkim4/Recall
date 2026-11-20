"""An engine with no memory port is the engine of commit 82f55e0, byte for byte.

The hashes below were taken at that commit, before the ports for the search by
meaning and for the digest were added.
"""

import hashlib
import json

from chat_engine.fake_llm import call, say
from chat_engine.ports import LLMRequest
from chat_engine.tools import TOOL_SPECS
from tests.chat_engine.helpers import all_text, make_engine, system_text

SPECS_AT_82F55E0 = "fe73fa1d7ccb0aedac549fa4192e1f9692e7e21a72fe504d61324ba9af39343b"
# taken again when the prompt began to ask a who-is answer for how it began
SYSTEM_AT_82F55E0 = "e17521b687fdda6bc58b0dd83da16cded769e27b05c186dcce8bd5b8020066d3"
# taken again when the seed began to read the first messages with the person a question names
REQUESTS_AT_82F55E0 = "ad12d1fe81ff85cf5f63ec9418f79b3c45a1cab87b6482fab9705a22531af7e2"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _specs_text(specs: tuple) -> str:
    return json.dumps([[spec.name, spec.description, spec.parameters] for spec in specs], ensure_ascii=False)


def _request_text(request: LLMRequest) -> str:
    return "\n".join([all_text(request), _specs_text(request.tools), str(request.allow_tools)])


def test_the_tool_list_is_unchanged() -> None:
    assert _hash(_specs_text(TOOL_SPECS)) == SPECS_AT_82F55E0


def test_the_system_prompt_is_unchanged() -> None:
    engine, fake, _ = make_engine([say("Nothing found.")])
    engine.ask("anything about zanzibar?")
    assert _hash(system_text(fake.requests[0])) == SYSTEM_AT_82F55E0
    assert fake.requests[0].tools == TOOL_SPECS


def test_every_request_of_a_question_with_lookups_is_unchanged() -> None:
    steps = [
        [
            call("search_messages", "s1", query="lease", limit=3),
            call("search_messages", "s2", query="sourdough starter zanzibar", match="any"),
            call("search_messages", "s3", query="lease", match="meaning"),
            call("conversation_digest", "s4", conversation="c1"),
            call("count_messages", "s5", group_by="year"),
        ],
        [call("read_around", "s6", reference="m1"), call("list_conversations", "s7", name="maya")],
        say("Maya signed the lease in 2021 [m1]."),
    ]
    engine, fake, _ = make_engine(steps)
    result = engine.ask("What did Maya say about the lease in 2021?")
    sent = "\n\n".join(_request_text(request) for request in fake.requests)
    # `limited` came later (the lookup limit of a question): it is the only key the old result did not have
    assert result.pop("limited") is False
    assert _hash(sent + json.dumps(result, ensure_ascii=False, sort_keys=True)) == REQUESTS_AT_82F55E0
