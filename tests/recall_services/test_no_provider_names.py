"""No sentence that the app shows names the provider, its models, the key's variable or the key file.

Owner: "dont even mention openai". The app calls it "API key" and points to
Settings > Archive > API key.
"""

import re
from types import SimpleNamespace

import pytest

from recall_services import (
    chat,
    digest_job,
    first_run,
    key_store,
    keys,
    meaning_failures,
    memory_prep_words,
    models,
    people_words,
    report_failures,
)
from report_engine.errors import LLMError

NAMED = re.compile(r"openai|gpt|\.env\b|OPENAI_API_KEY", re.IGNORECASE)
CODES = (
    "no_key", "key_rejected", "unknown_model", "rate_limit", "offline", "timeout", "model_error", "archive_error",
    "cache_unreadable", "cancelled", "run_failed", "model_not_run", "store_error", "store_newer", "embed_failed",
    "empty", "too_large", "something_else",
)  # fmt: skip
PROVIDER = models.provider_name("openai")
MODEL = "gpt-6-luna"


def clean(sentence: str) -> str:
    assert not NAMED.search(sentence), sentence
    return sentence


def test_the_provider_has_no_name():
    assert clean(models.provider_name("openai")) == clean(models.SOME_PROVIDER)


def test_the_key_place_is_the_screen_of_the_app():
    assert keys.KEY_PLACE == "Settings > Archive > API key"
    assert "Settings > Archive > API key" in clean(keys.NO_KEY)


@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("sentence_of", [digest_job.sentence_of, memory_prep_words.sentence_of, people_words.sentence_of])
def test_the_sentences_of_the_jobs(sentence_of, code):
    clean(sentence_of(code, PROVIDER, MODEL))


@pytest.mark.parametrize("code", CODES)
def test_the_sentences_of_chat(code):
    config = SimpleNamespace(provider="openai", model="gpt-6.1-sol")
    clean(chat._sentence(code, "", config, None))
    clean(chat._sentence(code, "", None, 3.0))


@pytest.mark.parametrize("status", [401, 403, 404, 429])
def test_the_sentences_of_a_report(status):
    found = report_failures._of_the_model(LLMError("no", status_code=status), PROVIDER, MODEL)
    assert found is not None
    clean(found[1])


@pytest.mark.parametrize("said", ["APITimeoutError", "APIConnectionError"])
def test_the_sentences_of_a_report_with_no_status(said):
    found = report_failures._of_the_model(LLMError(said), PROVIDER, MODEL)
    assert found is not None
    clean(found[1])


@pytest.mark.parametrize(
    "sentence",
    [
        report_failures.NO_KEY,
        first_run.NO_KEY,
        meaning_failures.NO_KEY,
        meaning_failures.KEY_REJECTED,
        meaning_failures.RATE_LIMIT,
        meaning_failures.OFFLINE,
        key_store.ENVIRONMENT_WINS,
        key_store.NOT_WRITTEN,
        key_store.NOT_REMOVED,
        key_store.NOT_A_KEY,
    ],
)
def test_the_fixed_sentences(sentence):
    clean(sentence)


def test_the_no_key_sentences_point_to_the_screen():
    for sentence in (report_failures.NO_KEY, first_run.NO_KEY, meaning_failures.NO_KEY,
                     chat._sentence("no_key", "", None, None), digest_job.sentence_of("no_key", PROVIDER, MODEL)):
        assert "Settings > Archive > API key" in sentence
