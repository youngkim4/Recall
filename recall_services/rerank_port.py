"""The reranker of chat: `chat_engine.model_reranker.ModelReranker` over the cheap model.

The model is the model of the memory (`memory_prep.model()`), asked for no
reasoning when it lists that level. A ranking is one short call, so it gets a
small output cap and a short wait. Its cost comes from the prices of
models.json, call by call, and the engine reports it as `rerank_usd`.

`chat.py` builds it only when its switch `RERANK` is on. A reranker that can
not be built (no key, a model that is not offered) is None: the question then
searches as without it.

Rules:
- Only masked text reaches the reranker: the engine masks the query and each
  candidate before the call.
- The model port comes from the factory that `chat.py` gives. It is never
  the port of the chat model, so a recorder of the chat calls does not see it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace

from chat_engine.model_reranker import ModelReranker
from chat_engine.ports import LLM, Usage
from recall_services import memory_prep, models
from recall_services.models import ChatModelConfig
from report_engine.ports import LLMUsage
from report_engine.pricing import call_cost_usd

logger = logging.getLogger(__name__)

NO_REASONING = "none"
OUTPUT_TOKENS = 600  # a list of 30 numbers needs about 150
TIMEOUT_SECONDS = 30  # a ranking that takes longer is not worth the wait: the order of the search stays

LLMFactory = Callable[[ChatModelConfig], LLM]


def config() -> ChatModelConfig:
    """What the model port of a ranking is made with."""
    model = memory_prep.model()
    levels = models.facts(model).reasoning_levels or ()
    chosen = models.chat_config(model, NO_REASONING if NO_REASONING in levels else "")
    return replace(chosen, max_output_tokens=OUTPUT_TOKENS, timeout_seconds=TIMEOUT_SECONDS)


def port(factory: LLMFactory) -> ModelReranker | None:
    """The reranker, or None when its model port can not be made now. A failure leaves one log line."""
    try:
        chosen = config()
        info = models.model_info(chosen.model)
        llm = factory(chosen)
    except Exception as error:  # noqa: BLE001 - no key, a model that can not be run: chat searches as before
        logger.warning("a reranker can not be built (%s); search results keep their order", type(error).__name__)
        return None

    def price(usage: Usage) -> float:
        return call_cost_usd(info, LLMUsage(usage.input_tokens, usage.cached_input_tokens, usage.output_tokens))

    return ModelReranker(llm, chosen.model, price=price, max_output_tokens=chosen.max_output_tokens)
