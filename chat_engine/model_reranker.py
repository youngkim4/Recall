"""A reranker over the model port: one call with no tools, that answers with JSON.

The prompt gives the question and the numbered candidates. The model answers
{"order": ["3", "1", ...]}: the numbers best first, at most `keep`. The answer
is read defensively (`parse_order`): an id that was not given is dropped, so
is a repeat, and the candidates the answer leaves out follow in their order.

It never raises. A model that fails, an answer that can not be read, or a
cancelled question gives the order of the candidates, with `failed` set. A
call that was made is paid for, also when its answer can not be read.

The texts that reach it are masked by the engine. Nothing here is logged.
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Optional, Sequence

from .cancel import CancelToken
from .ports import LLM, ChatMessage, Completed, LLMRequest, RerankItem, RerankResult, TextDelta, Usage

OUTPUT_TOKENS = 600  # a list of 30 numbers needs about 150
RERANK_SYSTEM = (
    "You rank search results from a person's message archive by how well each one helps to answer a "
    "question. You get the question and numbered candidates. Each candidate is a short piece of one "
    "conversation: the first line names the conversation and the day, and a line that starts with * is "
    "the message the search found. Prefer a candidate that answers the question or holds the facts it "
    "asks about, over one that only shares a word with it. "
    'Answer with JSON only, in this form: {"order": ["3", "1", "2"]}. Give the numbers of the candidates, '
    "best first. Use only numbers that you were given, each one time. Give at most the number of "
    "candidates that you are asked for. Do not explain."
)


def rerank_messages(query: str, candidates: Sequence[RerankItem], keep: int) -> tuple[ChatMessage, ...]:
    """The prompt of one ranking."""
    listed = "\n\n".join(f"[{item.id}] {item.text}" for item in candidates)
    user = f"Question: {query}\n\nGive at most {keep} candidates, best first.\n\nCandidates:\n\n{listed}"
    return ChatMessage("system", RERANK_SYSTEM), ChatMessage("user", user)


def parse_order(text: str, ids: Sequence[str]) -> Optional[tuple[str, ...]]:
    """The known ids of an answer, best first, each one time. None when the answer has no list of ids.

    The answer can be {"order": [...]} or a bare list, alone or inside other text.
    A number is read as its id.
    """
    value = _json_in(text)
    if isinstance(value, dict):
        value = value.get("order")
    if not isinstance(value, list):
        return None
    known = set(ids)
    found = (_as_id(item) for item in value)
    return tuple(dict.fromkeys(item for item in found if item in known))


def _json_in(text: str) -> Any:
    """The first JSON object or list in the text, or None."""
    source = str(text or "").strip()
    for opening, closing in (("{", "}"), ("[", "]")):
        start, end = source.find(opening), source.rfind(closing)
        if 0 <= start < end:
            try:
                return json.loads(source[start:end + 1])
            except ValueError:
                continue
    return None


def _as_id(item: object) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, int) and not isinstance(item, bool):
        return str(item)
    return ""


def _completed(order: Sequence[str], ids: Sequence[str], keep: int) -> tuple[str, ...]:
    """The order of the answer, then the ids it left out in their order, cut to `keep`."""
    return tuple(dict.fromkeys([*order, *ids]))[:max(0, keep)]


class ModelReranker:
    """`ports.Reranker` over one model.

    price: what one call cost in USD, from its usage. A price that fails counts 0.
    """

    def __init__(
        self,
        llm: LLM,
        model: str,
        *,
        price: Callable[[Usage], float],
        max_output_tokens: int = OUTPUT_TOKENS,
    ) -> None:
        self._llm = llm
        self._model = model
        self._price = price
        self._max_output_tokens = max_output_tokens

    def rerank(
        self, query: str, candidates: Sequence[RerankItem], keep: int, cancel: Optional[CancelToken] = None
    ) -> RerankResult:
        ids = tuple(item.id for item in candidates)
        if not ids or keep < 1:
            return RerankResult(ids[:max(0, keep)])
        if cancel is not None and cancel.cancelled:
            return RerankResult(ids[:keep], failed=True, reason="cancelled")
        request = LLMRequest(
            model=self._model,
            messages=rerank_messages(query, candidates, keep),
            tools=(),
            max_output_tokens=self._max_output_tokens,
            allow_tools=False,
            cancel=cancel,
        )
        try:
            text, usage = self._call(request)
        except Exception as error:  # the order of the search stays; the question goes on
            return RerankResult(ids[:keep], failed=True, reason=type(error).__name__)
        usd = self._cost(usage)
        order = parse_order(text, ids)
        if order is None:
            return RerankResult(ids[:keep], usage, usd, failed=True, reason="the answer had no list of ids")
        return RerankResult(_completed(order, ids, keep), usage, usd)

    def _call(self, request: LLMRequest) -> tuple[str, Usage]:
        parts: list[str] = []
        usage = Usage()
        stream = iter(self._llm.stream(request))
        try:
            for event in stream:
                if isinstance(event, TextDelta) and not event.commentary:
                    parts.append(event.text)
                elif isinstance(event, Completed):
                    usage = event.usage
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        return "".join(parts), usage

    def _cost(self, usage: Usage) -> float:
        try:
            usd = float(self._price(usage))
        except Exception:  # a price that can not be read costs nothing here; the call still counts its tokens
            return 0.0
        return usd if math.isfinite(usd) and usd >= 0 else 0.0
