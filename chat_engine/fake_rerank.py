"""A scripted reranker for tests. It calls no model and uses no network.

By default it gives the candidates in reverse order, so a test sees a change.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

from .cancel import CancelToken
from .ports import RerankItem, RerankResult, Usage

Order = Union[Sequence[str], Callable[[str, Sequence[RerankItem], int], Sequence[str]]]


def preferring(*words: str) -> Callable[[str, Sequence[RerankItem], int], list[str]]:
    """An order that puts first the candidates whose text holds one of `words`. The rest keep their order."""

    def order(_query: str, items: Sequence[RerankItem], _keep: int) -> list[str]:
        wanted = [item.id for item in items if any(word in item.text for word in words)]
        return [*wanted, *(item.id for item in items if item.id not in wanted)]

    return order


class FakeReranker:
    """`order` gives the ids: a list, or a function of the query, the items and `keep`.

    None: the items in reverse order. `error` is raised by every call. Each
    call costs `usd` and is recorded in `calls`. The result is not cut to
    `keep`: the engine must cut it.
    """

    def __init__(
        self,
        order: Optional[Order] = None,
        *,
        error: Optional[Exception] = None,
        usd: float = 0.0,
        failed: bool = False,
    ) -> None:
        self._order = order
        self._error = error
        self._usd = usd
        self._failed = failed
        self.calls: list[dict[str, Any]] = []

    def rerank(
        self, query: str, candidates: Sequence[RerankItem], keep: int, cancel: Optional[CancelToken] = None
    ) -> RerankResult:
        items = list(candidates)
        self.calls.append({"query": query, "items": items, "keep": keep, "cancel": cancel})
        if self._error is not None:
            raise self._error
        if self._order is None:
            ids = [item.id for item in reversed(items)]
        elif callable(self._order):
            ids = list(self._order(query, items, keep))
        else:
            ids = list(self._order)
        return RerankResult(tuple(ids), Usage(input_tokens=100, output_tokens=10), self._usd, self._failed)
