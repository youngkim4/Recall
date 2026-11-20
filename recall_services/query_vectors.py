"""The vectors of the last descriptions, one cache for the process.

The search of people (`people_search.py`) and the search by meaning
(`meaning_port.py`) both embed the text of a question. With the same model
and dimensions they share its vector: a question's text is embedded once.

The key is the model, the dimensions and the SHA-256 of the text, so no text
is kept. The last `QUERY_VECTORS` vectors are kept. A reply that a search can
not use (not one vector, another size, a number that is not finite) raises
ValueError and is not kept. A failure of the embedder is raised as it is.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

QUERY_VECTORS = 64

EmbedderFactory = Callable[[str, int], Any]
_Key = tuple[str, int, str]

_kept: OrderedDict[_Key, tuple[float, ...]] = OrderedDict()
_lock = threading.Lock()


def _one_vector(reply: Sequence[Sequence[float]], dimensions: int) -> tuple[float, ...]:
    rows = list(reply)
    if len(rows) != 1:
        raise ValueError("the embedder must give one vector for one description")
    vector = tuple(float(value) for value in rows[0])
    if len(vector) != dimensions or not all(math.isfinite(value) for value in vector):
        raise ValueError("the embedder must give one vector of finite numbers with the dimensions asked for")
    return vector


def query_vector(text: str, model: str, dimensions: int, factory: EmbedderFactory) -> tuple[float, ...]:
    """The vector of `text` with this model and these dimensions. `factory(model, dimensions)` makes the
    embedder only when the vector is not kept."""
    key = (model, dimensions, hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest())
    with _lock:
        found = _kept.get(key)
        if found is not None:
            _kept.move_to_end(key)
            return found
    made = _one_vector(factory(model, dimensions).embed([text]), dimensions)
    with _lock:
        _kept[key] = made
        _kept.move_to_end(key)
        while len(_kept) > QUERY_VECTORS:
            _kept.popitem(last=False)
    return made


def forget() -> None:
    """Let go of every kept vector."""
    with _lock:
        _kept.clear()
