"""The facts of the embedding models: ids, dimensions, limits and prices.

The facts are in `models.json` beside this file. A fact that was not read is
`null`: no number is made up for it. A model that is not in the file is a
typed error. It is never priced as another model.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meaning_index.errors import ModelFactsError

FACTS_PATH = Path(__file__).with_name("models.json")


@dataclass(frozen=True)
class EmbedderFacts:
    """What an estimate needs to know about one model at one size."""

    model: str
    dimensions: int
    usd_per_mtok: float
    max_input_tokens: int
    tokenizer_family: str


def _whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _entry(raw: Mapping[str, Any], model: str) -> Mapping[str, Any]:
    for entry in raw["models"]:
        if isinstance(entry, Mapping) and entry.get("id") == model:
            return entry
    raise ModelFactsError(f"the embedding model {model!r} is not in the facts file")


def _checked(entry: Mapping[str, Any], model: str, dimensions: object) -> EmbedderFacts:
    native, largest, price = entry.get("native_dimensions"), entry.get("max_input_tokens"), entry.get("usd_per_mtok")
    if not _whole(native) or not _whole(largest):
        raise ModelFactsError(f"the facts of {model!r} have no dimensions or no largest input")
    if isinstance(price, bool) or not isinstance(price, (int, float)) or price < 0:
        raise ModelFactsError(f"the facts of {model!r} have no price")
    size = native if dimensions is None else dimensions
    if not _whole(size) or size > native:
        raise ModelFactsError(f"{model!r} gives 1 to {native} dimensions")
    if size != native and entry.get("can_shorten") is not True:
        raise ModelFactsError(f"{model!r} gives {native} dimensions only")
    return EmbedderFacts(
        model=model,
        dimensions=int(size),
        usd_per_mtok=float(price),
        max_input_tokens=int(largest),
        tokenizer_family=str(entry.get("tokenizer_family") or ""),
    )


def load_facts(path: str | os.PathLike[str] | None = None) -> Mapping[str, Any]:
    """The facts file as it is written. A file that can not be used is a ModelFactsError."""
    target = Path(path) if path is not None else FACTS_PATH
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelFactsError(f"the embedding facts file can not be read ({type(exc).__name__})") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list) or not raw["models"]:
        raise ModelFactsError("the embedding facts file names no model")
    default = raw.get("default")
    if not isinstance(default, dict) or not isinstance(default.get("model"), str):
        raise ModelFactsError("the embedding facts file names no default model")
    _checked(_entry(raw, default["model"]), default["model"], default.get("dimensions"))
    return raw


def embedder_facts(
    model: str | None = None,
    dimensions: int | None = None,
    *,
    path: str | os.PathLike[str] | None = None,
) -> EmbedderFacts:
    """Facts of one model. With no model: the default model at the default dimensions.

    A model that is named with no dimensions has the default dimensions when it
    is the default model, and its own dimensions when it is not.
    """
    raw = load_facts(path)
    default = raw["default"]
    chosen = default["model"] if model is None else model
    size = dimensions
    if size is None and chosen == default["model"]:
        size = default.get("dimensions")
    return _checked(_entry(raw, chosen), chosen, size)
