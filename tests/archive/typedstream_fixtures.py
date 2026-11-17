"""Load the typedstream fixtures made by scripts/make_typedstream_fixtures.swift."""

from __future__ import annotations

import json
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "typedstream"


@lru_cache(maxsize=1)
def manifest() -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))


def entries() -> dict[str, dict[str, Any]]:
    return {entry["name"]: entry for entry in manifest()["fixtures"]}


def fixture_names() -> list[str]:
    return sorted(entries())


@cache
def blob(name: str) -> bytes:
    text = (FIXTURE_DIR / f"{name}.hex").read_text(encoding="ascii")
    return bytes.fromhex("".join(text.split()))


def counter_text(size: int) -> str:
    """Same recipe as counterText in the Swift script: "00000|00001|..." cut to size bytes."""
    blocks = (size // 6) + 2
    return "".join(f"{index:05d}|" for index in range(blocks))[:size]


def expected_text(name: str) -> str:
    entry = entries()[name]
    if "text" in entry:
        return str(entry["text"])
    recipe = str(entry["recipe"])
    if not recipe.startswith("counter:") or "+" in recipe:
        raise AssertionError(f"fixture {name} has no text and no plain counter recipe")
    return counter_text(int(recipe.split(":", 1)[1]))
