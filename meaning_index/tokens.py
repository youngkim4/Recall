"""The default token counter of a build and of an estimate.

It counts with tiktoken's `cl100k_base`. That family is an assumption: see
`models.json`. tiktoken fetches a vocabulary file the first time it is used on
a machine. When the encoding can not load, the count is rough: one token for
three bytes. The log says so one time.

The app can give a counter of its own to `build_index` and `estimate_build`.
Nothing here loads at import time.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

ENCODING = "cl100k_base"
ROUGH_BYTES_PER_TOKEN = 3


def rough_count(text: str) -> int:
    """One token for three bytes, rounded up. English text has about four letters in a token."""
    size = len(text.encode("utf-8", "surrogatepass"))
    return -(-size // ROUGH_BYTES_PER_TOKEN)


@lru_cache(maxsize=1)
def _encoding() -> Any:
    """The encoding, or None when it can not load. Asked for one time."""
    try:
        import tiktoken

        return tiktoken.get_encoding(ENCODING)
    except Exception as exc:  # noqa: BLE001 - any load failure means the same thing here
        logger.warning(
            "tokenizer %s could not load (%s); counting tokens by bytes", ENCODING, type(exc).__name__
        )
        return None


def forget_encoding() -> None:
    """Load the encoding again at the next count."""
    _encoding.cache_clear()


def default_count_tokens(text: str) -> int:
    encoding = _encoding()
    if encoding is None:
        return rough_count(text)
    return len(encoding.encode_ordinary(text))
