"""The search of people of chat: `chat_engine.ports.PeopleSearch` over `recall_services.people`.

`port(archive, names)` gives the port for one question, or None: with no key,
when the service is not in this build, when its status can not be read, or
when the archive has no profile yet. The engine then offers no `find_people`.

Rules:
- The description that reaches `PeoplePort.search` is masked by the engine.
- A search that did not run (no index, no key, the description could not be
  embedded) raises `PeopleNotAvailable`. The engine then gives the model one
  line, and the question goes on.
- No log line holds a description, a name, a handle or a path.

WARNING: `recall_services.people` belongs to the people index. The lines that
depend on it are `SERVICE`, the call of `status` and the keys "state" and
"profiles" it reads, the call of `search` (with `conversations` only in a
scope) and its `reason`, and the keys of a row in `_hit`.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from chat_engine.ports import PersonHit
from recall_services import keys

logger = logging.getLogger(__name__)

SERVICE = "recall_services.people"


class PeopleNotAvailable(Exception):
    """The search of people did not run. `reason` is the word of the service, such as "no_index"."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"the search of people did not run: {reason}")
        self.reason = reason


def _hit(row: Mapping[str, Any]) -> PersonHit:
    """A row of the service in the form of the port. A missing field is empty; the engine checks every value."""

    def listed(key: str) -> list[Any]:
        value = row.get(key)
        return list(value) if isinstance(value, (list, tuple)) else []

    def text(key: str) -> str:
        value = row.get(key)
        return value if isinstance(value, str) else ""

    figures = row.get("figures")
    score = row.get("score")
    return {
        "key": text("key"),
        "conversations": listed("conversations"),
        "label": text("label"),
        "kind": text("kind"),
        "status": text("status"),
        "summary": text("summary"),
        "turning_points": listed("turning_points"),
        "figures": dict(figures) if isinstance(figures, Mapping) else {},
        "score": float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0,
    }


class PeoplePort:
    """`chat_engine.ports.PeopleSearch` over the profiles of one archive."""

    def __init__(self, archive: Any, names: Any, service: Any) -> None:
        self._archive = archive
        self._names = names
        self._service = service

    def search(
        self,
        description: str,
        *,
        kinds: Sequence[str] | None,
        since: str | None,
        until: str | None,
        limit: int,
        conversations: Sequence[str] | None = None,
    ) -> list[PersonHit]:
        """`conversations` (the scope) is sent only when there is one."""
        scoped = {} if conversations is None else {"conversations": list(conversations)}
        found = self._service.search(
            self._archive,
            self._names,
            description,
            kinds=None if kinds is None else list(kinds),
            since=since,
            until=until,
            limit=limit,
            **scoped,
        )
        reason = getattr(found, "reason", None)
        if reason is not None:
            raise PeopleNotAvailable(str(reason))
        return [_hit(row) for row in found if isinstance(row, Mapping)]


def _service() -> Any | None:
    """The service of the app, or None when this build does not hold it."""
    try:
        return importlib.import_module(SERVICE)
    except ImportError as error:
        logger.info("the people index is not part of this build (%s); chat has no people search", type(error).__name__)
        return None


def _has_profiles(state: object) -> bool:
    if not isinstance(state, Mapping) or state.get("state") == "none":
        return False
    try:
        return int(state.get("profiles") or 0) > 0
    except (TypeError, ValueError):
        return False


def port(archive: Any, names: Any, *, service: Any = None) -> PeoplePort | None:
    """The port for one question, or None when the search of people can not work now. A test gives `service`."""
    if not keys.api_key():
        return None
    chosen = service if service is not None else _service()
    if chosen is None:
        return None
    try:
        state = chosen.status(archive, names)
    except Exception as error:  # noqa: BLE001 - whatever a store raises, the question goes on without people
        logger.warning("the people index can not be read (%s); chat has no people search", type(error).__name__)
        return None
    return PeoplePort(archive, names, chosen) if _has_profiles(state) else None
