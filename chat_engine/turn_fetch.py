"""The slow part of a lookup, run beside the stream.

A tool blocks the turn while it runs. The `fetch` of a lookup (a call to a
port) runs on a worker thread instead. It touches nothing of the engine: no
registry, no labeler, no `ChatArchive`. The port can read its own store there
(the digest port reads the conversation from the archive, which opens one
connection for each thread). Its reports go through a queue to the turn,
which gives each one as an event while the work goes on. The turn waits until
the work ends, so the events of a lookup all come before the next status line
and before `done`.

A cancel goes to the work through the cancel token of the question. The port
looks at it at its own steps: a read of its store that has begun ends first.
When the caller closes the stream while the work runs, the token is cancelled
so the work stops; nothing waits for it, and nothing it gives is read.
"""

from __future__ import annotations

import queue
import threading
from typing import Generator, Optional

from .cancel import CancelToken
from .tool_context import Fetch
from .turn_parts import Event

_END = object()


def run_beside(fetch: Fetch, cancel: CancelToken) -> Generator[Event, None, None]:
    """Run `fetch.work` on a worker thread. Yield the event of each of its reports, in order, while it runs.

    The generator ends when the work ends. What the work raises is raised here.
    `fetch.start` runs first, here on the thread of the turn.
    """
    if fetch.start is not None:
        fetch.start()
    reports: queue.SimpleQueue[object] = queue.SimpleQueue()
    failure: list[Exception] = []

    def work() -> None:
        try:
            fetch.work(reports.put)
        except Exception as error:  # raised again on the thread of the turn
            failure.append(error)
        finally:
            reports.put(_END)

    threading.Thread(target=work, name="chat-lookup", daemon=True).start()
    ended = False
    try:
        while not ended:
            item = reports.get()
            ended = item is _END
            event: Optional[Event] = None if ended else fetch.event(item)
            if event is not None:
                yield event
    finally:
        if not ended:  # the reader left: the work stops at its next step
            cancel.cancel()
    if failure:
        raise failure[0]
