"""The services against the server's own payload builders at this commit.

`ui_server` is imported here and in no service module. It is imported when
this file is collected, as the base tests do: imported inside a test, it would
keep the output stream of that one test.

Each test gives one invented export to the server's builder and, through the
legacy bridge, to the service, and compares the two payloads.
"""

from __future__ import annotations

import ui_server
from recall_services import messages


def test_the_text_limit_is_the_one_of_the_server() -> None:
    assert messages.MESSAGE_TEXT_LIMIT == ui_server.MESSAGE_TEXT_LIMIT
