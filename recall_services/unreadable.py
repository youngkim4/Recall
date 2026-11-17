"""Why a Messages database can not be read, in the words the app uses today.

The server has these sentences in `why_unreadable` and `closed_to_recall` in
ui_server.py. A service can not import that module, so the words are here a
second time. `tests/recall_services/test_server_parity.py` fails when the two
differ.

The check tells a database that macOS keeps closed from one that is not
there: without Full Disk Access macOS can show the file as missing.
`setup_status.probe_database` holds that rule.
"""

from __future__ import annotations

from setup_status import DbProbe

ARCHIVE_KEPT = "The archive was not changed."
# a busy database is tried: Messages may only be in the middle of a write
_TRIED = ("readable", "locked")


def closed_to_recall(probe: DbProbe) -> str:
    """What to do about a database that macOS does not let this app open."""
    if probe.kind == "live":
        return (
            "Recall cannot read Messages. Give it Full Disk Access in System Settings > "
            "Privacy & Security > Full Disk Access, then open Recall again."
        )
    return (
        f"Recall cannot read the copy at {probe.path}: macOS keeps that folder closed to it. "
        "Move the copy to your home folder and choose it again."
    )


def _what_stops(probe: DbProbe) -> str:
    live = probe.kind == "live"
    if probe.status == "fda_blocked":
        return closed_to_recall(probe)
    if probe.status == "missing":
        return "There is no Messages database on this Mac." if live else f"There is no file at {probe.path}."
    if probe.status == "empty":
        return "Messages holds no messages on this Mac." if live else f"The copy at {probe.path} holds no messages."
    said = f" ({probe.detail})" if probe.detail else ""
    if live:
        return f"The Messages database on this Mac could not be read{said}."
    return f"The copy at {probe.path} is not a Messages database{said}."


def why(probe: DbProbe) -> str | None:
    """The sentence that says what stops an import before it starts, or None when the database can be tried."""
    if probe.status in _TRIED:
        return None
    return f"{_what_stops(probe)} {ARCHIVE_KEPT}"
