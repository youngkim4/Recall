"""The words the model gets: the system prompt and the question block."""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

from .ports import ArchiveSummary

LIMIT_NOTE = (
    "You have reached the limit on lookups for this question. Do not call a tool. "
    "Answer now from what you have, and say what you could not check."
)

_RULES = """How to work
- Some lookups were already run for the question. Their results are in the question block. If they answer the question, answer at once and call no tool.
- Otherwise call tools. Search with few, distinctive words. Narrow with days and conversations. Read around a hit before you rely on it.
- Use count_messages for every number: how many, how often, who most, which month. Never estimate a number from the messages you happen to see.
- For the first or the last time something happened, search with order "oldest" or "newest".
- Work out relative dates ("last summer", "two years ago") from today's date before you set since and until.
- Write no text before or between tool calls. Write only the final answer.

How to answer
- Answer from the evidence. Say what the messages state. Say separately what you infer from them.
- When the user tries to recall what was said, lead with the exact words in quotation marks. Inside quotation marks, copy the message text exactly. Never reword a quote.
- After each claim, cite the message that supports it with its reference, like [m3] or [m3][m7]. Use only references you were shown.
- Address the user as "you". Lines marked "You:" are the user's own messages. Call other people by the name shown.
- Never write a raw phone number or email address as the identity of a person. A person without a saved name is "a number ending in 1234" or "an unsaved email contact", as shown.
- If the archive does not hold the answer, say so plainly and say what you searched. Do not guess.
- Be brief unless the user asks for detail."""


def timezone_label(now: datetime, configured: str = "") -> str:
    """The zone of the message times, for the prompt."""
    if configured:
        return configured
    local = now.astimezone() if now.tzinfo is None else now
    offset = local.strftime("%z")
    name = local.tzname() or "local time"
    return f"{name} (UTC{offset[:3]}:{offset[3:]})" if offset else name


def archive_line(summary: ArchiveSummary) -> str:
    messages = int(summary.get("messages") or 0)
    conversations = int(summary.get("conversations") or 0)
    first = str(summary.get("first_timestamp") or "")[:10]
    last = str(summary.get("last_timestamp") or "")[:10]
    span = f", from {first} to {last}" if first and last else ""
    return f"The archive holds {messages:,} messages in {conversations:,} conversations{span}."


def scope_line(where: Optional[str], since: Optional[str], until: Optional[str]) -> str:
    """`where` names the conversations of the scope; None means all."""
    if where is None and not (since or until):
        return "Scope: all conversations."
    parts = []
    if where is not None:
        parts.append(f"only {where}")
    if since and until:
        parts.append(f"only {since} to {until}")
    elif since:
        parts.append(f"only from {since}")
    elif until:
        parts.append(f"only up to {until}")
    return (
        f"Scope: {', '.join(parts)}. The user chose this scope. The tools cannot look outside it. "
        "If the answer needs something outside it, say so."
    )


def system_prompt(*, now: datetime, timezone: str, summary: ArchiveSummary, scope: str) -> str:
    return "\n".join(
        [
            "You are Recall. You answer questions about the user's own iMessage archive. "
            "The archive is on their Mac. You reach it only through the tools.",
            "",
            f"Today is {now.strftime('%A')}, {now.date().isoformat()}. "
            f"Time zone: {timezone}. Message times are local times in this zone.",
            archive_line(summary),
            scope,
            "",
            _RULES,
        ]
    )


def question_block(
    question: str, lookups: Sequence[tuple[str, str]], carried: str = "", note: str = ""
) -> str:
    """The user message: evidence carried over, lookups already run, the question."""
    parts: list[str] = []
    if carried:
        parts.append(
            "Evidence from earlier in this chat. You can cite it again or read around it:\n" + carried
        )
    if lookups:
        blocks = [
            f"[lookup {index}] {label}\n{result}" for index, (label, result) in enumerate(lookups, start=1)
        ]
        parts.append("Lookups already run for this question:\n\n" + "\n\n".join(blocks))
    else:
        parts.append("No lookup was run for this question yet.")
    if note:
        parts.append(note)
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)
