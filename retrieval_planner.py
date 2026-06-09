"""LLM retrieval planner.

Turns a chat question into a retrieval plan over the user's conversations, so
references are grabbed by the *meaning* of the question (which person/thread,
which topic) instead of string-matching the literal query words.

Example: "who is this 720" should route to the conversation whose handle is a
720-area-code number, not surface every message that contains the word "who".

The conversation catalog sent to the model uses short aliases (c0, c1, ...) and
masked handles (area code + last 4), never full phone numbers.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

import pandas as pd
from openai import OpenAI

from ai_config import DEFAULT_MODEL
from openai_client import _call_openai

PLAN_TEXT_FORMAT = {
    "type": "json_schema",
    "name": "retrieval_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["identity", "topical", "timeline", "general"],
                "description": (
                    "identity = asking who a person/number is; topical = a subject or theme; "
                    "timeline = time-based; general = anything else."
                ),
            },
            "conversation_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Catalog ids (like 'c3') to search. Empty means search across all conversations.",
            },
            "search_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Meaning-expanded keywords (synonyms / related words) to find relevant messages. "
                    "NOT the literal words of the question. Empty for pure identity lookups."
                ),
            },
            "prefer_recent": {"type": "boolean"},
        },
        "required": ["intent", "conversation_ids", "search_terms", "prefer_recent"],
    },
}


def _digit_runs(text: str) -> list[str]:
    return re.findall(r"\d{3,}", str(text or ""))


NAME_STOPWORDS = {
    "who", "whos", "whose", "what", "when", "where", "which", "why", "how",
    "is", "are", "was", "were", "the", "this", "that", "these", "those", "and", "for",
    "with", "about", "from", "you", "your", "our", "they", "them", "his", "her", "she",
    "him", "me", "my", "mine", "we", "did", "does", "talk", "talked", "talking",
    "say", "said", "tell", "told", "person", "people", "contact", "number", "message",
    "messages", "msg", "guy", "girl", "name", "named", "called", "anymore", "still",
}


def _name_tokens(question: str) -> list[str]:
    """Word tokens from the question that could name a contact (drops question words)."""
    return [
        tok
        for tok in re.findall(r"[a-z][a-z'’]{2,}", str(question or "").lower())
        if tok not in NAME_STOPWORDS
    ]


def build_catalog(
    conversations: pd.DataFrame,
    question: str,
    name_lookup: Optional[Callable[[list[str]], dict[str, str]]] = None,
    top_n: int = 60,
) -> pd.DataFrame:
    """Top conversations by volume, plus any the question points at:
    a number/area-code in the handle, or a word that matches a contact's name.
    So 'who is yaoye lin' surfaces the Yaoye Lin thread even if it's low-volume.

    `conversations` must have columns: chat_id, message_count, last_msg.
    """
    if conversations is None or conversations.empty:
        return pd.DataFrame(columns=["chat_id", "message_count", "last_msg", "display_name"])

    convo = conversations.copy()
    convo["chat_id"] = convo["chat_id"].astype(str)
    convo = convo.sort_values("message_count", ascending=False)

    # resolving every name is cheap (cache + one address-book pass) and lets us
    # match a named contact the question asks about, however small the thread.
    names = name_lookup(convo["chat_id"].tolist()) if name_lookup else {}

    selected = convo.head(top_n)

    runs = _digit_runs(question)
    if runs:
        digits = convo["chat_id"].map(lambda cid: re.sub(r"\D", "", cid))
        selected = pd.concat([selected, convo[digits.map(lambda d: any(run in d for run in runs))]])

    tokens = _name_tokens(question)
    if tokens and names:
        def name_hit(cid: str) -> bool:
            label = str(names.get(cid, "")).lower()
            return bool(label) and any(tok in label for tok in tokens)

        matched = convo[convo["chat_id"].map(name_hit)]
        if not matched.empty:
            selected = pd.concat([selected, matched.head(10)])

    selected = selected.drop_duplicates("chat_id").copy()
    selected["display_name"] = selected["chat_id"].map(names).fillna("") if names else ""
    return selected


def _masked_handle(chat_id: str) -> str:
    if str(chat_id).startswith("chat"):
        return "group chat"
    if "@" in str(chat_id):
        return "email"
    digits = re.sub(r"\D", "", str(chat_id))
    if len(digits) >= 7:
        return f"{digits[-10:-7] or digits[:3]}-***-{digits[-4:]}"
    return "contact"


def catalog_payload(catalog: pd.DataFrame) -> tuple[list[dict], dict[str, str]]:
    """Return (rows_for_model, alias_to_chat_id). Aliases keep handles off the wire."""
    rows: list[dict] = []
    alias_map: dict[str, str] = {}
    for index, (_, row) in enumerate(catalog.iterrows()):
        alias = f"c{index}"
        chat_id = str(row["chat_id"])
        alias_map[alias] = chat_id
        name = str(row.get("display_name") or "").strip()
        rows.append(
            {
                "id": alias,
                "name": name or "(unsaved)",
                "handle": _masked_handle(chat_id),
                "messages": int(row.get("message_count") or 0),
                "last": str(row.get("last_msg") or "")[:10],
            }
        )
    return rows, alias_map


def plan_retrieval(
    question: str,
    catalog: pd.DataFrame,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_MODEL,
) -> Optional[dict]:
    """Ask the model which conversations + meaning-expanded terms to retrieve.

    Returns a plan dict with chat_ids already mapped back to real ids, or None on
    any failure (callers fall back to literal keyword search).
    """
    rows, alias_map = catalog_payload(catalog)
    if not rows:
        return None

    system = (
        "You plan retrieval over a person's private iMessage archive. "
        "Given their question and a catalog of conversations, decide which conversations to pull from "
        "and what to look for, based on the MEANING of the question, not its literal words. "
        "If they ask 'who is this <number>' or about an area code, pick the conversation whose handle matches. "
        "If they describe a person ('my old roommate', 'the recruiter'), pick the best-matching conversation(s). "
        "For topical questions, expand into related search terms (synonyms and associated words), "
        "never just repeat the question's own words. "
        "Return catalog ids like 'c3'. Empty conversation_ids means search across everything."
    )
    user = (
        f"Question: {question}\n\n"
        f"Conversation catalog (JSON array):\n{json.dumps(rows, ensure_ascii=False)}"
    )

    try:
        content = _call_openai(
            client or OpenAI(),
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model or DEFAULT_MODEL,
            max_retries=1,
            reasoning_effort="low",
            verbosity="low",
            text_format=PLAN_TEXT_FORMAT,
        )
        plan = json.loads(content) if content else None
    except Exception:
        return None

    if not isinstance(plan, dict):
        return None

    chat_ids = [
        alias_map[alias]
        for alias in plan.get("conversation_ids", [])
        if alias in alias_map
    ]
    terms = [str(term).strip() for term in plan.get("search_terms", []) if str(term).strip()]
    return {
        "intent": str(plan.get("intent") or "general"),
        "chat_ids": chat_ids,
        "search_terms": terms[:12],
        "prefer_recent": bool(plan.get("prefer_recent", False)),
    }
