"""AI-assisted event extraction and conversation summary generation."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from ai_config import (
    DEFAULT_MODEL,
    EVENTS_TEXT_FORMAT,
    MAX_CONTEXT_TOKENS_EVENTS,
    MAX_CONTEXT_TOKENS_SUMMARY,
    get_token_budget,
)
from conversation import (
    ConversationStats,
    chunk_by_year,
    estimate_tokens,
    format_all_messages,
    truncate_to_tokens,
)
from openai_client import _call_openai

logger = logging.getLogger(__name__)

# Same voice + privacy contract as the chat pipeline: the report belongs to the
# archive's owner, and raw handles must never appear in prose.
VOICE_PRIVACY_PROMPT = (
    " Write for the archive's owner: address them as 'you' (lines marked ME are theirs), and refer to "
    "the other person by their name. Never output a raw phone number or email address in the analysis; "
    "use the person's name, or 'an unsaved contact' if there is none."
)


def _parse_json_events(content: str) -> List[Dict]:
    """Parse JSON events from AI response, handling markdown code fences."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse JSON from AI response: %s", e)
        logger.debug("Raw content: %s", content[:500])
        return []

    if isinstance(data, dict):
        events = data.get("events")
        if not isinstance(events, list):
            logger.warning("Expected 'events' to be a list in AI response")
            return []
        return events
    if isinstance(data, list):
        return data
    logger.warning("Unexpected JSON structure from AI: %s", type(data))
    return []


def _extract_events_for_period(
    client,
    contact: str,
    period_label: str,
    chunk_text: str,
    events_per_chunk: int,
    prior_context: str = "",
    model: str = DEFAULT_MODEL,
) -> List[Dict]:
    """Extract events from a single time period, with optional context from prior periods."""
    if prior_context:
        prior_context = truncate_to_tokens(prior_context, MAX_CONTEXT_TOKENS_EVENTS)
    context_section = ""
    if prior_context:
        context_section = f"## Context from Previous Periods\n\n{prior_context}\n\n"

    system = (
        "You are analyzing a specific time period of an iMessage conversation. "
        "Identify the most significant events, turning points, and meaningful moments. "
        "If context from previous periods is provided, use it to understand ongoing themes and references."
        + VOICE_PRIVACY_PROMPT
    )
    user = (
        f"# Event Extraction - {period_label}\n\n"
        f"**Contact:** {contact}\n\n"
        f"{context_section}"
        f"## All Messages from {period_label}\n\n{chunk_text}\n\n"
        f"## Task\n\n"
        f"Identify up to {events_per_chunk} significant events from {period_label}.\n\n"
        f"Return JSON with an 'events' array. Each event must include: date, title, detail, "
        f"category, score, and quote."
    )

    content = _call_openai(
        client, [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
        text_format=EVENTS_TEXT_FORMAT,
    )

    return _parse_json_events(content)


def ai_extract_events(
    contact: str,
    stats: ConversationStats,
    conv: pd.DataFrame,
    target_events: int,
    pbar: tqdm = None,
    precomputed_chunks: Optional[List[Tuple[str, pd.DataFrame, str]]] = None,
    total_tokens: Optional[int] = None,
    all_messages: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """Extract key events, using time-based chunking for large conversations."""
    client = OpenAI()

    if total_tokens is None or all_messages is None:
        all_messages = all_messages or format_all_messages(conv)
        total_tokens = total_tokens or estimate_tokens(all_messages)

    token_budget = get_token_budget(model)

    if total_tokens < token_budget:
        if pbar:
            pbar.set_description("Extracting events")

        system = (
            "You are analyzing a complete iMessage conversation to identify the most significant events. "
            "You have access to every message. Identify real turning points, not just busy days."
            + VOICE_PRIVACY_PROMPT
        )
        user = (
            f"# Event Extraction Request\n\n"
            f"**Contact:** {contact}\n"
            f"**Total messages:** {stats.total_messages:,}\n"
            f"**Date range:** {stats.first_timestamp} to {stats.last_timestamp}\n\n"
            f"## Complete Conversation\n\n{all_messages}\n\n"
            f"## Task\n\n"
            f"Identify the {target_events} most significant events/moments.\n\n"
            f"Return JSON with an 'events' array. Each event must include: date, title, detail, "
            f"category, score, and quote."
        )

        content = _call_openai(
            client, [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model,
            text_format=EVENTS_TEXT_FORMAT,
        )
        items = _parse_json_events(content)
    else:
        chunks = precomputed_chunks if precomputed_chunks else chunk_by_year(conv, max_tokens=token_budget)
        num_chunks = len(chunks)
        events_per_chunk = max(5, target_events // num_chunks + 2)

        if pbar:
            pbar.set_description(f"Extracting events ({num_chunks} periods)")

        items = []
        max_workers = min(5, num_chunks)
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for period_label, _chunk_df, chunk_text in chunks:
                future = executor.submit(
                    _extract_events_for_period,
                    client, contact, period_label, chunk_text, events_per_chunk,
                    "", model,
                )
                futures[future] = period_label

            for future in as_completed(futures):
                period_label = futures[future]
                if pbar:
                    pbar.set_description(f"Events: {period_label}")
                try:
                    items.extend(future.result())
                except Exception as exc:
                    # one failed period must not destroy the whole (expensive) report
                    logger.warning(
                        "Event extraction failed for %s: %s -- continuing without it",
                        period_label, exc,
                    )

    rows = []
    for it in items:
        date_val = pd.to_datetime(it.get("date"), errors="coerce")
        if pd.isna(date_val):
            continue
        rows.append({
            "date": date_val.date(),
            "title": str(it.get("title", ""))[:200],
            "detail": str(it.get("detail", ""))[:1000],
            "category": str(it.get("category", ""))[:50],
            "score": float(it.get("score", 0.5)),
            "quote": str(it.get("quote", ""))[:500],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.reset_index(drop=True)
    df["event_month"] = pd.to_datetime(df["date"]).dt.to_period("M")

    months = df["event_month"].unique()
    num_months = len(months) if len(months) > 0 else 1
    min_per_month = max(2, min(10, target_events // num_months))

    selected_indices = set()
    for month in sorted(months):
        month_events = df[df["event_month"] == month].sort_values("score", ascending=False)
        selected_indices.update(month_events.head(min_per_month).index.tolist())

    if len(selected_indices) < target_events:
        remaining = df[~df.index.isin(selected_indices)].sort_values("score", ascending=False)
        for idx in remaining.index:
            if len(selected_indices) >= target_events:
                break
            selected_indices.add(idx)

    selected_df = df.loc[list(selected_indices)].copy()

    if len(selected_df) > target_events:
        base_indices = []
        for _month, group in selected_df.groupby("event_month"):
            base_indices.append(group.sort_values("score", ascending=False).index[0])
        base_df = selected_df.loc[base_indices]

        if len(base_df) > target_events:
            base_df = base_df.sort_values("score", ascending=False).head(target_events)
        else:
            remaining = selected_df.drop(index=base_indices).sort_values("score", ascending=False)
            needed = target_events - len(base_df)
            if needed > 0:
                base_df = pd.concat([base_df, remaining.head(needed)])
        selected_df = base_df

    selected_df = selected_df.drop(columns=["event_month"], errors="ignore")
    return selected_df.sort_values("date").head(target_events)


def _summarize_period(
    client,
    contact: str,
    period_label: str,
    chunk_text: str,
    msg_count: int,
    prior_context: str = "",
    model: str = DEFAULT_MODEL,
) -> Tuple[str, str]:
    """
    Summarize a single time period, with optional context from prior periods.
    Returns (summary, context_for_next_period).
    """
    if prior_context:
        prior_context = truncate_to_tokens(prior_context, MAX_CONTEXT_TOKENS_SUMMARY)
    context_section = ""
    if prior_context:
        context_section = f"## Context from Previous Periods\n\n{prior_context}\n\n"

    system = (
        "You are analyzing a specific time period of an iMessage conversation. "
        "Provide an insightful summary that captures the essence of this period. "
        "If context from previous periods is provided, note how themes continue, evolve, or change."
        + VOICE_PRIVACY_PROMPT
    )
    user = (
        f"# {period_label} - {contact}\n\n"
        f"**Messages in this period:** {msg_count:,}\n\n"
        f"{context_section}"
        f"## All Messages from {period_label}\n\n{chunk_text}\n\n"
        f"## Task\n\n"
        f"Provide TWO things:\n\n"
        f"### SUMMARY\n"
        f"Summarize {period_label} (~600 words):\n"
        f"- What was the relationship like during this period?\n"
        f"- Key themes, topics, and emotional tone\n"
        f"- Notable quotes (include 3-5 verbatim with dates)\n"
        f"- Any significant events, shifts, or turning points\n"
        f"- How does this period connect to previous periods (if context provided)?\n\n"
        f"### CONTEXT FOR NEXT PERIOD\n"
        f"In 2-3 sentences, what should someone analyzing the next time period know? "
        f"Include unresolved tensions, ongoing topics, relationship status, and key events they might reference."
    )

    content = _call_openai(
        client, [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
    )

    context_markers = [
        "### CONTEXT FOR NEXT PERIOD",
        "## CONTEXT FOR NEXT PERIOD",
        "**CONTEXT FOR NEXT PERIOD**",
        "CONTEXT FOR NEXT PERIOD:",
        "Context for next period:",
    ]

    summary = content
    context_for_next = ""

    for marker in context_markers:
        if marker.lower() in content.lower():
            idx = content.lower().find(marker.lower())
            summary = content[:idx].strip()
            context_for_next = content[idx + len(marker):].strip()
            break

    for header in ["### SUMMARY", "## SUMMARY", "**SUMMARY**"]:
        summary = summary.replace(header, "").strip()

    return summary, context_for_next


def ai_summary(
    contact: str,
    stats: ConversationStats,
    conv: pd.DataFrame,
    key_events: pd.DataFrame = None,
    pbar: tqdm = None,
    precomputed_chunks: Optional[List[Tuple[str, pd.DataFrame, str]]] = None,
    total_tokens: Optional[int] = None,
    all_messages: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate narrative summary, using time-based chunking for large conversations."""
    client = OpenAI()

    if total_tokens is None or all_messages is None:
        all_messages = all_messages or format_all_messages(conv)
        total_tokens = total_tokens or estimate_tokens(all_messages)

    att_context = ""
    if stats.attachments.total > 0:
        att_context = f"\nAttachments: {stats.attachments.photos} photos, {stats.attachments.videos} videos, {stats.attachments.audio} audio"

    events_context = ""
    if key_events is not None and not key_events.empty:
        events_list = "\n".join([
            f"- {row['date']}: {row['title']} - {str(row.get('detail', ''))[:100]}"
            for _, row in key_events.iterrows()
        ])
        events_context = f"\n\nKey events identified:\n{events_list}"

    token_budget = get_token_budget(model)

    if total_tokens < token_budget:
        if pbar:
            pbar.set_description("Generating summary")

        system = (
            "You are analyzing a complete iMessage conversation history. You have access to every message. "
            "Produce a deeply insightful, emotionally intelligent analysis of this relationship."
            + VOICE_PRIVACY_PROMPT
        )
        user = (
            f"# Conversation Analysis Request\n\n"
            f"**Contact:** {contact}\n\n"
            f"## Statistics\n"
            f"- Total messages: {stats.total_messages:,}\n"
            f"- Sent by me: {stats.sent_count:,} | Received: {stats.received_count:,}\n"
            f"- Date range: {stats.first_timestamp} to {stats.last_timestamp}\n"
            f"- Active days: {stats.active_days} (avg {stats.avg_messages_per_day:.1f} msgs/day)\n"
            f"- Busiest day: {stats.busiest_day} ({stats.busiest_day_count:,} messages)\n"
            f"- Longest gap: {stats.longest_gap_days:.1f} days"
            f"{att_context}{events_context}\n\n"
            f"## Complete Conversation\n\n{all_messages}\n\n"
            f"## Task\n\n"
            f"Write a comprehensive analysis with these sections:\n\n"
            f"### Timeline\nBreak down the relationship chronologically. What were the distinct phases?\n\n"
            f"### Dynamics\nWho initiates more? Power dynamics? Communication patterns?\n\n"
            f"### Notable Moments\nQuote specific messages that reveal something important.\n\n"
            f"### Takeaways\nWhat does this conversation reveal about the relationship?\n\n"
            f"Be specific, quote messages, cite dates."
        )

        return _call_openai(
            client, [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model,
        )

    chunks = precomputed_chunks if precomputed_chunks else chunk_by_year(conv, max_tokens=token_budget)
    num_chunks = len(chunks)

    if pbar:
        pbar.set_description(f"Summarizing ({num_chunks} periods)")

    period_summaries = []
    prior_context = ""

    for period_label, chunk_df, chunk_text in chunks:
        if pbar:
            pbar.set_description(f"Summary: {period_label}")

        try:
            summary, context_for_next = _summarize_period(
                client, contact, period_label, chunk_text, len(chunk_df), prior_context,
                model=model,
            )
        except Exception as exc:
            # keep the report alive; flag the gap instead of losing everything
            logger.warning("Summary failed for %s: %s -- continuing without it", period_label, exc)
            summary, context_for_next = f"*(Analysis for {period_label} could not be generated.)*", ""
        period_summaries.append(f"## {period_label}\n\n{summary}")

        if context_for_next:
            prior_context += f"\n**{period_label}**: {context_for_next}"

    if pbar:
        pbar.set_description("Synthesizing across all years")

    combined_summaries = "\n\n---\n\n".join(period_summaries)

    system = (
        "You are synthesizing yearly summaries of an iMessage conversation into a cohesive analysis. "
        "Create a unified narrative that captures the full arc of the relationship over time."
        + VOICE_PRIVACY_PROMPT
    )
    user = (
        f"# Final Synthesis - {contact}\n\n"
        f"## Overall Statistics\n"
        f"- Total messages: {stats.total_messages:,}\n"
        f"- Sent by me: {stats.sent_count:,} | Received: {stats.received_count:,}\n"
        f"- Date range: {stats.first_timestamp} to {stats.last_timestamp}\n"
        f"- Active days: {stats.active_days}"
        f"{att_context}{events_context}\n\n"
        f"## Yearly Summaries\n\n{combined_summaries}\n\n"
        f"## Task\n\n"
        f"Synthesize these yearly summaries into a comprehensive relationship analysis:\n\n"
        f"### The Arc\nHow did this relationship evolve from beginning to now? What were the major phases?\n\n"
        f"### Dynamics\nOverall communication patterns. Who initiates? How has that changed?\n\n"
        f"### Defining Moments\nThe most significant quotes and events across all years.\n\n"
        f"### The Big Picture\nWhat story does this conversation tell? What does it reveal?\n\n"
        f"Create a cohesive narrative that flows across years, not a list of years."
    )

    return _call_openai(
        client, [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
    )
