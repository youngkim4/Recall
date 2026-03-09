#!/usr/bin/env python3
"""
Module: analysis

Purpose:
- Analyze conversations exported to CSV with full context AI analysis
- Produce timeline CSV, Markdown report, and optional HTML with charts
- Generate AI-assisted narrative, key events from FULL conversation context
"""
import argparse
import logging
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError
import tiktoken

logger = logging.getLogger(__name__)

load_dotenv()

MAX_CONTEXT_TOKENS_EVENTS = 12000
MAX_CONTEXT_TOKENS_SUMMARY = 20000
TOKEN_BUDGET = 170_000
SECONDS_PER_DAY = 86_400.0


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AttachmentStats:
    total: int = 0
    photos: int = 0
    videos: int = 0
    audio: int = 0
    gifs: int = 0
    documents: int = 0
    other: int = 0
    sent_attachments: int = 0
    received_attachments: int = 0


@dataclass
class ReactionStats:
    total: int = 0
    loves: int = 0
    likes: int = 0
    dislikes: int = 0
    laughs: int = 0
    emphasis: int = 0
    questions: int = 0
    sent_reactions: int = 0
    received_reactions: int = 0


@dataclass
class ConversationStats:
    chat_id: str
    total_messages: int
    sent_count: int
    received_count: int
    unknown_direction_count: int
    first_timestamp: Optional[pd.Timestamp]
    last_timestamp: Optional[pd.Timestamp]
    active_days: int
    avg_messages_per_day: float
    busiest_day: Optional[pd.Timestamp]
    busiest_day_count: int
    longest_gap_days: float
    longest_gap_start: Optional[pd.Timestamp]
    longest_gap_end: Optional[pd.Timestamp]
    attachments: AttachmentStats = field(default_factory=AttachmentStats)
    reactions: ReactionStats = field(default_factory=ReactionStats)


# =============================================================================
# Data Loading
# =============================================================================

def sanitize_filename(value: str) -> str:
    safe = "".join(c for c in value if c.isalnum() or c in ("-", "_", "+"))
    return safe or "conversation"


def load_messages(csv_path: str, since: datetime = None, until: datetime = None) -> pd.DataFrame:
    """Load messages CSV with optional date filtering."""
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError("messages CSV must contain a 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    if since:
        df = df[df["timestamp"] >= pd.to_datetime(since)]
    if until:
        df = df[df["timestamp"] <= pd.to_datetime(until)]
    
    if "is_from_me" not in df.columns:
        df["is_from_me"] = pd.NA
    if "service" not in df.columns:
        df["service"] = pd.NA
    return df


def load_attachments(csv_path: str) -> pd.DataFrame:
    """Load attachments CSV if it exists."""
    p = Path(csv_path)
    att_path = p.with_name(p.stem + "_attachments.csv")
    if att_path.exists():
        return pd.read_csv(str(att_path))
    return pd.DataFrame()


def load_reactions(csv_path: str) -> pd.DataFrame:
    """Load reactions CSV if it exists."""
    p = Path(csv_path)
    react_path = p.with_name(p.stem + "_reactions.csv")
    if react_path.exists():
        return pd.read_csv(str(react_path))
    return pd.DataFrame()


def filter_conversation(df: pd.DataFrame, contact: str) -> pd.DataFrame:
    """Return a DataFrame for matching chat_id, sorted by time."""
    if "chat_id" not in df.columns:
        raise ValueError("messages CSV must contain a 'chat_id' column")
    exact = df[df["chat_id"].astype(str) == contact]
    if not exact.empty:
        return exact.sort_values("timestamp")
    contains = df[df["chat_id"].astype(str).str.contains(contact, na=False, regex=False)]
    return contains.sort_values("timestamp")


# =============================================================================
# Stats Computation
# =============================================================================

def compute_attachment_stats(attachments_df: pd.DataFrame, contact: str) -> AttachmentStats:
    """Compute attachment statistics for a contact."""
    stats = AttachmentStats()
    if attachments_df.empty or "chat_id" not in attachments_df.columns:
        return stats
    
    filtered = attachments_df[attachments_df["chat_id"].astype(str) == contact]
    if filtered.empty:
        return stats
    
    stats.total = len(filtered)
    if "category" in filtered.columns:
        cats = filtered["category"].value_counts()
        stats.photos = int(cats.get("photo", 0))
        stats.videos = int(cats.get("video", 0))
        stats.audio = int(cats.get("audio", 0))
        stats.gifs = int(cats.get("gif", 0))
        stats.documents = int(cats.get("document", 0))
        stats.other = int(cats.get("other", 0)) + int(cats.get("unknown", 0))
    
    if "is_from_me" in filtered.columns:
        stats.sent_attachments = int(filtered["is_from_me"].fillna(0).astype(int).sum())
        stats.received_attachments = stats.total - stats.sent_attachments
    
    return stats


def compute_reaction_stats(reactions_df: pd.DataFrame, contact: str) -> ReactionStats:
    """Compute reaction statistics for a contact."""
    stats = ReactionStats()
    if reactions_df.empty or "chat_id" not in reactions_df.columns:
        return stats
    
    filtered = reactions_df[reactions_df["chat_id"].astype(str) == contact]
    if filtered.empty:
        return stats
    
    if "is_add" in filtered.columns:
        filtered = filtered[filtered["is_add"].eq(True)]
    
    stats.total = len(filtered)
    if "reaction_type" in filtered.columns:
        types = filtered["reaction_type"].value_counts()
        stats.loves = int(types.get("love", 0))
        stats.likes = int(types.get("like", 0))
        stats.dislikes = int(types.get("dislike", 0))
        stats.laughs = int(types.get("laugh", 0))
        stats.emphasis = int(types.get("emphasis", 0))
        stats.questions = int(types.get("question", 0))
    
    if "is_from_me" in filtered.columns:
        stats.sent_reactions = int(filtered["is_from_me"].fillna(0).astype(int).sum())
        stats.received_reactions = stats.total - stats.sent_reactions
    
    return stats


def compute_stats(conv: pd.DataFrame, attachments_df: pd.DataFrame = None, reactions_df: pd.DataFrame = None) -> ConversationStats:
    """Compute summary statistics for the conversation."""
    if conv.empty:
        return ConversationStats(
            chat_id="", total_messages=0, sent_count=0, received_count=0,
            unknown_direction_count=0, first_timestamp=None, last_timestamp=None,
            active_days=0, avg_messages_per_day=0.0, busiest_day=None,
            busiest_day_count=0, longest_gap_days=0.0, longest_gap_start=None,
            longest_gap_end=None,
        )

    conv = conv.copy()
    conv["date"] = conv["timestamp"].dt.date
    total = len(conv)
    direction_col = conv["is_from_me"] if "is_from_me" in conv.columns else pd.Series([pd.NA] * len(conv))
    known_direction = direction_col.dropna().astype(int)
    sent = int(known_direction.sum()) if not known_direction.empty else 0
    unknown_dir = total - len(known_direction)
    received = max(0, total - sent - unknown_dir)
    first_ts = pd.to_datetime(conv["timestamp"].min())
    last_ts = pd.to_datetime(conv["timestamp"].max())
    per_day = conv.groupby("date").size()
    active_days = int(per_day.shape[0])
    avg_per_day = float(per_day.mean()) if active_days > 0 else 0.0
    busiest_day = pd.to_datetime(per_day.idxmax()) if active_days > 0 else None
    busiest_day_count = int(per_day.max()) if active_days > 0 else 0

    timestamps = conv["timestamp"].sort_values().dropna().reset_index(drop=True)
    if len(timestamps) >= 2:
        diffs = timestamps.diff()
        max_idx = int(diffs.idxmax())
        longest_gap = diffs.iloc[max_idx]
        gap_days = float(longest_gap.total_seconds() / SECONDS_PER_DAY)
        gap_start = pd.to_datetime(timestamps.iloc[max_idx - 1]) if max_idx - 1 >= 0 else None
        gap_end = pd.to_datetime(timestamps.iloc[max_idx])
    else:
        gap_days, gap_start, gap_end = 0.0, None, None

    chat_id = str(conv["chat_id"].iloc[0]) if "chat_id" in conv.columns and not conv.empty else ""
    att_stats = compute_attachment_stats(attachments_df, chat_id) if attachments_df is not None else AttachmentStats()
    react_stats = compute_reaction_stats(reactions_df, chat_id) if reactions_df is not None else ReactionStats()

    return ConversationStats(
        chat_id=chat_id, total_messages=total, sent_count=sent, received_count=received,
        unknown_direction_count=unknown_dir, first_timestamp=first_ts, last_timestamp=last_ts,
        active_days=active_days, avg_messages_per_day=avg_per_day, busiest_day=busiest_day,
        busiest_day_count=busiest_day_count, longest_gap_days=gap_days,
        longest_gap_start=gap_start, longest_gap_end=gap_end,
        attachments=att_stats, reactions=react_stats,
    )


def progression_series(conv: pd.DataFrame) -> pd.DataFrame:
    """Aggregate conversation by month and compute sent ratio over time."""
    conv = conv.copy()
    conv["month"] = conv["timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly = conv.groupby("month").agg(
        total=("text", "size"),
        sent=("is_from_me", lambda s: int(pd.Series(s).fillna(0).astype(int).sum())),
    )
    monthly["received"] = monthly["total"] - monthly["sent"]
    monthly["sent_ratio"] = monthly.apply(lambda r: (r["sent"] / r["total"]) if r["total"] else 0.0, axis=1)
    return monthly.sort_index().reset_index()


# =============================================================================
# AI Analysis (Full Context)
# =============================================================================

def format_all_messages(conv: pd.DataFrame) -> str:
    """Format all messages for GPT context."""
    conv = conv.copy()
    is_from_me = conv["is_from_me"] if "is_from_me" in conv.columns else pd.Series([pd.NA] * len(conv), index=conv.index)
    conv["direction"] = is_from_me.apply(
        lambda x: "ME" if pd.notna(x) and int(x) == 1 else ("THEM" if pd.notna(x) and int(x) == 0 else "??")
    )
    conv["ts"] = conv["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    lines = conv.apply(lambda r: f"[{r['ts']}] {r['direction']}: {str(r['text'])}", axis=1).tolist()
    return "\n".join(lines)


_tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4/5 encoding


def estimate_tokens(text: str) -> int:
    """Count tokens using tiktoken (exact)."""
    return len(_tokenizer.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to last max_tokens tokens to preserve recent context."""
    if not text or max_tokens <= 0:
        return ""
    tokens = _tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _tokenizer.decode(tokens[-max_tokens:])


def _call_openai(client, messages, model="gpt-5-mini", max_retries=2) -> str:
    """Call OpenAI API with retry logic for transient errors."""
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            content = resp.choices[0].message.content
            if content is None:
                logger.warning("OpenAI returned None content")
                return ""
            return content.strip()
        except RateLimitError:
            if attempt < max_retries:
                wait = 2 ** attempt * 5
                logger.warning("Rate limited, retrying in %ss (attempt %s/%s)", wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                logger.error("Rate limit exceeded after %s retries", max_retries)
                raise
        except (APIConnectionError, APITimeoutError) as e:
            if attempt < max_retries:
                wait = 2 ** attempt * 2
                logger.warning("Connection error, retrying in %ss (attempt %s/%s)", wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                logger.error("Connection failed after %s retries: %s", max_retries, e)
                raise
        except APIError as e:
            logger.error("OpenAI API error: %s", e)
            raise
    return ""


def chunk_by_year(conv: pd.DataFrame, max_tokens: int = TOKEN_BUDGET) -> List[Tuple[str, pd.DataFrame, str]]:
    """
    Split conversation by year. If a year exceeds token limit, split into smaller periods.
    Hierarchy: Year -> Half -> Quarter -> Month -> Weeks -> Days
    Returns list of (period_label, dataframe, formatted_text) tuples.
    All messages are included - nothing is dropped.
    """
    conv = conv.copy()
    is_from_me = conv["is_from_me"] if "is_from_me" in conv.columns else pd.Series([pd.NA] * len(conv), index=conv.index)
    conv["direction"] = is_from_me.apply(
        lambda x: "ME" if pd.notna(x) and int(x) == 1 else ("THEM" if pd.notna(x) and int(x) == 0 else "??")
    )
    conv["ts"] = conv["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    conv["formatted"] = conv.apply(lambda r: f"[{r['ts']}] {r['direction']}: {str(r['text'])}", axis=1)
    conv["year"] = conv["timestamp"].dt.year
    conv["month"] = conv["timestamp"].dt.month
    conv["week"] = conv["timestamp"].dt.isocalendar().week.astype(int)
    conv["day"] = conv["timestamp"].dt.day
    
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    chunks = []
    
    def add_week_chunk(label, df):
        """Add a week chunk, splitting by day if still too large."""
        text = "\n".join(df["formatted"].tolist())
        tokens = estimate_tokens(text)
        
        if tokens <= max_tokens:
            chunks.append((label, df, text))
        else:
            # Week still too large - split by day
            for d in sorted(df["day"].unique()):
                d_df = df[df["day"] == d]
                if d_df.empty:
                    continue
                d_text = "\n".join(d_df["formatted"].tolist())
                chunks.append((f"{label} D{d}", d_df, d_text))
    
    def add_month_chunk(label, df):
        """Add a month chunk, splitting by week if still too large."""
        text = "\n".join(df["formatted"].tolist())
        tokens = estimate_tokens(text)
        
        if tokens <= max_tokens:
            chunks.append((label, df, text))
        else:
            # Month still too large - split by week
            for w in sorted(df["week"].unique()):
                w_df = df[df["week"] == w]
                if w_df.empty:
                    continue
                add_week_chunk(f"{label} W{w}", w_df)
    
    def add_quarter_chunk(label, df):
        """Add a quarter chunk, splitting by month if still too large."""
        text = "\n".join(df["formatted"].tolist())
        tokens = estimate_tokens(text)
        
        if tokens <= max_tokens:
            chunks.append((label, df, text))
        else:
            # Quarter too large - split by month
            for m in sorted(df["month"].unique()):
                m_df = df[df["month"] == m]
                if m_df.empty:
                    continue
                year = m_df["year"].iloc[0]
                add_month_chunk(f"{month_names[m-1]} {year}", m_df)
    
    for year in sorted(conv["year"].unique()):
        year_df = conv[conv["year"] == year].copy()
        year_text = "\n".join(year_df["formatted"].tolist())
        year_tokens = estimate_tokens(year_text)
        
        if year_tokens <= max_tokens:
            chunks.append((str(year), year_df, year_text))
        else:
            # Year too large - split into halves (H1/H2)
            year_df["half"] = year_df["month"].apply(lambda m: "H1" if m <= 6 else "H2")
            
            for half in ["H1", "H2"]:
                half_df = year_df[year_df["half"] == half].copy()
                if half_df.empty:
                    continue
                    
                half_text = "\n".join(half_df["formatted"].tolist())
                half_tokens = estimate_tokens(half_text)
                
                if half_tokens <= max_tokens:
                    chunks.append((f"{year} {half}", half_df, half_text))
                else:
                    # Half too large - split by quarter
                    half_df["quarter"] = half_df["month"].apply(lambda m: (m - 1) // 3 + 1)
                    for q in sorted(half_df["quarter"].unique()):
                        q_df = half_df[half_df["quarter"] == q].copy()
                        if q_df.empty:
                            continue
                        add_quarter_chunk(f"{year} Q{q}", q_df)
    
    return chunks


def _extract_events_for_period(
    client,
    contact: str,
    period_label: str,
    chunk_text: str,
    events_per_chunk: int,
    prior_context: str = "",
    model: str = "gpt-5-mini"
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
    )
    user = (
        f"# Event Extraction — {period_label}\n\n"
        f"**Contact:** {contact}\n\n"
        f"{context_section}"
        f"## All Messages from {period_label}\n\n{chunk_text}\n\n"
        f"## Task\n\n"
        f"Identify up to {events_per_chunk} significant events from {period_label}.\n\n"
        f"Return JSON with 'events' array. Each event:\n"
        f"- date: YYYY-MM-DD\n"
        f"- title: Short title (5-10 words)\n"
        f"- detail: What happened (2-3 sentences)\n"
        f"- category: [milestone, conflict, reconciliation, turning_point, intimacy, external_event]\n"
        f"- score: Significance 0-1\n"
        f"- quote: A verbatim message that captures this moment\n\n"
        f"JSON only."
    )
    
    content = _call_openai(
        client, [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
    )

    return _parse_json_events(content)


def _parse_json_events(content: str) -> List[Dict]:
    """Parse JSON events from AI response, handling markdown code fences."""
    cleaned = content.strip()
    # Strip markdown code fences if GPT wraps the JSON
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


def ai_extract_events(
    contact: str,
    stats: ConversationStats,
    conv: pd.DataFrame,
    target_events: int,
    pbar: tqdm = None,
    precomputed_chunks: Optional[List[Tuple[str, pd.DataFrame, str]]] = None,
    total_tokens: Optional[int] = None,
    all_messages: Optional[str] = None,
    model: str = "gpt-5-mini"
) -> pd.DataFrame:
    """Extract key events, using year-based chunking for large conversations."""

    client = OpenAI()
    
    # Use precomputed values if provided, compute if needed
    if total_tokens is None or all_messages is None:
        all_messages = all_messages or format_all_messages(conv)
        total_tokens = total_tokens or estimate_tokens(all_messages)
    
    # If small enough, do single request (170K limit with buffer for system/output)
    if total_tokens < TOKEN_BUDGET:
        if pbar:
            pbar.set_description("Extracting events")
        
        system = (
            "You are analyzing a complete iMessage conversation to identify the most significant events. "
            "You have access to EVERY message. Identify real turning points, not just busy days."
        )
        user = (
            f"# Event Extraction Request\n\n"
            f"**Contact:** {contact}\n"
            f"**Total messages:** {stats.total_messages:,}\n"
            f"**Date range:** {stats.first_timestamp} to {stats.last_timestamp}\n\n"
            f"## Complete Conversation\n\n{all_messages}\n\n"
            f"## Your Task\n\n"
            f"Identify the {target_events} most significant events/moments.\n\n"
            f"Return JSON with 'events' array. Each event:\n"
            f"- date: YYYY-MM-DD\n"
            f"- title: Short title (5-10 words)\n"
            f"- detail: What happened (2-3 sentences)\n"
            f"- category: [milestone, conflict, reconciliation, turning_point, intimacy, external_event]\n"
            f"- score: Significance 0-1\n"
            f"- quote: A verbatim message\n\n"
            f"JSON only."
        )
        
        content = _call_openai(
            client, [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model,
        )
        items = _parse_json_events(content)
    else:
        # Year-based chunking with cross-period context
        chunks = precomputed_chunks if precomputed_chunks else chunk_by_year(conv)
        num_chunks = len(chunks)
        events_per_chunk = max(5, target_events // num_chunks + 2)
        
        if pbar:
            pbar.set_description(f"Extracting events ({num_chunks} periods)")
        
        items = []

        # Parallel event extraction - each period is independent
        max_workers = min(5, num_chunks)
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for period_label, chunk_df, chunk_text in chunks:
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
                chunk_events = future.result()
                items.extend(chunk_events)
    
    # Parse all events with their source period
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
    
    # Ensure chronological coverage: don't just take top scores globally
    # Group by month, take top N from each month, then fill remaining with highest scores
    df = df.reset_index(drop=True)  # Ensure clean indices
    df["event_month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    
    months = df["event_month"].unique()
    num_months = len(months) if len(months) > 0 else 1
    # Reasonable per-month limit: at least 2, at most 10, scaled by target
    min_per_month = max(2, min(10, target_events // num_months))
    
    selected_indices = set()
    for month in sorted(months):
        month_events = df[df["event_month"] == month].sort_values("score", ascending=False)
        selected_indices.update(month_events.head(min_per_month).index.tolist())
    
    # If we still have room, add more high-scoring events not already selected
    if len(selected_indices) < target_events:
        remaining = df[~df.index.isin(selected_indices)].sort_values("score", ascending=False)
        for idx in remaining.index:
            if len(selected_indices) >= target_events:
                break
            selected_indices.add(idx)
    
    # Build final dataframe from selected indices
    selected_df = df.loc[list(selected_indices)].copy()

    # If we overshot, keep coverage by ensuring at least one per month
    if len(selected_df) > target_events:
        base_indices = []
        for month, group in selected_df.groupby("event_month"):
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
    model: str = "gpt-5-mini"
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
    )
    user = (
        f"# {period_label} — {contact}\n\n"
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
        f"In 2-3 sentences, what should someone analyzing the NEXT time period know? "
        f"Include: unresolved tensions, ongoing topics, relationship status, key events they might reference. "
        f"Be specific - names, dates, quotes if relevant."
    )
    
    content = _call_openai(
        client, [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
    )

    # Parse out the context section - try multiple header formats
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
            # Case-insensitive split
            idx = content.lower().find(marker.lower())
            summary = content[:idx].strip()
            context_for_next = content[idx + len(marker):].strip()
            break
    
    # Clean up summary - remove header if present
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
    model: str = "gpt-5-mini"
) -> str:
    """Generate narrative summary, using year-based chunking for large conversations."""

    client = OpenAI()
    
    # Use precomputed values if provided, compute if needed
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
    
    # If small enough, do single request (170K limit with buffer for system/output)
    if total_tokens < TOKEN_BUDGET:
        if pbar:
            pbar.set_description("Generating summary")
        
        system = (
            "You are analyzing a complete iMessage conversation history. You have access to EVERY message. "
            "Produce a deeply insightful, emotionally intelligent analysis of this relationship."
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
            f"## Your Task\n\n"
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

    # Year-based chunking with cross-period context
    chunks = precomputed_chunks if precomputed_chunks else chunk_by_year(conv)
    num_chunks = len(chunks)
    
    if pbar:
        pbar.set_description(f"Summarizing ({num_chunks} periods)")
    
    period_summaries = []
    prior_context = ""  # Accumulate AI-generated context from previous periods
    
    for i, (period_label, chunk_df, chunk_text) in enumerate(chunks):
        if pbar:
            pbar.set_description(f"Summary: {period_label}")
        
        summary, context_for_next = _summarize_period(
            client, contact, period_label, chunk_text, len(chunk_df), prior_context,
            model=model,
        )
        period_summaries.append(f"## {period_label}\n\n{summary}")
        
        # Use AI-generated context for next period
        if context_for_next:
            prior_context += f"\n**{period_label}**: {context_for_next}"
    
    # Final synthesis across all years
    if pbar:
        pbar.set_description("Synthesizing across all years")
    
    combined_summaries = "\n\n---\n\n".join(period_summaries)
    
    system = (
        "You are synthesizing yearly summaries of an iMessage conversation into a cohesive analysis. "
        "Create a unified narrative that captures the full arc of the relationship over time."
    )
    user = (
        f"# Final Synthesis — {contact}\n\n"
        f"## Overall Statistics\n"
        f"- Total messages: {stats.total_messages:,}\n"
        f"- Sent by me: {stats.sent_count:,} | Received: {stats.received_count:,}\n"
        f"- Date range: {stats.first_timestamp} to {stats.last_timestamp}\n"
        f"- Active days: {stats.active_days}"
        f"{att_context}{events_context}\n\n"
        f"## Yearly Summaries\n\n{combined_summaries}\n\n"
        f"## Your Task\n\n"
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


# =============================================================================
# Report Writing
# =============================================================================

def write_markdown_report(
    out_dir: str, 
    contact: str, 
    stats: ConversationStats, 
    monthly: pd.DataFrame, 
    summary_text: str,
    events_df: pd.DataFrame = None
) -> str:
    """Write a Markdown report."""
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"analysis_{sanitize_filename(contact)}.md")
    
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"# Conversation Analysis — {contact}\n\n")
        f.write("## Overview\n")
        f.write(f"- Total messages: {stats.total_messages:,}\n")
        f.write(f"- Sent / Received: {stats.sent_count:,} / {stats.received_count:,}\n")
        f.write(f"- First / Last: {stats.first_timestamp} / {stats.last_timestamp}\n")
        f.write(f"- Active days: {stats.active_days:,} (avg {stats.avg_messages_per_day:.2f} msgs/day)\n")
        f.write(f"- Busiest day: {stats.busiest_day} ({stats.busiest_day_count:,} msgs)\n")
        f.write(f"- Longest gap: {stats.longest_gap_days:.1f} days\n\n")
        
        if stats.attachments.total > 0:
            f.write("## Attachments\n")
            f.write(f"- 📷 Photos: {stats.attachments.photos:,}\n")
            f.write(f"- 🎬 Videos: {stats.attachments.videos:,}\n")
            f.write(f"- 🎤 Audio: {stats.attachments.audio:,}\n\n")
        
        if stats.reactions.total > 0:
            f.write("## Reactions\n")
            f.write(f"- ❤️ Loves: {stats.reactions.loves:,}\n")
            f.write(f"- 👍 Likes: {stats.reactions.likes:,}\n")
            f.write(f"- 😂 Laughs: {stats.reactions.laughs:,}\n\n")

        f.write("## Monthly Progression\n")
        f.write("| Month | Total | Sent | Received | Ratio |\n")
        f.write("|-------|-------|------|----------|-------|\n")
        for _, row in monthly.iterrows():
            f.write(f"| {row['month'].date()} | {row['total']} | {row['sent']} | {row['received']} | {row['sent_ratio']:.0%} |\n")
        f.write("\n")
        
        if events_df is not None and not events_df.empty:
            f.write("## Key Events\n")
            for _, ev in events_df.iterrows():
                f.write(f"### {ev['date']} — {ev['title']}\n")
                f.write(f"{ev['detail']}\n")
                if ev.get('quote'):
                    f.write(f"> \"{ev['quote']}\"\n")
                f.write(f"*Category: {ev['category']}*\n\n")

        f.write("## Summary\n")
        f.write(summary_text + "\n")
    return fname


# =============================================================================
# Main CLI Runner
# =============================================================================

def run_cli(
    messages_csv: str,
    contact: str,
    out_dir: str,
    since: datetime = None,
    until: datetime = None,
    html: bool = False,
    model: str = "gpt-5-mini"
) -> Tuple[Optional[str], str, Optional[str], Optional[str]]:
    """Run single-contact analysis with full context."""
    df = load_messages(messages_csv, since, until)
    conv = filter_conversation(df, contact)
    if conv.empty:
        raise ValueError(f"No messages found for contact '{contact}'")

    attachments_df = load_attachments(messages_csv)
    reactions_df = load_reactions(messages_csv)
    
    stats = compute_stats(conv, attachments_df, reactions_df)
    monthly = progression_series(conv)

    os.makedirs(out_dir, exist_ok=True)
    
    # Precompute chunks once (avoids redundant expensive operations)
    all_messages = format_all_messages(conv)
    total_tokens = estimate_tokens(all_messages)
    
    chunks = None
    if total_tokens > TOKEN_BUDGET:
        chunks = chunk_by_year(conv)
        period_labels = [label for label, _, _ in chunks]
        print(f"📊 Large conversation ({len(conv):,} msgs, ~{total_tokens:,} tokens)")
        print(f"   Will analyze by period: {', '.join(period_labels)}")
    else:
        print(f"📊 Sending {len(conv):,} messages to GPT (~{total_tokens:,} tokens)...")
    
    with tqdm(total=2, desc="Analyzing", unit="step") as pbar:
        # Extract events (pass precomputed data)
        target_events = min(100, max(20, len(conv) // 1000))
        events_df = ai_extract_events(
            contact, stats, conv, target_events, pbar,
            precomputed_chunks=chunks, total_tokens=total_tokens, all_messages=all_messages,
            model=model,
        )
        pbar.update(1)
        
        # Generate summary (pass precomputed data)
        summary_text = ai_summary(
            contact, stats, conv, events_df, pbar,
            precomputed_chunks=chunks, total_tokens=total_tokens, all_messages=all_messages,
            model=model,
        )
        pbar.update(1)
    
    # Write reports
    report_path = write_markdown_report(out_dir, contact, stats, monthly, summary_text, events_df)
    
    events_path = None
    if events_df is not None and not events_df.empty:
        events_path = os.path.join(out_dir, f"events_timeline_{sanitize_filename(contact)}.csv")
        events_df.to_csv(events_path, index=False)
    
    html_path = None
    if html:
        from html_report import write_html_report
        html_path = write_html_report(out_dir, contact, stats, monthly, summary_text, None, events_df)
    
    return None, report_path, events_path, html_path


def main():
    parser = argparse.ArgumentParser(description="Analyze iMessage history with full context AI")
    parser.add_argument("--messages", default="messages.csv", help="Path to messages CSV")
    parser.add_argument("--contact", required=True, help="Contact chat_id")
    parser.add_argument("--out", default="out", help="Output directory")
    parser.add_argument("--since", type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help="Start date")
    parser.add_argument("--until", type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help="End date")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI model (default: gpt-5-mini)")
    args = parser.parse_args()

    _, report_path, events_path, html_path = run_cli(
        args.messages, args.contact, args.out,
        since=args.since, until=args.until, html=args.html, model=args.model,
    )
    print(f"✅ Report: {report_path}")
    if events_path:
        print(f"✅ Events: {events_path}")
    if html_path:
        print(f"✅ HTML: {html_path}")


if __name__ == "__main__":
    main()
