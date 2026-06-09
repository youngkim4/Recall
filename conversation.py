"""Conversation loading, normalization, statistics, formatting, and chunking."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import tiktoken

from ai_config import SECONDS_PER_DAY, TOKEN_BUDGET


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


def sanitize_filename(value: str) -> str:
    safe = "".join(c for c in value if c.isalnum() or c in ("-", "_", "+"))
    return safe or "conversation"


def parse_boolish(value) -> Optional[int]:
    """Normalize iMessage direction values from CSV/SQLite into 1, 0, or NA."""
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value == 1)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return 1
    if text in {"0", "false", "f", "no", "n"}:
        return 0
    return pd.NA


def count_sent(series: pd.Series) -> int:
    return int(series.map(parse_boolish).dropna().astype(int).sum())


def direction_label(value) -> str:
    normalized = parse_boolish(value)
    if pd.isna(normalized):
        return "??"
    return "ME" if int(normalized) == 1 else "THEM"


def load_messages(csv_path: str, since: datetime = None, until: datetime = None) -> pd.DataFrame:
    """Load messages CSV with optional date filtering."""
    df = pd.read_csv(csv_path, dtype={"chat_id": "string", "sender": "string"})
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
        stats.sent_attachments = count_sent(filtered["is_from_me"])
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
        stats.sent_reactions = count_sent(filtered["is_from_me"])
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
    normalized_direction = direction_col.map(parse_boolish)
    known_direction = normalized_direction.dropna().astype(int)
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
        sent=("is_from_me", count_sent),
    )
    monthly["received"] = monthly["total"] - monthly["sent"]
    monthly["sent_ratio"] = monthly.apply(lambda r: (r["sent"] / r["total"]) if r["total"] else 0.0, axis=1)
    return monthly.sort_index().reset_index()


def _speaker_labels(conv: pd.DataFrame, sender_names: Optional[Dict[str, str]] = None) -> pd.Series:
    """Per-row speaker labels: ME for the owner; everyone else by resolved name.
    In multi-party (group) chats an unresolved sender gets a masked per-person
    label so distinct people never collapse into one 'THEM'."""
    is_from_me = conv["is_from_me"] if "is_from_me" in conv.columns else pd.Series([pd.NA] * len(conv), index=conv.index)
    direction = is_from_me.apply(direction_label)
    senders = (
        conv["sender"].fillna("").astype(str)
        if "sender" in conv.columns
        else pd.Series([""] * len(conv), index=conv.index)
    )
    multi_party = senders[senders != ""].nunique() > 1

    def label(index) -> str:
        if direction.loc[index] == "ME":
            return "ME"
        sender = senders.loc[index]
        if sender_names and sender:
            name = str(sender_names.get(sender) or "").strip()
            if name:
                return name
        if multi_party and sender:
            digits = re.sub(r"\D", "", sender)
            return f"Unsaved (…{digits[-4:]})" if digits else "Unsaved"
        return direction.loc[index]

    return pd.Series([label(i) for i in conv.index], index=conv.index)


def _formatted_lines(conv: pd.DataFrame, sender_names: Optional[Dict[str, str]] = None) -> pd.Series:
    ts = conv["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    who = _speaker_labels(conv, sender_names)
    text = conv["text"].astype(str)
    return "[" + ts + "] " + who + ": " + text


def format_all_messages(conv: pd.DataFrame, sender_names: Optional[Dict[str, str]] = None) -> str:
    """Format all messages for GPT context."""
    return "\n".join(_formatted_lines(conv, sender_names).tolist())


def _load_tokenizer():
    try:
        # gpt-5.x / gpt-4o tokenizer; cl100k overcounts these models ~10-20%
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


_tokenizer = _load_tokenizer()


def estimate_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    return len(_tokenizer.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to last max_tokens tokens to preserve recent context."""
    if not text or max_tokens <= 0:
        return ""
    tokens = _tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _tokenizer.decode(tokens[-max_tokens:])


def chunk_by_year(
    conv: pd.DataFrame,
    max_tokens: int = TOKEN_BUDGET,
    sender_names: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, pd.DataFrame, str]]:
    """
    Split conversation into chronological chunks that fit the token budget.
    Years are the base unit (split Year -> Half -> Quarter -> Month -> Weeks ->
    Days when one year alone exceeds the budget), then consecutive small
    periods are PACKED together up to the budget -- a 12-year history becomes a
    few large chunks instead of 12 tiny calls, keeping cross-year arcs intact.
    Returns list of (period_label, dataframe, formatted_text) tuples.
    All messages are included - nothing is dropped.
    """
    conv = conv.copy()
    conv["formatted"] = _formatted_lines(conv, sender_names)
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
                    half_df["quarter"] = half_df["month"].apply(lambda m: (m - 1) // 3 + 1)
                    for q in sorted(half_df["quarter"].unique()):
                        q_df = half_df[half_df["quarter"] == q].copy()
                        if q_df.empty:
                            continue
                        add_quarter_chunk(f"{year} Q{q}", q_df)

    return _pack_chunks(chunks, max_tokens)


def _pack_chunks(
    chunks: List[Tuple[str, pd.DataFrame, str]],
    max_tokens: int,
) -> List[Tuple[str, pd.DataFrame, str]]:
    """Greedily merge consecutive chunks up to the budget so the model sees
    multi-year arcs in one pass instead of fragmented per-year calls."""
    packed: List[List] = []
    for label, df, text in chunks:
        tokens = estimate_tokens(text)
        if packed and packed[-1][3] + tokens + 1 <= max_tokens:
            first_label = packed[-1][0].split(" – ")[0]
            packed[-1] = [
                f"{first_label} – {label}",
                pd.concat([packed[-1][1], df]),
                packed[-1][2] + "\n" + text,
                packed[-1][3] + tokens + 1,
            ]
        else:
            packed.append([label, df, text, tokens])
    return [(label, df, text) for label, df, text, _ in packed]
