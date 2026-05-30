"""Markdown report generation."""

import os

import pandas as pd

from conversation import ConversationStats, sanitize_filename


def write_markdown_report(
    out_dir: str,
    contact: str,
    stats: ConversationStats,
    monthly: pd.DataFrame,
    summary_text: str,
    events_df: pd.DataFrame = None,
) -> str:
    """Write a Markdown report."""
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"analysis_{sanitize_filename(contact)}.md")

    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"# Conversation Analysis - {contact}\n\n")
        f.write("## Overview\n")
        f.write(f"- Total messages: {stats.total_messages:,}\n")
        f.write(f"- Sent / Received: {stats.sent_count:,} / {stats.received_count:,}\n")
        f.write(f"- First / Last: {stats.first_timestamp} / {stats.last_timestamp}\n")
        f.write(f"- Active days: {stats.active_days:,} (avg {stats.avg_messages_per_day:.2f} msgs/day)\n")
        f.write(f"- Busiest day: {stats.busiest_day} ({stats.busiest_day_count:,} msgs)\n")
        f.write(f"- Longest gap: {stats.longest_gap_days:.1f} days\n\n")

        if stats.attachments.total > 0:
            f.write("## Attachments\n")
            f.write(f"- Photos: {stats.attachments.photos:,}\n")
            f.write(f"- Videos: {stats.attachments.videos:,}\n")
            f.write(f"- Audio: {stats.attachments.audio:,}\n\n")

        if stats.reactions.total > 0:
            f.write("## Reactions\n")
            f.write(f"- Loves: {stats.reactions.loves:,}\n")
            f.write(f"- Likes: {stats.reactions.likes:,}\n")
            f.write(f"- Laughs: {stats.reactions.laughs:,}\n\n")

        f.write("## Monthly Progression\n")
        f.write("| Month | Total | Sent | Received | Ratio |\n")
        f.write("|-------|-------|------|----------|-------|\n")
        for _, row in monthly.iterrows():
            f.write(f"| {row['month'].date()} | {row['total']} | {row['sent']} | {row['received']} | {row['sent_ratio']:.0%} |\n")
        f.write("\n")

        if events_df is not None and not events_df.empty:
            f.write("## Key Events\n")
            for _, ev in events_df.iterrows():
                f.write(f"### {ev['date']} - {ev['title']}\n")
                f.write(f"{ev['detail']}\n")
                quote = ev.get("quote")
                if quote is not None and pd.notna(quote) and str(quote) != "":
                    f.write(f"> \"{ev['quote']}\"\n")
                f.write(f"*Category: {ev['category']}*\n\n")

        f.write("## Summary\n")
        f.write(summary_text + "\n")
    return fname
