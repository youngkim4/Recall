#!/usr/bin/env python3
"""
Module: analysis

Purpose:
- Backward-compatible facade for conversation analysis
- Orchestrates loading, stats, AI analysis, and report generation
"""

import argparse
import logging
import os
from datetime import datetime
from typing import Optional, Tuple

from tqdm import tqdm

from ai_analysis import (
    _extract_events_for_period,
    _parse_json_events,
    _summarize_period,
    ai_extract_events,
    ai_summary,
)
from ai_config import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_VERBOSITY,
    EVENTS_TEXT_FORMAT,
    LEGACY_TOKEN_BUDGET,
    MAX_CONTEXT_TOKENS_EVENTS,
    MAX_CONTEXT_TOKENS_SUMMARY,
    MODEL_CONTEXT_WINDOWS,
    OUTPUT_TOKEN_RESERVE,
    PROMPT_TOKEN_RESERVE,
    SECONDS_PER_DAY,
    TOKEN_BUDGET,
    get_token_budget,
)
from conversation import (
    AttachmentStats,
    ConversationStats,
    ReactionStats,
    chunk_by_year,
    compute_attachment_stats,
    compute_reaction_stats,
    compute_stats,
    contact_label,
    conversation_label,
    count_sent,
    direction_label,
    estimate_tokens,
    filter_conversation,
    format_all_messages,
    given_name,
    load_attachments,
    load_messages,
    load_reactions,
    parse_boolish,
    progression_series,
    sanitize_filename,
    truncate_to_tokens,
)
from openai_client import (
    _call_openai,
    _prepare_responses_input,
    response_output_text,
    supports_reasoning_controls,
)
from reports import write_markdown_report

logger = logging.getLogger(__name__)


def write_events_timeline(out_dir: str, contact: str, events_df) -> Optional[str]:
    """Write the events a run found beside its report, and return the path.
    A run that found none leaves no file: one left by an earlier report on the
    same conversation would be shown beside the new report as its own."""
    events_path = os.path.join(out_dir, f"events_timeline_{sanitize_filename(contact)}.csv")
    if events_df is None or events_df.empty:
        try:
            os.remove(events_path)
        except FileNotFoundError:
            pass
        return None
    events_df.to_csv(events_path, index=False)
    return events_path


def run_cli(
    messages_csv: str,
    contact: str,
    out_dir: str,
    since: datetime = None,
    until: datetime = None,
    html: bool = False,
    model: str = DEFAULT_MODEL,
) -> Tuple[Optional[str], str, Optional[str], Optional[str]]:
    """Run single-contact analysis with full context."""
    df = load_messages(messages_csv, since, until)
    conv = filter_conversation(df, contact)
    if conv.empty:
        raise ValueError(f"No messages found for contact '{contact}'")

    # the same days as the messages, so the report's counts are for its dates
    attachments_df = load_attachments(messages_csv, since, until)
    reactions_df = load_reactions(messages_csv, since, until)

    stats = compute_stats(conv, attachments_df, reactions_df)
    monthly = progression_series(conv)

    os.makedirs(out_dir, exist_ok=True)

    # prompts get saved names, never raw handles; file paths keep using the raw
    # id so report locations stay stable. Senders resolve too, so group-chat
    # transcripts attribute each line to its actual speaker.
    sender_names: dict = {}
    try:
        from contact_names import resolve_contact_names

        senders = (
            conv["sender"].dropna().astype(str).unique().tolist()
            if "sender" in conv.columns
            else []
        )
        sender_names = dict(resolve_contact_names(sorted(set(senders + [contact]))))
    except Exception as exc:
        logger.warning("Contact names could not be read: %s", exc)
    # the name the conversation goes by: saved in Contacts, or given to a group
    known_name = sender_names.get(contact, "") or given_name(conv)
    # The model is told that name. With no name it is told a label that holds
    # no whole phone number and no address.
    contact_display = contact_label({"displayName": known_name, "chatId": contact})

    all_messages = format_all_messages(conv, sender_names)
    total_tokens = estimate_tokens(all_messages)

    chunks = None
    token_budget = get_token_budget(model)
    if total_tokens > token_budget:
        chunks = chunk_by_year(conv, max_tokens=token_budget, sender_names=sender_names)
        period_labels = [label for label, _, _ in chunks]
        print(f"📊 Large conversation ({len(conv):,} msgs, ~{total_tokens:,} tokens)")
        print(f"   Will analyze by period: {', '.join(period_labels)}")
    else:
        print(f"📊 Sending {len(conv):,} messages to GPT (~{total_tokens:,} tokens)...")

    with tqdm(total=2, desc="Analyzing", unit="step") as pbar:
        target_events = min(100, max(20, len(conv) // 1000))
        events_df = ai_extract_events(
            contact_display, stats, conv, target_events, pbar,
            precomputed_chunks=chunks, total_tokens=total_tokens, all_messages=all_messages,
            model=model,
        )
        pbar.update(1)

        summary_text = ai_summary(
            contact_display, stats, conv, events_df, pbar,
            precomputed_chunks=chunks, total_tokens=total_tokens, all_messages=all_messages,
            model=model,
        )
        pbar.update(1)

    report_path = write_markdown_report(
        out_dir, contact, stats, monthly, summary_text, events_df, since=since, until=until,
    )
    events_path = write_events_timeline(out_dir, contact, events_df)

    html_path = None
    if html:
        from html_report import write_html_report
        html_path = write_html_report(
            out_dir, contact, stats, monthly, summary_text, None, events_df,
            title=conversation_label(known_name, contact),
        )

    return None, report_path, events_path, html_path


def main():
    parser = argparse.ArgumentParser(description="Analyze iMessage history with full context AI")
    parser.add_argument("--messages", default="messages.csv", help="Path to messages CSV")
    parser.add_argument("--contact", required=True, help="Contact chat_id")
    parser.add_argument("--out", default="out", help="Output directory")
    parser.add_argument("--since", type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help="Start date")
    parser.add_argument("--until", type=lambda s: datetime.strptime(s, "%Y-%m-%d"), help="End date")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
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
