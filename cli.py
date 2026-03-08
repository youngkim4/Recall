#!/usr/bin/env python3
"""
Module: cli

Purpose:
- End-to-end wrapper that extracts messages from chat.db and runs analysis
- List contacts, run full-context AI analysis on conversations

Usage:
  python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db
  python cli.py --list-contacts --db ~/Library/Messages/chat.db
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from parse_imessage import extract_messages, list_contacts, extract_attachments, extract_reactions
from analysis import run_cli


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d")


MODEL_PRICING = {
    # model: (input_per_1M, output_per_1M)
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5": (1.25, 10.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}


def estimate_cost(messages_csv: str, contact: str, model: str = "gpt-5-mini") -> dict:
    """Estimate cost for full-context analysis using exact token counting."""
    from analysis import format_all_messages, estimate_tokens, filter_conversation, load_messages

    df = load_messages(messages_csv)
    contact_df = filter_conversation(df, contact)

    msg_count = len(contact_df)
    if msg_count == 0:
        return {"msg_count": 0, "input_tokens": 0, "output_tokens": 0,
                "estimated_cost": 0, "needs_chunking": False, "years": []}

    all_text = format_all_messages(contact_df)
    input_tokens = estimate_tokens(all_text)

    # Check if chunking needed (170K token limit with buffer for system/output)
    from analysis import TOKEN_BUDGET
    needs_chunking = input_tokens > TOKEN_BUDGET

    # Estimate periods (years, or half-years for large years)
    years = []
    if needs_chunking and not contact_df.empty:
        years = sorted(contact_df["timestamp"].dt.year.dropna().unique())
    num_periods = len(years) if years else 1

    # With chunking: each period gets processed, plus synthesis calls
    if needs_chunking:
        effective_input = input_tokens + (num_periods * 5000)
        output_tokens = num_periods * 3000 + 5000
    else:
        effective_input = input_tokens
        output_tokens = 15000

    input_rate, output_rate = MODEL_PRICING.get(model, (0.25, 2.00))
    estimated_cost = (effective_input / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate

    return {
        "msg_count": msg_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": estimated_cost,
        "needs_chunking": needs_chunking,
        "years": years,
    }


def main():
    parser = argparse.ArgumentParser(
        description="iMessage Memories - Full-context AI analysis of your message history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List your top contacts
  python cli.py --list-contacts --db ~/Library/Messages/chat.db

  # Analyze a conversation (sends ALL messages to GPT for deep analysis)
  python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db

  # Analyze with date range
  python cli.py --contact +12165551234 --since 2024-01-01 --db ~/Library/Messages/chat.db

  # Generate HTML report with charts
  python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db --html
        """
    )
    
    # Source options
    parser.add_argument("--db", help="Path to chat.db")
    parser.add_argument("--messages", help="Path to existing messages.csv")
    parser.add_argument("--out", default="out", help="Output directory (default: out)")
    
    # Mode options
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--contact", help="Contact chat_id (e.g., +12165551234)")
    mode_group.add_argument("--list-contacts", action="store_true", help="List top contacts")
    
    # Filtering options
    parser.add_argument("--since", type=parse_date, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--until", type=parse_date, help="End date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=30, help="Contacts to show (default: 30)")
    
    # Analysis options
    parser.add_argument("--no-confirm", action="store_true", help="Skip cost confirmation")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI model (default: gpt-5-mini)")
    
    args = parser.parse_args()

    # Validate inputs
    if args.list_contacts:
        if not args.db:
            print("Error: --list-contacts requires --db")
            sys.exit(1)
    elif not args.db and not args.messages:
        print("Error: Provide --db or --messages")
        sys.exit(1)

    messages_csv = args.messages

    # Extract from DB if provided
    if args.db:
        if not os.path.exists(args.db):
            print(f"Error: DB not found at {args.db}")
            sys.exit(1)
        
        # List contacts mode
        if args.list_contacts:
            df = list_contacts(args.db, args.limit)
            if df.empty:
                print("No contacts found.")
                sys.exit(0)
            print("\n📱 Top Contacts:\n")
            print("-" * 80)
            for _, row in df.iterrows():
                group_tag = " [GROUP]" if row["is_group"] else ""
                print(f"  {row['chat_id']}{group_tag}")
                print(f"    Messages: {row['message_count']:,}  |  {row['first_msg']} → {row['last_msg']}")
            print("-" * 80)
            print(f"\nUse --contact <chat_id> to analyze.")
            sys.exit(0)
        
        # Export DB to CSV
        messages_csv = messages_csv or os.path.join(os.getcwd(), "messages.csv")
        print(f"📥 Extracting from {args.db}...")
        
        with tqdm(total=3, desc="Extracting", unit="step") as pbar:
            df = extract_messages(args.db)
            pbar.update(1)
            
            if df.empty:
                print("No messages found.")
                sys.exit(1)
            
            attachments_df = extract_attachments(args.db)
            pbar.update(1)
            
            reactions_df = extract_reactions(args.db)
            pbar.update(1)
        
        df.to_csv(messages_csv, index=False)
        print(f"✅ Exported {len(df):,} messages")
        
        if not attachments_df.empty:
            csv_p = Path(messages_csv)
            att_path = csv_p.with_name(csv_p.stem + "_attachments.csv")
            attachments_df.to_csv(str(att_path), index=False)
            print(f"✅ Exported {len(attachments_df):,} attachments")

        if not reactions_df.empty:
            csv_p = Path(messages_csv)
            react_path = csv_p.with_name(csv_p.stem + "_reactions.csv")
            reactions_df.to_csv(str(react_path), index=False)
            print(f"✅ Exported {len(reactions_df):,} reactions")

    # Single-contact analysis (full context)
    if not args.contact:
        print("Error: Provide --contact or --list-contacts")
        sys.exit(1)

    print(f"\n🔍 Analyzing {args.contact}...")
    
    # Estimate cost
    cost_info = estimate_cost(messages_csv, args.contact, model=args.model)
    print(f"\n📊 {cost_info['msg_count']:,} messages → ~{cost_info['input_tokens']:,} tokens")
    if cost_info['needs_chunking']:
        years_str = ", ".join(str(y) for y in cost_info['years'])
        print(f"   📅 Will analyze by year: {years_str}")
    print(f"💰 Estimated cost: ~${cost_info['estimated_cost']:.2f}")
    
    if not args.no_confirm:
        confirm = input("\nProceed? [Y/n] ").strip().lower()
        if confirm and confirm != "y":
            print("Cancelled.")
            sys.exit(0)
    
    print(f"🤖 Model: {args.model}")
    _, report_path, events_path, html_path = run_cli(
        messages_csv,
        args.contact,
        args.out,
        since=args.since,
        until=args.until,
        html=args.html,
        model=args.model,
    )
    
    print("\n" + "=" * 50)
    print("✅ Analysis complete!")
    if events_path:
        print(f"   📅 Events:  {events_path}")
    print(f"   📄 Report:  {report_path}")
    if html_path:
        print(f"   🌐 HTML:    {html_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
