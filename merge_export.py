#!/usr/bin/env python3
"""Merge a frozen chat.db snapshot with the live export.

Messages-in-iCloud "Optimize Storage" offloads older messages off the local
Messages database, so a fresh export from the live DB can be *missing* history
that an earlier snapshot still holds. This unions the two and keeps every
unique message.

Dedup is content-based — (chat_id, timestamp, is_from_me, text) — because the
ROWID/message_id numbering is not comparable across two different databases.

Usage:
  # preview (writes merged copy, never touches messages.csv):
  python merge_export.py --static chat.db --live messages.csv --out messages.merged.csv

  # apply (backs up messages.csv, then replaces it):
  python merge_export.py --static chat.db --live messages.csv --apply
"""
import argparse
import datetime
import pathlib
import shutil

import pandas as pd

from parse_imessage import extract_messages

COLUMNS = [
    "message_id", "timestamp", "sender", "text", "chat_id",
    "chat_display_name", "is_from_me", "service", "attachment_types", "attachment_files",
]
RECOVERED_ID_OFFSET = 1_000_000_000  # keep recovered ids distinct from live ROWIDs


def _key_hash(df: pd.DataFrame) -> pd.Series:
    """Vectorized content hash per row, stable across databases."""
    norm = pd.DataFrame({
        "c": df["chat_id"].astype(str),
        "t": df["timestamp"].astype(str).str.replace("T", " ", regex=False).str.strip(),
        "f": df["is_from_me"].astype(str),
        "x": df["text"].astype(str),
    })
    return pd.util.hash_pandas_object(norm, index=False)


def merge(static_db: str, live_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (merged, recovered). recovered = static rows absent from live."""
    live = pd.read_csv(live_csv, dtype=str, keep_default_na=False)
    static = extract_messages(static_db)
    for frame in (static,):
        for col in COLUMNS:
            if col not in frame.columns:
                frame[col] = ""
        frame[COLUMNS] = frame[COLUMNS].astype(object)

    live_keys = set(_key_hash(live).tolist())
    static_keys = _key_hash(static)
    recovered = static.loc[~static_keys.isin(live_keys), COLUMNS].copy()

    # keep recovered ids from colliding with live ROWIDs
    rid = pd.to_numeric(recovered["message_id"], errors="coerce").fillna(0).astype("int64")
    recovered["message_id"] = (rid + RECOVERED_ID_OFFSET).astype(str)

    merged = pd.concat([live[COLUMNS], recovered], ignore_index=True)
    merged = merged.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return merged, recovered


def _count_for(df: pd.DataFrame, chat_id: str) -> int:
    return int((df["chat_id"].astype(str) == str(chat_id)).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge a chat.db snapshot into the live export")
    ap.add_argument("--static", default="chat.db", help="Frozen snapshot DB (default: chat.db)")
    ap.add_argument("--live", default="messages.csv", help="Live export CSV (default: messages.csv)")
    ap.add_argument("--out", default="messages.merged.csv", help="Preview output path")
    ap.add_argument("--apply", action="store_true", help="Replace --live in place (backs it up first)")
    ap.add_argument("--inspect-chat", default=None, help="Print before/after count for this chat_id")
    args = ap.parse_args()

    live_before = pd.read_csv(args.live, dtype=str, keep_default_na=False)
    merged, recovered = merge(args.static, args.live)

    print(f"live rows     : {len(live_before):,}")
    print(f"recovered     : {len(recovered):,}  (in snapshot, missing from live)")
    print(f"merged rows   : {len(merged):,}")
    if args.inspect_chat:
        b = _count_for(live_before, args.inspect_chat)
        a = _count_for(merged, args.inspect_chat)
        print(f"chat {args.inspect_chat[:4]}*** : {b:,} -> {a:,}  (+{a - b:,})")

    if args.apply:
        src = pathlib.Path(args.live)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = src.with_name(f"{src.stem}.backup-{stamp}{src.suffix}")
        shutil.copy2(src, backup)
        merged.to_csv(src, index=False)
        print(f"applied       : backed up -> {backup.name}; wrote {src.name}")
    else:
        merged.to_csv(args.out, index=False)
        print(f"preview       : wrote {args.out} (live export untouched)")


if __name__ == "__main__":
    main()
