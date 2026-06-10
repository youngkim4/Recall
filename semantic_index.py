"""Local semantic index over conversation windows.

Messages are grouped into small consecutive windows per conversation, embedded
once with OpenAI's text-embedding-3-small (truncated to 512 dims), and stored
on disk as normalized float16 vectors plus light metadata. Retrieval is an
in-memory dot product -- no external services, nothing leaves the machine
except the window text sent to the embeddings API at build time.

The index is keyed to the messages file signature: refreshing the export makes
it stale and the chat pipeline silently falls back to keyword retrieval until
it is rebuilt (a cheap one-time job, ~$0.10-0.30 for a 500k-message archive).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 512
WINDOW_SIZE = 12
BATCH_WINDOWS = 128
MAX_LINE_CHARS = 200
EMBED_PRICE_PER_MTOK = 0.02

VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"


def _outbound(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "1.0"}


def _window_text(rows: pd.DataFrame) -> str:
    lines = []
    for _, row in rows.iterrows():
        text = str(row.get("text") or "").replace("\n", " ").strip()
        if not text:
            continue
        ts = str(row.get("timestamp") or "")[:10]
        who = "Me" if _outbound(row.get("is_from_me")) else "Them"
        lines.append(f"[{ts}] {who}: {text[:MAX_LINE_CHARS]}")
    return "\n".join(lines)


def build_windows(df: pd.DataFrame) -> list[dict]:
    """Consecutive WINDOW_SIZE-message windows per conversation, text included
    transiently for embedding (not persisted)."""
    windows: list[dict] = []
    real = df[df["text"].fillna("").astype(str).str.strip() != ""]
    if real.empty:
        return windows
    for chat_id, convo in real.groupby(real["chat_id"].astype(str), sort=False):
        convo = convo.sort_values("timestamp", kind="stable")
        for start in range(0, len(convo), WINDOW_SIZE):
            rows = convo.iloc[start:start + WINDOW_SIZE]
            text = _window_text(rows)
            if len(text) < 24:
                continue
            anchor = rows.iloc[len(rows) // 2]
            windows.append(
                {
                    "c": str(chat_id),
                    "a": str(anchor.get("message_id") or ""),
                    "t0": str(rows["timestamp"].iloc[0])[:19],
                    "t1": str(rows["timestamp"].iloc[-1])[:19],
                    "text": text,
                }
            )
    return windows


def estimate_build(df: pd.DataFrame) -> dict:
    """Window count + rough token/cost estimate for the status endpoint."""
    text_chars = int(df["text"].fillna("").astype(str).str.len().sum()) if "text" in df.columns else 0
    tokens = text_chars // 4
    return {
        "windows": max(1, int(len(df) / WINDOW_SIZE)),
        "tokens": tokens,
        "estimatedCost": round(tokens / 1_000_000 * EMBED_PRICE_PER_MTOK, 4),
    }


def build_index(
    df: pd.DataFrame,
    signature: tuple | list | None,
    out_dir: Path,
    client=None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Embed the archive and persist vectors + metadata. Returns meta summary."""
    from openai import OpenAI

    client = client or OpenAI()
    say = progress or (lambda _msg: None)

    say("Building conversation windows…")
    windows = build_windows(df)
    if not windows:
        raise ValueError("No text messages to index")
    say(f"{len(windows):,} windows to embed")

    import time

    from openai import APIConnectionError, APITimeoutError, RateLimitError

    vectors = np.zeros((len(windows), EMBED_DIM), dtype=np.float16)
    done = 0
    for start in range(0, len(windows), BATCH_WINDOWS):
        batch = windows[start:start + BATCH_WINDOWS]
        response = None
        for attempt in range(10):
            try:
                response = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=[w["text"] for w in batch],
                    dimensions=EMBED_DIM,
                )
                break
            except RateLimitError:
                # the TPM window resets within a minute; wait it out
                wait = min(45, 6 * (attempt + 1))
                say(f"Rate limited at {done:,}/{len(windows):,}; waiting {wait}s…")
                time.sleep(wait)
            except (APIConnectionError, APITimeoutError):
                time.sleep(4)
        if response is None:
            raise RuntimeError(f"Embedding batch failed after retries at window {start}")
        for offset, item in enumerate(response.data):
            vec = np.asarray(item.embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vec)) or 1.0
            vectors[start + offset] = (vec / norm).astype(np.float16)
        done += len(batch)
        if done % (BATCH_WINDOWS * 8) == 0 or done == len(windows):
            say(f"Embedded {done:,}/{len(windows):,} windows")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / VECTORS_FILE, vectors)
    meta = {
        "signature": list(signature) if signature else None,
        "model": EMBED_MODEL,
        "dim": EMBED_DIM,
        "builtAt": pd.Timestamp.now().isoformat(timespec="seconds"),
        "windows": [{k: w[k] for k in ("c", "a", "t0", "t1")} for w in windows],
    }
    (out_dir / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    say(f"Index saved: {len(windows):,} windows, {vectors.nbytes / 1e6:.1f} MB")
    return {"windows": len(windows), "sizeMB": round(vectors.nbytes / 1e6, 1), "builtAt": meta["builtAt"]}


class SemanticIndex:
    def __init__(self, vectors: np.ndarray, meta: dict):
        self.vectors = vectors
        self.meta = meta
        windows = meta.get("windows", [])
        self.chat_ids = np.array([w["c"] for w in windows])
        self.anchors = [w["a"] for w in windows]
        self.t0 = np.array([w["t0"] for w in windows])
        self.t1 = np.array([w["t1"] for w in windows])

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 10,
        chat_ids: Optional[set[str]] = None,
        date_from: str = "",
        date_to: str = "",
    ) -> list[dict]:
        mask = np.ones(len(self.anchors), dtype=bool)
        if chat_ids:
            mask &= np.isin(self.chat_ids, list(chat_ids))
        if date_from:
            mask &= self.t1 >= date_from
        if date_to:
            mask &= self.t0 <= date_to + "T23:59:59"
        candidates = np.where(mask)[0]
        if candidates.size == 0:
            return []
        scores = self.vectors[candidates].astype(np.float32) @ query_vec
        order = np.argsort(scores)[::-1][:k]
        return [
            {
                "chat_id": str(self.chat_ids[candidates[i]]),
                "anchor_id": self.anchors[candidates[i]],
                "score": float(scores[i]),
            }
            for i in order
        ]


def load_index(out_dir: Path, signature: tuple | list | None) -> Optional[SemanticIndex]:
    """Load the index when present AND built from the current messages file."""
    vec_path = out_dir / VECTORS_FILE
    meta_path = out_dir / META_FILE
    if not vec_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if signature is not None and meta.get("signature") != list(signature):
            return None
        vectors = np.load(vec_path)
        if vectors.shape[0] != len(meta.get("windows", [])):
            return None
        return SemanticIndex(vectors, meta)
    except Exception:
        return None


def index_status(out_dir: Path, signature: tuple | list | None) -> dict:
    meta_path = out_dir / META_FILE
    if not meta_path.exists():
        return {"state": "none"}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {"state": "none"}
    fresh = signature is not None and meta.get("signature") == list(signature)
    return {
        "state": "fresh" if fresh else "stale",
        "windows": len(meta.get("windows", [])),
        "builtAt": meta.get("builtAt"),
        "model": meta.get("model"),
    }


def embed_query(text: str, client=None) -> np.ndarray:
    from openai import OpenAI

    client = client or OpenAI()
    response = client.embeddings.create(model=EMBED_MODEL, input=[text[:2000]], dimensions=EMBED_DIM)
    vec = np.asarray(response.data[0].embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec)) or 1.0
    return vec / norm
