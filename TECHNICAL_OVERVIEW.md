# Technical Overview

This project reads your Mac Messages database, extracts conversations, and uses the OpenAI Responses API with GPT-5.5 by default to generate deep narrative summaries and extract key life events using full conversation context.

## Requirements

- macOS with Messages database at `~/Library/Messages/chat.db`
- Python 3.10+
- OpenAI API key with GPT-5.5 access (`OPENAI_API_KEY` environment variable)
- `pip install -r requirements.txt`

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│ parse_imessage  │────▶│  messages.csv │────▶│   analysis.py  │
│   (SQLite)      │     │  + attachments│     │   (GPT-5.5)    │
│                 │     │  + reactions  │     │  (Responses)   │
└─────────────────┘     └──────────────┘     └────────────────┘
                                                     │
                              ┌──────────────────────┼──────────────────────┐
                              ▼                      ▼                      ▼
                        events.csv            report.md              report.html
                              ▲                      ▲                      ▲
                              └──────────── ui_server.py / ui ──────────────┘
```

### Core Modules

| File | Purpose |
|------|---------|
| `cli.py` | End-to-end CLI: extraction + analysis |
| `parse_imessage.py` | Extracts messages, attachments, reactions from `chat.db` |
| `analysis.py` | Compatibility facade and single-contact runner |
| `conversation.py` | CSV loading, normalization, statistics, token counting, chunking |
| `ai_config.py` | Model defaults, context budgets, structured-output schema |
| `openai_client.py` | Responses API wrapper, `store=false`, retry behavior |
| `ai_analysis.py` | Event extraction, chunked summarization, final synthesis |
| `reports.py` | Markdown report generation |
| `html_report.py` | Generates HTML reports with Chart.js visualizations |
| `ui_server.py` | Local web server, JSON APIs, background export/analysis jobs |
| `local_store.py` | Local SQLite store for analysis and preview cache state |
| `app/` | React + TypeScript + Vite frontend shell served by the local `ui_server.py` API when built |
| `ui/` | Dependency-free browser UI retained as a fallback surface |

### Current App Surface

- **Analyze**: conversation search, selected-thread preview, model selector, cost estimate, recent messages, monthly activity, local report launch.
- **Reports**: saved report list plus sectioned reader for Story, Events, Patterns, Media, Activity, and generated Files.
- **Settings**: local setup status, model preference, contact-name refresh, message export refresh, and preview-cache clearing.
- **Caching**: preview payloads are cached in both the browser and the local SQLite store. Cache keys include the message export and companion attachment/reaction export file signatures, so changed exports invalidate old previews.

### Product Architecture Direction

Recall should be a local-first desktop app, not a website that asks users to upload private message archives. The desktop app owns message extraction, Contacts permission, local report generation, and local cache. Cloud services should only handle account, license, billing, update, and optional sync metadata unless a user explicitly opts in to cloud storage.

### Product Stack Decision

The app should move in phases instead of jumping straight to a full SaaS rewrite.

| Layer | Current | Product target | Reason |
|---|---|---|---|
| Desktop shell | Swift + WKWebView wrapper | Tauri v2 + React/TypeScript UI | Keeps the app local-first, ships a small native desktop app, and gives us a standard web UI stack inside the shell. |
| Local engine | Python local HTTP server | Keep Python initially; later package as a Tauri sidecar or migrate hot paths to Rust | Existing extraction/analysis code works and has tests. Do not rewrite the engine before the product workflow is stable. |
| Local persistence | `saves/recall_store.sqlite3` | SQLite with explicit migrations | Fits local cache, report index, job history, preferences, and contact-name cache without requiring cloud. |
| Desktop UI | React + TypeScript + Vite build, with legacy static fallback | React + TypeScript + Vite | React gives cleaner state, reusable components, keyboard UX, and testability while the old UI remains a safe fallback. |
| Cloud control plane | None | Next.js App Router | Good fit for account pages, billing, downloads, release notes, and marketing without touching local message data. |
| Cloud database | None | Supabase/Postgres | Store users, licenses, device activations, subscription state, and optional encrypted sync metadata. |
| Payments | None | Stripe Checkout + Billing | Use hosted checkout for subscriptions first; keep payment logic out of the desktop app. |

Supporting references: [Tauri v2](https://v2.tauri.app/), [Next.js App Router](https://nextjs.org/docs/app), [Supabase local development](https://supabase.com/docs/guides/local-development), and [Stripe Checkout subscriptions](https://docs.stripe.com/payments/checkout/build-subscriptions).

### Build Plan

1. **Stabilize the current desktop product**
   - Keep the Swift/WKWebView wrapper until the core UX is no longer changing every session.
   - Continue using the Python local server as the engine boundary.
   - Store local app state in SQLite: report cache, report index, preferences, job history, and contact-name metadata.

2. **Introduce a real frontend stack**
   - Continue the Vite + React + TypeScript UI in `app/` that talks to the same local API.
   - Build reusable primitives for app shell, nav, conversation list, report reader, settings, buttons, tabs, segmented controls, and charts.
   - Add UI tests for hover/selected states, report loading, cache hits, and long conversation lists.

3. **Package as a proper desktop app**
   - Move from the temporary Swift wrapper to Tauri v2 once the local API shape is stable.
   - Bundle the Python engine as a sidecar first to avoid a risky rewrite.
   - Add app signing, notarization, auto-update, crash/error reporting, and local permission onboarding.

4. **Add cloud only where it makes product sense**
   - Build a Next.js account/dashboard app for login, downloads, billing, license management, and docs.
   - Use Supabase/Postgres for account and license state, not raw message content.
   - Use Stripe Checkout/Billing for subscription purchase and renewal.
   - Sync only encrypted report metadata or user-approved artifacts if we decide cloud sync is worth the privacy tradeoff.

### Product Boundaries

- **Do not build this as web upload-first.** iMessage history, Contacts, and local files make this a desktop product.
- **Do not rewrite Python analysis yet.** The current engine is covered by tests and already handles the hard data work.
- **Do not add cloud storage by default.** Monetization should start with licensing/subscriptions, not hosting private archives.
- **Do make the UI stack professional.** The current static UI is a prototype surface; React/TypeScript is the next real app layer.

## How It Works

### 1. Extraction (`parse_imessage.py`)

Queries SQLite `chat.db` and exports:
- `messages.csv` — timestamp, text, chat_id, is_from_me, service
- `messages_attachments.csv` — file types, sizes, categories
- `messages_reactions.csv` — loves, likes, laughs, etc.

Handles Apple's `attributedBody` blob for newer macOS versions where `text` column is empty.

### 2. Analysis (`analysis.py` facade, `conversation.py`, `ai_analysis.py`)

**Full Context**: Every message is included in analysis. For GPT-5.5 and GPT-5.4, the runner uses a long-context budget before chunking; smaller or older models chunk earlier.

**Time-Based Chunking**: For conversations that exceed the selected model budget, the system automatically splits by time:
```
Year → Half (H1/H2) → Quarter → Month → Week
```
Each chunk stays under the token limit while preserving all messages.

**Cross-Period Context**: When analyzing in chunks, each period receives:
- Summary of key events from previous periods
- Ongoing themes and unresolved topics
- Specific names, dates, quotes for continuity

**Event Selection**: Uses Structured Outputs for event JSON and then guarantees chronological coverage rather than just picking highest-scored events globally.

**Privacy**: Responses API calls set `store=false` so conversation content is not retained for response state.

### 3. Output

| File | Contents |
|------|----------|
| `analysis_<contact>.md` | Stats, monthly progression, AI narrative |
| `events_timeline_<contact>.csv` | Key events with dates, titles, quotes, scores |
| `analysis_<contact>.html` | Interactive charts (message volume, sent ratio over time) |

## Commands

```bash
# Start the local UI
venv/bin/python ui_server.py

# Start the React frontend prototype
cd app && /opt/homebrew/Cellar/node/25.4.0/bin/npm run dev

# List top contacts
python cli.py --list-contacts --db ~/Library/Messages/chat.db

# Analyze a conversation (full context)
python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db

# With date range
python cli.py --contact +12165551234 --since 2024-01-01 --until 2024-12-31 --db ~/Library/Messages/chat.db

# Generate HTML report
python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db --html

# Skip cost confirmation
python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db --no-confirm

# Use existing CSV (skip extraction)
python analysis.py --contact +12165551234 --messages messages.csv --out out
```

## Cost Estimation

Before analysis, the CLI shows:
- Message count and token estimate
- Whether chunking is needed (and which periods)
- Estimated cost using the selected model's current pricing table

Example for ~90K messages:
```
📊 89,435 messages → ~1,050,000 tokens
   📅 Will analyze by year: 2024 H1, Jul 2024, Aug 2024, Sep 2024, 2024 Q4
💰 Estimated cost: ~$2.15
```

## Common Issues

| Problem | Solution |
|---------|----------|
| `chat.db` locked | Quit Messages app or copy DB to another location |
| Missing messages | Newer macOS stores text in `attributedBody` blob—handled automatically |
| Rate limits | Large conversations are chunked to stay under model/API limits |
| No API key | Set `OPENAI_API_KEY` in your environment |

## Stats Tracked

- Total messages, sent/received counts
- Active days, average messages per day
- Busiest day, longest communication gap
- Attachments by type (photos, videos, audio, documents)
- Reactions (loves, likes, laughs, emphasis)
- Monthly progression with sent/received ratios
