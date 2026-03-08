# imessage-memories

Extract your iMessage history and generate AI-driven relationship analysis with visual reports.

## Features

- **Contact discovery** — list top contacts by message count, first/last dates, group vs 1:1
- **Full-context AI analysis** — sends your entire conversation to GPT for deep relationship insights
- **Key events extraction** — milestones, turning points, and memorable moments with verbatim quotes
- **Attachment stats** — photos, videos, audio, GIFs, documents breakdown
- **Reaction stats** — loves, likes, laughs, emphasis, questions
- **Monthly progression** — sent/received volume and ratio over time
- **HTML reports** — dark-themed visual reports with interactive Chart.js charts
- **Smart chunking** — conversations exceeding 170K tokens are split by year/half/quarter/month automatically

## Requirements

- macOS with Messages data available
- Python 3.10+
- Full Disk Access granted to Terminal/IDE to read `~/Library/Messages/chat.db`
- OpenAI API key (`OPENAI_API_KEY` in `.env` or environment)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
# List your top contacts
python cli.py --list-contacts --db ~/Library/Messages/chat.db

# Analyze a contact
python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db

# With HTML report
python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db --html

# Date range filter
python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db --since 2024-01-01 --until 2024-12-31

# Skip cost confirmation
python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db --no-confirm

# Use a different model
python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db --model gpt-5
```

## Outputs

- `out/analysis_<contact>.md` — Markdown report with stats, events, and narrative summary
- `out/analysis_<contact>.html` — Visual HTML report with charts (with `--html`)
- `out/events_timeline_<contact>.csv` — Key events timeline
- `messages.csv` — Extracted messages
- `messages_attachments.csv` — Attachment metadata
- `messages_reactions.csv` — Tapback reactions

## CLI Options

| Flag | Description |
|------|-------------|
| `--db` | Path to chat.db |
| `--messages` | Path to pre-exported messages.csv |
| `--contact` | Contact chat_id to analyze |
| `--list-contacts` | List top contacts by message count |
| `--since` | Start date filter (YYYY-MM-DD) |
| `--until` | End date filter (YYYY-MM-DD) |
| `--html` | Generate HTML report with charts |
| `--model` | OpenAI model (default: gpt-5-mini) |
| `--no-confirm` | Skip cost estimate confirmation |
| `--out` | Output directory (default: out) |
| `--limit` | Number of contacts to show (default: 30) |

## Notes on `chat_id`

For 1:1 chats, `chat_id` is usually the phone number (e.g., `+12165551234`). For group chats, it starts with `chat` followed by numbers. Use `--list-contacts` to discover available IDs.

## Troubleshooting

- **DB locked**: Close Messages app or copy the DB first:
  ```bash
  cp ~/Library/Messages/chat.db ./chat.db
  python cli.py --contact +12165551234 --db ./chat.db
  ```
- **Timestamp issues**: Assumes nanosecond timestamps (standard on modern macOS). Adjust the divisor in `parse_imessage.py` if your DB differs.

## Data Privacy

- All extraction and processing runs locally on your machine
- Your full conversation history is sent to the OpenAI API for analysis — cost estimates are shown before any API call
- Outputs contain sensitive message content; handle accordingly

## License

MIT
