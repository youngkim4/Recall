# Recall

Extract iMessage history and generate AI relationship analysis with visual reports.

## Features

- List top contacts by message volume and date range
- Full-context AI analysis via GPT
- Key event extraction with quotes
- Attachment and reaction breakdowns
- Monthly sent/received progression
- Reports with Chart.js charts
- Auto-chunking for conversations exceeding 170K tokens

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key-here" > .env # Need your own OpenAI API key
```

## Usage

```bash
# List top contacts
python cli.py --list-contacts --db ~/Library/Messages/chat.db

# Analyze a contact
python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db

# With HTML report
python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db --html

# Date range filter
python cli.py --contact +12165551234 --since 2024-01-01 --until 2024-12-31

# Skip cost confirmation
python cli.py --contact +12165551234 --no-confirm
```

| Flag | Description |
|------|-------------|
| `--db` | Path to chat.db |
| `--messages` | Path to pre-exported messages.csv |
| `--contact` | Contact chat_id to analyze |
| `--list-contacts` | List top contacts by message count |
| `--since` / `--until` | Date range filter (YYYY-MM-DD) |
| `--html` | Generate HTML report with charts |
| `--model` | OpenAI model (default: gpt-5-mini) |
| `--no-confirm` | Skip cost estimate confirmation |
| `--out` | Output directory (default: out) |
| `--limit` | Contacts to show (default: 30) |

## Outputs

- `out/analysis_<contact>.md` -- Markdown report
- `out/analysis_<contact>.html` -- HTML report (with `--html`)
- `out/events_timeline_<contact>.csv` -- Key events timeline
- `messages.csv`, `messages_attachments.csv`, `messages_reactions.csv` -- Extracted data

## Notes

- `chat_id` is usually the phone number for 1:1 chats, or starts with `chat` for group chats. Use `--list-contacts` to discover IDs.
- If the DB is locked, close Messages or copy it first: `cp ~/Library/Messages/chat.db ./chat.db`
- Full conversation history is sent to the OpenAI API -- cost estimates are shown before any API call.

## License

MIT
