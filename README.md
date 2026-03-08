# imessage-memories

iMessage analysis with AI driven insights: extract your message history, discover contacts, compare relationships, analyze group chats, and generate beautiful visual reports.

## Features

### 📱 Contact Discovery
- List your top contacts by message count
- See first/last message dates at a glance
- Identify group chats vs 1:1 conversations

### 💬 Single Contact Analysis
- Message statistics: total, sent/received, active days, busiest day
- Attachment breakdown: photos, videos, audio, GIFs, documents
- Reaction patterns: loves, likes, laughs, emphasis, questions
- Monthly progression with sent/received ratio
- AI-generated sentiment timeline showing emotional tone over time
- Key events extraction highlighting milestones and memorable moments

### 📊 Comparison Mode
- Compare multiple contacts side by side
- Identify who you message most, engagement patterns
- AI insights on relationship dynamics

### 👥 Group Chat Analysis
- Participant breakdown with message share
- Activity visualization per member
- AI analysis of group dynamics and roles

### 🎨 HTML Reports
- Beautiful dark-themed visual reports
- Interactive charts for message volume over time
- Sentiment visualization
- Shareable single-file output

## Requirements

- macOS with Messages data available
- Python 3.10+
- Full Disk Access granted to Terminal/IDE to read `~/Library/Messages/chat.db`
- OpenAI `OPENAI_API_KEY` with GPT-5.2 access
- Install Python deps:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Discover Your Contacts

```bash
python cli.py --list-contacts --db ~/Library/Messages/chat.db
```

### Analyze a Single Contact

```bash
# Basic analysis
python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db

# With HTML report
python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db --html

# Analyze specific date range
python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db --since 2024-01-01 --until 2024-12-31
```

### Compare Multiple Contacts

```bash
python cli.py --compare +14155551234 +12165559999 +18005551212 --db ~/Library/Messages/chat.db
```

### Analyze a Group Chat

```bash
python cli.py --group chat123456789 --db ~/Library/Messages/chat.db --html
```

### Skip Cost Confirmation

```bash
python cli.py --contact +12165551234 --db ~/Library/Messages/chat.db --no-confirm
```

## Outputs

- `out/analysis_<contact>.md` — Markdown report with stats, progression, sentiment, and summary
- `out/analysis_<contact>.html` — Visual HTML report with charts (when `--html` is used)
- `out/events_timeline_<contact>.csv` — Key events extracted by GPT-5.2
- `messages.csv` — Extracted messages
- `messages_attachments.csv` — Attachment metadata
- `messages_reactions.csv` — Tapback reactions

## CLI Options

| Flag | Description |
|------|-------------|
| `--db` | Path to chat.db |
| `--messages` | Path to pre-exported messages.csv |
| `--contact` | Single contact chat_id to analyze |
| `--list-contacts` | List top contacts by message count |
| `--compare` | Compare multiple contacts |
| `--group` | Analyze a group chat |
| `--since` | Start date filter (YYYY-MM-DD) |
| `--until` | End date filter (YYYY-MM-DD) |
| `--html` | Generate HTML report with charts |
| `--no-confirm` | Skip cost estimate confirmation |
| `--events` | Override number of key events to extract |
| `--out` | Output directory (default: out) |
| `--limit` | Number of contacts to show (default: 30) |

## Notes on `chat_id`

For 1:1 chats, `chat_id` is usually the other party's phone number (e.g., `+12165551234`). For group chats, it starts with `chat` followed by numbers. Use `--list-contacts` to discover available chat IDs.

## Troubleshooting

- **DB locked**: Close the Messages app and try again, or copy the DB first:

```bash
cp ~/Library/Messages/chat.db ./chat.db
python cli.py --contact +12165551234 --db ./chat.db
```

- **Timestamp issues**: This project assumes nanosecond-based timestamps. Adjust the divisor in `parse_imessage.py` if needed.

## Data Privacy & Security

- All extraction/processing runs locally
- GPT-5.2 receives only stats and small message samples (not your full history)
- Cost estimates shown before API calls
- Handle outputs carefully as they contain sensitive content

## License

MIT
