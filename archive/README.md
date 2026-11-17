# archive

Recall's own archive of the user's iMessage history.

One SQLite file holds every message that an import has seen. The importer reads Apple's `chat.db`. The query API reads the archive. Other packages use the query API only.

```python
from archive import Archive, import_chat_db

import_chat_db("~/Library/Messages/chat.db", archive_path, contacts=records)

with Archive.open(archive_path) as archive:
    archive.summary()
    archive.search("ferry tickets", limit=20)
```

## Rules

- The archive only grows. A message stays when `chat.db` no longer has it.
- A message id is Apple's `message.guid`. A ROWID is never an id.
- Stored time is UTC. Local time is the time zone of the process. It is made only when a row is shown or a day is turned into a range.
- A message has one kind: `text`, `attachment` or `system`.
- A tapback is a reaction. It is never a message and it is never counted.
- A group event (`item_type` other than 0) is kind `system`. It is never counted. A query returns it only with `include_system=True`.
- A conversation id is `chat_identifier`. An id of `chat` followed by digits only is a group. Any other id is one person.
- The package uses only the standard library. `archive/compat.py` is the one module that imports pandas.

## Import

```python
import_chat_db(source_path, archive_path, *, contacts=None, full=False, progress=None, cancel=None) -> ImportResult
```

The importer does these steps:

1. It opens the source read-only and starts a read transaction. This fixes one consistent state.
2. It copies that state into a private temp file with the SQLite backup API.
3. It reads the copy in batches of 1,000 rows. Memory use does not grow with the size of the database.
4. It writes the batches in one transaction.
5. It deletes the copy.

The source file is never changed. The copy is in `.recall-import-work` next to the archive. A copy that a killed run left is deleted by the next run.

Arguments:

- `contacts`: a list of records `{name, phones, emails}`, the shape of the Contacts cache JSON. `None` keeps the people that the archive has. The AddressBook database is never read.
- `full=True` reads every row. The default reads less. See "Incremental import".
- `progress` receives `{"phase": str, "done": int, "total": int}`. The phases are `copy`, `messages`, `changes` and `finish`.
- `cancel` is a callable that returns `bool`, or an object with `is_set()`.

Result and errors:

- `ImportResult` is a frozen dataclass. `as_dict()` gives a JSON-ready dict.
- `status` is `ok` or `cancelled`. `mode` is `full` or `incremental`: what the run read.
- `skipped` maps a reason to the number of source rows that are not stored.
- A failure raises a subclass of `ArchiveError`: `SourceError` or `SourceFormatError` when the source cannot be read, `ImportBusyError` when another import is writing, `ImportFailed` for all other failures.
- A failed or cancelled run leaves the archive data as it was. Each run is one row of `import_runs`, also a failed one.

How a source row is stored:

| Source row | Stored as |
| --- | --- |
| `associated_message_type` 2000 to 2007 | reaction, added |
| `associated_message_type` 3000 to 3007 | reaction, removed |
| `item_type` other than 0 | message of kind `system` |
| text in `text` or in `attributedBody` | message of kind `text` |
| no text, one or more attachments | message of kind `attachment` |
| no text, no attachment, `date_retracted` set | message of kind `text`, empty, `unsent` |
| all other rows | not stored: reason `empty` |

Rows that are not stored: no guid (`no_guid`), no usable date (`no_date`), no row in `chat_message_join` (`no_conversation`), a tapback with no target (`no_target`), nothing to show (`empty`).

More rules:

- Text comes from the `text` column. When that column is empty, the text comes from `attributedBody`.
- The attachment placeholder character (U+FFFC) and outer white space are removed from text.
- An own message has no sender. `handle_id` of an own message is ignored.
- Members come from `chat_handle_join`. A member that left stays in the archive.
- Attachment metadata is kept: mime type, file name, path, size. Attachment files are never read.
- A column that the source does not have is read as NULL. This applies to `display_name`, `thread_originator_guid`, `date_edited`, `date_retracted`, `associated_message_emoji` and more.
- An Apple date is seconds or nanoseconds since 2001-01-01 UTC. A value of 10^11 or more is nanoseconds.

What the archive never gives up:

- An older copy of a message does not undo an edit or an unsend.
- Text that the source lost does not blank the stored text.
- An unknown value does not replace a known value.

### Incremental import

The default run reads three sets of rows:

1. Rows with a ROWID above the mark of an earlier run.
2. The 2,000 rows below that mark. A recent row can still change: an attachment arrives late.
3. Every row below that with an edit or unsend mark.

ROWIDs change when Messages rebuilds its database. Each run stores the guids of its 8 highest ROWIDs. A later run trusts a mark only when the source has the same guid at that ROWID. If no mark fits, the run reads every row.

### People

- Handles that match the same contact record are one person.
- A phone number matches by its digits, the last 10 when there are more. An email address matches in lower case.
- Two records with the same name are two people. One exception: two records with the same name that share a phone number or an email address are two cards of one person.
- A handle that two different names share goes to the first record.
- A handle with no record is its own person with an empty name.
- A person id stays the same while the handles of that person stay together. When two people become one, the lower id stays.

## Query contract

`Archive.open(path, *, readonly=True) -> Archive`

- `path=None` uses `recall_paths.SAVES_DIR / "archive.sqlite3"`.
- `readonly=True` needs an existing file. A missing file raises `ArchiveNotFoundError`, a `FileNotFoundError`.
- `readonly=False` creates an empty archive when the file is missing.
- An archive that older code made is migrated when it is opened. A newer archive raises `ArchiveVersionError`.
- `Archive` is safe to use from several threads. Each thread gets its own read-only connection. The connection is closed when its thread ends.
- `Archive` is a context manager. `close()` closes every connection.
- No query writes to the archive.

### Shapes

Each value is a plain dict that `json.dumps` accepts.

`Message`

| Key | Type | Meaning |
| --- | --- | --- |
| `id` | str | the guid |
| `conversation` | str | `chat_identifier` |
| `timestamp` | str | local time, `YYYY-MM-DD HH:MM:SS` |
| `sent_at` | int | UTC unix seconds |
| `is_from_me` | bool | |
| `sender` | str | handle of the writer, `""` for an own message |
| `text` | str | never `None`, `""` when there is no text |
| `kind` | str | `text`, `attachment` or `system` |
| `attachments` | list of str | mime types in order, `application/octet-stream` when unknown |
| `reply_to` | str or None | guid of the first message of the thread |
| `edited` | bool | |
| `unsent` | bool | |

`Conversation`: `id`, `is_group`, `name` (group title or `""`), `message_count`, `sent_count`, `received_count`, `first_timestamp`, `last_timestamp`, `members` (handles), `person_id` (`None` for a group).

`Person`: `id`, `name`, `handles`, `conversations` (the one-to-one conversations of that person, most messages first).

### Methods

`since` and `until` are local days, `YYYY-MM-DD`. A range includes all of its first day and all of its last day. Any other text raises `ValueError`.

| Method | Returns |
| --- | --- |
| `summary()` | `{"messages", "conversations", "first_timestamp", "last_timestamp"}` |
| `conversations(*, limit=None, offset=0, order="count")` | list of Conversation. `order` is `count` or `recent`. |
| `conversation(conversation_id)` | Conversation or `None` |
| `iter_messages(conversation_id, *, since=None, until=None)` | a generator of Message, oldest first |
| `messages(conversation_id, *, around=None, before=None, after=None, limit=50)` | `{"messages": [...], "has_older": bool, "has_newer": bool}`, oldest first |
| `context(message_id, *, radius=6)` | list of Message: the message and its neighbours in the same conversation |
| `search(query, *, conversations=None, since=None, until=None, from_me=None, limit=50, offset=0, order="newest")` | `{"total": int, "hits": [{"message": Message, "ranges": [[start, end], ...]}]}` |
| `count(*, conversations=None, since=None, until=None, from_me=None, text=None, group_by=None)` | int, or a list of `{"key", "total", "sent", "received"}` |
| `stats(conversation_id, *, since=None, until=None)` | dict, see "stats" |
| `people()` | list of Person, named people first |
| `person(person_id)` | Person or `None` |
| `reactions(message_ids)` | `{message_id: [{"kind", "is_from_me", "sender"}]}` |

Additions to the contract. None of them changes a name, an argument or a shape:

- `include_system=False` on `iter_messages`, `messages`, `context`, `search`, `count` and `stats`.
- `include_empty=False` on `conversations`. A conversation with no message is listed only on request. `summary()["conversations"]` counts the same set.
- `iter_messages`, `messages` and `stats` also accept a list of conversation ids. The result is one thread in time order. Use it with `Person["conversations"]` to show the phone thread and the email thread of one person as one.
- `message(message_id)` returns one Message or `None`.
- `conversations` in `search` and `count` is a list of ids. One id as a string is also accepted. `[]` matches nothing.

### messages() and context()

- With no anchor, the page is the newest one.
- `before` and `after` exclude the anchor.
- `around` includes the anchor. Up to `(limit - 1) // 2` older messages come before it. The rest come after it. When one side is short, the other side fills the page.
- An unknown conversation or an unknown anchor gives an empty page with both flags `False`. `context()` gives `[]`.
- More than one anchor raises `ValueError`.
- Messages of the same second keep the order of the source.

### search()

- Words are AND-ed.
- "Quoted text" is a phrase.
- A trailing `*` is a prefix: `fam*`, `"good mor"*`.
- Case and diacritics do not matter.
- Text that a user typed never raises. Each term goes to FTS5 as a quoted string. `AND`, `OR`, `NOT`, `NEAR`, brackets and colons are plain words.
- An empty query gives `{"total": 0, "hits": []}`.
- `total` is every match. `hits` is the page.
- `order` is `newest`, `oldest` or `relevance`.
- `ranges` are `[start, end)` offsets in Unicode code points of `message["text"]`, the same as Python string indexes. JavaScript counts UTF-16 units. Use `archive.search.utf16_ranges(text, ranges)` before ranges go to the frontend.

The index tokenizer (`unicode61`) cannot find a word inside Chinese, Japanese, Korean or Thai text. It does not index emoji or punctuation. A query term with such characters, or with no letter or digit, is matched as a substring. A substring term reads message text without the index.

### count()

- With no `group_by`, the result is an int.
- `text` uses the rules of `search()`. `count(text=q)` is equal to `search(q)["total"]`.
- Keys: `day` is `YYYY-MM-DD`, `month` is `YYYY-MM`, `year` is `YYYY`. All are local time.
- `weekday_hour` is `W-HH`. `W` is 0 for Monday to 6 for Sunday, as `datetime.weekday()`. `HH` is the local hour, `00` to `23`.
- `conversation` keys are conversation ids. `sender` keys are handles. The key `""` is the owner.
- Time keys are in ascending order. `conversation` and `sender` rows are in order of `total`, largest first.
- A bucket with no message is not in the list.

### stats()

The keys are the field names of `ConversationStats` in `conversation.py`:

```
conversation, total_messages, sent_count, received_count,
first_timestamp, last_timestamp, active_days, avg_messages_per_day,
busiest_day, busiest_day_count,
longest_gap_days, longest_gap_start, longest_gap_end
```

- `avg_messages_per_day` is the average for each active day.
- A tie goes to the earliest day and to the earliest gap, as in `compute_stats`.
- An empty or unknown conversation gives zeros and `None`.
- The gap is elapsed time. `compute_stats` subtracts wall-clock times. The two differ by one hour when the longest gap crosses a daylight-saving change.

### reactions()

- Each requested id is a key. A message with no reaction has `[]`.
- `kind` is `love`, `like`, `dislike`, `laugh`, `emphasis`, `question`, `emoji` or `sticker`.
- A reaction of kind `emoji` has one more key, `emoji`.
- One person has at most one reaction on a message: the last one. A removed tapback is not listed.

## Bridge to the CSV world

`archive.compat.messages_frame(archive) -> pandas.DataFrame`

The columns are those of the CSV export: `message_id`, `timestamp`, `sender`, `text`, `chat_id`, `chat_display_name`, `is_from_me`, `service`, `attachment_types`, `attachment_files`. The current server can run on this frame before each feature is ported.

Differences from the export:

- `message_id` is the guid.
- Tapbacks and group events are not rows.
- `sender` is the writer. It is missing for an own message. The export had the handle of the other person on own one-to-one messages.

The frame loads every message into memory. New code must use the query API.

## Decoder

`archive.typedstream.decode_attributed_body(blob) -> str`

`attributedBody` is an `NSAttributedString` written by `NSArchiver`. The decoder reads the length of the text with the real integer rule: a byte below `0x80` is the value, `0x81` is followed by a 2-byte little-endian integer, `0x82` by a 4-byte little-endian integer.

The test fixtures come from Apple's encoder. `scripts/make_typedstream_fixtures.swift` makes them.

## File and schema

- Default path: `recall_paths.SAVES_DIR / "archive.sqlite3"`. The file mode is 0600.
- WAL mode. A reader never waits for an import.
- `PRAGMA user_version` holds the schema version. `archive/migrate.py` holds the migrations. Add a migration at the end of the list. Do not edit a migration that has shipped.
- Tables: `import_runs`, `people`, `handles`, `conversations`, `conversation_members`, `messages`, `attachments`, `reactions`, and the FTS5 index `messages_fts`.
- `messages_fts` is an external-content index with the tokenizer `unicode61 remove_diacritics 2`. Triggers keep it in step.
- `messages.sent_ns` is UTC unix nanoseconds and gives the order of messages. `messages.sent_at` is the same moment in seconds.
- Each conversation row holds its own totals. The importer counts them again for each conversation that a run touched.
- The package needs SQLite 3.44 or newer with FTS5. It raises `SqliteFeatureError` when it starts without them.

## Scripts

| Script | Use |
| --- | --- |
| `scripts/verify_archive_import.py` | The owner runs it on their own database. It imports into a temp archive and prints numbers only. |
| `scripts/archive_scale_check.py` | Makes a large synthetic database and prints import time, peak memory and query times. |
| `scripts/make_typedstream_fixtures.swift` | Makes the decoder fixtures with `NSArchiver`. |

## Tests

```bash
./venv/bin/python -m pytest tests/archive --cov=archive --cov-report=term-missing
```

The tests use synthetic data only. `tests/archive/conftest.py` sets a dummy API key, a temp data folder and a fixed time zone. It blocks every socket that leaves the machine.
