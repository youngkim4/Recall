# archive

Recall's own archive of the user's iMessage history.

One SQLite file mirrors the Messages database. The importer reads Apple's `chat.db`. The query API reads the archive. Other packages use the query API only.

```python
from archive import Archive, import_chat_db

import_chat_db("~/Library/Messages/chat.db", archive_path, contacts=records)

with Archive.open(archive_path) as archive:
    archive.summary()
    archive.search("ferry tickets", limit=20)
```

## Rules

- The archive mirrors the source. The source database is the truth. After a full pull the archive holds exactly the messages of the source.
- A message id is Apple's `message.guid`. A row that the source still holds keeps its id across pulls.
- The source ROWID is not an id. It is kept, and `message_by_source_rowid()` resolves it.
- Text is stored as Messages has it. Nothing is trimmed. No invisible character is dropped. Trim for display only.
- Stored time is UTC. Local time is the time zone of the process. It is made only when a row is shown or a day is turned into a range.
- A message has one kind: `text`, `attachment` or `system`.
- A tapback is a reaction. It is never a message and it is never counted.
- A group event (`item_type` other than 0) is kind `system`. It is never counted. A list, a search or a count returns it only with `include_system=True`. A lookup by id returns it.
- A conversation id is `chat_identifier`. It matches exactly, never in part. An id of `chat` followed by digits only is a group. Any other id is one person.
- A message that the source links to no chat belongs to no conversation. It is stored, and its `conversation` is `""`. No conversation lists it or counts it. A query returns it only on an explicit request. See "Messages with no conversation".
- The package uses only the standard library. `archive/compat.py` is the one module that imports pandas.

## Import

```python
import_chat_db(source_path, archive_path, *, contacts=None, full=True,
               max_removed_share=0.2, progress=None, cancel=None) -> ImportResult
```

The importer does these steps:

1. It copies the database file and its `-wal` and `-journal` files into a private folder next to the archive. It takes no lock and opens no connection on the source for this copy.
2. It checks that the source did not change during the copy. If the source changed, it copies again, three times at most.
3. If the source changed each time, it reads the source through a read-only connection inside one read transaction. It does this only when that cannot make a file in the source folder. If it can, the run fails with `SourceError`. A later run can succeed.
4. It reads the copy in batches of 1,000 rows. Memory use does not grow with the size of the database.
5. It writes the batches in one transaction.
6. It deletes the copy.

Limits of the copy:

- Step 2 compares the inode, the size and the write time of the files. A file system that keeps write times in whole seconds (HFS+) can miss a change. APFS is safe.
- Step 3 checks that `-wal` and `-shm` exist before it opens the source. If the writer closes its last connection in the same moment, SQLite makes these two files again in the source folder. The window is some microseconds, and it opens only after three copies failed.
- The copy opens and closes `chat.db` in the process that runs the import. POSIX then drops the file locks of that process on `chat.db`. Do not keep another SQLite connection to the same `chat.db` open in that process during an import.

The source file is never changed, and no file is made in its folder (but see the second limit). The copy is in `.<archive file name>.import-work` next to the archive, with mode 0700. Each archive has its own folder. A copy that a killed run left is deleted by the next run of the same archive.

### Full pull

A full pull is the default. It reads every row of the source.

- A new row is added.
- A row that changed is updated. The source wins: the stored row takes every value that the source has now.
- A row that did not change is not written again.
- A message whose guid is gone from the source is removed, with its attachment records and the tapbacks on it. This step runs before the rows are read.
- A row that changes its kind (a message that the source now marks as a tapback, or the reverse) is stored one time, as its new kind.
- A tapback is stored only when the archive holds its target message.
- The attachment records of a message are those that the source has now.
- Conversation titles, services and members are those of the source. A conversation or a handle that the source no longer has, and that no stored row uses, is removed.

The safety valve: when the source lacks more than `max_removed_share` of the stored messages, the pull raises `RemovalLimitError` before it changes anything. The default is 0.2 (20 percent). The error has `would_remove`, `stored` and `max_share`. The caller decides what to do: call again with a larger share, or with 1 to switch the valve off. A pull into an archive with no messages never trips the valve. This protects the archive from a database that is only half synced.

### Incremental run

`full=False` reads less. It cannot see a message that the source deleted, so it does not remove one. It reads three sets of rows:

1. Rows with a ROWID above the mark of an earlier run.
2. Rows with a ROWID up to 2,000 below that mark. A recent row can still change: an attachment arrives late.
3. Every older row with an edit or unsend mark.

What an incremental run leaves as it is:

- A message, a tapback, a member, a conversation or a handle that the source removed.
- An old row that changed with no edit or unsend mark (a late attachment, a new chat link, a text that the source filled in). Only a full pull reads it again.

One exception: a row that changed its kind is stored one time, as its new kind, also in an incremental run.

ROWIDs change when Messages rebuilds its database. Each run stores the guids of its 8 highest ROWIDs. A later run trusts a mark only when the source has the same guid at that ROWID. Only the marks of the last full pull and of the runs after it are used: a full pull can remove the rows that an older mark names. If no mark fits, the run is a full pull. A row that was skipped and that the source fills in later, more than 2,000 ROWIDs below the mark, is read only by a full pull.

### Arguments, result and errors

- `contacts`: a list of records `{name, phones, emails}`, the shape of the Contacts cache JSON. `None` keeps the people that the archive has. The AddressBook database is never read.
- `progress` receives `{"phase": str, "done": int, "total": int}`. The phases are `copy`, `messages`, `changes` and `finish`. Each phase is reported at least one time. A phase with no work reports `done` 0 and `total` 0.
- `cancel` is a callable that returns `bool`, or an object with `is_set()`.
- `ImportResult` is a frozen dataclass. `as_dict()` gives a JSON-ready dict. It can be pickled.
- `status` is `ok` or `cancelled`. `mode` is `full`, `incremental` or `legacy_csv`: what the run read. The counters of a cancelled run say how far it got. None of that work is kept.
- Counters: `rows_seen`, `messages_seen`, `reactions_seen`, `system_seen`, `messages_added`, `messages_updated`, `messages_removed`, `reactions_added`, `reactions_updated`, `reactions_removed`, `system_added`, `system_updated`, `system_removed`, `attachments_added`, `attachments_removed`, `conversations_added`, `handles_added`, `blob_texts`, `decode_failures`, `oversize_texts`. `messages_*` count messages of kind `text` and `attachment`. `system_*` count group events.
- `skipped` maps a reason to the number of source rows that are not stored. `counts` holds every counter of the run.
- A failure raises a subclass of `ArchiveError`: `SourceError` or `SourceFormatError` when the source cannot be read, `ImportBusyError` when another import is writing, `RemovalLimitError` for the safety valve, `ImportFailed` for all other failures.
- A failed or cancelled run leaves the archive data as it was. Each run is one row of `import_runs`, also a failed one.

### How a source row is stored

| Source row | Stored as |
| --- | --- |
| `associated_message_type` 2000 to 2007 | reaction, added |
| `associated_message_type` 3000 to 3007 | reaction, removed |
| another `associated_message_type` from 2000 to 3999 | not stored: reason `unknown_tapback` |
| `item_type` other than 0 | message of kind `system` |
| text in `text` or in `attributedBody` | message of kind `text` |
| no visible text, and an attachment or an attachment placeholder | message of kind `attachment` |
| no text, no attachment, `date_retracted` set | message of kind `text`, empty, `unsent` |
| all other rows | not stored: reason `empty` |

Rows that are not stored: no guid (`no_guid`), no usable date (`no_date`), a tapback with no target or with a target that the archive does not hold (`no_target`), a tapback code that this version does not know (`unknown_tapback`), nothing to show (`empty`), a value that cannot be used (`bad_row`). A row that is skipped does not remove a stored row: only a guid that is gone from the source does.

A message with no row in `chat_message_join` is stored with no conversation. `ImportResult.counts["without_conversation"]` is the number of such rows that the run read.

More rules:

- Text comes from the `text` column. When that column is empty, the text comes from `attributedBody`. On a current macOS nearly every row has its text only in `attributedBody`.
- Two characters are never stored: the attachment placeholder U+FFFC, because each attachment is a row of its own, and NUL. All other characters are stored as they are, also outer white space and emoji joiners.
- A message of kind `attachment` can have white space as its text.
- An unsent message has empty text, as in the source.
- An own message has no sender. `handle_id` of an own message is ignored.
- Members come from `chat_handle_join`.
- Attachment metadata is kept: mime type, file name, path, size. Attachment files are never read.
- A column that the source does not have is read as NULL. This applies to `display_name`, `thread_originator_guid`, `date_edited`, `date_retracted`, `associated_message_emoji` and more.
- An Apple date is seconds or nanoseconds since 2001-01-01 UTC. A value of 10^11 or more is nanoseconds.
- The send time is `date`. A row with no `date` uses `date_delivered`, then `date_read`. `extra` then holds `date_source`.

### Oversize texts

A text longer than `archive.INDEXED_TEXT_CHARS` (10,000 characters) is an oversize text. Real exports hold a few texts of many megabytes that no person typed.

- The row is kept and `Message["text"]` is the whole text. No query shape changes.
- `search()` and `count(text=...)` read only the first 10,000 characters of it. The full-text index holds only that part.
- `ImportResult.oversize_texts` is the number of such texts that the run read.
- A caller recognises one with `len(message["text"]) > archive.INDEXED_TEXT_CHARS`.

### People

- Handles that match the same contact record are one person.
- A phone number matches by its digits, the last 10 when there are more. An email address matches in lower case.
- Two records with the same name are two people. One exception: two records with the same name that share a phone number or an email address are two cards of one person.
- A handle that two different names share goes to the first record.
- A handle with no record is its own person with an empty name.
- A person id stays the same while the handles of that person stay together. When people merge, the id that already holds most of the handles stays. A tie goes to the lower id.

## Legacy export bridge

```python
import_legacy_csv(csv_path, archive_path, *, contacts=None, progress=None, cancel=None) -> ImportResult
```

The bridge loads ONE legacy export (`messages.csv` from `parse_imessage.py`) into an EMPTY archive. It exists so that other packages can be built and measured before the importer reads the Messages database. It is not a product feature.

- The file must have the ten columns that `parse_imessage.py` writes: `message_id`, `timestamp`, `sender`, `text`, `chat_id`, `chat_display_name`, `is_from_me`, `service`, `attachment_types`, `attachment_files`. A file without one of them, or with a column name two times, raises `SourceFormatError`.
- The file is read as strict CSV. A quote that is never closed raises `SourceFormatError`. A row with too few or too many fields, with no usable `message_id` or with no `is_from_me` is not stored: reason `bad_row`.

- An archive that holds anything is refused with `ArchiveNotEmptyError`.
- A later `import_chat_db` replaces every row that the bridge loaded. That first pull is always a full pull. Bridge rows have no guid of the source, so the safety valve compares totals for them: when the source holds fewer rows in all than the bridge rows less the share, the pull stops with `RemovalLimitError` before it changes anything. Message ids change in that pull. Conversation ids do not.
- A merged export (`merge_export.py`) also holds rows that the database no longer has, with ids above 1,000,000,000. The bridge loads them. The first pull from the database removes them: the database is the truth.
- A bridge archive has no group events, no unsent messages, no edit marks, no reply links and no reactions. It has no conversation and no handle without a message.
- A row with a time at or before 2001-01-01 UTC has no date: reason `no_date`. The old exporter wrote that time for a row with no date.
- A row gets the id `legacy:<ROWID>`. `message_by_source_rowid()` resolves the ROWID.
- The export has no member list. The members of a conversation are the senders seen in it, plus the handle that names a one-to-one conversation.
- The export does not mark tapbacks. It keeps only the English summary that Messages writes into a tapback row. One function holds the rule for such a text: `archive.tapbacks.is_reaction_phrase(text)`. A row whose whole text is a reaction phrase is stored as kind `system`: it is hidden by default and is not counted. `ImportResult.counts["reaction_phrases"]` is their number. The forms, where the target is text in double quotes (straight or curly), `an image`, `a movie`, `an audio message`, `an attachment` or `a contact`:
  - `Loved`, `Liked`, `Disliked`, `Laughed at`, `Emphasized` or `Questioned`, then the target.
  - `Removed a heart from`, `Removed a like from`, `Removed a dislike from`, `Removed a laugh from`, `Removed an exclamation from` or `Removed a question mark from`, then the target.
  - `Reacted <emoji> to` or `Removed <emoji> from`, then text in double quotes. The emoji is 1 to 16 characters with no space, no ASCII letter and no ASCII digit, so `Reacted badly to "the news"` is a message.
- The whole text must match, and the rule is case-sensitive. A text that starts like a reaction phrase and does not end as one (`Loved it`, `Loved "Dune" a lot`, `Loved a sticker`) is a message: it is stored as text and counted. A real text of the exact form of a phrase is hidden.
- A row that holds only the attachment placeholder, or the `[Attachment]` label of the old exporter, is kind `attachment`.
- A row with no `chat_id` is a message with no conversation.
- Time stamps of the export are local with no zone. The bridge reads them in the zone of this machine.
  - A local time that occurs two times (the clocks went back) is read as its first pass. The two passes then are one moment, and the source ROWID gives the order inside it.
  - A local time that does not exist (the clocks went forward) is read as the time one hour later. The message then comes after the messages of that later time with a lower ROWID.
- One ROWID two times in the file: the last row wins.
- The two attachment lists of a row are paired by position. When their lengths differ, no attachment of that row gets a type.
- Text is what the old exporter wrote. Text that the exporter cut or cleaned is not restored.
- The progress of phase `messages` is counted in bytes of the file.

## Query contract

`Archive.open(path, *, readonly=True) -> Archive`

- `path=None` uses `recall_paths.SAVES_DIR / "archive.sqlite3"`.
- `readonly=True` needs an existing archive. A missing or empty file raises `ArchiveNotFoundError`, a `FileNotFoundError`.
- `readonly=False` creates an empty archive when the file is missing.
- An archive that older code made is migrated when it is opened. A newer archive raises `ArchiveVersionError`.
- `Archive` is safe to use from several threads. Each thread gets its own read-only connection. The connection is closed when its thread ends.
- `Archive` is a context manager. `close()` closes every connection. It can be called at any time: it waits until a query that runs in another thread ends. Every later query raises `ArchiveError`.
- No query writes to the archive.
- A method with several queries reads one state of the archive. `iter_messages()` is the exception: it reads in chunks and holds no read transaction between them.

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
| `sender` | str | handle of the writer. `""` for an own message, and for a received message that has no sender in the source. |
| `text` | str | never `None`, `""` when there is no text |
| `kind` | str | `text`, `attachment` or `system` |
| `attachments` | list of str | mime types in order, `application/octet-stream` when unknown |
| `reply_to` | str or None | guid of the first message of the thread |
| `edited` | bool | |
| `unsent` | bool | |

`Conversation`: `id`, `is_group`, `name` (group title or `""`), `message_count`, `sent_count`, `received_count`, `first_timestamp`, `last_timestamp`, `members` (handles), `person_id` (`None` for a group).

The order of `members` is fixed: the member who wrote the most messages in that conversation is first. Members with the same number are in the order of their handle text. Members who never wrote come last, in the order of their handle text. The owner is not a member.

`Message["sent_at"]` is true UTC. A difference of two `sent_at` values is elapsed time, also across a clock change.

`Person`: `id`, `name`, `handles`, `conversations` (the one-to-one conversations of that person, most messages first).

### Methods

`since` and `until` are local days, `YYYY-MM-DD`. A range includes all of its first day and all of its last day. Any other text raises `ValueError`. A valid day before 1970 or after 2199 is a bound with nothing beyond it: `until="9999-12-31"` means no upper limit.

| Method | Returns |
| --- | --- |
| `summary()` | `{"messages", "conversations", "first_timestamp", "last_timestamp", "messages_without_conversation"}` |
| `conversations(*, limit=None, offset=0, order="count")` | list of Conversation. `order` is `count` or `recent`. |
| `conversation(conversation_id)` | Conversation or `None`. A conversation with no message is returned too. |
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

- `include_system=False` on `iter_messages`, `messages`, `context`, `search`, `count`, `stats` and `day`.
- `include_empty=False` on `conversations`. A conversation with no message is listed only on request. `summary()["conversations"]` counts the same set.
- `iter_messages`, `messages`, `stats` and `day` also accept a list of conversation ids. The result is one thread in time order. Use it with `Person["conversations"]` to show the phone thread and the email thread of one person as one.
- `message(message_id)` returns one Message or `None`.
- `message_by_source_rowid(rowid)` returns the Message that had this ROWID in the database of the last pull, or `None`. Saved chats of the current app hold ROWIDs. The answer is right only while the Messages database is the one that the ROWID came from: a rebuilt database numbers its rows again. The ROWID of a tapback gives `None`. In an archive that the bridge loaded, the ROWID of a tapback phrase gives that row, of kind `system`.
- `messages(..., oldest=True)` gives the oldest page: the start of the conversation. It cannot be given with an anchor.
- `day(conversation_id, day, *, limit=200, offset=None, around=None)` gives one local day. See "day".
- `conversations` in `search` and `count` is a list of ids. One id as a string is also accepted. `[]` matches nothing.
- A reaction of kind `emoji` has one more key, `emoji`.
- `summary()` has one more key, `messages_without_conversation`.
- `no_conversation=False` on `iter_messages`, `search` and `count`. See "Messages with no conversation".

Argument checks: a number that is not an `int` in range, or a flag that is not a `bool`, raises `ValueError`. An id that is not text raises `TypeError`. `limit` has no upper bound below 2^63: `messages()` and `context()` read the page into memory, so use `iter_messages()` for a whole conversation.

### messages() and context()

- With no anchor, the page is the newest one. With `oldest=True`, it is the oldest one.
- `before` and `after` exclude the anchor.
- `around` includes the anchor. Up to `(limit - 1) // 2` older messages come before it. The rest come after it. When one side is short, the other side fills the page.
- An unknown conversation or an unknown anchor gives an empty page with both flags `False`. `context()` gives `[]`.
- More than one anchor raises `ValueError`.
- The order is the send time, in nanoseconds. Messages with the same send time are in the order of their source ROWID. This order is the same in `iter_messages`, `day`, `context` and the time orders of `search`.

### Messages with no conversation

The source can hold a message with no row in `chat_message_join`. A real export of 565,821 rows holds 155 of them.

- The message is stored. `Message["conversation"]` is `""`.
- `""` is not a conversation id. `conversation("")` is `None`. `messages("")`, `stats("")` and `day("", ...)` give nothing.
- `summary()["messages"]`, `conversations()`, `count()`, `search()`, `stats()`, `day()` and every total of a conversation leave these messages out.
- `summary()["messages_without_conversation"]` is their number.
- The explicit request is `no_conversation=True`. It gives these messages and no other message:
  - `count(no_conversation=True)`, also with `since`, `until`, `from_me`, `text` and `group_by`. With `group_by="conversation"` the key is `""`.
  - `search(query, no_conversation=True)`.
  - `iter_messages(None, no_conversation=True)`.
- `no_conversation=True` with a conversation id raises `ValueError`.
- `message(id)` and `message_by_source_rowid(rowid)` return such a message: the caller named it. `context(id)` gives that message alone.
- `compat.messages_frame()` has these messages with a missing `chat_id`, as the export has.
- When a later pull finds a chat for the message, the message moves into that conversation.

### day()

`day()` backs a day view: one day of one conversation.

```python
{
    "day": "2024-06-03",
    "total": 8,
    "offset": 0,
    "messages": [...],
    "previous_day": "2024-06-01",
    "next_day": "2024-06-04",
}
```

- `messages` are the messages of that local day, oldest first.
- `total` is the number of messages in the day. It is equal to `count(conversations=id, since=day, until=day)`.
- A day longer than `limit` is given in part. `offset` in the answer is the place of the first message given.
- A place named with `offset` is used as it is, from 0 to `total`. The answer can then hold fewer messages than `limit`, and a place past the end gives none.
- With no `offset`, the part is a whole window around the local time `around` (`YYYY-MM-DD HH:MM:SS`), or from the start of the day. It never runs past either end of the day. `around` is compared with the time stamps as text, as the day view of the app does. Text that is not such a time is ignored. A value that is not text raises `TypeError`.
- A `limit` below 1 raises `ValueError`. A `limit` above 400 is 400.
- `previous_day` and `next_day` are the nearest days that hold messages, or `None`.
- An unknown conversation gives a day with no messages and no neighbours.

### search()

- Words are AND-ed.
- "Quoted text" is a phrase.
- A trailing `*` is a prefix: `fam*`, `"good mor"*`.
- Case and diacritics do not matter.
- Text that a user typed never raises. Each term goes to FTS5 as a quoted string. `AND`, `OR`, `NOT`, `NEAR`, brackets and colons are plain words.
- An empty query gives `{"total": 0, "hits": []}`. A query never lists messages without a term: use `iter_messages()` for that.
- A query is cut to 1,000 characters and 32 terms. A character that cannot be stored (a lone surrogate) is read as `?`.
- `total` is every match. `hits` is the page.
- `order` is `newest`, `oldest` or `relevance`. The relevance order is the rank of the full-text index, then the newest first. It is the archive's own order: it can differ from another search and it can change.
- `ranges` are `[start, end)` offsets in Unicode code points of `message["text"]`, the same as Python string indexes. JavaScript counts UTF-16 units. Use `archive.search.utf16_ranges(text, ranges)` before ranges go to the frontend.

The index splits text on spaces, punctuation, symbols and emoji. A word next to an emoji is found by the word alone. The index cannot find a word inside Chinese, Japanese, Korean or Thai text, and it does not hold emoji or punctuation. A query term with such characters, or with no letter or digit, is matched as a substring. A term that mixes letters with a symbol or an emoji (`c++`, `$5`) must match as words and as a substring. Limits of a substring term: it reads message text without the index, case matters only for A to Z, and an accent must be the same as in the text.

### count()

- With no `group_by`, the result is an int.
- `text` uses the rules of `search()`. `count(text=q)` is equal to `search(q)["total"]`.
- Keys: `day` is `YYYY-MM-DD`, `month` is `YYYY-MM`, `year` is `YYYY`. All are local time.
- `weekday_hour` is `W-HH`. `W` is 0 for Monday to 6 for Sunday, as `datetime.weekday()`. `HH` is the local hour, `00` to `23`.
- `conversation` keys are conversation ids.
- `sender` keys are handles. The key `""` is the owner. The key `unknown` holds received messages that have no sender in the source.
- Time keys are in ascending order. `conversation` and `sender` rows are in order of `total`, largest first.
- A bucket with no message is not in the list.

### stats()

`stats()` gives the figures of `compute_stats` in `conversation.py` with these keys:

```
conversation, total_messages, sent_count, received_count,
first_timestamp, last_timestamp, active_days, avg_messages_per_day,
busiest_day, busiest_day_count,
longest_gap_days, longest_gap_start, longest_gap_end
```

Differences from `ConversationStats`:

- The key is `conversation`, not `chat_id`.
- Time stamps and `busiest_day` are strings, not `pandas.Timestamp`.
- There is no `unknown_direction_count`, `attachments` or `reactions`.
- `avg_messages_per_day` is the average for each active day.
- A tie goes to the earliest day and to the earliest gap, as in `compute_stats`.
- An empty or unknown conversation gives zeros and `None`.
- `longest_gap_days` is elapsed time. `compute_stats` subtracts wall-clock times. The two differ by one hour (0.0417 days) when the longest gap crosses a clock change.

### reactions()

- Each requested id is a key. A message with no reaction has `[]`.
- `kind` is `love`, `like`, `dislike`, `laugh`, `emphasis`, `question`, `emoji` or `sticker`.
- A reaction of kind `emoji` has one more key, `emoji`.
- One person has at most one reaction on a message: the last one. A removed tapback is not listed. A person is the contact: two tapbacks from two handles of one contact are one reaction. Each tapback with no sender is listed.
- A reaction with no sender in the source has `sender` `""`. Use `is_from_me` to tell it from an own reaction.

### Local time

- `Message["timestamp"]` comes from SQLite's `localtime`. Days and hours of `count()`, `stats()` and `day()` come from a table of the UTC offset changes of the local zone from 2001 to 2200.
- The two agree for dates up to 2037. For a later date (a wrong device clock) they can differ by the daylight-saving hour, because SQLite maps such a year to an earlier one. For the same reason the bridge can give back a time stamp after 2037 one hour off.

## Bridge to the CSV world

`archive.compat.messages_frame(archive) -> pandas.DataFrame`

The columns are those of the CSV export: `message_id`, `timestamp`, `sender`, `text`, `chat_id`, `chat_display_name`, `is_from_me`, `service`, `attachment_types`, `attachment_files`. The current server can run on this frame before each feature is ported.

Differences from the export:

- `message_id` is the guid.
- Tapbacks and group events are not rows.
- An unsent message with no text is not a row.
- A message that is only attachments has the `[Attachment] ...` label of the old exporter as its text. The current exporter writes the placeholder U+FFFC for most of these rows.
- `sender` is the writer. It is missing for an own message. The export had the handle of the other person on own one-to-one messages.
- `attachment_types` and `attachment_files` have one item for each attachment, also when the source has no mime type or no path. The export left such items out.

The frame loads every message into memory. New code must use the query API.

## Decoder

`archive.typedstream.decode_attributed_body(blob) -> str`

`attributedBody` is an `NSAttributedString` written by `NSArchiver`. The decoder reads the length of the text with the real integer rule: a byte below `0x80` is the value, `0x81` is followed by a 2-byte little-endian integer, `0x82` by a 4-byte little-endian integer. The old exporter read these lengths wrong and cut long texts.

- The decoder returns the text as it is in the blob. It trims nothing and drops nothing.
- Data that is not a typedstream raises `TypedStreamError`. The decoder never returns text from it.
- An error message never holds bytes of the blob.

The test fixtures come from Apple's encoder. `scripts/make_typedstream_fixtures.swift` makes them.

## File and schema

- Default path: `recall_paths.SAVES_DIR / "archive.sqlite3"`. The file mode is 0600.
- WAL mode. A reader never waits for an import.
- `PRAGMA user_version` holds the schema version. `archive/migrate.py` holds the migrations. Add a migration at the end of the list. Do not edit a migration that has shipped.
- Version 1 is the layout at the commit that hands this package over. It changed while the package was built. An archive that an earlier build made raises `ArchiveVersionError`: delete the file and pull again. Such a file holds nothing that a pull or the bridge cannot make again.
- Tables: `import_runs`, `people`, `handles`, `conversations`, `conversation_members`, `messages`, `attachments`, `reactions`, and the FTS5 index `messages_fts`.
- `messages_fts` is an external-content index over `messages.search_text`, the first 10,000 characters of the text. Triggers keep it in step.
- The tokenizer is `unicode61 remove_diacritics 2` with a list of separators: every symbol, punctuation mark and emoji above ASCII. Without the list, the tokenizer takes an emoji newer than Unicode 6.1 for a letter. The list is made from the Unicode data of the Python that creates the archive, and it is stored in the file.
- `messages.sent_ns` is UTC unix nanoseconds and gives the order of messages. `messages.sent_at` is the same moment in seconds.
- `messages.source_rowid` is indexed.
- Each conversation row holds its own totals. A full pull counts all of them again.
- The package needs SQLite 3.44 or newer with FTS5. It raises `SqliteFeatureError` when it starts without them.

## Scripts

| Script | Use |
| --- | --- |
| `scripts/verify_archive_import.py` | The owner runs it on their own database. It imports into a temp archive and prints numbers only. |
| `scripts/archive_scale_check.py` | Makes a large synthetic database and prints import time, peak memory and query times. |
| `scripts/archive_bridge_check.py` | Loads a legacy export, given as an argument, into a temp archive and prints numbers only. |
| `scripts/make_typedstream_fixtures.swift` | Makes the decoder fixtures with `NSArchiver`. |

## Tests

```bash
./venv/bin/python -m pytest tests/archive --cov=archive --cov-report=term-missing
```

The tests use synthetic data only. `tests/guards.py` sets a dummy API key and a temp data folder, and it blocks every socket that leaves the machine. `tests/archive/conftest.py` adds a fixed time zone.
