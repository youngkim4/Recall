# recall_services

The functions that the server calls. Each one reads the archive and returns the JSON shape that its route returns today.

`ui_server.py` calls these functions for every route that reads messages. "The archive and the services" in `CONTRACTS.md` holds the rules of the hook-up.

## Rules

- No module here imports `ui_server`.
- No module here reads the Contacts database of the Mac. Names come in as data.
- Every read goes through one `Archive`. The services do not open a second handle.
- A function that takes a model id reads the model from `models.py`. An unknown id is a typed error.
- Each failure that a person can see is one plain sentence and a code.

## What the server keeps

```python
from recall_services import names, places

state = places.state_for(messages_path)           # the archive that stands for the file that the app named
saved = names.cached(CONTACTS_CACHE_PATH)         # read again only when the file changes
```

Ask for `state.archive()` and for `names.cached(...)` at the start of each request. Do not keep the answers between requests.

`places.py` says which archive stands for a `messagesPath`: nothing named, the default export or the default archive is the archive at the default place (`archive_state.shared()`); a name that ends in `.sqlite3` is the archive itself; any other name is an export of an earlier build, and its archive stands beside it with the ending `.sqlite3`.

`keys.py` gives the key of the model provider: `OPENAI_API_KEY` in the environment, or in the file `.env` in the data folder, read at each call. `key_store.py` writes the key that a person gives in the app to that file. See "First run".

## Functions

| Route or use | Call |
|---|---|
| `/api/defaults` (models) | `models.defaults_payload()` |
| `/api/contacts` | `conversations.payload(archive, saved, limit, messages_path)` |
| `/api/search` | `search.answer(archive, saved, query_string)`, which calls `search.payload(archive, saved, query, contact, limit, offset, as_you_type=..., filters=..., order=...)` |
| `/api/stats` | `stats.answer(archive, saved, query_string)`, which calls `stats.payload(archive, saved, filters, words=..., by=..., limit=..., prefix=..., habits=...)` |
| `/api/day` | `day.payload(archive, saved, contact, day, around, offset, limit)` |
| `/api/ask/stream` | `chat.stream(question, contact, model, history, cancel, archive=..., names=..., use_ai=...)` |
| `/api/ask` | `chat.ask(...)`, the same arguments |
| `/api/preview` (estimate) | `reports.estimate(contact, model, since, until, archive=..., names=...)` |
| job `analyze` | `reports.run(contact, model, since, until, on_progress, cancel, archive=..., names=...)` |
| `/api/meaning` | `meaning.payload(archive, names)` |
| job `meaning` | `meaning.build(archive, names, cancel, on_progress)` |
| the refresh after an import | `meaning.built(archive_file)`, `meaning.estimate(archive, names)`, `meaning.small_enough(estimate)` |
| job `digest` | `digest_job.run(conversation, since, until, on_progress, archive=..., names=...)` |
| `/api/memory` | `memory_prep.payload(archive, names)` |
| job `memory` | `memory_prep.run(on_progress, cancel, archive=..., names=...)` |
| the reading of the memory after an import | `memory_prep.completed_before(archive_file)`, `memory_prep.payload(archive, names)`, `memory_prep.worth_starting(found, completed_before=True)` |
| the marker of a completed `memory` job | `memory_prep.mark_completed(archive_file)` |
| `/api/people` | `people.payload(archive, names)` |
| job `people` | `people.build(archive, names, cancel, on_progress)` |
| the build of the people index after a `memory` job or an import | `people.completed_before(archive_file)`, `people.payload(archive, names)`, `people.worth_starting(found)` |
| find people by description (chat) | `people.search(archive, names, description, kinds=..., since=..., until=..., limit=..., conversations=...)`, `people.status(archive, names)` |
| `GET`, `POST`, `DELETE /api/key` | `key_store.state()`, `key_store.save(key)`, `key_store.remove()`, and `key_store.same_origin(host, origin)` before each |
| `/api/first-run` | `first_run.estimate(archive, names)`, or `first_run.nothing()` with no archive |
| job `first_run` | `first_run_job.run(steps, work, on_progress, archive_file=...)` with the work of the jobs `meaning`, `memory` and `people` |
| several digests at once | `digests.call_pool(model, max_parallel)`, then `digests.build(..., pool=pool)` in each thread |
| the two optional ports of the chat engine | `memory.ports(archive, names)` |
| the saved report of a report file | `reports.saved_report(contact, report_text)` |
| the moments of a report | `reports.moments_for(report, archive)` |
| a story that the server shows plain | `reports.shown_plain(report_text)` |
| the key of a cache that is kept on disk | `state.file_mark()` |
| figures on pandas | `frame.messages_frame(state)`, `frame.attachments_frame(...)`, `frame.reactions_frame(...)` |
| job `export` (update) | `state.pull_from_messages(db_path, records)` |
| first start with an export | `state.ensure_from_export(messages_csv)` |
| an id that the browser saved | `messages.find_saved(archive, saved_id, chat_id=..., timestamp=...)` |

## The archive state

`archive_state.ArchiveState` holds the place of the archive, one read handle and the two ways to fill the archive.

- `archive()` gives the shared handle. It raises `ArchiveNotFoundError` when there is no archive.
- `reopen()` closes the handle. The next `archive()` opens the file that is there then.
- `generation` goes up each time the content of the archive changed.

`ensure_from_export(messages_csv)` is the bridge from an export:

1. It reads the signature of the export: path, write time in nanoseconds, size.
2. It loads the export when there is no archive, or when the signature differs from the recorded one.
3. It loads into a new file and moves that file into place in one step.
4. It records the signature in `archive.sqlite3.source.json`.

WARNING: An export never replaces an archive that a pull from Messages filled. The answer is then `kept`.

Limits of the bridge:

- A file at the place of the archive that Recall can not open is not replaced. The answer is `failed`. A person must move the file away.
- Two states of an export with the same size and the same write time are one state.
- The note beside the archive can be lost. An archive with a message from a pull is still `kept`. An archive that a pull emptied, with no note, is loaded from the export.

`pull_from_messages(db_path, contacts)` is a full pull. The answer is one of five types:

| Type | Meaning |
|---|---|
| `PullDone` | The archive holds what the database holds. `counts` has the numbers. |
| `PullRefused` | The database lacks more of the stored messages than the limit. Nothing changed. |
| `PullBusy` | Another import writes the archive. |
| `PullCancelled` | The caller stopped the pull. Nothing changed. |
| `PullFailed` | The pull did not run or did not end. `reason` is for the person. |

The caller names the database. `database_for` in the server decides which one a job reads. The note beside the archive holds no path of a database.

`PullFailed.detail` is for the log only. It can hold a path. `as_dict()` leaves it out.

### Another process

The command line importer can write the archive too. The lock of the state holds inside one process only.

- A load takes the write lock of the old archive before the new file takes its place. When another process is in a write or a read, the answer is `busy` and nothing changed.
- The lock is let go just before the move. A writer that starts in that instant loses its write.
- A first pull that did not end removes the archive only when the archive holds nothing. `PullBusy` removes nothing.

### Caches

- `generation` is for a cache in memory. It starts at 0 at each start of the server.
- `file_mark()` is for a cache on disk: the place, and the write time and size of the database and of its write-ahead file. It can change while the content is the same. It does not stay the same when the content changed.
- Do not call the state from inside a query of its own handle.

## Names

`names.Names` is the one place for names. It follows "Names" in `CONTRACTS.md`.

- `saved(handle)`: the saved name, or `""`.
- `conversation(record)`: a saved name, the title of a group, a name made from the saved members with the busiest first, or `""`.
- `chat_labeler(conversations)`: the labeler for the chat engine. It gives a saved name or a mask.
- `report_labeler(record)`: the labeler for a report. It gives `""` for a sender who is not saved. `reports.py` also masks each unsaved handle in the text of a message, as chat does.

A handle matches a record as the archive matches people: a phone number by its last 10 digits, an address in lower case.

A group that nobody named gets a name from its saved people. Two differences from the list of today:

- A member who never wrote is a person of the group. The export knew only the writers.
- A person with two handles stands where the busier handle stands. The export rule added both.

## Search and counts

`search.py` answers `/api/search`. `stats.py` answers `/api/stats`. Both read the filters of `search_filters.py`.

### Filters

Each filter is optional. All of them are combined with AND. A value of a filter, of `order`, `prefix` or `by` that is not allowed is a `ValueError` with one sentence, and the route answers 400 with it: `{"error": "kind must be direct, group or empty."}`. That sentence never repeats the value. A `limit` or an `offset` that is no whole number gives the error of `inputs.whole`, as before: `limit must be a whole number, got 'many'`.

| Parameter | Values |
|---|---|
| `contacts` | conversation ids, comma separated. `contact` (one id) still works and is added first. An id matches whole. An unknown id matches nothing. |
| `since`, `until` | local days, `YYYY-MM-DD`, both included. A `since` after `until` matches nothing. |
| `from` | `me`, `them` or empty |
| `kind` | `direct`, `group` or empty |
| `attachments` | `any`, `none`, `image`, `video`, `audio`, `other` or empty |
| `query` | the words. A star is always a prefix. |
| `prefix` | `false`: every word is whole. `true`: the last word is a prefix. No `prefix` (or an empty one): `true` on `/api/search`, the search of the app of today as the reader types; `false` on `/api/stats`. Send `prefix=false` to `/api/search` for whole words. |
| `order` | `/api/search` only: `newest` (default), `oldest`, `relevance` |

- The kind of an attachment comes from its mime type: `image/*`, `video/*`, `audio/*`. `other` is every other type and an attachment with no type. A GIF is an image.
- `any` is a message with one attachment record or more. A message of kind `attachment` with no record (an old export) is `none`.
- A message with an image and a video is in `image` and in `video`.
- `relevance` with no words is `newest`, and `order` in the payload says `newest`.

### `/api/search`

```json
{
  "query": "hi",
  "contact": "",
  "count": 2,
  "total": 2,
  "offset": 0,
  "results": [
    {
      "messageId": "…", "chatId": "…", "displayName": "…", "senderName": "…",
      "timestamp": "2025-01-03 19:30:00", "sender": "…", "text": "this is which hi",
      "attachment": "", "isFromMe": true,
      "ranges": [[14, 16]]
    }
  ],
  "filters": {"contacts": [], "since": "", "until": "", "from": "", "kind": "", "attachments": ""},
  "order": "newest",
  "prefix": false
}
```

- That answer is for `query=hi&prefix=false`: the word "hi" only, not "this", "which", "hit" or "him". With no `prefix`, `prefix` is `true` and "hit" and "him" match too ("this" and "which" never: a prefix is the start of a word).
- `prefix` says what the search applied.

- `filters` holds every key. A filter that was not applied is `""` or `[]`. `contacts` holds `contact` too.
- `ranges` are `[start, end)` in UTF-16 units of `text`, as JavaScript counts. They are the ranges that the index matched: a prefix marks the whole word. They follow the shown text: no outer space, no attachment placeholder. A word after the cut of a long text (600 characters) has no range. A range across the cut stops at the cut.
- With no words, `ranges` is `[]`.
- No words and a filter list the messages of the filter, newest first. `total` counts all of them. The pages have no depth limit.
- Typed text with no word (`*`, `""`) matches nothing, also with a filter. `/api/stats` counts 0 for it.

### `/api/stats`

```json
{
  "by": "person",
  "query": "pizza",
  "prefix": false,
  "filters": {"contacts": [], "since": "", "until": "", "from": "", "kind": "", "attachments": ""},
  "totals": {"total": 5, "sent": 1, "received": 4},
  "first": "2025-01-01",
  "last": "2025-01-05",
  "days": 4,
  "rows": [
    {"key": "+15550000001", "label": "Avery Stone", "total": 2, "sent": 1, "received": 1,
     "conversations": ["+15550000001"]}
  ],
  "rowsTotal": 3
}
```

- `by`: `day`, `week`, `month`, `year`, `weekday`, `hour`, `weekday_hour`, `person`, `conversation`, or empty. Empty gives `rows` `[]` and the totals only.
- `totals` covers the whole filtered set. `first` and `last` are its first and last local day, or `null` when the set is empty. `received` is the count of the archive. In the archive every message is sent or received (`is_from_me` is 0 or 1), so its `received` is `total` less `sent`; the service adds the counts of the archive and never works one out itself.
- `days` is the number of local days that hold a message of the filtered set. It is 0 when the set is empty. It is not the length of the range from `first` to `last`.
- `rowsTotal` is the number of rows before `limit`.
- No `prefix` is `false` here: `query=hi` counts the word "hi" only, never "hit", "him" or "hiking". `prefix=true` counts the words that start with the last word. `prefix` says what was applied.
- Time rows: `{"key", "label", "total", "sent", "received"}`.
  - `day` (`YYYY-MM-DD`), `week` (the Monday, `YYYY-MM-DD`), `month` (`YYYY-MM`), `year` (`YYYY`): every bucket from `since` to `until`, in time order. With no `since`, the range starts at `first`. With no `until`, it ends at `last`. The range never goes past the first or the last day of the archive. An empty bucket is a row of zeros. The label is the key.
  - `weekday` (`0` is Monday to `6`, label `Monday`), `hour` (`00` to `23`, label `09:00`), `weekday_hour` (`W-HH`, label `Monday 09:00`): every bucket, always.
  - With an empty set and no `since` or `until`, `day` to `year` have no row.
- People rows: the time row and `conversations`, the ids that the row counts.
  - `conversation`: one row per conversation.
  - `person`: one row per person. The phone thread and the email thread of one person are one row: the archive joins handles of one contact record (`Conversation["person_id"]`). A group chat is no person and is left out: it is listed by `conversation` only. (Once a group was its own row here, and the two lists matched row for row.) The key is the id of the busiest thread in the set.
  - The label is `conversation_label(names.conversation(record), id)` from `conversation.py`: the words of `conversationName` in the app. A saved name, a group name, "Unnamed group", "Number ending in 1234", or an email address as it is. No full phone number of a person who is not saved.
  - The busiest come first (a tie: the key). `limit` is 50 by default, held between 1 and 500.
- `habits=true` adds `"habits"` to the payload, and to each row of `person` and `conversation` (`habits.py`, counted by `archive/habits.py`). False by default: habits read every message of the set. A value other than `true` or `false` is a 400.
  - `{"messages", "youReply", "theyReply", "replies": {"you", "they"}, "youStart", "talks", "streak": {"days", "first", "last"}, "lateNight", "sentLength", "receivedLength"}`.
  - A talk starts after 6 hours of quiet in a one-to-one conversation. Its first message says who started it. A reply is the first message from the other side in the talk. `youReply` and `theyReply` are the median seconds from the start of the talk, in elapsed time. A group has no talks.
  - Late night is midnight to 5 a.m. local. A length is the mean characters of a text (kind `text`). The streak is the longest run of local days in a row with a message; a tie goes to the earliest.
  - The words, `from` and `attachments` narrow the messages, not the talks: with only the messages that hold a word, nobody replies to anybody. The contacts, the days and `kind` narrow both.
  - A figure with nothing to count is `null`, never 0: a reply at once is 0 seconds.
  - A person row joins the habits of its threads before the medians and the streak.
- One answer reads one state of the archive (`Archive.snapshot()`): the totals and the rows agree also while an import commits.
- `GET /api/stats/words` (the most used words) is not built. On the invented archive below, a count of the words of every message took 3.9 s in Python (2.4 s of it to read the texts), and `fts5vocab` with a filter of one year took 1.7 s. Both miss the limit of 2 s.

### Speed

Measured on an invented archive of 545,000 messages and 1,400 conversations (`tests/archive/synthetic.py` with 1,300 people and 100 groups, 2017 to 2037), on the Mac of the build, best of three:

- `/api/stats` on the whole archive: 212 ms (`person`) to 626 ms (`year`). The total alone: 235 ms. A grouping by time makes two counts: the days, and the grouping.
- `/api/stats` with filters (one year, `from=me&kind=group`, `attachments=image`, a word): 6 ms to 244 ms.
- `/api/stats?by=person&limit=10&habits=true` on the user's archive (545,000 messages): 1.1 s to 1.2 s. One year: 0.35 s. A word: 0.3 s.
- `/api/search` with words: 27 ms to 88 ms, also with filters and with `prefix=true`. With no words and a filter: 16 ms (one year) to 506 ms (`attachments=none`, 490,673 messages).


## Chat

- A "user" row of the history can hold `contact`: the conversation that its question was asked of.
- A row with no `contact` was asked of everything. Its answer does not go to a question that has a scope.
- A question over 8,000 characters is the error `question_too_long`.
- `cancel` is a `CancelToken` that the route makes for each request.
- The engine gets the two ports of `memory.ports(archive, names)`. See "The two ports of chat".
- No question starts a reading outside itself. `chat.stream` takes no function of the server for it.
- While a digest is read inside the question, `chat.stream` gives one `progress` event for each change of a part:
  `{"type": "progress", "tool": "conversation_digest", "conversation": <label as shown>, "part": i, "of": n, "label": <period label>, "state": "start"|"done"|"cached"|"failed"}`.
  The events come after the status line of the digest and before `done`. `/api/ask/stream` writes each one as a `data:` frame. `chat.ask` and `/api/ask` drop them.
- With a digest source, the payload has `memory`: `{"notReady": [conversation ids]}`. These are the conversations whose story had more parts that are not read yet than one question reads (`DIGEST_PARTS_LIMIT`). Nothing reads them by itself: the app can offer it. With no digest source (no key, or the local answer) the payload has no `memory`.

## The index of meanings

`meaning.py` gives the routes the index of `meaning_index`. `meaning_failures.py` holds its failure sentences.

A segment of the index is a short exchange. A line for a person says "exchange".

```python
from recall_services import meaning

meaning.index_folder(archive.path)   # saves/meaning/<16 hex digits>: the folder of the index of this archive
meaning.status(archive, names)       # {"state", "segments", "model", "dimensions", "builtAt", "coveredUntil"}
meaning.estimate(archive, names)     # {"segments", "newSegments", "staleSegments", "tokens", "estimatedCost", "upperBound"}
meaning.build(archive, names, cancel=None, on_progress=None)   # {"segments", "embedded", "reused", "removed", "tokens"}
meaning.payload(archive, names)      # {"status", "estimate", "error", "code"}: the answer of GET /api/meaning
meaning.small_enough(estimate)       # can the server bring the index up to date by itself
```

One index for one archive:

- The folder of an index is `saves/meaning/<name>`. The name is the first 16 hex digits of the SHA-256 of the resolved path of the archive file.
- `index_folder` is the one function that builds this path. Each function takes the folder from the archive that it was given.
- A build for one archive does not touch the index of another archive.

What a build sends:

- The messages come through `reports._Masked`, with the saved names of the server. A handle that is not saved is masked in the text, by the same function as in a report and in a digest.
- A message with no text is in no segment.
- A contact that is saved later is no longer masked. The segments that held its handle then have a new content hash, and the next build embeds them again.

Status:

- `state` is `none`, `ready` or `behind`.
- `status` is the quick look. Its `behind` means that the archive holds a later message with text than the index covers. A later photo with no caption does not make the state `behind`.
- `payload` also uses the estimate. Its `behind` means, in addition, that a build would embed a segment (`newSegments`) or write a segment whose id the index does not hold (`staleSegments`).
- `coveredUntil` is the local time of the last message in the index. `builtAt` is UTC.
- An index that this build of the app can not read has the state `none`. The estimate and the build then fail with the code `index_newer`.
- With no index folder the status asks for no file. An import with no index thus opens no database of the index.

Estimate:

- The estimate reads every conversation. It makes no request and no file. It needs no key.
- WARNING: Do not ask for `GET /api/meaning` on a timer. Ask when the screen opens and after a job.
- `newSegments` are the segments that a build embeds. They cost. `estimatedCost` is in USD, from the price in `meaning_index/models.json`.
- `staleSegments` are the segments whose id the index does not hold. After the first pull over a loaded export each message has a new id: every segment is stale and none is new, so the build costs nothing.
- `upperBound` is true when the vectors that a stopped build paid for could not be read. The build then sends at most this much. The app shows "at most" before the cost.
- An estimate and a build use the token counter of `meaning_index` (`default_count_tokens`), so their numbers agree.

Build:

- A build embeds only the segments that are new. It uses the default model of `meaning_index/models.json`.
- A build asks for the key when the first text must be embedded. A build with nothing new needs no key.
- `cancel` is a callable that returns a bool, or a token with `cancelled`.
- `on_progress` gets `{"progress": {"step", "of", "label", "done", "total"}}`. A step that starts, and the end of the build, also give `log`: one line for the job. A line holds counts only.
- The three steps are `Reading your conversations`, `Reading the meaning of your messages` and `Saving the index`.

The refresh of the server:

- After an import the server brings an index that is there up to date by itself. It takes the estimate first.
- `small_enough` says yes when `newSegments` is at most 2,000, or at most 10 percent of the segments, whichever is more (`REFRESH_SEGMENTS`, `REFRESH_SHARE`).
- Above that the server starts nothing. The state is `behind`, and the reader starts the build and sees the cost first.
- A `meaning` job that a reader starts always builds.

A failure is a `MeaningFailure` with a code and one plain sentence:

| Code | Cause |
|---|---|
| `no_key` | No key is set. |
| `key_rejected` | The provider refused the key. |
| `rate_limit` | The provider limits requests, or the account has no credit. |
| `offline` | The provider could not be reached. |
| `model_error` | Any other failure of the embedder. |
| `busy` | Another build holds the folder. |
| `cancelled` | The caller stopped the build. |
| `archive_error` | The archive could not be read. |
| `index_newer` | A newer build of the app wrote the index. |
| `index_error` | The files of the index could not be read or written. |
| `build_failed` | Any other failure, such as a facts file that can not be read. |

- After a failure the index is as it was. The vectors that were paid for are kept for the next build.
- The error of the archive or of the provider is not chained to the failure. The log gets the type of the error, and the HTTP status and the code of the provider.

Tests put a stand-in at `meaning.embedder_for` and at `meaning.default_facts`, or give `store`, `facts`, `embedder_factory` and `count_tokens` to the call.

Known limits:

- The day in a segment is a local day. After a change of the time zone of the Mac some segments have a new day, and a build embeds them again. The refresh of the server is bounded by `small_enough`; above it the reader sees the cost first.
- A `meaning` job can not be stopped. An import that is asked for while it runs is refused with 409 and a plain sentence (`ONE_AT_A_TIME`); the reader starts the import again when it has ended. The same holds for a `digest` job and the `memory` job.
- The first use of the token counter fetches a vocabulary file. When that fails, the count is one token for three bytes.
- The status compares times to the second. A new message in the same second as `coveredUntil` does not make the quick status `behind`.
- An index of an archive file that was moved or deleted stays in `saves/meaning`. Nothing removes it.
- An estimate reads every row of the index a second time, to compare the ids. For 100,000 segments that is about half a second more.

## The two ports of chat

`memory.ports(archive, names)` gives `{"meaning": ..., "digests": ...}` for `ChatEngine`. `chat.py` calls it for each question that uses the model.

| Port | Class | None when |
|---|---|---|
| `meaning` | `MeaningPort` in `meaning_port.py`, over `meaning_index` | No key is set. The archive has no whole index. The folder can not be read. |
| `digests` | `DigestPort` in `memory.py`, over `digests.py` | No key is set. The model of the digest can not be run. The notes of earlier report runs can not be opened. |

With a port that is None the engine does not offer its tool. A reason goes to the log as one line with no private text.

`MeaningPort`:

- It searches the index of the archive of the question (`meaning.index_store(archive.path)`).
- It embeds the description with the model and the dimensions of the index, not with the default model.
- Only the description goes to the provider. The engine masked it. The conversation ids filter the search on this machine.
- The process keeps one loaded index for each folder. It asks `IndexStore.meta()` for each search and loads again only when the build on disk is another one (`IndexMeta.build_id`).
- While a build swaps the files, the folder holds no whole build. The index that was loaded before answers then. A question that finds no loaded index gets no port.
- A load is checked: when the build on disk changed during the load, the load is thrown away and tried again.
- The process keeps the vectors of the last 64 descriptions (`QUERY_VECTORS`). The seeded search and a repeat do not pay twice.
- A hit with the content hash of a better hit of the same conversation is left out: the same words said again. The hash holds no conversation, so the same words in two conversations are two hits.
- A request for a vector has a timeout of 10 seconds (`QUERY_TIMEOUT_SECONDS`). A build has 120 seconds for one request.
- Every failure is raised. The engine then ranks by words alone.

`DigestPort`:

- `estimate` and `digest` pass the conversation and the days to `digests.estimate` and `digests.build`.
- `digest` reads at most `DIGEST_PARTS_LIMIT` (4) parts that are not read yet. It gives that number to `digests.build` as `max_new_periods`. A digest that needs more raises `too_many_parts`. When the check before the first call refuses it, no call is made. The report engine can also stop a digest at the limit after calls that were paid for: the failure then carries their usage, and `usd` and the spend line say what they cost. Do not take a refusal to mean that nothing was paid: look at the usage. `chat.py` gives the engine the same constant (`EngineConfig.digest_parts_limit` in `_events`), so the two can not differ.
- `digest` returns `usd`: the cost of the usage of the run. The log `recall.spend` gets one line with counts and the cost. `log_to_the_server_log` in `ui_server.py` lets that line through.
- A digest that was paid for and failed, or that was stopped, says what it cost too. `digest` takes the usage from the failure of `digests.build` (`ReportFailure.usage`), writes the line `a digest inside a question gave nothing (<code>): N model calls, X USD`, sets `usd` on the failure and raises the same failure again. A failure before any call has no usage: its `usd` is 0.0 and it leaves no line.
- One lock for the archive path, the conversation and the model is held around the build (`digest_lock`). A second question waits and then finds the parts read. A question that is cancelled while it waits stops waiting (`DigestStopped`).
- The model port of a digest inside a question has a timeout of 120 seconds and one more try (`question_llm`, `DIGEST_TIMEOUT_SECONDS`, `DIGEST_RETRIES`). A report job keeps 300 seconds and two more tries.
- `digest` tells its `progress` callback each change of a part (`progress_relay`): a map event of the report engine with the state `start`, `done`, `cached` or `failed` becomes `{"part": index, "of": total, "label": period, "state": state}`. The events `prepare`, `waiting`, `reduce` and `done` of the run are not told.
- One question reads the parts of one conversation one at a time, in order: the carry-over notes of a part go to the next part. `digest` gives `digests.build` no `pool`.
- Nothing here starts a `digest` job.
- The cancel token of chat sets a cancel token of the report engine (`_report_cancel`). The two classes have the same name and are not the same class.
- A `ReportFailure` is raised. The engine then prints its one line for a digest that failed.
- `DIGEST_MODEL` is the model of a digest: `gpt-6-luna`. The digest only sums up the periods. The chat model answers and checks against the messages.
- WARNING: A digest shares the period cache with a report only when the report was run with the same model. A report by another model saves a digest nothing.
- The moments of a period come in the order of the report engine. That is the order of the reply of the model, not the strongest first.

`memory.forget()` lets go of each loaded index and each kept vector.

A test gives `store`, `embedder_factory` and `digest` (`DigestSettings`) to `memory.ports`, or puts its own function at `memory.ports`.

Known limits:

- `/api/ask/stream` sees the reader leave without a write: a thread (`chat-reader-watch`, `watching_the_reader` in `ui_server.py`) waits on the socket with a selector and looks every 0.2 s (`READER_POLL_SECONDS`). When a peek finds the end of the connection, or the connection was reset, it cancels the token of the question. So no part starts after the reader left; the part in flight is paid for. The thread ends with the stream. It sees nothing on a connection that stays open: a proxy that holds the socket, or a client that half-closes and still reads. Bytes that the reader sends during the answer end the watch. In those cases only a failed write sees the close, and the first write after a close succeeds.
- `DigestPort.digest` reads the messages of the conversation from the archive on that worker thread before its first model call (the archive opens one connection for each thread). A cancel does not stop that read: the source looks at the token only before each model call.
- A question reads at most 3 digests, one after another, and each digest reads at most 4 parts that are not read yet, one at a time: at most 12 paid parts in a row. On `gpt-6-luna` a part takes about 20 s, so one digest takes about 80 s and a question about 4 minutes at most when the provider answers. When the provider hangs, one part takes at most two tries of 120 s and one wait of at most 60 s between them (`DIGEST_TIMEOUT_SECONDS`, `DIGEST_RETRIES` and the longest wait of `OpenAIResponsesLLM`): 5 minutes for a part, 20 minutes for a digest, an hour for a question. A cancel, or a reader who leaves, stops this before the next part.
- A question and the `digest` job do not share a lock. A part is paid for twice only when two runs send it at the same moment, before either has saved it; so a question beside a `digest` job pays again for at most the parts of its own limit (4), and two runs with no limit pay again only for the parts whose calls overlap. In the worst case two runs with no limit that move in step each pay for every part, because nothing marks a part as being read. The mechanism: just before a part is sent, its key is looked up one more time (`report_engine/map_step.py`, for a report run too).
- The clock times in a transcript are local times. After a change of the time zone of the Mac the parts of a digest are new, and they are read and paid for again.
- A digest under a scope with days can pay for a period that is then not shown. The app sets no days in a scope today.

## The digest job

`digest_job.run(conversation_id, since, until, on_progress, archive=..., names=...)` reads one conversation for chat, outside a question. The `digest` job of the server calls it.

- It uses `DIGEST_MODEL`, so a question finds the parts in the notes. It has no limit of parts, and the model port of a report.
- The result has `conversation`, `periods`, `periodsBuilt`, `periodsCached`, `periodsFailed`, `usd` and `usage`.
- A listener gets `{"progress": {"step", "of", "label"}}` and a line under `log`, as in a report job. A line holds the label of a part and counts.
- A failure is a `DigestFailure`: a `ReportFailure` with the code of the failure of `digests.build`, its usage, and a sentence of this job (`sentence_of`). No sentence names a report. A missing conversation and a bad day are a ValueError.
- The job starts only when a request asks for it. No question starts it.

The sentences of a `digest` job that failed:

| Code | Sentence |
|---|---|
| `no_key` | No API key is set. Add one in Settings > Archive > API key, then ask again. |
| `key_rejected` | The AI service refused the API key. Check the API key in Settings > Archive > API key, then ask again. |
| `unknown_model` | The AI service does not know the model that reads conversations. The conversation was not read. |
| `rate_limit` | The AI service is limiting requests, or the account has no credit left. Wait a moment, then ask again. |
| `offline` | The AI service could not be reached. Check the internet connection, then ask again. |
| `timeout` | The model took too long to answer. Ask again. |
| `model_error` | The model could not read the conversation. Ask again. |
| `empty` | This conversation has no messages with text in these dates. |
| `too_large` | The model can not read this conversation with its limits. The conversation was not read. |
| `archive_error` | The messages of this conversation could not be read. Update the archive, then ask again. |
| `cache_unreadable` | What earlier runs read could not be read or kept. The conversation was not read. |
| `cancelled` | The reading was stopped. What was read is kept. |
| `run_failed` | The reading failed. What was read is kept: ask again. |
| any other code | The conversation could not be read for chat. Ask again. |

Known limits:

- A `digest` job can not be stopped. An import that is asked for while it runs is refused with 409 and a plain sentence (`ONE_AT_A_TIME`); the reader starts the import again when it has ended.
- A job that failed for a reason that passes, such as no connection, is not started again by a question until an import changed the archive or the server started again. A reader can still start the job.

## Several digests at once

`digests.build` reads the parts of one conversation one at a time: the prompt of a part holds the notes of the part before it (see "Several conversations at once" in `report_engine/README.md`). Several conversations are read at once with one build in each thread and one pool for all of them:

```python
pool = digests.call_pool(model, max_parallel)   # at most max_parallel requests in flight, one budget of tokens
found = digests.build(
    conversation_id, since, until, cancel, on_progress,
    archive=archive, names=names, model=model, pool=pool,
    places=..., llm_factory=..., options=..., counter=..., runtime=..., max_new_periods=...,
)
```

- `pool`: a `report_engine.CallPool` from `call_pool`, with the same `model` as each build. `pool.stop()` stops every build before its next request; the requests in flight end and their parts are kept. With no pool a build has a pool of its own with one slot. `call_pool(..., clock=runtime.clock)` takes the clock of a test runtime.
- `cancel`: a token with `cancelled`. It stops this build only.
- `max_new_periods`: the limit of this build. A request counts when it is sent, and the one that would pass the limit is refused before it is sent.
- `on_progress`: each `report_engine.ProgressEvent`, in the thread of the build. Give each build a function of its own to know its conversation. The events of a part have `step == "map"`, `period` (the label of the part), `index` (its number, from 1) and `total` (the number of parts of the dates), and `state`:
  - `start`: its request is sent now;
  - `done`: it is read and in the notes;
  - `cached`: it was in the notes;
  - `failed`: it gave no result (also a part that is too large, with no `start` before it);
  - `waiting`: the build waits for room in the tokens per minute.
  Before the parts come `prepare` `start` and `prepare` `done` (`total` is the number of parts); after them `done` `done`, when the digest is whole.

## The memory job

The memory of chat is the digest of each conversation with at least `PREPARE_MIN_MESSAGES` (200) messages, read with `DIGEST_MODEL`. `memory_prep.py` reads it ahead of time, so a question finds each part in the notes. `memory_prep_read.py` holds the tasks and the counts, and `memory_prep_words.py` the lines and the sentences.

- `memory_prep.payload(archive, names)` is the answer of `GET /api/memory`. It takes `digests.estimate` for each conversation of the memory. It makes no model call, needs no key and makes no file.
- `memory_prep.run(on_progress, cancel, archive=..., names=...)` is the `memory` job. It reads each conversation that is not read whole, busiest first, `PREPARE_PARALLEL` (4) at a time, with the model port of a report.
- The parts of one conversation are read one at a time: the prompt of a part holds the notes of the part before it. The conversations share one pool of calls (`digests.call_pool`): at most 4 requests are in flight, and they share the tokens per minute of the model.
- A conversation that fails is counted, and the others are read. A failure that each conversation would have stops the job (`FATAL`: `no_key`, `key_rejected`, `unknown_model`, `rate_limit`, `offline`, `cache_unreadable`). No new request is sent; the requests in flight end, and their parts are kept.
- `memory_prep.completed_before(archive_file)` says whether a `memory` job of this archive completed before: its marker is there (`memory_prep_marker.py`, `saves/memory/<16 hex digits>.json`, private to its owner, holding the time and `minMessages`). `prepare_memory` in `ui_server.py` writes it with `mark_completed(archive_file, min_messages)` when the job ends completed. The marker keeps the lower of the threshold it holds and the new one. `memory_prep.marker_threshold(archive_file)` reads it: None for a marker of an earlier build. A question, a report and a `digest` job never write it.
- A `memory` job reads with `minMessages` of its request when it has one (a whole number, 1 or more; anything else is refused with 400), else `PREPARE_MIN_MESSAGES`. The job that the server starts after an import names the threshold of the marker.
- `memory_prep.worth_starting(found, completed_before=...)` says whether the server starts the job by itself after an import: a `memory` job of this archive completed before, a part is not read, and the parts that are not read cost at most `PREPARE_AUTO_USD` (0.25 USD).
- Each end of `run` writes one line to the log `recall.spend` with counts and the cost: `the memory job: 4 parts read now, 0 from the notes, 0 not read, 4 model calls, 0.0123 USD`, or `the memory job stopped (<code>): ...` for a failure. A `MemoryFailure` carries what was spent in `usage`. An error of no known kind in a task is wrapped as `MemoryFailure("run_failed", ...)` with that usage.
- No line, progress label or sentence holds text of a message, a name, a handle, an id or the label of a part.

`GET /api/memory?messagesPath=...` answers:

```json
{
  "state": "partial",
  "model": "gpt-6-luna",
  "minMessages": 200,
  "conversations": 184,
  "conversationsRead": 120,
  "conversationsSkipped": 0,
  "parts": 332,
  "partsRead": 200,
  "partsNotRead": 132,
  "estimatedCost": 0.81,
  "error": null,
  "code": null,
  "jobId": null
}
```

- `state` is `none` when no part is in the notes, `partial`, or `ready` when every part of every conversation of the memory is in the notes. An archive with no conversation of the memory is `none`.
- `conversationsRead` counts the conversations with every part in the notes. `conversationsSkipped` counts the conversations whose parts could not be planned (for example `too_large`); their parts are in no count.
- `estimatedCost` is the cost in USD of the parts that are not read, with `DIGEST_MODEL`.
- `jobId` is the id of the `memory` job that runs, or null.
- When the archive can not be read or the model can not be run, the counts are 0, `estimatedCost` is 0.0, `state` is `none`, and `error` and `code` say why (`archive_error`, or the `code` of the ModelChoiceError). With no archive the route answers as the other routes do.

The `memory` job is asked for as every job: `POST /api/jobs` with `{"action": "memory", "messagesPath": "..."}` (`messagesPath` is optional). The answer is 202 with `{"job": {...}}`, and the job record has `contact` "".

Its `progress`:

```json
{"step": 1, "of": 2, "label": "Looking at your conversations", "done": 0, "total": 184}
{"step": 2, "of": 2, "label": "Reading your conversations", "done": 12, "total": 132}
```

In step 2, `total` is the number of parts that are not read, and `done` counts the parts that ended, read or failed. At the end `done` is `total`.

Its lines, in this order:

```text
Looking at 184 conversations with 200 messages or more
132 parts of 64 conversations are not read yet, 200 are read
A conversation is read: 3 parts read now, 1 from earlier runs, 0 not read (5 of 64)
A conversation could not be read (6 of 64)
The memory is read: 132 parts read now, 200 from earlier runs, 0 not read.
```

There is one line for each conversation that is read, in the order in which they end.

Its `result`:

```json
{
  "conversations": 184,
  "conversationsRead": 183,
  "conversationsFailed": 1,
  "periodsBuilt": 131,
  "periodsCached": 200,
  "periodsFailed": 1,
  "usd": 0.81,
  "usage": {"input_tokens": 2400000, "cached_input_tokens": 0, "output_tokens": 260000, "cost_usd": 0.81, "calls": 131}
}
```

- `conversations` counts the conversations of the memory, skipped ones too. `conversationsFailed` counts the conversations that gave no digest and the skipped ones.
- `periodsCached` counts the parts that were in the notes, also of the conversations that were read whole before. `periodsFailed` counts the parts that failed, and the parts that were not read of a conversation that failed.
- `usd` and `usage` are what the job spent, also on conversations that failed after paid calls.

The job record has `spent`, what the job spent, when it ended completed or failed after the reading started:

```json
{"usd": 0.81, "usage": {"input_tokens": 2400000, "cached_input_tokens": 0, "output_tokens": 260000, "cost_usd": 0.81, "calls": 131}}
```

A job that failed before any call has `{"usd": 0.0, "usage": null}`. A job that failed before the reading started (no archive) has no `spent`.

A refusal is 409 with `{"error": sentence}`:

| Asked for | Held up by | Sentence |
|---|---|---|
| `memory` | `memory` | Recall is reading the memory of chat. Wait until it ends. |
| `memory` | an import | An import is running. Wait until it ends, then read the memory. |
| an import | `memory` | Recall is reading the memory of chat. Wait until it ends. |
| `memory` | `digest` | Recall is reading a conversation for chat. Wait until it ends, then read the memory. |
| `digest` | `memory` | Recall is reading the memory of chat, and this conversation with it. Wait until it ends. |

A `memory` job and a `digest` job read with `DIGEST_MODEL` and no limit of parts: side by side they would pay for each part of one conversation twice. A report, the `meaning` job and chat run beside the `memory` job.

The sentences of a `memory` job that failed:

| Code | Sentence |
|---|---|
| `no_key` | No API key is set. Add one in Settings > Archive > API key, then read the memory again. |
| `key_rejected` | The AI service refused the API key. Check the API key in Settings > Archive > API key, then read the memory again. What was read is kept. |
| `unknown_model` | The AI service does not know the model that reads conversations. What was read is kept. |
| `rate_limit` | The AI service is limiting requests, or the account has no credit left. Wait a moment, then read the memory again. What was read is kept. |
| `offline` | The AI service could not be reached. Check the internet connection, then read the memory again. What was read is kept. |
| `cache_unreadable` | What earlier runs read could not be read or kept. What was read is kept. |
| `archive_error` | The message archive could not be read. Update the archive, then read the memory again. |
| `cancelled` | The reading was stopped. What was read is kept. |

A model that can not be run fails the job with the sentence of its ModelChoiceError, as in the other jobs.

After an import that changed the archive, `refresh_memory` in `ui_server.py` starts one `memory` job when `completed_before` is true and `worth_starting` says yes. With no marker it starts nothing and takes no estimate: the first reading is the choice of the reader. Above the cost it writes one line with the count of parts to the log of the server, and starts nothing.

`GET /api/memory` keeps its counts in the server process for one key (`memory_key` in `ui_server.py`): the signature of the archive, the Contacts cache file, the notes file and its `-wal` file (time and size), `DIGEST_MODEL` and `PREPARE_MIN_MESSAGES`. A part that is saved changes the notes file, so the counts move at the next request; while a `memory` job runs, each request after a saved part works them out again (about 12 s on a large archive). Show the `progress` of the job while it runs. One lock is held while the counts are worked out, so requests that arrive together wait for one answer. `jobId` is read fresh each time.

Known limits:

- A `memory` job can not be stopped from the app. An import that is asked for while it runs is refused with 409 and a plain sentence.
- A report with `DIGEST_MODEL` (the low level) can run beside the `memory` job and pay for the parts of one conversation twice, when both send them at the same moment.
- The job and a question do not share a lock. A part is paid twice only when both send it at the same moment (see "A digest inside a question is bounded" in `CONTRACTS.md`).
- A conversation whose new parts fail for a key or a rate limit, while its other parts are in the notes, gives a digest with failed parts and no failure. The job then goes on to the next conversation; each request is refused, and nothing is paid.

## The people index

The people index answers "which people" questions. It holds one profile for each person and each group, and it is searched by meaning. `people.py` is the service. The steps are in `people_inputs.py` (who, and what the model reads), `people_look.py` and `people_plans.py` (a look that reads no message), `people_figures.py`, `people_prompt.py`, `people_reply.py`, `people_calls.py`, `people_build.py`, `people_store.py`, `people_search.py` and `people_words.py`. `query_vectors.py` keeps the vectors of descriptions for the process.

### What a profile is

- A person is the archive's `person_id` with all of its direct threads (phone and email), busiest first. Its key is `person:<person_id>`. A group conversation is a profile of its own, and so is a direct conversation with no `person_id`. Their key is `conversation:<conversation id>`.
- The label is the name that chat gives the busiest conversation of the profile (`names.chat_labeler`): a saved name, a group title or name, or a mask such as "a number ending in 0171".
- The figures come from the archive only: one stream of the messages of all the conversations of the profile, group events left out (`people_figures.py`). A message with no text is a message.
- The rest comes from one model call: `kind`, `status`, `summary` (at most 600 characters) and `turning_points` (at most 5 of `{date, what}`). The model is the model of the memory (`memory_prep.model()`, `gpt-6-luna`). It reads the period summaries and the moments (title and detail, no quote) of the digests of the profile's own conversations, the figures and the label. No message is read again by a model.
- A profile is made only when every part of every conversation of the profile is in the notes. The notes are read with no model call and no mark of use: a digest with `max_new_periods=0` over a cache that does not mark what it reads. A profile whose notes are not all read is left out of a build and counted in `notRead`; its row from an earlier build stays. A profile with no message with text has no notes: it is not built and not counted. With no notes file nothing is laid out, so such a profile is counted too, as not read.

### The prompt and the reply

- The prompt (`people_prompt.py`) defines each kind and each status in one sentence. A kind says what the relationship was, and the status how it stands: a talking stage can be ongoing. It says that the summary states what the notes show and nothing else, and that a relationship is judged only from its own conversation: what the owner said about a person to someone else is not that relationship.
- The reply is a strict JSON object: `kind` (one of `KINDS`), `status` (one of `STATUSES`), `summary`, and `turning_points` with `date` and `what`. The request asks for no reasoning (`PROFILE_REASONING = "none"`) and at most `MAX_OUTPUT_TOKENS` (1,664).
- A reply in the wrong shape is asked for once more. A second one fails the profile, not the build. A kind or a status that is not one of the words is `unclear`. A group is always `group`, and only a group is one. A turning point whose date is no day or is outside the span of the figures is dropped. Text that is too long is cut at a word.
- The label and the notes of a request, and the text of a reply, are masked by `Planner.mask`: `chat_engine/handle_mask.mask_every_handle` with `names.chat_labeler`, as the chat digest masks a summary. A handle of the archive becomes its saved name or its mask; any other phone number or email address becomes "a phone number" or "an email address". A date, a span of years and a clock time stay.
- Text of a reply loses the line references of the report engine (`report_engine/chat_digest_refs.py`).

### What is paid for again

- Each row holds the hash of its inputs, its fingerprint (`people_plans.profile_fingerprint`): the prompt version, the model, the label, the counts and the first and last times of its conversations (they decide the figures) and the cache keys of its parts (they decide the notes). It is made with no message read. A build calls the model only for a profile whose notes are read and whose fingerprint is not in the store.
- The keys of the parts of each conversation are kept in `plans.sqlite3` beside the store, with a fingerprint of the conversation as the archive lists it (id, kind, name, members, counts, first and last time) and of the settings of the reading (the prompt version of a part, the model, the options, the token counter, and the saved name of each handle of the archive). A look lays out again only a conversation whose fingerprint changed: the way a digest does (`span_plan`), which reads every message of it. A change of the Contacts lays out every conversation again once. The file is a cache: one that can not be read is made again.
- Each row also holds the hash of its embedded text with the embedding model and the dimensions. A row whose text, model or dimensions changed is embedded again. That costs no model call.
- A new message changes the figures, so the profile is written again once its new part is read. A contact that is saved later changes the label and the names in the parts, so the profiles that held it are written again. A handle that only a summary of a part holds does not change a fingerprint.

### The estimate

- `estimatedCost` is for one try of each profile. A reply in the wrong shape is asked once more (`TRIES`, 2), so a build spends at most `TRIES` times the estimate; the server compares `TRIES` times the estimate with `PEOPLE_AUTO_USD` before it builds by itself.
- The input of a profile is an upper bound from the lengths of its notes, with no prompt built and no token counted (`people_plans.input_bound`): the fixed text, 600 characters for the figures, the label and the frames, and for each part 80 characters, its summary, and 20 characters with the title and the detail of each moment, also a moment that the check later drops. At 3 characters for one token (English text has about 4), this is not below the real input. Text in another script can have fewer characters for one token: a known limit.
- The output of a profile is sized from the limits of the reply: the fullest reply that the limits allow (`fullest_reply`: the longest kind and status, a summary of 600 characters, 5 turning points of 160 characters) is 1,661 characters of JSON. At 4 characters for one token that is `OUTPUT_TOKENS` = 416. The request asks for no reasoning, so no hidden output is added. A real reply is shorter, so this part is an upper bound. The limit of the request (`MAX_OUTPUT_TOKENS`) is not used for the estimate: the estimate of the memory used the limit of 4,000 tokens for a part, where the real mean was about 1,230, and said 4.60 USD for a run that cost 1.87.
- The embedding of a new profile counts `OUTPUT_TOKENS` tokens at the price of the embedding model. A row that is only embedded again counts its text at 4 characters for one token.

### The store

- `saves/people/<16 hex digits of the SHA-256 of the resolved path of the archive file>/people.sqlite3` (`people_store.people_folder`), and `plans.sqlite3` beside it (conversation ids and keys of parts, no text). The folder is 0700 and the files 0600.
- One row for each profile: key, conversation ids, label, figures, kind, status, summary, turning points, input hash, text hash and vector (float32, length 1, or none until it is embedded). The table `meta` holds the format, the model, the embedding model, the dimensions, the time of the build and the time of the last build that completed.
- A write makes a new file and puts it in place with one rename. A write that fails leaves the old file. A store of a newer format is not read and not replaced (`store_newer`).
- A build that stops (a failure that each profile would have, or a cancel) saves the profiles that it paid for with no vector, and raises. The next build embeds them and pays nothing for them again.

### `people.search`

```python
people.search(archive, names, description, *, kinds=None, since=None, until=None, limit=10,
              conversations=None, store=None, embedder_factory=None) -> people.Found
```

- `Found` is a `list` of rows, best first, with three attributes:
  - `reason`: None when the search ran; `no_index` with no profile or a store that can not be read; `not_embedded` when the store has profiles but none of those asked for has a vector yet (an embedding failed: the payload counts them in `toEmbed`); `no_key` or `embed_failed` when the description could not be embedded.
  - `not_searched`: the profiles asked for that have no vector yet. When some have one, those are searched, `reason` is None, and `not_searched` says how many were left out.
  - `index_missing`: `reason == "no_index"`.
- The description is embedded with the embedding model and the dimensions of the store, and the profiles are ranked by cosine. The caller masks the description first.
- The vector of a description is kept in `query_vectors.py`: one cache for the process, 64 entries, keyed by the model, the dimensions and the SHA-256 of the text, which the search by meaning (`meaning_port.py`) uses too. A question's text is embedded once. A search waits 5 s at most for the embedder (`SEARCH_TIMEOUT_SECONDS`).
- `kinds` keeps the profiles of those kinds. `conversations` keeps the profiles that hold at least one of those conversation ids (the scope of a question). `since` and `until` keep the profiles whose span (first to last day) overlaps them.
- It raises ValueError only for arguments that can not be used: an empty description, a kind that is not in `people.KINDS`, `conversations` that is not a list of ids, a day that is not `YYYY-MM-DD`, `since` after `until`, a `limit` that is not a whole number from 1 to 50. Nothing else raises.

```python
query_vectors.query_vector(text, model, dimensions, factory) -> tuple[float, ...]   # factory(model, dimensions) -> embedder
query_vectors.forget() -> None
```

A row:

```json
{
  "key": "person:12",
  "conversations": ["+12025550143", "mina@example.com"],
  "label": "Mina",
  "kind": "talking_stage",
  "status": "faded",
  "summary": "Coffee dates and a valentine in early 2023, then it faded.",
  "turning_points": [{"date": "2023-02-14", "what": "You sent a valentine."}],
  "figures": {
    "first_day": "2023-01-02",
    "last_day": "2023-02-20",
    "messages": 8,
    "sent": 4,
    "received": 4,
    "active_days": 4,
    "longest_gap_days": 39.45,
    "busiest_month": "2023-01",
    "wrote_first_share": 0.5
  },
  "score": 0.4123
}
```

`people.status(archive, names)` reads the store only:

```json
{"state": "ready", "profiles": 1180, "builtAt": "2026-11-18T10:00:00+00:00", "model": "gpt-6-luna",
 "embedding": {"model": "text-embedding-3-large", "dimensions": 1024}}
```

With no profile it is `{"state": "none", "profiles": 0, "builtAt": null, "model": null, "embedding": null}`.

### `GET /api/people` and the `people` job

`GET /api/people?messagesPath=...` answers `people.payload` with `jobId` (`people_answer` in `ui_server.py`):

```json
{
  "state": "partial",
  "model": "gpt-6-luna",
  "people": 1200,
  "profiles": 1180,
  "current": 1170,
  "toWrite": 12,
  "toEmbed": 0,
  "notRead": 18,
  "estimatedCost": 0.0061,
  "builtAt": "2026-11-18T10:00:00+00:00",
  "error": null,
  "code": null,
  "jobId": null
}
```

- `people` counts the profiles with text. `profiles` counts the rows of the store. `current` counts the profiles whose row has the hash of their inputs now. `toWrite` counts the profiles that a build would write now, and `estimatedCost` is their cost for one try with the embeddings. `toEmbed` counts the rows that are kept but have no vector for the embedding model and its dimensions (an embedding that failed); a build embeds them with no model call. `notRead` counts the profiles whose notes are not all read.
- `state` is `none` with no row, `ready` when nothing is to write, to embed or left out, and `partial` otherwise.
- When the store or the archive can not be read, or the model can not be run, the counts are 0, `state` is `none`, and `error` and `code` say why. It makes no model call and needs no key. It reads no message of a conversation whose plan is kept, and writes only the plans that it made.
- Time on an invented archive of 545,000 messages in 300 conversations (`review-g/payload_bench.py`): 0.01 s with no notes file; 0.02 s when the notes hold every part and the plans are kept; 23.4 s the first time after the notes were read, when every conversation is laid out once. The server does that first look after a `memory` job (`refresh_people`), so a request finds the answer.
- The answer is kept in the server process for one key (`people_counts`): the signature of the archive, the Contacts cache file, the count of parts in the notes (`people.notes_mark`: a chat digest or a report that only marks a part as used does not change it; a new part does), the store file and the model. `jobId` is read fresh.

The job is asked for as every job: `POST /api/jobs` with `{"action": "people", "messagesPath": "..."}` (`messagesPath` is optional). The answer is 202 with `{"job": {...}}`, and the record has `contact` "".

Its `progress`:

```json
{"step": 1, "of": 3, "label": "Looking at the people in your conversations", "done": 40, "total": 1204}
{"step": 2, "of": 3, "label": "Writing a profile of each person", "done": 5, "total": 12}
{"step": 3, "of": 3, "label": "Saving the people index", "done": 1, "total": 1}
```

Its lines:

```text
Looking at 1,200 people and groups
12 profiles to write, 1,170 kept as they are, 18 left out because their memory is not read
The people index is saved: 12 profiles written now, 1,170 kept, 0 failed, 18 left out, 2 removed.
```

Its `result`, and its record under `spent` as `{"usd", "usage"}`, also when it failed after paid calls:

```json
{
  "people": 1200,
  "built": 12,
  "kept": 1170,
  "failed": 0,
  "notRead": 18,
  "removed": 2,
  "embedded": 12,
  "embedTokens": 1630,
  "usd": 0.0049,
  "usage": {"input_tokens": 30000, "cached_input_tokens": 0, "output_tokens": 3100, "cost_usd": 0.0046, "calls": 12}
}
```

- `failed` counts the profiles whose replies were in the wrong shape twice, or whose call failed on its own. Their rows from an earlier build stay. `removed` counts the rows of profiles that the archive no longer holds, or that have no text now.
- `usage` is what the model calls spent. `usd` adds the embeddings, counted with the token counter of the index of meanings.
- Each end of a build writes one line to the log `recall.spend`: `the people job: 12 profiles written, 1170 kept, 0 failed, 12 model calls, 0.0049 USD`, or `the people job stopped (<code>): ...`.

A refusal is 409 with `{"error": sentence}`:

| Asked for | Held up by | Sentence |
|---|---|---|
| `people` | `people` | The people index is being built. Wait until it ends. |
| `people` | an import | An import is running. Wait until it ends, then build the people index. |
| an import | `people` | The people index is being built. Wait until it ends. |
| `people` | `memory` | Recall is reading the memory of chat. Wait until it ends, then build the people index. |

A `memory` job is not refused by a `people` job: a profile whose notes the memory job reads later is written by the build that starts after it. A report, the `meaning` job, a `digest` job and chat run beside a `people` job.

The sentences of a `people` job that failed (`code` is the first column):

| Code | Sentence |
|---|---|
| `no_key` | No API key is set. Add one in Settings > Archive > API key, then build the people index again. |
| `key_rejected` | The AI service refused the API key. Check the API key in Settings > Archive > API key, then build the people index again. The profiles that were written are kept. |
| `unknown_model` | The AI service does not know the model that writes the profiles. The profiles that were written are kept. |
| `rate_limit` | The AI service is limiting requests, or the account has no credit left. Wait a moment, then build the people index again. The profiles that were written are kept. |
| `offline` | The AI service could not be reached. Check the internet connection, then build the people index again. The profiles that were written are kept. |
| `embed_failed` | The profiles could not be made searchable. The profiles that were written are kept. Build the people index again. |
| `archive_error` | The message archive could not be read. Update the archive, then build the people index again. |
| `cache_unreadable` | What earlier runs read could not be read. Nothing was written. |
| `store_error` | The people index could not be read or written. Check the free space on the disk, then build the people index again. |
| `store_newer` | A newer version of Recall made the people index. Update Recall, then try again. |
| `cancelled` | The build was stopped. The profiles that were written are kept. |
| `run_failed` | The people index could not be built. The profiles that were written are kept. |

`no_key`, `key_rejected`, `unknown_model`, `rate_limit` and `offline` stop the build: no new call starts, the calls in flight end, and what they paid for is saved. A failure of the embedder has the same codes.

After a `memory` job that completed, or an import that changed the archive, `refresh_people` in `ui_server.py` works the answer of the route out (`people_counts`). It starts one `people` job when a build of this archive completed before (`people.completed_before`: `completed_at` in the store) and `people.worth_starting` says yes: a profile is to write or to embed, and `TRIES` times the estimate is at most `PEOPLE_AUTO_USD` (0.10 USD). After an import that started a `memory` job, it waits: the end of that job does it. With no completed build it starts nothing: the first build is the choice of the reader. Above the cost it writes one line with the count of profiles to the log of the server.

Known limits:

- A `people` job can not be stopped from the app, and an import is refused while it runs. A first build of about 1,400 profiles makes about 1,400 calls, 4 at a time. At about 4 s a call that is about 23 minutes of calls. The reading of the notes and the figures runs in the same tasks: with the fake model, a first build of 300 profiles of 545,000 messages took 73 s (the look 23 s, then about 0.25 s a profile). So a first build takes about 25 minutes. A later build writes only the profiles that changed.
- `ongoing` and `faded` are judged from the notes, not from the date of the end of the archive: that date would change the hash of every profile at each import.
- A `memory` job that ends while a `people` job runs does not start another one (the second is refused). The next `memory` job or import does.
- The first look after the notes were read, and the first look after a change of the Contacts, lay out every conversation (23 s on the archive above).

## First run

A new person has no key and no index. The app asks for the key, shows what getting chat ready costs, and starts one job that builds the three things that chat needs. After that job, the builds that the server starts by itself after an import work for the person as they work for the owner.

### The key: `/api/key`

`GET /api/key` answers where the key is. It never gives the key or a part of it:

```json
{"present": true, "source": "file", "place": "Settings > Archive > API key"}
```

- `source` is `environment` (the key is set where the server starts), `file` (the file `.env` of the data folder) or `null` (no key).
- `POST /api/key` with `{"key": "sk-..."}` saves the key in the file. The answer is the same shape. A key is text, trimmed, 20 to 300 characters, starts with `sk-`, and holds only `A-Z`, `a-z`, `0-9`, `-` and `_`.
- `DELETE /api/key` removes only the line of the key from the file. The answer is the same shape.
- The next question, report or job uses the new key. The server does not start again: each model client reads `keys.api_key()` at the call.

The file:

- It is written whole to a temporary file in the same folder, then put in place with one rename. Its mode is 0600. The data folder is made when it is missing.
- Every other line stays as it is, byte for byte. The first line of the key is replaced in its place, and any later line of the key is dropped. With no line of the key, the line is added at the end.

The answers:

| Status | When | `error` |
|---|---|---|
| 200 | the key is saved or removed, or the state is asked for | none: the shape above |
| 400 | the key is not text | The key must be text. |
| 400 | the key does not have the shape of a key | This is not an API key. A key starts with sk- and has 20 to 300 letters, digits, - or _. |
| 403 | the request comes from another origin | Forbidden: only the page of this server may ask |
| 409 | `OPENAI_API_KEY` is set in the environment | An API key is set where the server starts, and that key is used. A key saved here would not be used. |
| 500 | the file can not be written | The API key could not be saved. Check the free space on the disk, then try again. |

WARNING: These three routes, and every `POST` and `DELETE` under `/api/`, have a stricter check than a `GET`. A request that sends an `Origin` header must come from the exact origin of this server: `http://` and the `Host` of the request, with its port (`is_own_write` in `ui_server.py`, `key_store.same_origin`). A request with no `Origin` passes (the Mac app, curl). A page of another local port is refused with 403: it could send a simple `text/plain` POST with no preflight and start a paid job (`first_run`, `memory`, `people`, `meaning`, `digest`) or ask a question. A `GET` other than `/api/key` keeps the check of `is_local_api_request`.

WARNING: The Vite dev server is a page of another local port. Its proxy with `changeOrigin: true` changes `Host` to the target, but it does not change `Origin`, so each write of the dev UI gets 403. The proxy of `/api` in `app/vite.config.ts` must give the origin of the target to a request that names the origin of the dev server:

```ts
configure: (proxy) => proxy.on('proxyReq', (req) => { if (req.getHeader('origin') === 'http://127.0.0.1:5173') req.setHeader('origin', target) })
```

Here `target` is the string of the proxy target (for example `http://127.0.0.1:8765`). A dev server on `localhost` sends `http://localhost:5173`: name that origin too.

No sentence, no log line, no job record and no traceback holds the key. A fault of the disk is told in one sentence, and its cause is dropped.

### `GET /api/first-run`

`GET /api/first-run?messagesPath=...` answers what getting chat ready costs and what is ready (`first_run_answer` in `ui_server.py`, `first_run.estimate`). It makes no model call and no request, and needs no key:

```json
{
  "key": {"present": true, "source": "file"},
  "archive": {"holdsMessages": true, "conversations": 1404, "messages": 550000},
  "steps": [
    {"name": "meaning", "state": "none", "estimatedCost": 0.85, "error": null, "code": null},
    {"name": "memory", "state": "none", "estimatedCost": 4.60, "error": null, "code": null},
    {"name": "people", "state": "none", "estimatedCost": 1.26, "error": null, "code": null}
  ],
  "estimatedCost": 6.71,
  "ready": false,
  "jobId": null
}
```

- The steps are always in this order: `meaning`, `memory`, `people`.
- `meaning`: the state and the cost of `GET /api/meaning`. `state` is `none`, `behind` or `ready`.
- `memory`: the state and the cost of `GET /api/memory` for every conversation (`MEMORY_MIN_MESSAGES` is 1). `state` is `none`, `partial` or `ready`.
- `people`: `none`, `partial` or `ready`.
  - Before the memory is read, the people index has nothing to measure. The cost is an upper bound from the plan of the memory (`first_run.people_bound`): for each profile, the parts of its conversations, each at most `first_run.notes_tokens_per_part` (the summary and the moments that the limits of `report_engine.digest` allow, or the output limit of a part when that is smaller), the frame of the request, the fullest reply of a profile (`OUTPUT_TOKENS`) and its embedding, times `TRIES`.
  - After the memory is read, the cost is `TRIES` times `estimatedCost` of `GET /api/people`.
  - The step is `ready` only when a build of this archive completed before and nothing is left to write or to embed. A profile whose memory can not be read does not hold it back.
- `error` and `code` of a step say why its cost could not be worked out. Its cost is then 0.
- `estimatedCost` is the sum of the steps that are not `ready`. `ready` is true when all three steps are `ready`.
- With no message in the archive, or no archive, `holdsMessages` is false, each step is `none` with cost 0, and `ready` is false.
- `key` has `present` and `source`, as `GET /api/key` gives them.
- `jobId` is the `first_run` job that runs now, read fresh.
- The answer is kept in the server process for one key (`first_run_counts`): the signature of the archive, the Contacts cache file, the notes file and its `-wal` file, the people store file, the id of the build of the index of meanings (`first_run.index_mark`) and the model of the memory. Each step that a job builds changes one of them, so a request after a job finds a fresh answer.
- WARNING: While a job writes the notes, each part that it saves changes the key, so each request plans every step again. While a `first_run` job is unfinished, the app polls `GET /api/jobs/<id>`, not `GET /api/first-run`. It asks `GET /api/first-run` again when the job has ended.

### The `first_run` job

`POST /api/jobs` with `{"action": "first_run", "messagesPath": "..."}` (`messagesPath` is optional). The answer is 202 with `{"job": {...}}`, and the record has `contact` "".

The job builds, in order, only the steps that are not `ready` (`get_chat_ready` in `ui_server.py`, `first_run_job.run`):

1. the index of meanings, with the work of the `meaning` job (`build_meaning_index`);
2. the memory of every conversation, with the work of the `memory` job (`prepare_memory` with `min_messages=1`). It writes the marker of the memory;
3. the people index, with the work of the `people` job (`build_people_index`).

- A step that is `ready` is skipped with one line. A memory that is read already gets its marker.
- After a `first_run` job that completed, `meaning.built`, `memory_prep.completed_before` and `people.completed_before` are true for its archive. So the builds after a later import start as they start for the owner.
- The marker of the memory saves the threshold of the reading (`minMessages`): 1 after a first run. `refresh_memory` plans and starts the `memory` job after an import with that threshold, so a message added to a small conversation is read, its profile is written again, and chat stays ready. A marker of an earlier build has no threshold: the reading uses `PREPARE_MIN_MESSAGES`. The marker keeps the lower of the threshold it holds and the new one, so a `memory` job with the threshold of the app does not take back what a first run read.
- A step that fails ends the job failed, with the sentence and the code of that step's own job. The steps that ended before stay saved: a new `first_run` job starts at the first step that is not ready.
- `spent` (`{"usd", "usage"}`) holds what all steps spent, also when a step failed. The index of meanings tells no money: its tokens are priced at the price of the embedding model.
- With an archive that holds no message the job fails with `no_messages`: The archive holds no message yet. Read your messages first, then get chat ready.

Its `progress`, with the label of each step's own job:

```json
{"step": 1, "of": 3, "label": "Reading the meaning of your messages", "done": 120, "total": 900}
{"step": 2, "of": 3, "label": "Reading your conversations", "done": 40, "total": 2300}
{"step": 3, "of": 3, "label": "Writing a profile of each person", "done": 5, "total": 1400}
```

Its lines are the lines of each step's job, and one line for each step that is skipped:

```text
The index for the search by meaning is ready: nothing to build.
The memory of chat is read: nothing to read.
The people index is ready: nothing to write.
```

Its `result` holds counts and money only. A step that was skipped is `null`:

```json
{
  "meaning": {"segments": 52000, "embedded": 52000, "reused": 0, "removed": 0, "tokens": 9100000},
  "memory": {"conversations": 1404, "conversationsRead": 1404, "conversationsFailed": 0, "periodsBuilt": 2300, "periodsCached": 0, "periodsFailed": 0, "usd": 4.4, "usage": {}},
  "people": {"people": 1200, "built": 1200, "kept": 0, "failed": 0, "notRead": 0, "removed": 0, "embedded": 1200, "embedTokens": 160000, "usd": 1.1, "usage": {}},
  "skipped": [],
  "usd": 6.2
}
```

A refusal is 409 with `{"error": sentence}`, and no job is made:

| Asked for | Held up by | Sentence |
|---|---|---|
| `first_run` | no key | No API key is set. Add one in Settings > Archive > API key, then get chat ready again. |
| `first_run` | an import | An import is running. Wait until it ends, then get chat ready. |
| `first_run` | `meaning` | The index for the search by meaning is being built. Wait until it ends. |
| `first_run` | `memory` | Recall is reading the memory of chat. Wait until it ends. |
| `first_run` | `digest` | Recall is reading a conversation for chat. Wait until it ends. |
| `first_run` | `people` | The people index is being built. Wait until it ends. |
| an import, `meaning`, `memory`, `digest`, `people`, `first_run` | `first_run` | Recall is getting chat ready. Wait until it ends. |

A report and chat run beside a `first_run` job.

Known limits:

- A `first_run` job can not be stopped from the app. On a large archive it takes as long as its three jobs together: the people step alone is about 25 minutes for 1,400 profiles.
- A job works the estimate out when it starts, if no request did it before (about as long as the three routes together).
- The upper bound of the people index is above what a build pays: each part counts the most notes that the limits of a digest allow. The app shows it as "at most".
- The bound counts 3 characters for one token, as `people_plans.input_bound` does. Text in other scripts (Chinese, Japanese, Korean, emoji) can have more tokens than that, and a build can then pay more than the bound.

## Reports

- The report file and the events file change together. When one write fails, the files of the older report stay.
- `cache_unreadable`: the notes of earlier runs can not be opened. A run stops before any call. An estimate counts every part.
- `estimate` reads the notes of earlier runs. A cache of the preview must have the file `report-cache.sqlite3` in its key, or the estimate must stay outside that cache.
- `saved_report` finds the saved report of a report file by its text. A report of the older kind has none: read its events file as today.
- Two conversation ids that differ only in characters such as `.` and `@` share one file name, as today.

## Message ids

`messageId` in a payload is the id of the archive.

- After a pull it is Apple's guid.
- In an archive that the bridge loaded it is `legacy:<ROWID>`.
- The export of today holds the ROWID as a number, and the browser saved that number.

`messages.find_saved` reads all three forms. Give it the saved conversation and the saved time too: a rebuilt Messages database numbers its rows again.

## Models

`models.json` holds each model: id, provider, label, level, limits, prices and the day its facts were read.

- `null` means that a fact was not read. No number is made up.
- A model that is not offered can not be run.
- `/api/defaults`, every route that takes a model and `app/src/lib/models.ts` must change in one step. The ids of today are not offered.
- The tokenizer family of the 6 series is an assumption. The report engine then adds its safety margin of 1.35 to every token count.

## Differences from the server at `89ab504`

Each one has a test that states it.

- A tapback is not a message. It is in no search result, no day and no total.
- `reactions_frame` holds the six classic kinds of tapback. A tapback with any emoji or a sticker is no row.
- A message that was unsent is a row of the archive with no text. It counts in the list, in a day and in a search with no words. `messages_frame` has no row for it. `tests/recall_services/test_frame_edges.py` states this open seam.
- `around` of a day must be written as `YYYY-MM-DD HH:MM` or longer. Other forms give the start of the day.
- The route reads the last word as a prefix unless the request says `prefix=false`: then `piz` does not find `pizza`, and `piz*` does. The service matches whole words unless `as_you_type=True`.
- A search with no words and no filter gives nothing. With a filter it lists the messages of the filter.
- `sender` of an own message is `""`. The export held the handle of the other person there.
- `senderName` is set only for a received message in a group.
- A chat answer has no `terms`, and no keyword answer takes the place of an answer that failed.
- A report run writes no HTML page. It removes an older page of the same conversation.

## Tests

```bash
venv/bin/python -m pytest tests/recall_services tests/integration --cov=recall_services --cov-report=term-missing
```

`tests/integration/test_chat_memory.py` runs whole questions through `chat.py` with the two ports on a real archive. `tests/test_meaning_routes.py` runs `GET /api/meaning` and the `meaning` job.

The tests use invented data, a scripted model and a dummy key. `tests/guards.py` blocks every socket that leaves the machine and points the home folder and the data folder at temp folders.

No Python linter and no type checker is configured in this repository.
