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

`keys.py` gives the key of the model provider: `OPENAI_API_KEY` in the environment, or in the file `.env` in the data folder, read at each call.

## Functions

| Route or use | Call |
|---|---|
| `/api/defaults` (models) | `models.defaults_payload()` |
| `/api/contacts` | `conversations.payload(archive, saved, limit, messages_path)` |
| `/api/search` | `search.payload(archive, saved, query, contact, limit, offset, as_you_type=True)` |
| `/api/day` | `day.payload(archive, saved, contact, day, around, offset, limit)` |
| `/api/ask/stream` | `chat.stream(question, contact, model, history, cancel, archive=..., names=..., use_ai=...)` |
| `/api/ask` | `chat.ask(...)`, the same arguments |
| `/api/preview` (estimate) | `reports.estimate(contact, model, since, until, archive=..., names=...)` |
| job `analyze` | `reports.run(contact, model, since, until, on_progress, cancel, archive=..., names=...)` |
| `/api/meaning` | `meaning.payload(archive, names)` |
| job `meaning` | `meaning.build(archive, names, cancel, on_progress)` |
| the refresh after an import | `meaning.built(archive_file)`, `meaning.estimate(archive, names)`, `meaning.small_enough(estimate)` |
| job `digest` | `digest_job.run(conversation, since, until, on_progress, archive=..., names=...)` |
| the two optional ports of the chat engine | `memory.ports(archive, names, start_digest=...)` |
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

## Chat

- A "user" row of the history can hold `contact`: the conversation that its question was asked of.
- A row with no `contact` was asked of everything. Its answer does not go to a question that has a scope.
- A question over 8,000 characters is the error `question_too_long`.
- `cancel` is a `CancelToken` that the route makes for each request.
- The engine gets the two ports of `memory.ports(archive, names, start_digest=...)`. See "The two ports of chat".
- `chat.stream(..., start_digest=...)` takes the function of the server that reads a long conversation outside the question.

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
- A `meaning` job can not be stopped, and an import waits for it.
- The first use of the token counter fetches a vocabulary file. When that fails, the count is one token for three bytes.
- The status compares times to the second. A new message in the same second as `coveredUntil` does not make the quick status `behind`.
- An index of an archive file that was moved or deleted stays in `saves/meaning`. Nothing removes it.
- An estimate reads every row of the index a second time, to compare the ids. For 100,000 segments that is about half a second more.

## The two ports of chat

`memory.ports(archive, names, start_digest=None)` gives `{"meaning": ..., "digests": ...}` for `ChatEngine`. `chat.py` calls it for each question that uses the model.

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
- `digest` reads at most `DIGEST_PARTS_LIMIT` (2) parts that are not read yet. It gives that number to `digests.build` as `max_new_periods`. A digest that needs more raises `too_many_parts`, and no call is made. `chat.py` gives the engine the same constant (`_digest_limits`), so the two can not differ.
- `digest` returns `usd`: the cost of the usage of the run. The log `recall.spend` gets one line with counts and the cost. `log_to_the_server_log` in `ui_server.py` lets that line through.
- One lock for the archive path, the conversation and the model is held around the build (`digest_lock`). A second question waits and then finds the parts read. A question that is cancelled while it waits stops waiting (`DigestStopped`).
- The model port of a digest inside a question has a timeout of 120 seconds and one more try (`question_llm`, `DIGEST_TIMEOUT_SECONDS`, `DIGEST_RETRIES`). A report job keeps 300 seconds and two more tries.
- `prepare` calls the `start_digest` function of the server and returns its answer. With none given, or when it fails, the answer is False.
- The cancel token of chat sets a cancel token of the report engine (`_report_cancel`). The two classes have the same name and are not the same class.
- A `ReportFailure` is raised. The engine then prints its one line for a digest that failed.
- `DIGEST_MODEL` is the model of a digest: `gpt-6-luna`. The digest only sums up the periods. The chat model answers and checks against the messages.
- WARNING: A digest shares the period cache with a report only when the report was run with the same model. A report by another model saves a digest nothing.
- The moments of a period come in the order of the report engine. That is the order of the reply of the model, not the strongest first.

`memory.forget()` lets go of each loaded index and each kept vector.

A test gives `store`, `embedder_factory` and `digest` (`DigestSettings`) to `memory.ports`, or puts its own function at `memory.ports`.

Known limits:

- A reader who leaves during a digest inside a question does not stop it: the turn gives no event while a tool runs. It reads 2 parts at most.
- The `digest` job does not hold the lock of a question. A question and the job can pay for the same part, 2 parts at most.
- The clock times in a transcript are local times. After a change of the time zone of the Mac the parts of a digest are new, and they are read and paid for again.
- A digest under a scope with days can pay for a period that is then not shown. The app sets no days in a scope today.

## The digest job

`digest_job.run(conversation_id, since, until, on_progress, archive=..., names=...)` reads one conversation for chat, outside a question. The `digest` job of the server calls it.

- It uses `DIGEST_MODEL`, so a later question finds the parts in the notes. It has no limit of parts, and the model port of a report.
- The result has `conversation`, `periods`, `periodsBuilt`, `periodsCached`, `periodsFailed`, `usd` and `usage`.
- A listener gets `{"progress": {"step", "of", "label"}}` and a line under `log`, as in a report job. A line holds the label of a part and counts.
- A failure is a `ReportFailure` with the sentences of a report run. A missing conversation and a bad day are a ValueError.
- `digest_starter` in `ui_server.py` makes the `start_digest` function of one question. It starts a `digest` job for the archive of the question.

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
- A search matches whole words. `piz` does not find `pizza`; `piz*` does. `as_you_type=True` makes the last word a prefix.
- A search with no words and no conversation gives nothing.
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
