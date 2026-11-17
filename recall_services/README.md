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
- `report_labeler(record)`: the labeler for a report. It gives `""` for a sender who is not saved.

A handle matches a record as the archive matches people: a phone number by its last 10 digits, an address in lower case.

A group that nobody named gets a name from its saved people. Two differences from the list of today:

- A member who never wrote is a person of the group. The export knew only the writers.
- A person with two handles stands where the busier handle stands. The export rule added both.

## Chat

- A "user" row of the history can hold `contact`: the conversation that its question was asked of.
- A row with no `contact` was asked of everything. Its answer does not go to a question that has a scope.
- A question over 8,000 characters is the error `question_too_long`.
- `cancel` is a `CancelToken` that the route makes for each request.

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

The tests use invented data, a scripted model and a dummy key. `tests/guards.py` blocks every socket that leaves the machine and points the home folder and the data folder at temp folders.

No Python linter and no type checker is configured in this repository.
