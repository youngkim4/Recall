# Technical overview

This document explains how Recall works inside. Read the [README](README.md) first: it shows the features, the privacy list and the stack. This document goes one level deeper, part by part. The rules that must hold across files are in [CONTRACTS.md](CONTRACTS.md).

## Contents

1. [Processes and parts](#processes-and-parts)
2. [The data folder](#the-data-folder)
3. [Import](#import)
4. [The archive query API](#the-archive-query-api)
5. [Services](#services)
6. [The local server](#the-local-server)
7. [Background jobs](#background-jobs)
8. [Models and the API key](#models-and-the-api-key)
9. [Chat](#chat)
10. [Meaning search](#meaning-search)
11. [Memory and the people index](#memory-and-the-people-index)
12. [Reports](#reports)
13. [First run](#first-run)
14. [The Mac shell](#the-mac-shell)
15. [Builds](#builds)
16. [The frontend](#the-frontend)
17. [Tests](#tests)
18. [Older files](#older-files)

## Processes and parts

Recall runs as two processes on one Mac:

1. The **Mac shell**, a Swift app. It starts the server, then shows the app in a `WKWebView` window.
2. The **local server**, a Python process (`ui_server.py`). It owns the archive, the indexes and the background jobs. It listens only on `127.0.0.1`.

The React app runs inside the window. It talks to the server only through `/api/*` routes.

| Part | Folder | Job |
|---|---|---|
| Importer | `archive/` | Mirrors Apple's `chat.db` into Recall's own SQLite archive |
| Query API | `archive/` | Reads the archive: messages, days, search, counts, habits |
| Services | `recall_services/` | One module for each group of routes; turns archive rows into JSON |
| Server | `ui_server.py` | Routes, request checks, background jobs, static files |
| Chat | `chat_engine/` | The agent loop, its tools, citation checks and masking |
| Meaning search | `meaning_index/` | Vectors of conversation segments, kept on disk |
| Reports | `report_engine/` | Map-reduce reports on one conversation, with a cache |
| Mac shell | `macos/Recall/` | The window and the server process |
| App | `app/` | The React interface |

Only chat, meaning search, memory, the people index and reports send data to the model provider. Each of them needs your API key.

## The data folder

`recall_paths.py` names every place that Recall writes to. All parts read these constants. No part builds its own path to the data folder.

`DATA_DIR` is found in this order:

1. The environment variable `RECALL_DATA_DIR`, when it is set.
2. `~/Library/Application Support/Recall`, when the server runs from the packaged app (`RECALL_BUNDLED=1` or a frozen build).
3. The repository folder, in development.

The layout:

```
DATA_DIR/
  .env                         your API key (mode 0600)
  messages.csv                 older export, read only by the bridge
  out/                         files that a report run writes
  saves/
    archive.sqlite3            the archive
    archive.sqlite3.source.json  which chat.db was read
    .archive.sqlite3.building    present while an import runs
    .archive.sqlite3.import-work/  private copy of chat.db for one import
    contact_names.json         names from Contacts
    Recall Contacts Exporter.app  helper that reads Contacts
    setup.json                 setup state
    recall_store.sqlite3       preview and analysis cache
    report-cache.sqlite3       cache of report periods
    reports/                   saved reports
    meaning/<hash>/            one meaning index for each archive
    memory/<hash>.json         marker: the memory of this archive is read
    people/<hash>/             one people index for each archive
```

The archive and its folders are created with mode 0600 for files and 0700 for folders.

## Import

The importer is in `archive/`. Its entry point is `import_chat_db` in `archive/importer.py`.

### The source snapshot

The importer never reads `chat.db` in place. It copies `chat.db` and its `-wal` and `-journal` files into the private work folder, then reads the copy. The Messages app can keep writing while an import runs.

### Full pull and incremental run

The importer has two modes.

A **full pull** makes the archive an exact mirror of the source:

1. It loads every source message guid into a temporary table.
2. It counts the messages that the pull would remove. If the share is more than `DEFAULT_MAX_REMOVED_SHARE` (0.2), it stops with `RemovalLimitError` before it changes anything. This protects the archive from a wrong or empty `chat.db`.
3. It removes messages, attachments and reactions whose guid is not in the source.
4. It writes new and changed rows, replaces the members of each conversation, and removes people and handles that nothing uses.
5. It counts the totals of every conversation again.

An **incremental run** removes nothing. It starts from a mark and reads forward:

- The mark comes from `import_runs.checkpoints`. Each run stores the 8 highest source `ROWID`s with their guids.
- A mark is trusted only when the source still has the same guid at that `ROWID` (`runs.verified_mark`). Apple can reuse a `ROWID` after a reset.
- The run reads again from `mark - OVERLAP_ROWS` (2,000 rows).

Both modes also read older rows that were edited or unsent (`date_edited` or `date_retracted` is set).

The app always asks for a full pull (`recall_services/archive_state.py`). The incremental mode is used by `scripts/archive_scale_check.py`.

One import runs at a time. The importer takes the write lock with `BEGIN IMMEDIATE`, and a busy archive gives `ImportBusyError`. The whole run is one transaction. A failure or a cancel rolls back all of it, and the run is recorded in `import_runs` with its status.

### Message text

Newer versions of macOS leave the `text` column empty and store the text in the `attributedBody` blob. `archive/typedstream.py` decodes that blob:

1. A fast path reads the known layouts.
2. Otherwise a parser walks the archived objects: the attributed string, its string, then its bytes.
3. If the parser fails, a scan looks for the byte marker of the string.

Each message records where its text came from in `text_source`: `column`, `blob`, `export` or `none`. A blob that cannot be decoded gives `none`, and the importer counts it.

### Ids

Apple's message `guid` is the only stable id. The archive keeps it in `messages.guid` (unique) and writes by guid. A second import updates rows in place and writes only the rows that changed.

`messages.id` is the archive's own id. All routes, citations and indexes use it. `source_rowid` keeps the source `ROWID` of the last pull.

### Reactions

A tapback is not a message. `archive/tapbacks.py` reads `associated_message_type`:

- `2000 + n` adds reaction kind `n`. `3000 + n` removes it.
- The kinds, in order: love, like, dislike, laugh, emphasis, question, emoji, sticker.
- Other codes from 2000 to 3999 are skipped and counted.

The target comes from `associated_message_guid`, after its part prefix. A reaction is stored only when its target message exists in the archive. Reactions that wait for their target are settled at the end of the run.

### People, handles and names

- **Handles.** `archive/handles.py` gives each handle a kind (phone, email or other) and a match key. A phone key uses the last 10 digits. An email key is lower case.
- **People.** Contacts enter the import as plain data: a list of records with a name, phones and emails. `archive/people.py` groups handles into people. One record is one person. Two records with the same name stay two people, unless they share a phone or an email. Person ids stay stable across imports.
- **Contacts.** The server does not read the Contacts database. A small Swift helper app exports the names into `saves/contact_names.json`. The services read that file through `recall_services/names.py` (`Names`), which reloads it only when the file changes.
- **Group names.** A group with no title gets a name from its members (`group_names.py`).

### Schema and full-text search

`archive/schema.py` defines the schema. All tables are `STRICT`.

| Table | Holds |
|---|---|
| `import_runs` | Each run: mode, status, counts, checkpoints |
| `people` | One row for each person |
| `handles` | Phone numbers, emails and other ids, with their person |
| `conversations` | One row for each chat, with totals and first and last dates |
| `conversation_members` | Handles in each conversation |
| `messages` | Text, sender, time, kind, reply target, edit and unsend state |
| `attachments` | Type, name, size and path of each attachment |
| `reactions` | Tapbacks, with their target guid and part |

Search uses an FTS5 table, `messages_fts`. It is an external-content table over `messages.search_text`, which is the first 10,000 characters of the text. Three triggers keep it in step with `messages` on insert, delete and update. The tokenizer is `unicode61` with diacritics removed and with symbols, punctuation and emoji as separators.

### Versions

The schema version is in `PRAGMA user_version`. `archive/migrate.py` holds a list of steps that only grows. When the importer opens the archive to write, it applies the missing steps in one transaction. A reader checks the version and the layout first. An archive from a newer version of Recall gives `ArchiveVersionError`, and Recall does not change it.

## The archive query API

`archive.Archive` (`archive/queries.py`) is the only way to read the archive.

- `Archive.open(path, readonly=True)` opens it. It keeps one connection for each thread. A read connection sets `query_only`. The writer uses WAL mode.
- Reads: `summary`, `conversations`, `conversation`, `people`, `person`, `messages`, `iter_messages`, `message`, `day`, `context`, `reactions`.
- Figures (`archive/query_figures.py`): `search`, `count`, `habits`, `stats`.

Every expected failure is a subclass of `ArchiveError`. Examples: `ArchiveNotFoundError`, `ArchiveVersionError`, `ImportBusyError`, `RemovalLimitError`. The services turn each one into a plain sentence and a fixed `code`.

## Services

`recall_services/` holds one module for each group of routes. A route handler in the server parses the request, then calls one function of one service. The function returns the JSON of the route. Examples:

- `GET /api/day` calls `day.payload(archive, names, ...)`.
- `GET /api/search` calls `search.answer(archive, names, query)`.
- `GET /api/stats` calls `stats.answer(...)`.

The modules, by purpose:

| Purpose | Modules |
|---|---|
| The archive and its files | `archive_state`, `places`, `archive_files`, `files`, `outcomes`, `unreadable` |
| Names and inputs | `names`, `inputs`, `messages`, `frame` |
| Read routes | `conversations`, `day`, `search`, `search_filters`, `stats`, `stats_time`, `habits` |
| Chat | `chat`, `memory`, `digests`, `digest_job`, `meaning_port`, `people_port`, `query_vectors` |
| Meaning search | `meaning`, `meaning_failures` |
| Memory | `memory_prep`, `memory_prep_marker`, `memory_prep_read`, `memory_prep_words` |
| People index | `people` and the `people_*` modules |
| First run | `first_run`, `first_run_job` |
| Reports | `reports`, `report_files`, `report_progress`, `report_failures` |
| Keys and models | `keys`, `key_store`, `models` |

`recall_services/archive_state.py` keeps one read handle of the archive for the whole server. It also starts each pull.

A failure that the person sees is one plain sentence in `error`, with a fixed word in `code`. The app shows the sentence and acts on the code. Sentences never name the model provider; a test checks every sentence.

## The local server

`ui_server.py` is a standard-library HTTP server with one thread for each request.

### Binding

- Host: `RECALL_UI_HOST`, default `127.0.0.1`.
- Port: `RECALL_UI_PORT`, default `8765`. The Mac shell always sets it.
- When it runs from the packaged app, the server writes its log to `~/Library/Logs/Recall/RecallBackend.log`. It stops when the shell process stops.

### Request checks

- Every `GET /api/*` must be local: the `Host` must be `localhost`, `127.0.0.1` or `::1`, and an `Origin`, when present, must be a local `http://` origin.
- Every `POST` and `DELETE` must also come from this server's own page (`is_own_write`). When a request names an `Origin`, it must equal `http://` plus the `Host`. A page on another local port cannot write.
- `/api/key` applies the stricter check to every method, `GET` too.
- `OPTIONS` always gives 403. The server sends no CORS headers.
- A request body can be at most 2,000,000 bytes.

### Routes

| Group | Routes |
|---|---|
| Setup | `GET /api/defaults`, `/api/setup/status`, `/api/permissions/fulldisk`, `/api/first-run`, `/api/first-words`; `POST /api/setup/complete` |
| Archive | `GET /api/overview`, `/api/memories`, `/api/contacts`, `/api/preview`, `/api/day`, `/api/search`, `/api/stats` |
| Names | `POST /api/contact-names` |
| Indexes | `GET /api/meaning`, `/api/memory`, `/api/people` |
| Jobs | `GET /api/jobs`, `/api/jobs/<id>`; `POST /api/jobs` |
| Chat | `POST /api/ask`, `/api/ask/stream`, `/api/ask/cancel` |
| Reports | `GET /api/reports`, `/api/report`, `/api/analysis` |
| Key | `GET`, `POST` and `DELETE /api/key` |
| Cache | `POST /api/cache/preview` |

`/api/ask/stream` sends server-sent events: each frame is `data: <json>` and a blank line. When the reader leaves, the server cancels the question.

### Static files

Other paths serve the app. The server serves `app/dist` when it holds a build. Otherwise it serves the older static interface in `ui/`. `RECALL_UI_FRONTEND` can force one of them. A path with no file extension gets `index.html`, so the app can route on its own. Every response has `Cache-Control: no-store`.

## Background jobs

Long work runs as a job in a thread of the server: imports, index builds and reports.

### Actions

`JOB_ACTIONS`: `analyze`, `export`, `setup_import`, `bridge`, `meaning`, `digest`, `memory`, `people`, `first_run`.

- `setup_import`, `export` and `bridge` fill the archive.
- `meaning` builds the meaning index.
- `digest` reads one conversation for chat.
- `memory` reads the memory of every conversation.
- `people` writes the people index.
- `first_run` does what chat needs, in order (see [First run](#first-run)).
- `analyze` writes a report.

### Start, poll and end

1. The app sends `POST /api/jobs` with `{"action": ..., ...}`. The server checks the request, then answers 202 with the job.
2. The app polls `GET /api/jobs/<id>`. A job has `status`, `progress`, `logs` (the last 300 lines), `result`, `error` and `code`.
3. The status goes from `queued` to `running`, then to `completed` or `failed`.

No route cancels a job. A job runs until it ends. Only a chat question can be cancelled (`POST /api/ask/cancel`).

Jobs are kept in memory only. The server keeps about 60 of them and removes the oldest ended ones.

### One at a time

`ONE_AT_A_TIME` lists which jobs cannot run together. The server checks it under `JOBS_LOCK` before it adds a job, and refuses with 409 and a sentence. In short:

- One import at a time.
- An index job does not run during an import or during a second job of its own kind.
- `digest` and `memory` do not run together. `people` waits for `memory`.
- `first_run` runs alone.

### After a job

`after_job` keeps the indexes current after a change:

- After an import that changed the archive, `refresh_meaning_index` updates an existing meaning index when the update is small.
- `refresh_memory` reads new conversations when a memory job completed before and the cost is at most `PREPARE_AUTO_USD`.
- `refresh_people` updates the people index when a people job completed before and the cost is at most `PEOPLE_AUTO_USD`.

These refreshes never start an index that the person did not build first.

## Models and the API key

### The key

`recall_services/keys.py` finds the key at each call:

1. The environment variable that `KEY_NAME` names, when it is set. It wins.
2. Otherwise the same line in `DATA_DIR/.env`.

The server reads only that one line. It never writes the key to the environment, and no log line or error holds it. `recall_services/key_store.py` writes the line (see [First run](#first-run)).

### Model facts

Model ids, prices and limits are not in the code. They are in two JSON files:

- `recall_services/models.json`: the models for chat, digests, memory, the people index and reports.
- `meaning_index/models.json`: the embedding models for meaning search.

Each model has a `level` (`low`, `medium`, `high` or `top`) and an `offered` flag. The app shows levels, not ids. `recall_services/models.py` reads the file once and checks each choice. An unknown or not-offered model, or a bad reasoning level, gives a `ModelChoiceError` with its own `code`. The `chat` block sets the output cap and the time limit for each reasoning effort.

### Requests

Each engine has its own small model client. All of them:

- send requests with storage turned off;
- retry a failed call a few times with a growing wait, and honour `retry-after`;
- retry a streamed call only before its first token, so an answer is never sent twice.

The meaning index embeds in batches and retries each batch on its own.

## Chat

Chat is in `chat_engine/`. `recall_services/chat.py` wires it to the archive, the names and the memory.

### The agent loop

`ChatEngine` (`chat_engine/engine.py`) answers one question as a `QuestionTurn` (`chat_engine/turn.py`):

1. It sends the status "Reading your archive".
2. It runs a few seeded look-ups that most questions need (`chat_engine/seed.py`).
3. It calls the model with the tools. The model calls tools, reads the results and calls again.
4. The turn ends when the model answers with no tool call, or when a limit is reached. At a limit, the engine adds a note and makes one last call with no tools, so the model must answer.

The limits are in `EngineConfig`:

| Limit | Value |
|---|---|
| Rounds with tools | 8, then one round that must answer |
| Tool calls in one round | 8 |
| Tokens of one tool result | 3,000 |
| Tokens of all tool results | 30,000 |
| Tokens of earlier turns | 6,000 |
| Characters of one message | 4,000 |
| Characters of a question | 4,000 |
| Digests in one question | 3, each under a cost limit |

A repeated tool call gets a short note, not a second result.

### Tools

`chat_engine/tool_specs.py` gives a tool to the model only when its port is present.

| Tool | What it does |
|---|---|
| `search_messages` | Word search with context. With the meaning port, it also searches by meaning. |
| `read_around` | Reads the messages before and after one message |
| `count_messages` | Exact counts, with an optional group |
| `list_conversations` | Finds conversations or people by name or by the last four digits |
| `conversation_stats` | Totals, active days, the busiest day, the longest gap |
| `conversation_digest` | The story of one conversation, period by period |
| `find_people` | Finds people by the kind of relationship and its story |

A search by meaning joins the meaning ranking and the word ranking with reciprocal rank fusion (`chat_engine/fusion.py`). If the meaning search fails, the word ranking is used alone.

### Citations

The model cites messages as `[mN]`. `CitationStream` (`chat_engine/citations.py`) numbers them `[1]`, `[2]` and so on while the answer streams. It numbers a reference only when the turn read that message and the message is in the scope of the question. It removes other references from the text and lists them as `invalid_references`.

`chat_engine/verify.py` checks each quote of 12 or more characters against the messages that the turn read. It folds case and quote marks first. A quote with an ellipsis must match its parts in order. A quote that does not match is listed in `unverified_quotes`. The answer keeps its text; the app marks the span.

### Masking

The model never sees a phone number or an email that you have not saved as a name. `MaskingLabeler` (`chat_engine/labeler.py`) replaces each one before a message leaves the Mac:

- "an unsaved contact", "an unsaved email contact";
- "a number ending in" and its last four digits;
- "the short code" and its digits;
- "a group chat", with its saved members.

Digest summaries get a second pass that masks any phone number or email left in the text (`chat_engine/handle_mask.py`). The answer keeps the masks. The citations show names, because the app builds them again from the archive with your saved names.

### Memory ports

Chat can use three optional ports (`chat_engine/ports.py`):

- `MeaningSearch` finds segments by meaning.
- `DigestSource` estimates and reads the digest of one conversation.
- `PeopleSearch` finds people in the people index.

`recall_services/memory.py` builds the first two. `recall_services/people_port.py` builds the third. With no API key, the ports are absent and chat uses only the archive tools.

## Meaning search

Meaning search is in `meaning_index/`.

### Segments

A segment is a run of messages from one conversation (`meaning_index/segments.py`). A new segment starts:

- after a gap of more than 120 minutes;
- at 16 messages;
- at 1,600 characters.

Each line is cut at 400 characters and starts with "Me" or "Them". No name goes into a segment. The id of a segment is a hash of its conversation and message ids. A second hash covers its text, the model and the dimensions; a new build reuses each vector whose hash it already has.

### The store

Each archive has its own index folder with three files:

- `segments.sqlite3`: the segments and the build record;
- `vectors.npy`: one row of unit length for each segment, in `float16`;
- `meta.json`: the format version, the `build_id`, the model, the dimensions and the date that the index covers.

All three files name the same `build_id`. `work.sqlite3` keeps the vectors that a stopped build already paid for, and `build.lock` stops two builds at once.

### Search

The server embeds the query with the model and the dimensions of the index. It keeps the last 64 query vectors. The search takes the dot product of the query with every row, in chunks, and keeps the best rows. Filters are the conversation and the dates.

### The safe swap

A new build never breaks a reader (`meaning_index/store_swap.py`):

1. The build writes its three files as `.new`.
2. The swap links each old file to `.old`, then replaces the files in a fixed order. If a step fails, the swap puts the old files back.
3. A reader accepts the files only when all three name the same `build_id`. Otherwise it reads the `.old` set.
4. At the next build, `recover` repairs a swap that stopped half way.

The server keeps the loaded index in memory and loads it again only when the `build_id` changes.

## Memory and the people index

**Memory** is a digest of each conversation: a summary and the moments of each period. The `memory` job (`recall_services/memory_prep.py`) reads every conversation with at least `PREPARE_MIN_MESSAGES` messages, four at a time. It writes the digests to the report cache, so chat finds them already read. A marker file records that the memory of an archive is read, with the lowest message count that it covered.

The **people index** (`recall_services/people.py`) holds one profile for each person or group. A profile is written from the memory of that person's conversations. A reply in the wrong shape is asked again once (`TRIES` = 2). The profiles are embedded, so chat can find people by the kind of relationship.

## Reports

Reports are in `report_engine/`. A report reads one conversation in four steps: prepare, map, select, reduce (`report_engine/pipeline.py`).

1. **Periods.** `plan_periods` packs calendar months, oldest first, under a token budget (80,000 by default). A month that is too large is split by days, and a day by lines. A period that is closed never changes, so its cache entry stays good.
2. **Map.** One model call for each period, oldest first. Each call gets the notes of the earlier periods (at most 4,000 tokens). The reply gives a summary and at most 8 moments. A period that fails is skipped.
3. **Checks.** `check_moments` checks each moment. Its date must be a day in the period; if not, it moves to the earliest cited day or the moment is dropped. Its quote must be in a cited message or in a message at most 3 days away. A quote that is not found is removed.
4. **Moments.** `select_moments` removes near duplicates, drops weak moments and spreads at most 40 across the whole conversation.
5. **Reduce.** One call writes four sections: the arc, the dynamics, the defining moments and the big picture. If the input is too large, `fit_reduce` first shortens the period lines, then drops the weakest moments.

### The cache

`PeriodCache` keeps the reply of each period in `saves/report-cache.sqlite3`. The key is a hash of the prompt version, the model, the labels and the text of the period. A second report, or a chat digest, pays only for periods it has not read.

### The estimate

`estimate` plans the same periods before a run. It skips cached periods. It counts each map call with its expected output, adds a margin for the tokenizer, and always counts the reduce call. `report_engine/pricing.py` prices the tokens from the model facts, with cached input and long input priced on their own.

### Errors

Every failure is a typed error from `report_engine/errors.py`. Examples: `EmptyConversationError`, `AllPeriodsFailed`, `ReduceError`, `ReportCancelled`. `recall_services/report_failures.py` turns each one into a sentence and a code.

## First run

Chat is ready when three steps are built, in this order:

1. `meaning`: the meaning index;
2. `memory`: the memory of every conversation;
3. `people`: the people index.

### The key routes

`/api/key` (`recall_services/key_store.py`):

- `GET` gives `present`, `source` (`environment`, `file` or none) and `place`. It never gives the key or a part of it.
- `POST` checks the shape of the key and writes it to `DATA_DIR/.env`. It writes a temporary file in the same folder with mode 0600, then moves it in place with one rename. Every other line of the file stays as it was. When the environment already sets a key, the server refuses with 409, because a saved key would not be used.
- `DELETE` removes only the key line.

### The estimate

`GET /api/first-run` (`recall_services/first_run.py`) gives the state and the cost of each step, and the sum of the steps that are not ready. It makes no model call and needs no key.

The people step cannot be measured before the memory is read. Its cost is then an upper bound (`people_bound`): each profile is counted with the largest notes that the digest limits allow and the largest reply, times `TRIES`. The app rounds the sum up and shows it as "at most".

### The job

`POST /api/jobs` with `{"action": "first_run"}` starts the job. The server refuses it with 409 when no key is set. The job (`recall_services/first_run_job.py`) builds each step that is not ready, in order, and adds up what it spends. A step that fails ends the job, and the steps that were done are kept.

The job does not stop at the estimate on its own. The estimate is built to be higher than the real cost, and `tests/test_first_run.py` checks that the spend stays under it.

## The Mac shell

The shell is one Swift file, `macos/Recall/RecallApp.swift`.

### Starting the server

The shell has two modes:

- **Packaged.** The app holds a frozen server at `Contents/Helpers/RecallServer.app`. The shell starts it with the data folder `~/Library/Application Support/Recall` and sets `RECALL_DATA_DIR`, `RECALL_BUNDLED=1`, `RECALL_UI_HOST` and `RECALL_UI_PORT`.
- **Development.** The shell runs `ui_server.py` from a checkout of the repository. It finds the checkout from `RECALL_APP_ROOT`, or from the `RecallRoot.path` file in the app. It finds Python from `RECALL_PYTHON`, then `venv/bin/python` in the checkout, then `/usr/bin/python3`. The server log goes to `~/Library/Logs/Recall/RecallBackend.log`.

In both modes the shell passes the path of the Contacts helper in `RECALL_CONTACTS_EXPORTER_APP`.

### The port

The shell picks a free port from 49152 to 65535 and remembers it. The packaged app and the development app remember different ports. The next launch uses the same port when it is free. This matters because the web view keeps saved state for each origin, and the port is part of the origin.

### Loading the app

The shell asks `GET /api/defaults` every 0.25 seconds, for about 60 seconds. When the server answers, the window loads `http://127.0.0.1:<port>/?app=mac`. If the server does not answer, the window shows an error page that points to the log.

### The bridge

The app calls the shell through the `recall` message handler. Its commands:

- open the Full Disk Access settings;
- pick a database;
- relaunch the app;
- save a page as PDF;
- play a haptic tick;
- move or zoom the window.
 The shell answers through `window.__recallBridge`. It also tells the app when the window becomes active again.

When the app quits, the shell stops the server.

## Builds

### Development build

`scripts/build_macos_app.sh` builds `saves/Recall.app`:

1. It builds the React app (`npm run build`).
2. It compiles the Swift shell.
3. It writes the path of the repository into `RecallRoot.path`, so the app runs the server from this checkout.
4. It builds the Contacts helper and the icon.

### Release build

`scripts/build_release_app.sh` builds `dist/Recall.app`, which runs on a Mac with no Python:

1. It builds the React app.
2. It freezes the server with PyInstaller (`packaging/recall_server.spec`, entry `packaging/server_entry.py`). The bundle holds `app/dist`, `ui/`, both `models.json` files and the tokenizer vocabularies, so the server starts with no network.
3. It compiles the shell, then puts `RecallServer.app` in `Contents/Helpers`.
4. It builds the Contacts helper and the icon.
5. It stops if a message export, a `chat.db` or a `.env` file is in the bundle.
6. It signs from the inside out with `RECALL_SIGN_IDENTITY`, then checks the signature.

With no identity, the signature is ad hoc. macOS then blocks the first launch until the person allows it in System Settings.

## The frontend

The app is React and TypeScript, built with Vite.

- `app/src/components/` holds one folder for each screen (`overview`, `insights`, `search`, `chat`, `reports`, `cards`, `settings`, `onboarding`, `firstrun` and more) and shared widgets in `ui/`.
- `app/src/lib/` holds the logic and the hooks, kept apart from the components so that tests can run them in Node.
- `app/src/lib/api.ts` makes every server call through one `recallApi` object. A refusal becomes an `ApiError` with the HTTP status, the `code` and the sentence of the server.
- The styles are plain CSS in cascade layers: `reset`, `base`, `controls`, `shell`, `components`, `views`. Layouts use container queries.
- The theme is the `data-palette` attribute on the root (Graphite is the default; the scene themes are Broadsheet, Shoreline, Canopy and Sky). Light and dark is the separate `data-theme` attribute. Colours are `light-dark()` pairs.
- In development, Vite serves the app on `127.0.0.1:5173` and sends `/api` to the server. It rewrites the `Origin` of its own requests to the server's origin, because the server accepts writes only from its own origin.

`CONTRACTS.md` holds the rules of the interface.

## Tests

### Run them

```bash
venv/bin/python -m pytest -q
```

```bash
cd app && npx tsc -b && npx eslint . --quiet && npm test
```

The frontend tests use Node's test runner on the compiled `*.test.ts` files.

### Guards

`tests/guards.py` runs before any project import (`tests/conftest.py`). It:

- blocks every socket connection to a host that is not the loopback, and blocks name look-ups;
- blocks the helper programs `open`, `osascript`, `swiftc`, `xcrun` and `codesign`, so no test starts the Contacts helper;
- sets a fake API key, points `RECALL_DATA_DIR` at a temporary folder and `HOME` at an empty folder;
- stops `.env` files from loading.

A fixture applies the guards again before each test, and checks that an outbound connection fails. A test that reads names gets a Contacts stub that fails if it is called.

### Invented data

No test reads real messages. Builders make Messages databases and archives from invented data:

| Builder | Makes |
|---|---|
| `tests/archive/builders.py` (`ChatDbBuilder`) | A `chat.db` with handles, chats, messages, tapbacks and attachments |
| `tests/archive/synthetic.py` | A large `chat.db` for scale checks |
| `tests/archive/worlds.py` | Small named archives for import tests |
| `tests/invented_db.py` | A Messages database for the server tests |
| `tests/recall_services/world.py`, `report_world.py` | Archives for the services and reports |
| `tests/chat_engine/tiny.py` | A tiny archive for chat |
| `tests/report_engine/synth.py` | Conversations for the report engine |
| `tests/integration/engine_data.py` | Data for the seams between the engines |

The model is a fake in every test. No test sends a request to the model provider.

## Older files

Some files at the root come from the first version of Recall, which exported messages to CSV and wrote reports from the command line.

- Still used by the server: `parse_imessage.py` (the conversation list before an archive exists), `conversation.py`, `contact_names.py`, `group_names.py`, `group_people.py`, `local_store.py`, `message_text.py`, `safe_html.py`, `setup_status.py`, `day_view.py`, and the older interface in `ui/`.
- Used only by the command line (`cli.py`): `analysis.py`, `ai_analysis.py`, `html_report.py`, `reports.py` and the root model client.
- Used by nothing: `merge_export.py`.

The `bridge` job reads an older CSV export into the archive. New installs do not need it.
