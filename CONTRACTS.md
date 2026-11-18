# Contracts

This file lists the rules that more than one file must obey.
A change to one side of a rule must keep the other side true.
Each rule names the code that holds it.
Add a rule here when a review finds a new one.

## Dates

- A date range includes all of its first day and all of its last day.
  - Server: `within_dates` in `conversation.py`.
  - Client: `countBetween` in `app/src/lib/days.ts`.
- A range that covers the whole conversation is sent as no dates. `tidyRange` in `app/src/lib/days.ts` does this.
- A timestamp is local time with no zone, from the export to the screen. `read_times` in `conversation.py` reads the times of a file. A time that cannot be read is no time: the server sends null for it, and `dayOfMessage` in `app/src/lib/dayParts.ts` gives no day for it.
- Calendar arithmetic on the client uses UTC. `app/src/lib/days.ts`, `months.ts`, `activity.ts` and `heat.ts` obey this. `timezones.test.ts` checks it in nine zones.
- The chosen dates belong to one conversation in one archive. `App.tsx` stores the dates with the archive path and the chat id. Dates stored for a different conversation are no dates.

## Figures

- `daily` in a preview covers the whole conversation. It feeds the calendar.
- `activity`, `stats`, `monthly`, `dynamics`, attachments and reactions follow the chosen dates.
- A month or a day with no messages keeps its place on a time axis. `fill_months` and `day_points` in `conversation.py` do this.
- A message of an export frame has one of three directions: sent, received, or unknown. The archive holds two: every message is sent or received.
  - `total` counts all three.
  - `sent` counts sent only. `received` counts received only.
  - Never compute received as `total` minus `sent`.
  - Server: `sent_mask` and `received_mask` in `conversation.py`. Client: `receivedCount` in `app/src/lib/conversations.ts`.
- `isFromMe` in a message is `true`, `false` or `null`. `messages.payload` in `recall_services/messages.py` writes it for a search result, a day and a citation; `sent_by_me` in `ui_server.py` writes it for the figures. `isOutbound` in `app/src/lib/conversations.ts` reads it, and also accepts the older forms that saved chats hold.
- The `direction` of a first message is `outgoing`, `incoming`, or any other value, which means unknown. Only `sentByMe` in `app/src/lib/direction.ts` reads it.
- `voicesOf` in `app/src/lib/conversations.ts` gives what each side wrote in a conversation of the list, or null when the list does not say. The list read from the Messages database before there is an archive has totals only. A row with no voices draws one bar in the neutral color.
- Rhythm rows start on Monday. `hour_by_weekday` writes them. `Rhythm.tsx` maps them to the week of the reader.
- Heat levels use one scale for the whole archive. The busiest count is always level 4. `heatSteps` and `heatLevel` in `app/src/lib/heat.ts` hold this.
- Search: `total` is the number of matches. `count` is the number of matches that the response holds, from the place `offset`. The screen shows `total`. `recall_services/search.py` states what a query matches: whole words in any order, a quoted phrase, and the last word as a prefix while the reader types.

## Names

- The server sends a saved name, a group name made from saved names, or an empty string.
- Only one function on each side labels a conversation that has no name.
  - Client: `conversationName` in `app/src/lib/conversations.ts`.
  - Server, for text that the user reads: `conversation_label` in `conversation.py` (the saved report of the hand tool uses it).
  - The two functions must use the same words.
- No screen shows a full telephone number.
- Text that goes to the AI model never holds a handle that is not saved. Chat uses `names.chat_labeler`: a saved name or a mask. A report uses `names.report_labeler` for the writers: a saved name, or a name that the report engine gives each unsaved writer. Both mask a handle that is not saved where it stands in the text of a message (`MaskingLabeler.redact`; `_Masked` in `recall_services/reports.py` for a report). The hand tool uses `contact_label` in `conversation.py`, and `tests/test_prompt_privacy.py` checks its prompts.
- The title of a saved HTML report is `conversation_label`. The file name and the heading of the Markdown report keep the conversation id.
- A group id is `chat` followed by digits only. Three places hold this rule and must change together:
  - `is_group_chat` and `group_chat_mask` in `conversation.py`.
  - `isGroupChat` in `app/src/lib/conversations.ts`.
  - The SQL in `list_contacts` in `parse_imessage.py`.
- Names follow the Contacts cache. The messages bundle signature and `preview_cache_key` include the signature of the Contacts cache file. See "The archive and the services", Names.

## A day of a conversation

- `/api/day` gives one day of one conversation. `Archive.day` decides which messages; `day.payload` in `recall_services/day.py` gives each message the shape that search results have (`messages.payload`). `day_window` in `day_view.py` is the same rule on a frame, and `tests/integration/test_day_against_day_view.py` holds the two together.
- The conversation id must match exactly. This route never matches part of an id. On a frame, `in_conversation` in `conversation.py` holds the rule, and `filter_conversation` uses it too.
- A message with no conversation id belongs to no conversation. It is in no day and in no moment.
- A place given as `offset` is used as it is, from 0 to the total. The answer can hold fewer rows than `limit`. A `limit` below 1 is refused; a `limit` above 400 is 400.
- The command line can name a conversation by a part of its id. `find_conversation` in `conversation.py` does this, and it refuses a part that several ids hold.
- A day longer than the limit is given in part. `total` is the number of messages in the day and `offset` is the place of the first message given.
- `previousDay` and `nextDay` are the nearest days that hold messages, or null.
- A moment in `/api/memories` carries `day`, and `previewAt` and `previewId` when it quotes a message. `previewAt` is written in the same form as `timestamp` in a message of `/api/day`. `previewId` is the `message_id` of the frame (`message_id_text`), which is the id that `messageId` holds. A moment of a report carries the same as `at` and `pickedId`.
- The `count` of an on-this-day moment is every message of that day in that conversation. It is the same number as `total` in `/api/day`.
- The client finds the quoted message with `quotedIndex` in `app/src/lib/dayParts.ts`: first by `previewId` equal to `messageId`, then by `previewAt` equal to `timestamp`.
- The client reads a long day in parts with `partToAsk`: it asks for the rows that are there and no more. `joinPart` adds a part only when it meets what is shown.
- `DayReader.tsx` in `app/src/components/ui` shows a day: the arrows, the messages, the parts, the place that the reader left each day at. `DaySheet.tsx` is the dialog around it. Search shows the same reader in a pane.
- `.dayview-scroll` has `overflow-anchor: none`. The reader puts the messages in place itself.
- An on-this-day card takes its day from `day`. It does not work the day out from the clock of the client.

## Reports

- The report heading carries the conversation id. `report_contact` in `ui_server.py` reads it back.
- The first line of the Overview section is the scope line. `report_engine/markdown.py` writes it (`reports.py` for the hand tool). `report_scope` in `ui_server.py` reads it first, and uses the First / Last line for older reports.
  - `- Scope: 2024-06-01 to 2024-06-30`
  - `- Scope: from 2024-06-01`
  - `- Scope: through 2024-06-30`
  - `- Scope: all dates`
- A saved report shows figures for the days in its scope.
- A run that finds no events removes the older events file. `write` in `recall_services/report_files.py` does this (`write_events_timeline` in `analysis.py` for the hand tool).
- The steps of an import have one order on the server (`import_messages`) and in the guide (`PHASES` in `ImportStep.tsx`): names, reading, archive. `tests/test_server_faults.py` holds the two together.
- The app shows an analysis only with the report file that it was read from. `App.tsx` keeps the file path with the analysis.
- An analysis answer with no `analysis` is a failure. A job watch that has ended changes nothing.

## Story HTML

WARNING: The story of a report is written by an AI model from the messages of other people. Treat it as hostile input.

- One function turns a piece of a story into HTML: `story_html` in `safe_html.py`.
- The saved HTML report of the hand tool uses `story_html`. The app page uses `story_page`, which cuts the story where a chapter starts (a line that begins with `### `) and gives each chapter to the same renderer. A story with more than `STORY_PARTS` chapters is one piece. Do not render a story in any other way.
- The result holds only the tags in `ALLOWED_TAGS`.
- The result holds only three attributes: `href` and `rel` on a link, and `start` on a numbered list.
- A link address starts with `http://`, `https://` or `mailto:`. Each link has `rel="noopener noreferrer"`.
- `story_html` limits its input: the length of the text (`STORY_LIMIT`), a run of one mark, and a run of empty lines.
- `story_html` limits its work: an estimate of the work before the renderer starts (`STORY_WORK_LIMIT`) and the processor time of the renderer (`STORY_SECONDS`). A piece past a limit, or nested too deep, is shown as plain paragraphs. The limits hold for each chapter of `story_page`, all the chapters of one page share `STORY_PAGE_SECONDS`, and the whole story is cut to `STORY_LIMIT` first. A chapter is cut at its line whatever stands around it (a code fence too): each piece is made safe by itself. `story_page` says whether a chapter is shown plain, and `/api/analysis` gives that as `shownPlain`.
- A picture becomes its caption.
- The client puts this HTML into the page in one place: `ReportStory.tsx` in `app/src/components/reports`.
- `readStory` in `app/src/lib/story.ts` cuts the story into an opening and chapters. It moves whole elements and changes heading levels only. It never makes markup from text.
- Do not add a markdown extension that lets text set an attribute.

## Report files

- `/api/report` serves these file names only: `analysis_*.md`, `analysis_*.html`, `events_timeline_*.csv`.
- The path is resolved before the name is checked.
- Each response has `X-Content-Type-Options: nosniff` and a fixed content type.
- An HTML response has `Content-Security-Policy: sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:`. No script runs in a saved report page, and the page fetches nothing. Do not add `allow-scripts` or `allow-same-origin`.
- `html_report.py` writes a page with no script and no outside file. The chart is an SVG in the page.

## Paper

- A report as paper is a state of the page: `<html data-paper>`. `usePaper.ts` in `app/src/components/reports` sets it and removes it.
- `[data-screen-only]` is not shown on paper. `[data-paper-only]` is shown on paper only.
- `app/src/styles/paper.css` holds the paper state and the print rules. A layout rule for the screen must not change the pages.
- In the Mac app, `window.print()` sends the `print` command to the shell. `saveAsPDF` in `macos/Recall/RecallApp.swift` saves the page as a PDF and then sends `afterprint` to the page.

## Import and update

### Which database is read

- The app starts one job for a first setup and for an update: `setup_import`. Both doors to an update (Settings and the Overview) open the setup guide, and the guide starts that job. The `export` job reads Messages too; only the old page in `ui/` starts it.
- An update never reads a copy. `database_for` in `ui_server.py` names the database that a job reads, and no other code selects a database for a job.
  - An archive that holds a message is updated from the database of Messages on this Mac, and from nothing else. A path from the client does not change this.
  - An import that names a copy (`dbPath`) for such an archive is refused with one plain sentence. `/api/jobs` answers 409, and no job is made.
  - Only a first setup can name a copy: the archive holds no message (`holds_messages`). The copy is read for that import only. With no copy named, the job reads Messages.
  - An export of an earlier build that waits to be loaded counts as an archive that holds a message. An archive file that cannot be opened is not taken for an empty one.
  - No other job reads a database. A report is written from the archive as it is.
- The guide holds the same rule: in an update it offers no other source (`offersCopy` in `app/src/lib/setupGuide.ts`). It says why Messages cannot be read and how to correct that.
- One test says whether the guide is an update: `holds_messages`. `/api/setup/status` gives it as `export.holdsMessages`, and `isUpdate` in `app/src/lib/setupGuide.ts` reads it. Do not decide this from the list of conversations.
- Each check of setup names the archive file that the app reads (`messagesPath`): at launch, when the guide opens, and in the guide.
- Known limit: a reader who works from copies only cannot update from a newer copy. They must remove the archive file first, and then do a first setup from the newer copy.
- `probe_database` in `setup_status.py` knows the database of Messages by the parts of its path (`in_messages_folder`). A folder whose name only starts like the folder of Messages holds a copy.
- The hand tool `parse_imessage.py` names no default database and no default file to write. WARNING: `cli.py --db` writes `messages.csv` in the current folder. In a checkout that file is the export that the archive is first loaded from.

### How a database is read

- An update is one full pull: `pull_into_archive` in `ui_server.py` calls `ArchiveState.pull_from_messages` in `recall_services/archive_state.py`, which calls `import_chat_db` of the package `archive`. After a pull the archive holds what the database holds: new rows are added, changed rows are changed, and rows that the database no longer has are removed. Nothing of a second source is merged.
- The importer copies the database to a work file of its own and reads that copy (`archive/source.py`). It opens the database to read only.
- The other readers of a Messages database open it with `open_read_only` in `setup_status.py`: the readers of the hand tool in `parse_imessage.py` (the list of `/api/contacts` before there is an archive uses one of them), and `probe_database`.
  - The database is opened to read only. A path that holds no file makes no file.
  - The path is quoted in the address, so a `?`, a `#` or a `%` in a file name is part of the name. The path is taken as the file it means: links are followed and `..` is worked out.
  - The database of Messages is not opened as immutable. Messages can write to it at any moment, and the reader also gets what is in the `-wal` file.
  - A copy with no `-wal` file and no `-journal` file beside it is opened as immutable. SQLite makes no file beside it.
- `/api/permissions/fulldisk` opens no database. It answers from `probe_database`.

### All or nothing

- A pull changes the archive or leaves it as it was. A database that cannot be read, a reader that stops, and a pull that is refused all leave the archive with the messages that it had.
- A first pull that does not end leaves no archive file (`_forget_empty_archive`).
- A database with no message to keep is not imported. A first pull of such a database leaves no archive file (`_forget_first_archive`). An update from one is stopped by the limit below.
- A pull that would remove more than 20% of the stored messages is refused before it changes anything (`DEFAULT_MAX_REMOVED_SHARE`). The job fails with the code `removal_limit` and one sentence with the numbers.
  - Only a request that says `allowRemoval: true` lifts the limit. Any other value does not.
  - The limit guards an archive that holds a message (`holds_messages`). An archive with none has nothing to lose, so a first setup is not held up by it.
  - The guide then says so in plain words and offers "Update anyway" (`failedImport` in `app/src/lib/setupGuide.ts`).
  - Rows that the bridge loaded (`legacy:` ids) are replaced by the first pull. They have no id of Messages, so the limit compares totals for them (`_check_bridge_rows` in `archive/mirror.py`): a database that holds far fewer rows than the export gave is refused in the same way. An export that reaches further back than Messages is not lost without a question.
- When the archive is written, the import did its work.
  - The server then drops each preview and each opened report that it keeps (`forget_previews`), and the guide clears the previews that the browser keeps (`finishOnboarding` in `App.tsx`). `useConversationPreview.ts` drops the figures that it holds in memory (`previewGeneration` in `app/src/lib/cache.ts`).
  - A later step of the job that fails does not make the import a failed one. The job ends as completed, with a plain `warning` in its result and in its log (`counted`). The Done step shows the warning.
- An import and the Overview give one count of messages: `what_the_archive_holds` uses `archive_summary`. A message in no conversation is shown nowhere in the app, and it is not counted.

### The export of an earlier build

- Earlier builds kept three CSV files: `messages.csv`, `messages_attachments.csv` and `messages_reactions.csv`. The server never writes, moves or deletes them now. They are the source of the bridge until the first pull.
- The bridge loads `messages.csv` into the archive one time (`ensure_from_export`). The job is `bridge` (`load_export`).
  - At the start of the server, `load_export_at_start` starts that job when an export waits (`export_waits`).
  - A request that needs the archive when only an export is there starts the same job (`archive_for`) and answers 503 with the code `archive_building` and the job. The app shows a waiting state (`ArchiveWait.tsx`) and loads its data again when the job ends.
  - A load that did not end well is not started again until the export file changes (`BRIDGE_FAILED`, `bridge_failure`). The answer is then 500 with the code `archive_failed` and the sentence. A load that met a busy archive is tried again after `BRIDGE_RETRY_SECONDS`.
  - A request also waits (503, `archive_building`, the import job) while a first import fills an archive that holds nothing yet.
- An export never replaces an archive that a pull filled.
- No code lists the folder of the archive. Do not add code that takes a file in that folder for an archive because of its name.

### Jobs

- One import writes the archive at a time. `create_job` in `ui_server.py` decides this once, with `JOBS_LOCK` held, when the job is made. `IMPORT_ACTIONS` holds the jobs that write the archive: `setup_import`, `export` and `bridge`.
- `ONE_AT_A_TIME` lists the jobs that do not run together. An import and the `meaning` job are in it. See "Chat memory".
- The first job runs. The second is refused: `/api/jobs` answers 409 with a plain sentence, and no job is made.
- `check_request` refuses a request that cannot be a job. `/api/jobs` answers 400, and no job is made.
  - The action is text, and it is one of `JOB_ACTIONS`.
  - A path (`messagesPath`, `outDir`, and `dbPath` for a `setup_import`) is text that the system can hold.
  - `includeContacts` is `true` or `false`.
- A body that is not JSON is answered with 400 on each POST route (`BadRequest`).
- The answer to a new job is a copy of the job as it was made (`job_copy`). Its status is `queued`. The answer about a job is a copy too. A copy shares no list with the job.
- A job always ends in the list. `run_job` marks a job as failed when its work is over and the list still says `queued` or `running`. A job that stays in one of these states holds up each later import.
- A failed job has `error`, one plain sentence, and `code`, a fixed word for the client or null (`TOLD_IN_A_SENTENCE`). A failure that nobody planned for says `JOB_FAILED` with the code `job_failed`: what was raised goes to the log of the server and to no line of the job.
- A step after the work of a job is a warning, not a failure: a count after an import (`counted`), the opening of a report that is written (`opened_report`).
- What a thread that answers a request prints is never a line of a job (`_JobStreamRouter.answering`).

### What the reader is told

- `recall_services/unreadable.py` is the one place for the sentences about a database that cannot be read. The server and the archive state import it. It uses `probe_database`, because macOS can show a protected file as missing.
- A busy database (`locked`) is tried. The importer waits for it.
- An import that stops raises `ImportStopped`. Its text is one plain sentence for the reader.
  - The sentence holds no text of a query. What a reader raised goes to the log of the server (`PullFailed.detail`). It does not go to the lines of the job.
  - The sentence ends with what is true of the archive (`archive_note`): nothing when there is no archive, or "The archive was not changed."
- `/api/contacts` answers a busy database with 503 and a database that fails with 500. The error is a plain sentence from `reading_stopped`.

### What is kept

- The server keeps no path to a database. `/api/defaults` names no database. `/api/contacts` takes no database from the client.
- The note beside the archive (`archive.sqlite3.source.json`) says only where the content came from: an export, or Messages. It holds no path of a database.
- The record of setup holds three things, each `true` or `false`: `completed`, `skipped` and `firstWordsShown`.
  - `setup_marker_updates` takes only `true` and `false` from the client.
  - `read_setup_marker` gives only `true` and `false`. It drops `pickedDbPath` and each value that an earlier build wrote as text.
  - `write_setup_marker` holds `SETUP_MARKER_LOCK` from its read to its write.
- The client keeps no path to a database. `RuntimePaths` holds the archive file and the reports folder only.

### The guide

- `guideStep` in `app/src/lib/setupGuide.ts` gives the step that the guide shows. `OnboardingWizard.tsx` keeps no step.
  - The step comes from the last check of the database (`db.status`), not from the state of setup (`state`). A first setup and an update use the same rule.
  - `canImport` is true for `readable` and for `locked`. The import is not offered for another status.
  - A copy that the reader chose is kept apart from the check of Messages. It is for one import. Back gives it up.
- `importRequest` builds the request for an import. It names the archive file that the app reads (`messagesPath`). In a first setup it names the path that the reader chose for a copy. In an update it names no database.
- The import step cannot be left while the request to start is on its way. The import can have begun.
- An import that the server refused has no id. The guide shows what the server said under "did not start".
- `SetupFrame.tsx` moves the focus to the step when the step changes. `import`, `import_running` and `import_failed` are three steps. `importView` gives the view of each.
- What the reader is told at a step belongs to that step (`StepNotes`). It goes when the step changes.
- `revealsFirstWords` decides whether the first words show: after the first setup, and not after an update.

## Message text

- The text of a message is stored as Messages has it. The export does not trim it.
  - Spaces and line breaks at the ends stay. A text of only spaces is a row.
  - Invisible characters stay: emoji joiners, variation selectors, direction marks.
- `extract_messages` in `parse_imessage.py` takes the text from the text column. When that column holds no text, it takes the text from `attributedBody` with `extract_text_from_attributed_body`.
  - That function gives the whole text or no text. It never gives a part of a text.
  - One rule cleans both sources (`_text_of_message`): only the C0 control characters other than tab and newline are dropped (`_DROPPED_CONTROLS`). Text with nothing left is no text.
- A text of only white space is a row of the export. Code that counts or shows text tests it with `strip()` or `display_text`, so such a message has no text for the reader.
- `display_text` in `message_text.py` is the code that trims. It removes the attachment mark (U+FFFC) and the spaces at the ends. `clip_text` shortens a text for a list.
- Code that shows a text, or that tests a text for emptiness, uses `display_text`. Do not trim a text when you write it to the export.

WARNING: Do not run `merge_export.py` on an archive that was written before the decoder fix (commit `5ac7086`). The tool compares the text in the archive with the text that the decoder gives. The texts that the earlier decoder cut or trimmed are different, so the tool adds those rows a second time.

## Tests

- No test reads the Messages database or the Contacts of this Mac, and no test connects to another host. `tests/guards.py` is the one guard of every suite. `tests/conftest.py` sets it for each test file:
  - Before the project is imported, it sets a key that is not a key, an empty home folder and a temp data folder. It removes the proxy and address settings. `load_dotenv` reads no file.
  - It refuses a connection, a send and a name lookup to a host that is not this machine. The error is `NetworkBlocked`, an `OSError`. Loopback and socket files stay open.
  - Each test starts with a check that the guard holds.
- A conftest of a package does not make a guard of its own. It adds only what its suite needs from `tests/guards.py`: the fixed time zone, the pinned data folder, the block of loopback.
- A child process has no socket guard. A test that starts one opens no socket there.
- Code that blocks the network for a run gives it back when the run ends (`network_blocked` in `chat_engine/eval/volume.py`).
- A test that needs a Messages database makes one with `messages_db` in `tests/invented_db.py`.
- No test starts a helper of the app. WARNING: `export_contact_names` starts the Contacts helper with `open`, outside every guard of a process, and that helper reads the real Contacts. `tests/guards.py` refuses to start `open`, `osascript` and the compilers (`HelperBlocked`). A trial run of the server must block them too.

## Caches

- `PREVIEW_PAYLOAD_VERSION` in `ui_server.py` is 8. The browser key in `app/src/lib/cache.ts` is `recall.previewCache.v8`. Change the two together when the shape of a preview changes.
- `preview_cache_key` holds `file_mark()` of the archive, the file of what earlier report runs read, and the Contacts cache.
- `ANALYSIS_PAYLOAD_VERSION` in `ui_server.py` is 6. Increase it when a stored analysis can differ for the same key. A change to `story_html` or `story_page` is such a change.
- `analysis_cache_key` holds `preview_cache_key`, so a stored analysis is not used after its figures change.
- A stale browser entry is safe. `useConversationPreview` always asks the server again.

## The archive and the services

- Every route reads one archive: a SQLite file of the package `archive`. The functions in `recall_services/` build each answer. `ui_server.py` calls them and adds no second way to read messages.
- The app names one file in each request (`messagesPath`). `recall_services/places.py` says which archive stands for it:
  - Nothing named, the default export or the default archive: `saves/archive.sqlite3` in the data folder.
  - A name that ends in `.sqlite3`: that file is the archive.
  - Any other name is an export of an earlier build. Its archive stands beside it with the ending `.sqlite3`.
  - `/api/defaults` gives the default archive file as `messagesPath`. Its `hasMessages` is `holds_messages`: an archive that a first import is still filling holds nothing yet, so setup is not taken for done.
- `archive_for` in `ui_server.py` gives the archive of a request, or raises `ArchiveNotReady`. Its codes are `archive_building` (503, with the job), `no_archive` (400) and `archive_failed` (500). Each route answers with that status and `{error, code}`.
- A read route with no archive gives an empty answer where the Overview needs one before setup (`bundle_or_none`): `/api/memories`, `/api/overview`, `/api/first-words`.
- `named_archive` reads the archive file of a request. A name that is not text, that no system can hold, or that names no file is 400.
- A request that fails in a way nobody planned for answers 500 with `SERVER_FAULT` and the code `server_error` (`server_fault`). What was raised goes to the log of the server. `/api/ask/stream` tells it as its last event when the stream is open.
- The request log holds the path of a request and no query (`log_message`): a query holds handles and search words.
- A message in no conversation is in no figure (`_parse_archive`), as it is in no list and no total.
- The figures on pandas read `frame.messages_frame(state)`. A cache in memory is keyed by `archive_signature` (the place, `generation` and `file_mark()`). A cache on disk is keyed by `file_mark()` only: `generation` starts at 0 at each start of the server.
- A message that was unsent is a row with no text in each list, each day and each figure. A payload gives it the label `UNSENT_LABEL` of `message_text.py` in `attachment`.
- A message id is text: Apple's guid after a pull, `legacy:<ROWID>` after the bridge. No code reads it as a number.

### Names

- Names come from the Contacts cache only: `saved_names` in `ui_server.py` calls `names.cached`. No request reads the Contacts database of the Mac.
- Two actions read Contacts into the cache, and each is started by the reader: `/api/contact-names`, and the first step of a `setup_import` (`names_from_contacts`).
- `recall_services/names.py` is the one place that names a conversation and a writer. A group that nobody named is named from all its saved members.

### Models

- `recall_services/models.json` is the one source for the offered models: id, label, level, limits and prices. `ai_config.py` and `cli.py` hold tables for the hand tools only.
- These three change together: `/api/defaults` (`models.defaults_payload`), each route and job that takes a model, and `app/src/lib/models.ts`, which builds its levels from `modelChoices`. No model id is written in the app.
- A model that is not in the file, or that is not offered, is an error with a code before any call. It is never run or priced as another model.
- `reasoning_levels` of a model are the efforts that its page at the provider lists, least first. `null` means that the page was not read. Do not write a level that the page does not list.
- `modelChoices` gives each offered model `reasoningLevels` and `reasoningDefault` (`models.offered`). `reasoningDefault` is `chat.reasoning_effort` of the file when the model lists it, and `null` when it does not.
- The packaged server ships `models.json` (`packaging/recall_server.spec`).

### The key of the provider

- `recall_services/keys.py` gives the key: `OPENAI_API_KEY` in the environment of the server, or in the file `.env` in the data folder. The file is read at each call, as it is written (nothing of the environment is filled in). No file above the data folder is looked for.
- Each model client is given the address of the provider, and takes no proxy from the environment (`build_client` in `report_engine/openai_llm.py`, `_ready_client` in `chat_engine/openai_llm.py`).
- WARNING: A run with a dummy key still sends its request. Tests and trial runs block the network.

### Chat

- `/api/ask/stream` sends the events of `recall_services/chat.py`: `status` (with `reset` when the text so far was a note of the model), `delta`, and then `done` or `error`. `askStream` in `app/src/lib/api.ts` and `app/src/lib/chatStore.ts` read them.
- A failure is one plain sentence and a `code`. No keyword answer stands in for an answer that failed.
- Each "user" row of the history holds `contact`: the conversation that its question was asked of. A row with no `contact` was asked of everything, and its answer does not go to a question that has a scope.
- The app gives each question a `requestId`. `/api/ask/cancel` stops the question with that id (`ASKING`). A write to a reader who left stops it too.
- A question can ask for a reasoning effort: `reasoningEffort` in the body of `/api/ask` and `/api/ask/stream`. These change together: `ask_arguments` in `ui_server.py`, `chat.stream`, `models.chat_config`, `askBody` in `app/src/lib/api.ts`, and `effortOf` in `app/src/lib/reasoning.ts`.
  - The effort must be one of the `reasoning_levels` of the model. Another word ends the question with the code `reasoning_level` before any call. `/api/ask` answers 400 (`ASK_REQUEST_FAULTS`). `/api/ask/stream` sends the code as its error event.
  - Reasoning is counted as output, and the model sends nothing while it reasons. Thus `chat.by_effort` in `models.json` gives an effort above medium more output and more time between two pieces of an answer (`chat_config`, `default_llm`). The limits of high, xhigh and max were not tried with a real call.
  - A question that asks for no effort gets `reasoningDefault` of the model. Do not lower `chat.reasoning_effort` below medium: at low, a model answered a question about a relationship from one keyword hit.
  - With the effort `none` the model port does not ask for carried reasoning (`chat.default_llm`).
  - The effort is for chat only. A report does not take it.
- Text for the model holds a saved name or a mask, never a handle that is not saved (`names.chat_labeler`).

### Reports

- The `analyze` job runs `recall_services/reports.run` (`write_report`). It writes the report file and the events file together, saves the report as JSON, and writes no HTML page.
- A report can be read when its conversation left the archive: `/api/analysis` then gives the story with no figures (`empty_preview`).
- A moment that was written on the rows of the bridge is found again after the first pull by its row number, when the conversation and the day agree (`_quoted` in `recall_services/reports.py`).
- `/api/analysis` reads a saved report by the text of its report file (`saved_report`). Each moment then carries `chatId`, `day`, `at` and `pickedId`, and the Reports page opens its day at the quoted message (`MomentDate.tsx`, `ReportDays.tsx`). A report of the older kind has no saved report: its events file is read with each cell as text (`events_of_file`).
- The estimate of a preview is the plan of the report engine (`reports.estimate`). `preview_cache_key` holds the file of what earlier runs read, so the estimate follows a run at once. With no plan, `estimate` is null and `estimateError` is one sentence.
- `shownPlain` in `/api/analysis` is true when a chapter of the story is shown as plain paragraphs. The Reports page then says so in one line (`ReportPage.tsx`).

### What the browser keeps

- A saved message is known by its conversation, its time and its text, never by its id (`bookmarkId` in `app/src/lib/bookmarks.ts`). A message has three ids in its life.
- A saved chat shows a citation from its stored copy and opens its day by conversation and time (`quotedIndex`).
- A stored model id that is not offered falls back to the default model (`preferredModel`).
- The reasoning level that a person sets is kept for each model under `recall.reasoning.v1` (`app/src/lib/reasoning.ts`). A kept level that the model does not list falls back to `reasoningDefault` (`effortOf`). The app sends only a level that the server listed, and the slider shows the level that is sent. Where the browser keeps nothing, a level holds while the page is open.

## Chat memory

Chat has two optional tools: a search by meaning, and a digest of one conversation.
Four parts hold them: `meaning_index/`, `report_engine/chat_digest.py` with `recall_services/digests.py`, the two ports of `chat_engine/ports.py`, and the wiring in `recall_services/memory.py`, `recall_services/meaning.py` and `ui_server.py`.

### Real messages are the proof

- An answer cites message references only. A reference is given by the evidence registry of the engine, for a message that the engine read from the archive.
- A hit of the index names two message ids. The engine reads the messages of the exchange from the archive (`_read_segment` in `chat_engine/tool_meaning.py`). The index gives no text.
- A moment of a digest names message ids that the report engine verified. The engine reads each one from the archive with `message` and shows it with a reference (`_with_evidence` in `chat_engine/tool_digest.py`). An id that the archive does not hold is dropped.
- A summary from a digest is never a quote and never a citation. The prompt says so, and the first line of each digest says so (`chat_engine/prompt.py`, `chat_engine/digest_text.py`).
- A quote of a moment is shown only when a shown message of the moment holds it.
- The ids in the index are the ids of the archive (`ArchiveSource` in `meaning_index/archive_source.py`). `ChatArchive.message` must find them. `tests/integration/test_chat_memory.py` holds this.

### The scope binds both ports

- The conversations and the days of a question go to both ports through `ToolContext.conversations` and `ToolContext.days`. A day is `YYYY-MM-DD` on each side: `Scope` in `chat_engine/scope.py`, `MeaningIndex.search`, and `_day` in `recall_services/reports.py`.
- The engine checks each hit of the search again. A hit outside the conversations or the days is dropped, also when a port gives one by mistake.
- A digest is built for one conversation of the scope. A period that reaches outside the days of the scope is not shown.
- `memory.py` adds no conversation and no day to a call.

### What leaves the machine

- The description of a search by meaning is masked by the engine before it reaches the port (`ask_port`). `MeaningPort` sends only the description to the embedding provider.
- A conversation id is a handle. It filters the search on this machine. It is in no request.
- A build sends the text of each new segment to the embedding provider: the day, then `Me:` or `Them:` and the text of each message. No name and no handle is added. A handle that a person typed into a message is part of that text.
- A digest sends what a report sends, with the same masks (`_Masked` and the labeler of `recall_services/reports.py`).
- Each client goes to the address of the provider and takes no proxy (`build_client`).
- No log line, no line of a job and no failure sentence holds text of a message, a name, a handle or an id. An error of the archive is not chained to a `MeaningFailure`.

### The index files

- The index is in `saves/meaning` (`meaning.index_store`). `segments.sqlite3` holds conversation ids and message ids. `vectors.npy` and `work.sqlite3` hold vectors. No file of the index holds text of a message.
- A build swaps the files in this order: segments, vectors, meta (`SWAP_ORDER` in `meaning_index/store_swap.py`). A put-back goes the other way. Each file names its build, and three files are an index only when they name one build.
- A reader takes files of one build only (`read_whole`). `MeaningPort` asks `IndexStore.meta()` for the build id after a load and loads again when it changed (`_load` in `recall_services/memory.py`).
- A process keeps one loaded index for each folder. It asks `IndexStore.meta()` for each search, which reads no row, and loads again only when the build on disk is another one. While the folder holds no whole build, the index that was loaded before answers.
- With no index folder the services ask for no file (`_held` in `recall_services/meaning.py`). So an import with no index opens no database of the index. `Reads` in `tests/test_update_reads_messages.py` names a database in the index folder `index`, and any other unknown database by its path.
- An index of a newer format is not read and not replaced. Chat then has no search by meaning, and the status of the index is `none`.

### What is embedded again

- The content hash of a segment is a hash of the text to embed, the model and the dimensions. A segment whose hash has a vector is not sent again.
- The segment rules are in `meaning_index/segments.py`: a gap of more than 120 minutes, 16 messages, 1,600 characters, a line of 400 characters, the day on the first line. WARNING: A change to a rule, to the model or to the dimensions makes the next build embed every segment again.
- The app builds with the default model of `meaning_index/models.json` (`meaning.default_facts`). `MeaningPort` embeds a description with the model and the dimensions of the index that it searches.
- The packaged server ships `meaning_index/models.json` (`packaging/recall_server.spec`). `tests/test_packaged_facts.py` holds this.

### One cache for reports and digests

- A digest and a report share `saves/report-cache.sqlite3` (`PeriodCache`). The key of a period is a hash of the prompt version, the model id, the speaker names and the transcript of the period.
- These must be equal on both sides, or the keys differ and a digest pays again: the source with its masks, the labeler, the model, the options, the token counter and the cache file. `recall_services/digests.py` takes each one from `recall_services/reports.py` (`_reading`).
- `DIGEST_MODEL` in `recall_services/memory.py` is `gpt-6-luna`: the digest only sums up the periods, and the chat model answers and checks against the messages. Thus a digest shares the period cache with a report only when the report was run with the same model.
- A report with dates reads only the messages of its dates. Its periods can differ from the periods of a digest.
- A digest writes no report and no file but the cache.

### The ports

- `memory.ports(archive, names)` gives the engine `meaning` and `digests` (`_events` in `recall_services/chat.py`). A port is None when its tool would fail at its first call: no key, no whole index, a folder that can not be read, a digest model that can not be run, notes of earlier runs that can not be opened.
- With a port that is None the tool is not offered and the prompt does not name it.
- A failure of a port never fails a question. The search falls back to the word ranking. A digest that failed is one line for the model.
- `report_engine.CancelToken` and `chat_engine.cancel.CancelToken` are two classes. `DigestPort` sets the first when the second is cancelled.

### The index job and its route

- `GET /api/meaning` answers `{status, estimate, error, code}` (`meaning.payload`). `status` has `state`, `segments`, `model`, `dimensions`, `builtAt` and `coveredUntil`. `estimate` has `segments`, `newSegments`, `tokens`, `estimatedCost` and `upperBound`, or it is null when `error` and `code` say why.
- `state` is `none`, `ready` or `behind`. The status and the estimate need no key and make no request.
- The `meaning` job builds the index or brings it up to date (`build_meaning_index` in `ui_server.py`). Its result has `segments`, `embedded`, `reused`, `removed` and `tokens`. Its `progress` has `step`, `of`, `label`, `done` and `total`. Its lines hold counts only.
- The `meaning` job and an import do not run together, and two `meaning` jobs do not run together (`ONE_AT_A_TIME`). The second one is refused with a plain sentence.
- After an import that changed the archive, the server starts one `meaning` job when an index is there (`refresh_meaning_index`, called by `run_job`). With no index it starts nothing: the first build is the choice of the reader. That job is made by `create_job`, so the same rule holds for it.
- `run_job` looks at the index after the import has ended in the list. A slow look or a fault there does not delay or fail the import.
- A failure of a build is a `MeaningFailure`. The job tells its sentence and its code.

## Mac shell

- What the page keeps in `localStorage` belongs to the origin `http://127.0.0.1:<port>`. `portForThisLaunch` in `macos/Recall/RecallApp.swift` remembers one port in the defaults of `local.recall.app` and asks for it at each launch: `RecallServerPort` for the packaged app, `RecallServerPortDev` for the development build. On another port the page starts with an empty store.
- The shell waits about 3 s for a remembered port that is in use (`rememberedPortChecks`, `rememberedPortPause`). `main` in `ui_server.py` must let go of the port in less time: it calls `server_close()` when it stops serving. Change the two sides together.
- A remembered port that is still in use is not replaced. That launch gets a drawn port and an empty store.
- The server binds with `SO_REUSEADDR` and without `SO_REUSEPORT` (`ThreadingHTTPServer`). The shell's check of a port (`isPortAvailable`) does the same.

## Interface

- The interface needs `light-dark()` colors and the `popover` attribute (Safari 17.5). `app/src/main.tsx` checks for both and shows a message if one is missing.
- A view asks for the width of the block that holds it, not for the width of the window (container queries). The blocks of a view are `workspace`, `detail`, `half`, `report-pane` and `chat`. A part that fits itself to its own width has its own block: `ranked`, `activity`, `days`, `hours`.
- One left edge: the title in the strip is 16px in. A page or a pane that fills its width starts on the same 16px. A column that sits in the middle has gutters of 32px or more.
- A page has one of two limits in `app/src/styles/tokens.css`:
  - `--page-max` is one column to read. A page keeps it while the workspace is narrower than 64rem.
  - `--page-wide` is a page of columns. `.page.wide` fills the workspace up to this limit when the workspace is 64rem or wider. Overview and Settings use it.
- In a wide workspace no screen keeps its content in one narrow column with the rest empty. A screen is a page of columns (Overview, Settings) or panes that fill the width (Chat, Analyze, Search, Reports, Jobs).
- The pages are open. A part of a page stands on the ground of the page. Space and hairlines set the parts apart. A card does not.
  - One block is 32px (`--s-8`) from the next block. The first block is 32px below the title strip.
  - No line stands between two sections. Space sets a section apart. The hairlines of a page are between the rows of a list, between figures, and around a card.
  - Four lines stay on purpose. A section with one line of words where a list would be has the hairline of a first row (`.section.ruled > .section-note`). The first year of a report's moments has a hairline below the strip of marks (`.report-strip + .report-timeline`). The files of a report are a footer with a hairline above it (`.report-files`). A chapter of a story after another has a hairline above it, with 32px on each side.
  - `.section.ruled` in `shell.css` is an open section. The heading is one line of 20px. It is 12px above the content. The people of a group (`analyze.css`) and the blocks of a report (`reports.css`) repeat this heading. Change the three together.
  - `.rows.ruled` in `lists.css` puts a hairline over each row, the first row also. A row is one band of 44px: 12px, a line of 20px, 12px. A card below a heading starts where the hairline of a first row would be.
  - Words keep 12px above and below them in a band, on one line or on more. A control of 32px keeps 6px.
  - Figures in a row have vertical hairlines between them. The first figure has no space before it. The last figure has no space after it. The tiles of the Overview, the figures of a report, the figures of a job and the facts of Analyze obey this rule.
  - Two columns are 40px (`--s-10`) apart. A figure keeps 20px on each side of its hairline. Thus the hairline in the middle of four figures is on the center of the gap between the two columns below it.
  - Two columns of a row end on one line where a list can take the height that is left. On the Overview and in Analyze, a column that ends with an open list has a hairline below the last row of that list. The two columns of Settings both end with a row and have no such line.
  - The moments do not grow to fill a column. With one or two moments, the column of the moments is shorter than the column beside it.
  - Only an object is a card: each moment of the Overview, the latest messages of Analyze, the chapters of a report, the log of a job.
  - A card holds one thing. Objects that are each opened by themselves are each a card, 12px apart. Do not put them as rows in one card. The contents of a report (`.report-contents`) is one thing: one list of the places in one story.
- The Overview is rows of sections. A column of sections in a row is `.overview-col`.
  - `momentsOf` in `app/src/lib/moments.ts` makes one list of moments. Each moment is a card of `.moment-cards`.
  - The list of the busiest conversations shows the rows that fit beside the moments. `useRowsThatFit` reads `--fit` on the box of the list. The style sheet sets `--fit` where the height of the box comes from outside. `rowsThatFit` in `app/src/lib/fit.ts` shows one more row if each row is then 5% shorter at most.
  - The row of the two heat maps is cut so that a day and an hour have the same size. The cut in `overview.css` and the cell sizes in `charts.css` use the same numbers: 53 weeks, 24 hours, gaps of 2px, 2rem of weekday names for each map, 2.5rem between the maps.
- A chart of the two voices that has a `heading` puts the heading, the day or month that is picked out, and the two totals in one band (`VoicesOverTime`, `VoicesSkeleton`). The day or month is `.readout-when` between the heading and `.readout`. It takes only the space that is left, so a pick does not move the totals to a second line. The rules for `.section-head > .readout` and `.section-head > .readout-when` in `charts.css` need this structure.
- A row of Settings with no value has an empty `.setting-value`. In a narrow workspace `settings.css` reads `:empty` to keep the label and the control on one line. Do not put a space or a comment in that element.
- The Quality menu of the chat (`QualityControl` with `reasoning`) has a slider below its levels: `LevelSlider` in `.choice-extra` of `ChoiceMenu`. The slider sets the reasoning level of the model in use. Its look is a thick track that is filled up to a round knob, with a mark at each step. The track is 30px high in `slider.css` (`--track`), and `INSET` in `LevelSlider.tsx` is half of it. Change the two together. The knob follows the pointer while it is held (`data-held`) and settles on the nearest step when it is let go.
- A placeholder has the structure and the classes of the content that replaces it. Thus no block moves when the content arrives.
  - `MomentsSkeleton` has the number of cards and the height that the list of moments had last. `app/src/lib/keptShape.ts` keeps these two numbers in the browser under `recall.shape.<name>`.
  - WARNING: `keptShape.ts` keeps numbers only. Do not keep a name or the text of a message there.
  - The cover and the figures of `ReportPageSkeleton` have the same classes, line for line, as the cover and the figures of `ReportPage`.
  - Where the size of the content is not known before it arrives, the placeholder is about that size. The blocks of a report under its figures are such a case.
- Code that scrolls a list from screen rectangles divides by the drawn scale (the width of the rectangle over `offsetWidth`). A panel that is opening is drawn smaller than its layout.
- A card in a page has the fill of the page. Its edge is the ring of `--shadow-raised`. This is the same in light and in dark.
  - `--bg-card` is the fill of a card in a page. A mark that must look cut out of a card uses `--bg-card` too.
  - `--bg-raised` is the lighter fill. Only a control (button, field, composer) and a layer that floats (popover, sheet, undo bar) use it.
  - In dark, `--ring` is firmer than `--line`, so the edge of a card leads the hairlines in it.
- A pane with nothing to show puts its words in the middle with `.pane-empty` in `app/src/styles/shell.css`.
- Each distance in a motion is multiplied by `--travel` in `app/src/styles/base.css`. Reduce Motion sets `--travel` to 0.
- Use one motion for each kind of change:
  - `.step` with `Step.tsx`: a change made with previous and next arrows.
  - `.arrive` and `.arrive-each`: a view or a detail pane that a click opens.
  - `.swap`: a value that changes in its place.
  - `.appear`: content that waited for a load.
- Do not put `.appear` on a direct child of `.arrive-each`. The two run together and `.appear` wins.
- A list drawn with no markers has `role="list"`. Safari removes the list role without it.
- A control at the end of its range uses `aria-disabled`, not `disabled`, so that it keeps the focus.
