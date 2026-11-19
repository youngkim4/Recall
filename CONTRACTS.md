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
- `ONE_AT_A_TIME` lists the jobs that do not run together. An import, the `meaning` job, the `digest` job, the `memory` job and the `people` job are in it. See "Chat memory".
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
Four parts hold them: `meaning_index/`, `report_engine/chat_digest.py` with `recall_services/digests.py`, the two ports of `chat_engine/ports.py`, and the wiring in `recall_services/` (`memory.py`, `meaning_port.py`, `meaning.py`, `digest_job.py`) and `ui_server.py`.

### Real messages are the proof

- An answer cites message references only. A reference is given by the evidence registry of the engine, for a message that the engine read from the archive.
- A hit of the index names two message ids. The engine reads the messages of the exchange from the archive (`_read_segment` in `chat_engine/tool_meaning.py`). The index gives no text.
- A moment of a digest names message ids that the report engine verified. The engine reads each one from the archive with `message` and shows it with a reference (`_with_evidence` in `chat_engine/tool_digest.py`). An id that the archive does not hold is dropped.
- A summary from a digest is never a quote and never a citation. The prompt says so, and the first line of each digest says so (`chat_engine/prompt.py`, `chat_engine/digest_text.py`).
- A quote of a moment is shown only when a shown message of the moment holds it.
- The ids in the index are the ids of the archive (`_Masked` in `recall_services/meaning.py`). `ChatArchive.message` must find them. `tests/integration/test_chat_memory.py` holds this.
- The ids of an archive can change under the same words: the first pull over a loaded export gives each message a new id. The estimate counts such segments as `staleSegments`, and the state is then `behind` until a build. That build costs nothing.

### The scope binds both ports

- The conversations and the days of a question go to both ports through `ToolContext.conversations` and `ToolContext.days`. A day is `YYYY-MM-DD` on each side: `Scope` in `chat_engine/scope.py`, `MeaningIndex.search`, and `_day` in `recall_services/reports.py`.
- The engine checks each hit of the search again. A hit outside the conversations or the days is dropped, also when a port gives one by mistake.
- A digest is built for one conversation of the scope. A period that reaches outside the days of the scope is not shown.
- `memory.py` adds no conversation and no day to a call.

### What leaves the machine

- The description of a search by meaning is masked by the engine before it reaches the port (`ask_port`). `MeaningPort` sends only the description to the embedding provider.
- A conversation id is a handle. It filters the search on this machine. It is in no request.
- A build sends the text of each new segment to the embedding provider: the day, then `Me:` or `Them:` and the text of each message. No name and no handle is added.
- A handle that a person typed into a message is masked in that text when it is not saved. The build, the estimate and the status read the messages through `reports._Masked`, with the saved names of the server: the same function as in a report and in a digest. Do not give `meaning_index` the archive directly.
- A contact that is saved later is no longer masked. The segments that held its handle get a new content hash and are embedded again.
- A digest sends what a report sends, with the same masks (`_Masked` and the labeler of `recall_services/reports.py`).
- Each client goes to the address of the provider and takes no proxy (`build_client`).
- No log line, no line of a job and no failure sentence holds text of a message, a name, a handle or an id. An error of the archive is not chained to a `MeaningFailure`.

### The index files

- Each archive has an index of its own, in `saves/meaning/<name>`. The name is the first 16 hex digits of the SHA-256 of the resolved path of the archive file. `meaning.index_folder` is the one function that builds this path: `meaning.py`, `memory.py` and `ui_server.py` take the folder from the archive that they were given.
- `segments.sqlite3` holds conversation ids and message ids. `vectors.npy` and `work.sqlite3` hold vectors. No file of the index holds text of a message.
- No test leaves an index in the data folder that the tests share (`no_index_in_the_shared_folder` in `tests/conftest.py` looks at `saves/meaning`).
- A build swaps the files in this order: segments, vectors, meta (`SWAP_ORDER` in `meaning_index/store_swap.py`). A put-back goes the other way. Each file names its build, and three files are an index only when they name one build.
- A reader takes files of one build only (`read_whole`). `MeaningPort` asks `IndexStore.meta()` for the build id after a load and loads again when it changed (`_load` in `recall_services/meaning_port.py`).
- A process keeps one loaded index for each folder. It asks `IndexStore.meta()` for each search, which reads no row, and loads again only when the build on disk is another one. While the folder holds no whole build, the index that was loaded before answers.
- With no index folder the services ask for no file (`_held` in `recall_services/meaning.py`). So an import with no index opens no database of the index. `Reads` in `tests/test_update_reads_messages.py` names a database in the index folder of its archive `index`, and any other unknown database by its path.
- An index of a newer format is not read and not replaced. Chat then has no search by meaning, and the status of the index is `none`.

### What is embedded again

- The content hash of a segment is a hash of the text to embed, the model and the dimensions. A segment whose hash has a vector is not sent again.
- The segment rules are in `meaning_index/segments.py`: a gap of more than 120 minutes, 16 messages, 1,600 characters, a line of 400 characters, the day on the first line. WARNING: A change to a rule, to the model or to the dimensions makes the next build embed every segment again.
- The app builds with the default model of `meaning_index/models.json` (`meaning.default_facts`). `MeaningPort` embeds a description with the model and the dimensions of the index that it searches.
- The packaged server ships `meaning_index/models.json` (`packaging/recall_server.spec`). `tests/test_packaged_facts.py` holds this.
- A hit with the content hash of a better hit is left out only inside one conversation (`_hits` in `recall_services/meaning_port.py`). The hash holds no conversation: the same words in two conversations are two hits.
- Known limit: the day in a segment and the clock times in a transcript are local times. After a change of the time zone of the Mac, segments have a new day and the parts of a digest are new. A build embeds those segments again, and a digest reads those parts again. The refresh of the server is bounded (see "The index job and its route"); the reader sees the cost of a larger build first.

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

### A digest inside a question is bounded

- One digest inside a question reads at most `DIGEST_PARTS_LIMIT` parts that are not read yet (4, in `recall_services/memory.py`). These change together: `DigestPort.digest` gives the number to `digests.build` as `max_new_periods`, and `_events` in `recall_services/chat.py` gives it to `EngineConfig.digest_parts_limit`. Both read the one constant.
- No answer is promised for later. A longer conversation is not read by the question, and nothing is started: no line, status, prompt rule or tool description says that a fuller answer comes later. The payload names the conversation in `memory: {"notReady": [conversation ids]}` (`ToolBox.memory_payload`), so the app can offer to read it.
- Nothing in a question starts a reading outside it. The `digest` job starts only when a request asks for it.
- While a digest is read inside the question, the stream gives one `progress` event for each change of a part: `{"type": "progress", "tool": "conversation_digest", "conversation": <label as shown>, "part": i, "of": n, "label": <period label>, "state": "start"|"done"|"cached"|"failed"}`. The turn runs the call to the source on a worker thread and yields the reports from a queue (`chat_engine/turn_fetch.py`). The events come before the next status line and before `done`; none comes after `done` or `error`. `DigestPort` makes each report from a map event of the report engine (`progress_relay`). With no digest source the stream is as it was.
- `DigestPort.digest` returns `usd`, the cost of the usage of the run, for the usage of the question. The log `recall.spend` gets one line with counts and the cost at the level INFO. `log_to_the_server_log` lets that level through for this log only.
- A digest inside a question that was paid for and failed, or that was stopped, says what it cost too. `DigestPort.digest` writes the line of the spend log from the usage that the failure carries, sets `usd` on the failure and raises it again. The engine adds that `usd` to `digest_usd`.
- A refusal does not mean that nothing was paid. The report engine can stop a digest at the limit after calls that were paid for (`DigestTooLarge` with `usage`). `_too_many` in `recall_services/digests.py` passes that usage on, and its sentence then names no count of the parts that are left. Code that reads a failure of a digest looks at the usage.
- One lock for the archive path, the conversation and the model is held around the build inside a question (`digest_lock`). A second question waits and then finds the parts read.
- The model port of a digest inside a question has a timeout of 120 s and one more try (`question_llm`). The `digest` job and a report keep the port of a report: 300 s and two more tries.
- Time bound: A question reads at most 3 digests, one after another, and each digest reads at most 4 parts that are not read yet, one at a time: at most 12 paid parts in a row. On `gpt-6-luna` a part takes about 20 s, so one digest takes about 80 s and a question about 4 minutes at most when the provider answers. When the provider hangs, one part takes at most two tries of 120 s and one wait of at most 60 s between them (`DIGEST_TIMEOUT_SECONDS`, `DIGEST_RETRIES` and the longest wait of `OpenAIResponsesLLM`): 5 minutes for a part, 20 minutes for a digest, an hour for a question. A cancel, or a reader who leaves, stops this before the next part.
- `/api/ask/stream` sees the reader leave without a write: a thread (`chat-reader-watch`, `watching_the_reader` in `ui_server.py`) waits on the socket with a selector and looks every 0.2 s (`READER_POLL_SECONDS`). When a peek finds the end of the connection, or the connection was reset, it cancels the token of the question. So no part starts after the reader left; the part in flight is paid for. The thread ends with the stream. It sees nothing on a connection that stays open: a proxy that holds the socket, or a client that half-closes and still reads. Bytes that the reader sends during the answer end the watch. In those cases only a failed write sees the close, and the first write after a close succeeds.
- `DigestPort.digest` reads the messages of the conversation from the archive on that worker thread before its first model call (the archive opens one connection for each thread). A cancel does not stop that read: the source looks at the token only before each model call.
- Known limits: the part that is sent when the reader leaves is paid for. A digest under a scope with days can pay for a period that is then not shown.
- A question and the `digest` job do not share a lock. A part is paid for twice only when two runs send it at the same moment, before either has saved it; so a question beside a `digest` job pays again for at most the parts of its own limit (4), and two runs with no limit pay again only for the parts whose calls overlap. In the worst case two runs with no limit that move in step each pay for every part, because nothing marks a part as being read. The mechanism: just before a part is sent, its key is looked up one more time (`report_engine/map_step.py`, for a report run too).

### The digest job

- The `digest` job reads one conversation for chat (`read_conversation` in `ui_server.py`, `digest_job.run`). The request has `conversation`, and `since` and `until` as days or null. It uses `DIGEST_MODEL` and has no limit of parts.
- Its result has `conversation`, `periods`, `periodsBuilt`, `periodsCached`, `periodsFailed`, `usd` and `usage`. Its `progress` has `step`, `of` and `label`, as in a report job. Its lines hold the label of a part and counts.
- One `digest` job runs at a time. A `digest` job and an import do not run together, and a `digest` job and the `memory` job do not run together (`ONE_AT_A_TIME`). A report and the `meaning` job run beside it.
- The record of a `digest` job says what it reads: `contact` is the conversation (`conversation` of the request, before `contact`), `archivePath` is the resolved path of the archive file, and `since` and `until` are the days or null (`job_record`). The record of any other job has no such field.
- A `digest` job that fails tells a sentence of its own (`sentence_of` in `recall_services/digest_job.py`), with the code of the failure of `digests.build`. No sentence names a report: the job writes none.
- Known limit: a `digest` job can not be stopped. An import that is asked for while it runs is refused with 409 and a plain sentence (`ONE_AT_A_TIME`); the reader starts the import again when it has ended.

### The index job and its route

- `GET /api/meaning` answers `{status, estimate, error, code}` (`meaning.payload`). `status` has `state`, `segments`, `model`, `dimensions`, `builtAt` and `coveredUntil`. `estimate` has `segments`, `newSegments`, `staleSegments`, `tokens`, `estimatedCost` and `upperBound`, or it is null when `error` and `code` say why.
- `state` is `none`, `ready` or `behind`. The status and the estimate need no key and make no request.
- `behind` in this answer means one of three things: the archive holds a later message with text; a build would embed a segment (`newSegments`); or the index does not hold the id of a segment (`staleSegments`). `meaning.status` alone knows only the first.
- The `meaning` job builds the index or brings it up to date (`build_meaning_index` in `ui_server.py`). Its result has `segments`, `embedded`, `reused`, `removed` and `tokens`. Its `progress` has `step`, `of`, `label`, `done` and `total`. Its lines hold counts only.
- The `meaning` job and an import do not run together, and two `meaning` jobs do not run together (`ONE_AT_A_TIME`). The second one is refused with a plain sentence.
- After an import that changed the archive, the server starts one `meaning` job when the archive has an index (`refresh_meaning_index`, called by `run_job`). With no index it starts nothing: the first build is the choice of the reader. That job is made by `create_job`, so the same rule holds for it.
- The server does not spend on the index without a word. `refresh_meaning_index` takes the estimate first, and starts the job only when `meaning.small_enough` says yes: `newSegments` is at most 2,000, or at most 10 percent of the segments, whichever is more (`REFRESH_SEGMENTS`, `REFRESH_SHARE`). Above that it starts nothing, and the state is `behind`. A `meaning` job that a reader starts always builds.
- Known limit: a `meaning` job can not be stopped. An import that is asked for while it runs is refused with 409 and a plain sentence (`ONE_AT_A_TIME`); the reader starts the import again when it has ended. The same holds for a `digest` job and the `memory` job.
- `run_job` looks at the index after the import has ended in the list. A slow look or a fault there does not delay or fail the import.
- A failure of a build is a `MeaningFailure`. The job tells its sentence and its code.

### The memory job and its route

- The memory of chat is the digest of each conversation with at least `PREPARE_MIN_MESSAGES` (200) messages, read with `DIGEST_MODEL` (`recall_services/memory_prep.py`). The `memory` job reads it ahead of time (`prepare_memory` in `ui_server.py`, `memory_prep.run`), so a question finds each part in the notes.
- The job reads each conversation that is not read whole, busiest first, `PREPARE_PARALLEL` (4) conversations at a time. The parts of one conversation are read one at a time: the prompt of a part holds the notes of the part before it (`map_step.py`). These change together: a prompt that no longer needs the notes of the part before it lets the parts of one conversation overlap.
- The conversations of one job share one `CallPool` (`digests.call_pool`): at most 4 requests are in flight, and they share the tokens per minute of the model. A run alone has a pool of its own with one slot, so a report sends as before.
- The limit of a digest (`max_new_periods`) holds with a pool: `Caller` counts a request when it is sent, under a lock, and refuses the one that would pass the limit before it is sent.
- A failure that each conversation would have (`FATAL` in `memory_prep_words.py`) stops the pool: no new request is sent, and the requests in flight end and are kept. Any other failure is counted for its conversation, and the job goes on.
- Its result has `conversations`, `conversationsRead`, `conversationsFailed`, `periodsBuilt`, `periodsCached`, `periodsFailed`, `usd` and `usage`. Its `progress` has `step`, `of`, `label`, `done` and `total`. Its lines and its progress hold counts only: no text, name, handle, id or label of a part.
- What the job spent is in its record under `spent`, `{"usd", "usage"}`, when it ended completed or failed after `memory_prep.run` started (`prepare_memory`). A failure of the job (`MemoryFailure`) carries the usage of the calls that were paid, also an error of no known kind, which is wrapped as `run_failed`. Each end of `memory_prep.run` writes one line to the log `recall.spend`, with counts and the cost only.
- `GET /api/memory` answers `memory_prep.payload` with `jobId` (`memory_answer` in `ui_server.py`): `state` (`none`, `partial`, `ready`), `model`, `minMessages`, the counts of conversations and parts, read and not read, `estimatedCost` of the parts not read, `error` and `code`. It makes no model call and needs no key. The shapes are in `recall_services/README.md`.
- The counts of `GET /api/memory` are kept in the server process for one key (`memory_key`): the signature of the archive, the Contacts cache file, the notes file and its `-wal` file (time and size), `DIGEST_MODEL` and `PREPARE_MIN_MESSAGES`. A part that is saved changes the notes file, so the next answer works the counts out again. One lock is held while they are worked out: requests that arrive together wait for one answer. `jobId` is read fresh for each request.
- One `memory` job runs at a time. The `memory` job and an import do not run together, and the `memory` job and a `digest` job do not run together (`ONE_AT_A_TIME`): both read with `DIGEST_MODEL` and no limit of parts, and side by side they would pay for each part of a conversation twice. A report, the `meaning` job and chat run beside it.
- After an import that changed the archive, `refresh_memory` (called by `after_job`) starts one `memory` job only when a `memory` job of this archive completed before, and the parts not read cost at most `PREPARE_AUTO_USD` (0.25 USD, `memory_prep.worth_starting`). The marker of a completed job is `saves/memory/<16 hex digits of the SHA-256 of the resolved path of the archive file>.json` (`memory_prep_marker.py`). `prepare_memory` writes it when the job ends completed. A question, a report and a `digest` job never write it. With no marker the server starts nothing and takes no estimate: the first reading is the choice of the reader. Above the cost it starts nothing. A fault there does not change how the import ended.
- The job and a question do not share a lock. The bound of a double spend in "A digest inside a question is bounded" holds for the `memory` job.
- Known limit: a report with the model of the digest (`DIGEST_MODEL`, the low level) can run beside the `memory` job and pay for the parts of one conversation twice, when both send them at the same moment. No lock for each conversation holds the two apart.
- Known limit: a `memory` job can not be stopped from the app. An import that is asked for while it runs is refused with 409 and a plain sentence.

### People

- The people index holds one profile for each person and each group (`recall_services/people.py`). A person is the archive's `person_id` with all of its direct threads. A group conversation, and a direct conversation with no `person_id`, is a profile of its own.
- The figures of a profile come from the archive only (`people_figures.py`). The kind, the status, the summary and the turning points come from one call of the model of the memory (`memory_prep.model()`), which reads the period summaries and the moments of the digests of the profile's own conversations, the figures and the label. No message is read again by a model, and no quote of a moment is sent.
- A profile is judged only from its own conversation. The prompt says so, and the model reads no digest of another conversation. A summary of a profile is never a quote and never a citation: chat confirms a person in their own conversation before it names them.
- A profile is made only when every part of its conversations is in the notes. The notes are read with no model call and no mark of use: a digest with no new part allowed, over a cache that does not mark what it reads. A profile whose notes are not all read keeps its row from an earlier build and is counted in `notRead`.
- What leaves the machine: the request of a profile holds the label as chat gives it, and notes in which every handle is masked by `Planner.mask` (`chat_engine/handle_mask.mask_every_handle` with `names.chat_labeler`): a handle of the archive becomes its saved name or its mask, any other phone number or email address a fixed phrase. The reply loses the line references of the report engine and is masked by the same function (`people_reply.py`). The embedding provider gets the text of each profile, and the description of a search.
- The fingerprint of a profile is its hash of the inputs (`people_plans.profile_fingerprint`): the prompt version (`people_prompt.PROMPT_VERSION`, a hash of the fixed text, the shape and the reasoning level), the model of the memory, the label, the counts and the first and last times of its conversations (they decide the figures) and the cache keys of its parts (they decide the notes). A build calls the model only for a profile whose fingerprint is not in the store. The embedding model and the dimensions (`meaning.default_facts()`) are in the hash of the embedded text; a change embeds again with no model call.
- The cache keys of the parts of each conversation are kept in `plans.sqlite3` beside the store (`people_plans.py`), with a fingerprint of the conversation as the archive lists it and of the settings of the reading (the prompt version of a part, the model, the options, the token counter, the saved name of each handle of the archive). A look lays out again only a conversation whose fingerprint changed (`people_look.py`). A change of the Contacts lays out every conversation again once.
- Each archive has a store of its own: `saves/people/<16 hex digits of the SHA-256 of the resolved path of the archive file>/people.sqlite3` (`people_store.people_folder`), and `plans.sqlite3` beside it. The folder is 0700 and the files 0600. A write of the store puts a whole new file in place with one rename. A store of a newer format is not read and not replaced.
- `people.search` never fails a question. With no profile, a store that can not be read, profiles with no vector, or a description that can not be embedded, it gives an empty list with `reason` (`no_index`, `not_embedded`, `no_key`, `embed_failed`). It searches the profiles that have a vector and counts the others in `not_searched`. With `conversations` it ranks only the profiles that hold one of them. It raises ValueError only for arguments that can not be used. The caller masks the description first. The shapes are in `recall_services/README.md`.
- One cache of the vectors of descriptions serves the process (`recall_services/query_vectors.py`, 64 entries, keyed by the model, the dimensions and the SHA-256 of the text). `people.search` and `MeaningPort` both use it, so the text of a question is embedded once. A search of people waits 5 s at most for the embedder (`SEARCH_TIMEOUT_SECONDS`), the search by meaning 10 s.
- The `people` job (`build_people_index` in `ui_server.py`, `people.build`) and an import do not run together, two `people` jobs do not run together, and a `people` job waits for a `memory` job (`ONE_AT_A_TIME`). A `memory` job is not refused by a `people` job. A report, the `meaning` job, a `digest` job and chat run beside it.
- `GET /api/people` answers `people.payload` with `jobId` (`people_answer`). It makes no model call and needs no key. It reads no message of a conversation whose plan is kept, and it writes only the plans it made. The answer is kept for one key (`people_counts`): the signature of the archive, the Contacts cache file, the count of parts in the notes (`people.notes_mark`; a mark of use does not change it), the store file and the model.
- After a `memory` job that completed, or an import that changed the archive, `refresh_people` (called by `after_job`) works the answer of the route out, so the first request finds it. It starts one `people` job only when a build of this archive completed before (`completed_at` in the store), and the profiles to write or to embed cost at most `PEOPLE_AUTO_USD` (0.10 USD) with every try: `TRIES` times `estimatedCost` (`people.worth_starting`). After an import that started a `memory` job it does nothing: the end of that job does it. A fault there does not change how the job ended.
- The estimate is for one try of each profile: the input is an upper bound from the lengths of its notes at 3 characters for one token (`people_plans.input_bound`), the output is the fullest reply that the limits of the reply allow (`OUTPUT_TOKENS`, 416), with no reasoning. A profile whose reply is in the wrong shape is asked once more (`TRIES`, 2): a build spends at most `TRIES` times the estimate.
- Known limit: a `people` job can not be stopped from the app, and an import is refused while it runs. A first build of about 1,400 profiles makes about 1,400 calls, 4 at a time: at about 4 s a call that is about 23 minutes of calls, plus the reading of the notes and the figures (about 75 s for 300 profiles of 545,000 messages with the fake model): about 25 minutes. A later build writes only the profiles that changed.

### People in chat

- `chat_engine` has a third optional port, `PeopleSearch` (`chat_engine/ports.py`), and its tool `find_people` (`chat_engine/tool_people.py`). With no port, the tool list, the prompt, the requests and the payload are as before: `tests/chat_engine/test_no_memory_ports.py`.
- `recall_services/people_port.py` gives the port over `recall_services.people.search`. It is None with no key, when the service is not in the build, or when `status` says there is no profile. `chat.py` adds it beside `memory.ports`.
- A search that did not run (`reason` is not None) raises `PeopleNotAvailable`. The engine gives the model one line.
- The scope binds the tool: a person none of whose conversations is in the scope is never shown. Only conversations inside the scope get an alias.
- A profile is text that a model wrote. The engine masks every phone number and every address in it, and in the description, before the model or the port gets them (`mask_every_handle`).
- A story of a person is never a quote and never a citation. The prompt tells the model to check each person in their own conversation before it names them.
- Names of the question (`chat_engine/question_names.py`, also with no port): a word of the name of a person is a name. A word that only a group title holds is a name only when at most `max(20, 0.5 percent of the messages)` hold it. Every candidate is looked at, at most 8 title words are counted, one time each, and the first 4 names are kept.
- The scope's conversations go to `PeopleSearch.search` (`conversations`), so the port ranks only the people of the scope. `PeoplePort` sends the keyword only when there is a scope.
- A search of people that failed is not asked again in the same question.

## Mac shell

- The page sends the command `tick` to the shell when a control with steps goes to another step (`recallBridge.tick` in `app/src/lib/bridge.ts`). The shell gives one tick of the trackpad (`NSHapticFeedbackManager`, `.levelChange`). The command has no answer. In a browser it does nothing.
- What the page keeps in `localStorage` belongs to the origin `http://127.0.0.1:<port>`. `portForThisLaunch` in `macos/Recall/RecallApp.swift` remembers one port in the defaults of `local.recall.app` and asks for it at each launch: `RecallServerPort` for the packaged app, `RecallServerPortDev` for the development build. On another port the page starts with an empty store.
- The shell waits about 3 s for a remembered port that is in use (`rememberedPortChecks`, `rememberedPortPause`). `main` in `ui_server.py` must let go of the port in less time: it calls `server_close()` when it stops serving. Change the two sides together.
- A remembered port that is still in use is not replaced. That launch gets a drawn port and an empty store.
- The server binds with `SO_REUSEADDR` and without `SO_REUSEPORT` (`ThreadingHTTPServer`). The shell's check of a port (`isPortAvailable`) does the same.

## Interface

- The app has palettes: Graphite (the first, in `tokens.css`) and the ones in `app/src/styles/palettes.css`. These three hold the same names and change together: `paletteOptions` in `app/src/lib/palette.ts`, the blocks of `palettes.css`, and the list in the script of `app/index.html`. `palette.test.ts` reads the two files and fails when they differ.
  - A palette sets only the grounds, the text steps, `--gray`, `--edge`, `--ink` and the hue (`--you`, `--you-hover`, `--you-ink`, `--them`). Fills, lines, rings and heat steps are shares of `--gray`, `--edge` and `--text` in `tokens.css`. Do not write a fill or a line as a colour of its own.
  - Each text step of each palette holds 4.5:1 on each ground, light and dark, a selected row also. `palette.test.ts` measures this from the style sheets. A new palette is not done until that test passes.
  - No colour is written outside `tokens.css` and `palettes.css`, other than white on a hue and the ramp of the reasoning slider.
  - Paper is Graphite: the blocks of `palettes.css` do not apply to `[data-paper]`.
  - `[data-palette]` on an element gives that element the colours of the palette. `PalettePicker` uses this for its rounds. Light or dark is the page's alone: no palette block sets `color-scheme`.
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
- The Quality menu of the chat (`QualityControl` with `reasoning`) has a slider below its levels: `LevelSlider` in `.choice-extra` of `ChoiceMenu`. The slider sets the reasoning level of the model in use. Its look is a track that is filled up to a round knob, with a mark at each step. The track is 22px high in `slider.css` (`--track`), and `INSET` in `LevelSlider.tsx` is half of it. Change the two together.
  - While a pointer holds the knob, `LevelSlider` writes `--at` on the track itself. React does not draw the slider again for each move. Do not put `--at` in the `style` of the JSX: a new drawing would take the knob from the pointer.
  - The fill is cut with `clip-path` and the knob is moved with `translate`. Neither changes layout. Do not move them with `width`, `left` or `right`.
  - WARNING: Do not move the knob with `transform`. The knob swells with `scale`, and `scale` also scales a `transform`: the knob then jumps to the side at each step.
  - A knob that is taken stays under the pointer where it was taken (`offset` in `LevelSlider.tsx`). It is not pulled toward a step while it is held.
  - A press away from the knob brings the knob to the pointer frame by frame (`chase` in `LevelSlider.tsx`). The knob has no CSS transition while a pointer holds it (`[data-held]`): a transition that is started again at each move makes the knob trail and then jump.
  - A held knob changes its step only when it is well past half way to the next step (`heldStep` in `app/src/lib/levelSlider.ts`). Without this, a hand that rests between two steps makes the step change back and forth, and each change ticks the trackpad.
  - On a new step the knob and the mark of the step swell one time, a ring goes out from the knob, and the page sends `tick` to the shell. The last step also sends a light along the fill.
  - The colours of the track are one ramp: `RAMP` in `LevelSlider.tsx`. The fill, the glow of the track and the ring read it (`--ramp`, `--tone`; `toneAt` and `rampCss` in `app/src/lib/tone.ts`). No other control uses these colours.
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
