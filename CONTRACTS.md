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
- A message has one of three directions: sent, received, or unknown.
  - `total` counts all three.
  - `sent` counts sent only. `received` counts received only.
  - Never compute received as `total` minus `sent`.
  - Server: `sent_mask` and `received_mask` in `conversation.py`. Client: `receivedCount` in `app/src/lib/conversations.ts`.
- `isFromMe` in a message is `true`, `false` or `null`. `sent_by_me` in `ui_server.py` writes it. `isOutbound` in `app/src/lib/conversations.ts` reads it, and also accepts the older forms that saved chats hold.
- The `direction` of a first message is `outgoing`, `incoming`, or any other value, which means unknown. Only `sentByMe` in `app/src/lib/direction.ts` reads it.
- `voicesOf` in `app/src/lib/conversations.ts` gives what each side wrote in a conversation of the list, or null when the list does not say. The list read from the Messages database has totals only. A row with no voices draws one bar in the neutral color.
- Rhythm rows start on Monday. `hour_by_weekday` writes them. `Rhythm.tsx` maps them to the week of the reader.
- Heat levels use one scale for the whole archive. The busiest count is always level 4. `heatSteps` and `heatLevel` in `app/src/lib/heat.ts` hold this.
- Search: `total` is the number of matches. `count` is the number of matches that the response holds. The screen shows `total`.

## Names

- The server sends a saved name, a group name made from saved names, or an empty string.
- Only one function on each side labels a conversation that has no name.
  - Client: `conversationName` in `app/src/lib/conversations.ts`.
  - Server, for text that the user reads: `conversation_label` in `conversation.py`.
  - The two functions must use the same words.
- No screen shows a full telephone number.
- Text that goes to the AI model uses `contact_label` in `conversation.py`. It never sends a handle that is not saved. A report run tells the model a saved name, or the given name of a group, or this label. `tests/test_prompt_privacy.py` checks each prompt.
- The title of a saved HTML report is `conversation_label`. The file name and the heading of the Markdown report keep the conversation id.
- A group id is `chat` followed by digits only. Three places hold this rule and must change together:
  - `is_group_chat` and `group_chat_mask` in `conversation.py`.
  - `isGroupChat` in `app/src/lib/conversations.ts`.
  - The SQL in `list_contacts` in `parse_imessage.py`.
- Names follow the Contacts cache. The messages bundle signature and `preview_cache_key` include the signature of the Contacts cache file.

## A day of a conversation

- `/api/day` gives one day of one conversation. `day_window` in `day_view.py` decides which messages; `day_payload` in `ui_server.py` gives each message the shape that search results have.
- The conversation id must match exactly. This route never matches part of an id. `in_conversation` in `conversation.py` holds the rule, and `filter_conversation` uses it too.
- A message with no conversation id belongs to no conversation. It is in no day and in no moment.
- A place given as `offset` is used as it is, from 0 to the total. The answer can hold fewer rows than `limit`. A `limit` below 1 is refused; a `limit` above 400 is 400.
- The command line can name a conversation by a part of its id. `find_conversation` in `conversation.py` does this, and it refuses a part that several ids hold.
- A day longer than the limit is given in part. `total` is the number of messages in the day and `offset` is the place of the first message given.
- `previousDay` and `nextDay` are the nearest days that hold messages, or null.
- A moment in `/api/memories` carries `day`, and `previewAt` and `previewId` when it quotes a message. `previewAt` is written in the same form as `timestamp` in a message of `/api/day`. `previewId` is written by `message_id_text`, the same code that writes `messageId`.
- The `count` of an on-this-day moment is every message of that day in that conversation. It is the same number as `total` in `/api/day`.
- The client finds the quoted message with `quotedIndex` in `app/src/lib/dayParts.ts`: first by `previewId` equal to `messageId`, then by `previewAt` equal to `timestamp`.
- The client reads a long day in parts with `partToAsk`: it asks for the rows that are there and no more. `joinPart` adds a part only when it meets what is shown.
- `DayReader.tsx` in `app/src/components/ui` shows a day: the arrows, the messages, the parts, the place that the reader left each day at. `DaySheet.tsx` is the dialog around it. Search shows the same reader in a pane.
- `.dayview-scroll` has `overflow-anchor: none`. The reader puts the messages in place itself.
- An on-this-day card takes its day from `day`. It does not work the day out from the clock of the client.

## Reports

- The report heading carries the conversation id. `report_contact` in `ui_server.py` reads it back.
- The first line of the Overview section is the scope line. `reports.py` writes it. `report_scope` in `ui_server.py` reads it first, and uses the First / Last line for older reports.
  - `- Scope: 2024-06-01 to 2024-06-30`
  - `- Scope: from 2024-06-01`
  - `- Scope: through 2024-06-30`
  - `- Scope: all dates`
- A saved report shows figures for the days in its scope.
- A run that finds no events removes the older events file. `write_events_timeline` in `analysis.py` does this.
- The app shows an analysis only with the report file that it was read from. `App.tsx` keeps the file path with the analysis.
- An analysis answer with no `analysis` is a failure. A job watch that has ended changes nothing.

## Story HTML

WARNING: The story of a report is written by an AI model from the messages of other people. Treat it as hostile input.

- One function turns a story into HTML: `story_html` in `safe_html.py`.
- The app page and the saved HTML report both use `story_html`. Do not render a story in any other way.
- The result holds only the tags in `ALLOWED_TAGS`.
- The result holds only three attributes: `href` and `rel` on a link, and `start` on a numbered list.
- A link address starts with `http://`, `https://` or `mailto:`. Each link has `rel="noopener noreferrer"`.
- `story_html` limits its input: the length of the text (`STORY_LIMIT`), a run of one mark, and a run of empty lines.
- `story_html` limits its work: an estimate of the work before the renderer starts (`STORY_WORK_LIMIT`) and the processor time of the renderer (`STORY_SECONDS`). A story past a limit, or nested too deep, is shown as plain paragraphs.
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
  - Only a first setup can name a copy: the archive file is not there, or it holds no row (`holds_messages`). The copy is read for that import only. With no copy named, the job reads Messages.
  - No other job reads a database. A report is written from the archive as it is.
- The guide holds the same rule: in an update it offers no other source (`offersCopy` in `app/src/lib/setupGuide.ts`). It says why Messages cannot be read and how to correct that.
- One test says whether the guide is an update: `holds_messages`. `/api/setup/status` gives it as `export.holdsMessages`, and `isUpdate` in `app/src/lib/setupGuide.ts` reads it. Do not decide this from the list of conversations.
- Each check of setup names the archive file that the app reads (`messagesPath`): at launch, when the guide opens, and in the guide.
- Known limit: a reader who works from copies only cannot update from a newer copy. They must remove the archive file first, and then do a first setup from the newer copy.
- An import does not merge a second source. After an import, the archive holds what the database holds. `merge_export.py` is a hand tool. The app does not call it.
- `probe_database` in `setup_status.py` knows the database of Messages by the parts of its path (`in_messages_folder`). A folder whose name only starts like the folder of Messages holds a copy.
- The hand tool `parse_imessage.py` names no default database and no default file to write. WARNING: `cli.py --db` writes `messages.csv` in the current folder. In a checkout that file is the archive of the app.

### How a database is read

- Each reader of a Messages database opens it with `open_read_only` in `setup_status.py`: the five readers in `parse_imessage.py`, `probe_database` in `setup_status.py`, and `group_display_names_from_database` in `ui_server.py`.
  - The database is opened to read only. A path that holds no file makes no file.
  - The path is quoted in the address, so a `?`, a `#` or a `%` in a file name is part of the name. The path is taken as the file it means: links are followed and `..` is worked out.
  - The database of Messages is not opened as immutable. Messages can write to it at any moment, and the reader also gets what is in the `-wal` file. For this, SQLite keeps a `-shm` file beside the database, and makes it and the `-wal` file when they are not there.
  - A copy with no `-wal` file and no `-journal` file beside it is opened as immutable. SQLite makes no file beside it, and the copy can be in a folder that cannot be written to.
- A reader runs a query with `read_frame` in `parse_imessage.py`. An error is the error that `sqlite3` raises. It holds no text of a query.
- A reader gives a table with all of its columns when the result is empty.
- `/api/permissions/fulldisk` opens no database. It answers from `probe_database`.

### The three files of the archive

- An archive is three files: the messages file, and beside it `<stem>_attachments.csv` and `<stem>_reactions.csv`. `archive_files` in `ui_server.py` names them.
- An import writes all three files each time. A database with no attachments or no reactions gives a file with only its header. `_load_companion` in `conversation.py` reads that file as an empty table.
- A database with no message to keep is not imported. The archive stays as it was.
- An import changes all three files or none of them. `export_database`, `settle_earlier` and `move_in` in `ui_server.py` do these steps in this order:
  1. Put back what an import before this one did not finish (see below), and remove the files that it left half written.
  2. Read the messages, the attachments and the reactions. No file is written before all three are read.
  3. Write each new file beside its place, with `.tmp-export` in its name.
  4. Keep each earlier file beside its place, with `.before-update` in its name, the messages first. The earlier file is a hard link, or a copy on a volume with no hard links.
  5. Move each new file to its place in one step: the attachments, then the reactions, then the messages.
  6. Remove the earlier messages file. This step finishes the import.
  7. Remove the other earlier files.
- A failure at any step leaves the three files as they were. When step 5 or step 6 fails, the files that moved are put back from the earlier files.
- The earlier messages file is the sign of an import that is not finished. It is kept first and removed last.
  - While it is there, the next import first puts each earlier file back, the messages last. This also undoes an import that the end of the process cut off during the moves.
  - The server does the same for its own archive file when it starts (`settle_default_archive`). An archive at another path is put back by its next import. Until then a request can read new attachments or new reactions beside the earlier messages.
  - An import that put the archive back and then stopped says "The archive is as it was before the import that did not finish."
  - When it is not there, the import before finished. The next import removes what is left of its earlier files.
- If an earlier file cannot be put back, it stays beside the archive with the earlier messages file, and the error names it (`ArchiveInParts`). The next import puts it back. An archive that still cannot be put back is not imported over.
- A request that the server answers during the moves reads a whole file at each of the three places.
  - For a moment it can read new attachments or new reactions beside the earlier messages.
  - It never reads new messages beside earlier attachments or earlier reactions, because the messages move last.
- When the three files are in place, the import did its work.
  - The server then drops each preview that it keeps (`forget_previews`), and the guide clears the previews that the browser keeps (`finishOnboarding` in `App.tsx`).
  - Known limit: `useConversationPreview.ts` holds the figures of the last conversation in memory. After an import, Analyze can show them for that conversation until the server answers.
  - A later step of the job that fails does not make the import a failed one. The job ends as completed, with a plain `warning` in its result and in its log. The Done step shows the warning.
- An import and the Overview give one count of messages: `what_the_archive_holds` uses `archive_summary`. A message in no conversation is shown nowhere in the app, and it is not counted. When that count fails, the result holds the number of rows that were written, with the warning.
- A companion file that is a link is replaced by a file.
- No code lists the folder of the archive. Do not add code that takes a file in that folder for an archive because of its name.

### Jobs

- One import writes the archive at a time. `create_job` in `ui_server.py` decides this once, with `JOBS_LOCK` held, when the job is made. `ONE_AT_A_TIME` holds the rules:
  - A second import does not run beside an import.
  - A second build of the meaning index (the `semantic` job) does not run beside a build.
  - An import and a build do not run together. The index is of one archive, so an import during a build would leave the index out of date.
- The first job runs. The second is refused: `/api/jobs` answers 409 with a plain sentence, and no job is made.
- `check_request` refuses a request that cannot be a job. `/api/jobs` answers 400, and no job is made.
  - The action is text, and it is one of `JOB_ACTIONS`.
  - A path (`messagesPath`, `outDir`, and `dbPath` for a `setup_import`) is text that the system can hold.
  - `includeContacts` is `true` or `false`.
- A body that is not JSON is answered with 400 on each POST route (`BadRequest`).
- The answer to a new job is a copy of the job as it was made (`job_copy`). Its status is `queued`. The answer about a job is a copy too. A copy shares no list with the job.
- A job always ends in the list. `run_job` marks a job as failed when its work is over and the list still says `queued` or `running`. A job that stays in one of these states holds up each later job of its group.

### What the reader is told

- `why_unreadable` in `ui_server.py` gives the reason when a database cannot be read. It uses `probe_database`, because macOS can show a protected file as missing.
- A busy database (`locked`) is tried. `export_when_free` tries it once more after `BUSY_WAIT_SECONDS`.
- An import that stops raises `ImportStopped`. Its text is one plain sentence for the reader. `why_unreadable`, `reading_stopped` and `writing_stopped` write the sentences.
  - The sentence holds no text of a query. What a reader raised goes to the log of the server. It does not go to the lines of the job.
  - The sentence ends with what is true of the archive (`archive_note`): nothing when there was no archive, "The archive was not changed.", or that the archive was put back.
- `/api/contacts` answers a busy database with 503 and a database that fails with 500. The error is a plain sentence from `reading_stopped`.

### What is kept

- The server keeps no path to a database. `/api/defaults` names no database. `/api/contacts` takes no database from the client.
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

## Caches

- `PREVIEW_PAYLOAD_VERSION` in `ui_server.py` is 7. The browser key in `app/src/lib/cache.ts` is `recall.previewCache.v7`. Change the two together when the shape of a preview changes.
- `ANALYSIS_PAYLOAD_VERSION` in `ui_server.py` is 5. Increase it when a stored analysis can differ for the same key. A change to `story_html` is such a change.
- `analysis_cache_key` holds `preview_cache_key`, so a stored analysis is not used after its figures change.
- A stale browser entry is safe. `useConversationPreview` always asks the server again.

## Interface

- The interface needs `light-dark()` colors and the `popover` attribute (Safari 17.5). `app/src/main.tsx` checks for both and shows a message if one is missing.
- A view asks for the width of the block that holds it, not for the width of the window. The blocks are `workspace`, `detail` and `half` (container queries).
- One left edge: the title in the strip is 16px in. A page or a pane that fills its width starts on the same 16px. A column that sits in the middle has gutters of 32px or more.
- A page has one of two limits in `app/src/styles/tokens.css`:
  - `--page-max` is one column to read. Settings uses it.
  - `--page-wide` is a page of columns. `.page.wide` fills the workspace up to this limit when the workspace is 64rem or wider. Overview and Jobs use it.
- In a wide workspace no screen keeps its content in one narrow column with the rest empty. A screen is a page of rows (Overview, Jobs) or panes that fill the width (Chat, Analyze, Search, Reports).
- The Overview is rows. The cards of a row have one height and end on one line.
  - The first row and the row of tiles use the same gap, so the middle of the page is one line.
  - The list of the busiest conversations shows the rows that fit beside the moments. `useRowsThatFit` reads `--fit` on the card; the style sheet sets it where the height of the card comes from outside.
  - The row of the two heat maps is cut so that a day and an hour have the same size. The cut in `overview.css` and the cell sizes in `charts.css` use the same numbers: 53 weeks, 24 hours, gaps of 2px, 2rem of weekday names.
- Code that scrolls a list from screen rectangles divides by the drawn scale (the width of the rectangle over `offsetWidth`). A panel that is opening is drawn smaller than its layout.
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
