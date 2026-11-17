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

- `database_for` in `ui_server.py` names the database that a job reads. No other code selects a database for a job.
  - An update (the `export` job) reads the database of Messages on this Mac. A path from the client does not change this.
  - The setup guide (the `setup_import` job) can name a copy. The copy is read for that import only. With no copy named, the job reads Messages.
  - No other job reads a database. A report is written from the archive as it is.
- An import does not merge a second source. After an import, the archive holds what the database holds. `merge_export.py` is a hand tool. The app does not call it.
- `export_database` reads the messages, the attachments and the reactions before it writes a file. It writes each file beside its place and then moves it. A failure leaves the three files as they were.
- `export_database` moves the messages file last, so the other two files are new before the messages file changes.
- `why_unreadable` gives the reason when a database cannot be read. It uses `probe_database`, because macOS can show a protected file as missing.
- The server keeps no path to a database. `read_setup_marker` drops `pickedDbPath`. `/api/defaults` names no database. `/api/contacts` takes no database from the client.
- The client keeps no path to a database. `RuntimePaths` holds the archive file and the reports folder only.
- The setup guide opens an update on the state of the database (`phaseForDatabase` in `OnboardingWizard.tsx`), not on the state of setup.

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
