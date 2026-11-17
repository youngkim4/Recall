# report_engine

`report_engine` writes the report on one conversation.

It replaces the pipeline in `analysis.py` (`run_cli`), `ai_analysis.py` and `reports.py`.
It does not import `ui_server` or any other module at the repository root.

## What it does

1. It reads the messages of one conversation from a source.
2. It splits the conversation into periods on calendar boundaries.
3. It calls the model one time for each period, oldest first (map step).
4. It checks every date, quote and reference against the messages.
5. It selects the moments of the report.
6. It calls the model one time to write four sections (reduce step).
7. It returns a `Report`. The caller saves it.

A period result goes into a cache. A second run calls the model only for periods whose text changed.

## Run a report

```python
from report_engine import (
    CancelToken,
    ModelInfo,
    PeriodCache,
    ReportStore,
    estimate,
    run_report,
)
from report_engine.frame_source import FrameSource
from report_engine.openai_llm import OpenAIResponsesLLM, build_client

model_info = ModelInfo(...)  # see "Model facts"
source = FrameSource(messages_frame)  # or ArchiveSource.open(path)
cache = PeriodCache(data_dir / "report-cache.sqlite3")
store = ReportStore(data_dir / "reports")
llm = OpenAIResponsesLLM(build_client(), model_info.id)
scope = {"since": "2024-06-01", "until": "2024-06-30"}  # or None for all dates

quote = estimate(source, chat_id, scope, model_info, cache, labeler=labeler)

cancel = CancelToken()
report = run_report(
    source=source,
    conversation=chat_id,
    scope=scope,
    llm=llm,
    model_info=model_info,
    labeler=labeler,
    cache=cache,
    stats=stats,  # any figures; stored as given
    on_progress=show_progress,  # optional
    cancel=cancel,  # optional
)
path = store.save(report)
```

Give `estimate` and `run_report` the same labeler, options and token counter.
Different values give different periods and different cache keys.

## What the caller supplies

| Part | Type | Notes |
|------|------|-------|
| Source | `ReportSource` | `messages(conversation, since, until)` yields messages, oldest first. |
| Labeler | `(sender: str) -> str` | The name a speaker has in text that the model reads. |
| Model | `LLM` | One model, bound to its id. |
| Model facts | `ModelInfo` | Limits, prices, rate limit, tokenizer family. |
| Cache | `PeriodCache` | One SQLite file. `None` is permitted. |
| Store | `ReportStore` | One folder. The engine does not write to it; the caller calls `save`. |

### Source

A message is a dict: `id` (str, not empty), `timestamp` (`YYYY-MM-DD HH:MM:SS`, local time, no zone), `is_from_me` (`True`, `False`, `1` or `0`), `sender` (str), `text` (str).
A message that does not have this shape stops the run with `SourceError` before any model call.
An error that the source or the labeler raises also stops the run with `SourceError`.
The message of this `SourceError` gives only the type of the error, because the text of the error can hold a handle or words of a message. The cause is in `__cause__`.

`since` and `until` are days (`YYYY-MM-DD`) or full timestamps. Both ends are included. A day means the whole day.
A scope with an unknown key or a bad date is a `ConfigError`. It never becomes "all dates".

- `FrameSource(frame)` reads a pandas DataFrame with the export columns `message_id`, `timestamp`, `sender`, `text`, `chat_id`, `is_from_me`. It matches `chat_id` exactly. It reads rows by position. It converts a timestamp that has a zone to the local time of the machine.
- `ArchiveSource(archive)` reads `archive.iter_messages(conversation_id, since=..., until=...)`. It sends days only, asks for one day more than `until`, and applies the exact bounds itself. It skips rows whose `kind` is `system`. It reads one conversation id, not a list of ids. `ArchiveSource.open(path)` imports the `archive` package at that moment. It does not catch the errors of `Archive.open`: the archive contract says that a missing file is a `FileNotFoundError`. This adapter is tested with a stand-in only.

### Labeler

The engine calls the labeler one time for each sender that is not the owner.
Own messages always get the name `You`.

The engine replaces a name with `Unsaved contact` when the name:

- is empty or is `You`;
- contains `@`;
- has 7 or more digits;
- has digits and no letter (a short code).

The check reads the full name. After the check, the name is cut to 40 characters.
When the cut makes the names of two people equal, the later name gets a number.
Two senders with the same full name are one person and have one name.
The conversation id never goes into a prompt.

In a one-to-one chat the source can give an empty sender. Return the contact's name for the empty sender.

### Model facts

The engine holds no model id, price or limit. All of them come from `ModelInfo`:

`id`, `context_window`, `max_input_tokens`, `max_output_tokens`, `input_usd_per_mtok`, `cached_input_usd_per_mtok`, `output_usd_per_mtok`, `long_context` (threshold and price multipliers, or `None`), `tokens_per_minute` (or `None`), `tokenizer_family`.

Prices are in USD for one million tokens.

### Model

`LLM.generate(messages, *, schema=None, max_output_tokens, reasoning=None)` returns `LLMResult` with `text`, `parsed`, `usage`, `status` and `attempts`.

- `usage.input_tokens` includes the cached tokens. `usage.cached_input_tokens` is the cached part.
- `status` is `completed`, `incomplete`, `refused` or `failed`.
- The port raises `LLMError` when it cannot answer. It does its own retries.
- An error of any other type from the port fails that call. The engine names only the type of the error.

`OpenAIResponsesLLM` sends `store=False`, a timeout and `max_output_tokens` with every request.
It retries a rate limit, a timeout, a connection error and a 5xx reply, 2 times by default.
It waits the time that `retry-after` gives. When that time is longer than 60 seconds it stops and raises.
It does not send a cut-off reply again.
Its errors hold the HTTP status, the error code and the request id. They hold no text from the service and no link to the request.
`build_client` always sets the address of the service. `OPENAI_BASE_URL` in the environment has no effect.

`FakeLLM` (`report_engine.fake_llm`) is a model for tests. It makes no network call.

## Options

`ReportOptions` holds the choices of a run. The defaults are:

| Option | Default | Meaning |
|--------|---------|---------|
| `period_token_budget` | 80,000 | Largest transcript of one period, in tokens. |
| `map_max_output_tokens` | 16,000 | Hard output limit of a map call. |
| `reduce_max_output_tokens` | 24,000 | Hard output limit of the reduce call. |
| `reasoning` | `None` | Reasoning effort, passed to the model as given. |
| `score_floor` | 0.4 | A moment below this score is not in the report. |
| `max_moments` | 40 | Largest number of moments in the report. |
| `max_message_chars` | 6,000 | A longer message keeps its start and its end. |
| `carry_over_token_budget` | 4,000 | Room for notes from earlier periods. |
| `tokenizer_mismatch_margin` | 1.35 | Safety margin for token counts. See "Tokens". |

The output limits include reasoning tokens. Set them high enough for the reasoning effort.
With a low `tokens_per_minute`, set them lower: one request is the input plus the output limit.

## Periods

The engine packs calendar months, oldest first, until the next month does not fit the budget.
A month that is larger than the budget stands alone and is split by days.
A day that is larger than the budget stands alone and is split by messages.

New messages change only the last period. A period that is closed does not change.

The budget is also limited by the model. One request must fit `max_input_tokens`, the context window and one minute of `tokens_per_minute`.
The engine keeps room for the fixed prompt text, the speaker names and the notes.

## Cache

The key of a period is a hash of four things: the prompt version, the model id, the speaker names in the period, and the period transcript.
The prompt version is a hash of the map prompt text and of its reply schema. A change to the prompt stops the use of old entries.

The key does not hold the notes from earlier periods. A change in an old period does not run the later periods again.
The key does not hold the reasoning effort. To force a new run, call `PeriodCache.clear()`.

A failed period has no cache entry. The next run tries it again.
The cache file holds summaries and quotes. The engine sets its mode to `0600`. Keep it with the archive.

## The report

`Report` is a pydantic model. `report.model_dump(mode="json")` gives the JSON form.

- `schema_version`, `id`, `conversation`, `scope` (`since`, `until`), `model`, `created_at` (always in UTC)
- `status`: `complete` or `partial`
- `warnings`: text lines
- `sections`: always `arc`, `dynamics`, `defining_moments`, `big_picture`, in this order. Each has `key`, `title`, `markdown`.
- `moments`: the selected moments, oldest first
- `periods`: one `PeriodSummary` for each period
- `usage`: `input_tokens`, `cached_input_tokens`, `output_tokens`, `cost_usd`, `calls` of this run
- `stats`: the figures from the caller, changed to plain JSON values

A `Moment` has `date`, `title`, `detail`, `category`, `score`, `quote`, `message_ids`, `quote_verified`, `quote_message_id`, `quote_speaker`.

- `message_ids` are real message ids. Use them to open the exchange. The list can be empty.
- `date` is always a day on which the period has a message.
- When `quote_verified` is true, `quote` is in the message `quote_message_id`, and `quote_speaker` is the name of the person who wrote that message.
- When `quote_verified` is false, `quote`, `quote_message_id` and `quote_speaker` are empty.
- `title`, `detail` and `quote` are plain text. Escape them when you put them into HTML.

Section `markdown` has no heading, link, image or HTML tag. The engine removes them from the model reply, also when they are nested.

A `PeriodSummary` has `label`, `first_day`, `last_day`, `summary`, `carry_over`, `moments`, `usage`, `cache_key`, `from_cache`, `error`, `message_count`.
For a period from the cache, `usage` is the cost of the call that made the entry. `report.usage` counts only the calls of this run.

## Checks

- A reference such as `m12` maps to a message of the period. A reference that does not exist is ignored.
- A date must be a day of the period. A wrong date moves to the earliest cited day. A moment with no usable date is dropped.
- A quote must be part of a cited message, or of a message within three days of the moment's date. A quote of fewer than 8 characters must be in a cited message.
- Two texts match when they are equal after white space, quote marks and Unicode composition are made the same. Letter case must match.
- In Latin, Greek and Cyrillic text a quote cannot start or end in the middle of a word. `ok` is not a quote from `tokens`.
- A quote that is not found is cleared. The report gets a warning with the count.
- The name in `quote_speaker` comes from the message, not from its text. Text such as `Mina: I owe you` in a message of another person does not make Mina the speaker.

## Moments

1. The engine removes duplicates. Two moments from different periods are one event when they hold the same message, or when their dates are 7 days apart or less and their titles have the same words. The moment with the higher score stays. Two moments of one period are never duplicates.
2. It removes moments below `score_floor`.
3. It takes the best moment of each period, then the second best of each period, until `max_moments`.

A quiet period gives no moment. The engine adds nothing to fill the list.

## Prompt safety

All text from messages, names, summaries and notes is data in a prompt.

- Each piece is put on one line.
- Each block of data is in a frame: `<transcript>`, `<notes>`, `<periods>`, `<moments>`.
- Text that contains a frame tag gets an invisible character in the tag. It cannot close a frame.
- The prompts tell the model not to follow instructions inside the data.

## Failures

| Event | Result |
|-------|--------|
| A period fails | The report is `partial`. A warning names the period. |
| All periods fail | `AllPeriodsFailed`. No report. |
| The summaries are too large for the model | The engine shortens them for the reduce call. A warning gives the count. |
| The reduce reply is empty, cut off or has an empty section | `ReduceError`. No report. |
| The cancel token is set | `ReportCancelled`. No report. |
| An error in the engine after the first call | `RunError`. No report. The cause is in `__cause__`. |
| The scope has no message with text | `EmptyConversationError`. No model call. |
| The source or the labeler raises an error, or a message has a bad shape | `SourceError`. No model call. |
| The model limits leave no room for a period or for the reduce call | `ConfigError`. No model call. |

`ReportCancelled`, `AllPeriodsFailed` and `ReduceError` are kinds of `RunError`. Each `RunError` has a `usage` attribute with the tokens and the cost of the run until that moment.
A call that ends with an error from the port gives no usage. The engine cannot count it.
Period results stay in the cache. The next run does not pay for them again.

## Cancel and progress

`CancelToken.cancel()` stops the run before the next model call. A call that is in progress completes first.

`on_progress` gets a `ProgressEvent` with `step`, `state`, `period`, `index` and `total`.

- `step` is `prepare`, `map`, `reduce` or `done`.
- `state` is `start`, `cached`, `waiting`, `done` or `failed`.
- `waiting` means that the run waits for the rate limit.

## Tokens and the rate limit

The engine counts tokens with `tiktoken`.
It uses the encoding with the name `ModelInfo.tokenizer_family` when `tiktoken` has it. If not, it uses `o200k_base`.
When the encoding is not the tokenizer family of the model, each count is multiplied by `tokenizer_mismatch_margin`.

`tiktoken` gets its vocabulary file from the network the first time it runs on a machine.
To stay offline, set `TIKTOKEN_CACHE_DIR` to a folder that has the file, or give `run_report` and `estimate` a `counter`.

For each run the engine writes the tokenizer and the margin to the log (`report_engine.plan`, level INFO).
When no vocabulary file can load, the engine counts by class of character and writes a warning to the log.
This count is 1.0 to 1.5 times the `o200k_base` count on chat text in the languages that were measured.
The engine does not load a tokenizer at import time.

Before each call the engine reserves the input estimate plus the output limit in a 60-second window.
It waits when the window is full. After the call it replaces the reserve with the real usage.
Retries inside the model port are counted after the call, at the time of the last send.

## Estimate

`estimate(source, conversation, scope, model_info, cache, *, labeler, options=None, counter=None)` returns:

| Key | Meaning |
|-----|---------|
| `input_tokens` | Tokens that the run sends: periods not in the cache, plus the reduce call. |
| `cached_tokens` | Tokens of the periods that are in the period cache. The run does not send them. |
| `output_tokens` | Reply tokens, from the options, not more than the output limits. |
| `cost_usd` | Cost of the calls at `ModelInfo` prices. The long-context rule applies to each call. |
| `periods` | Number of periods. |
| `periods_cached` | Number of periods in the cache. |

`labeler` is necessary: the speaker names are part of the cache key.
An empty scope gives zeros. `estimate` raises the same `ConfigError` as `run_report` when the model limits leave no room.
The engine has no spend limit of its own. Call `estimate` and ask the user before a run.

## Store

`ReportStore(root)` keeps one folder for each conversation and one JSON file for each run:

```
<root>/<conversation folder>/<report id>.json
```

- `save(report)` writes the file in one step. It raises `StoreError` when the id exists. It never replaces a report.
- `list_reports(conversation=None)` gives `ReportEntry` items, newest first.
- `load_report(conversation, report_id)` gives a `Report`, or raises `ReportNotFound`.
- `latest_report(conversation)` gives a `Report` or `None`.

Report files have mode `0600`. New folders have mode `0700`.

`render_markdown(report)` gives the layout that the current Reports page and server read:

- Line 1: `# Conversation Analysis - <conversation id>`
- `## Overview`. Its first line is the scope line: `- Scope: 2024-06-01 to 2024-06-30`, `- Scope: from 2024-06-01`, `- Scope: through 2024-06-30` or `- Scope: all dates`.
- `## Key Events`. One `### <date> - <title>` block for each moment. The section is empty when the report has no moment.
- `## Summary`. It is the last section and holds the four sections under `###` headings.

In this layout `#`, `[`, `]` and `\` in titles, details, quotes and warnings have a backslash before them.
Text from the model or from a message cannot make a heading, a link or an image.

`write_text_atomic(path, text)` writes a text file in one step.

## Wiring in `ui_server.py`

This package does not change the server. The wiring step must do these things.

1. Make one `ModelInfo` for each model that the UI offers. Today the numbers are in `ai_config.py` and `cli.py`.
2. Make a labeler from `resolve_contact_names`. Return the contact's name for the empty sender of a one-to-one chat.
3. In `run_job`, action `analyze`, replace the call to `run_cli`:
   1. Make the source: `FrameSource` on the cached frame, or `ArchiveSource` when the archive is in place.
   2. Call `run_report`.
   3. Call `ReportStore.save`.
   4. Until the Reports page reads JSON, write the markdown file: `write_text_atomic(out_dir / f"analysis_{sanitize_filename(contact)}.md", render_markdown(report))`.
4. Make the events for `build_analysis_payload` from `report.moments`. The engine does not write `events_timeline_*.csv`.
5. Send `ProgressEvent` to the job progress. Keep one `CancelToken` for each job and add a route that calls `cancel()`.
6. Catch `RunError`, `EmptyConversationError`, `ConfigError` and `SourceError`. All of them are kinds of `ReportEngineError`. After the first model call a run ends only with a `RunError`. Record `error.usage` of a `RunError`. Show `report.warnings` when `status` is `partial`.
7. In `preview_payload`, replace `estimate_cost` with `estimate`. Give it the same labeler as the run.
8. Serve the JSON to the Reports page: the list from `list_reports`, one report from `load_report(...).model_dump(mode="json")`.
9. Give `since` and `until` as days. The engine includes the whole `until` day. Today's server compares against midnight and drops that day.

The engine does not write the HTML report of `html_report.py`.

## Tests

```bash
venv/bin/python -m pytest tests/report_engine --cov=report_engine --cov-report=term-missing
```

`tests/report_engine/conftest.py` sets a dummy `OPENAI_API_KEY` before any import, removes proxy and address settings from the environment, and blocks all sockets that do not go to the local machine.
The tests use invented conversations and `FakeLLM` only.
